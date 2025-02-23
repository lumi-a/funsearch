# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A single-threaded implementation of the FunSearch pipeline."""

import logging
import queue
import threading
import time
from concurrent import futures
from typing import TYPE_CHECKING

from funsearch import code_manipulation
from funsearch.evaluator import Evaluator
from funsearch.sampler import LLM, Sampler

if TYPE_CHECKING:
  from funsearch.programs_database import ProgramsDatabase


def extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(code_manipulation.yield_decorated(specification, "funsearch", "run"))
  if len(run_functions) != 1:
    msg = "Expected 1 function decorated with `@funsearch.run`."
    raise ValueError(msg)
  evolve_functions = list(code_manipulation.yield_decorated(specification, "funsearch", "evolve"))
  if len(evolve_functions) != 1:
    msg = "Expected 1 function decorated with `@funsearch.evolve`."
    raise ValueError(msg)
  return evolve_functions[0], run_functions[0]


def sampler_runner(sampler: Sampler, iterations: int) -> None:
  try:
    if iterations < 0:
      while True:
        sampler.sample()
    else:
      for _ in range(iterations):
        sampler.sample()
  except KeyboardInterrupt:
    logging.info("Keyboard interrupt in sampler thread.")


# We pass in llm_name because there doesn't seem to be a good way of getting the class of
# a model from its string. You could do:
# |  a = get_model(llm_name)
# |  b = a.__class__(a.model_id)
# but it feels like we're lying to the caller there, who passes an instance of Model.
# We could ask the caller to pass a class, but then we'd *also* need them to ask for the id.
# So let's just ask them for the id directly and be a bit inefficient upfront.
def run(database: "ProgramsDatabase", llm_name: str, iterations: int = -1) -> None:
  """Launches a FunSearch experiment in parallel using threads."""
  database.print_status()

  # Stores (program, island_id, version_generated, index) per LLM-call
  llm_responses: queue.Queue[tuple[str, int, int, int]] = queue.Queue()
  analysation_results: queue.Queue[tuple[code_manipulation.Function, dict[float | int | str, float]]] = (
    queue.Queue()
  )

  # Keep track of how many llm requests you made, to not
  # exceed `iterations` (TODO: rename parameter, also on callsites of `run`)
  # and to pass to the llm-prompting to make logging to files safe
  llm_prompt_index = 0
  llm_prompt_index_lock = threading.Lock()

  # The maximum size of the llm_responses queue
  dynamic_max_queue_size = 30
  dynamic_max_queue_lock = threading.Lock()

  def llm_response_worker(stop_event: threading.Event, llm: LLM) -> None:
    """Worker thread that continuously makes web requests as long as we haven't reached `iterations`.

    Waits if the output queue has size >= dynamic_max_queue_size.
    """
    global llm_prompt_index

    while not stop_event.is_set():
      # Check if we've reached the maximum number of web requests (if applicable)
      with llm_prompt_index_lock:
        if iterations != -1 and llm_prompt_index >= iterations:
          break
        current_index = llm_prompt_index
        llm_prompt_index += 1

      # Check dynamic max queue size and wait if needed
      with dynamic_max_queue_lock:
        current_max = dynamic_max_queue_size
      while llm_responses.qsize() >= current_max and not stop_event.is_set():
        time.sleep(0.1)

      # Perform the web request and enqueue the result
      prompt = database.get_prompt()
      # TODO: Is this not blocking?
      sample = llm.draw_sample(prompt.code, current_index)
      llm_responses.put((sample, prompt.island_id, prompt.version_generated, current_index))

    logging.info("LLM response worker stopped.")

  def analysation_dispatcher(
    stop_event: threading.Event, executor: futures.ProcessPoolExecutor, evaluator: Evaluator
  ) -> None:
    """Dispatcher thread that pulls web results from the queue and submits CPU tasks to the process pool.

    Completed tasks have their results pushed into `analysation_results`
    """
    while not stop_event.is_set():
      try:
        sample, island_id, version_generated, current_index = llm_responses.get(timeout=0.1)
      except queue.Empty:
        with llm_prompt_index_lock:
          # If `iterations` is reached, we can exit now
          # TODO: Do we need to check llm_responses.empty() here again?
          if iterations != -1 and llm_prompt_index >= iterations and llm_responses.empty():
            break
        continue

      future = executor.submit(evaluator.analyse, sample, version_generated, current_index)
      future.add_done_callback(lambda fut: analysation_results.put(fut.result()))

    logging.info("Analysation-dispatcher exiting.")

  # TODO: Consider passing `max_workers=os.cpu_count()` to ProcessPoolExecutor.
  # This might help because the cpu-heavy task involves a subprocess-call itself.
  with futures.ProcessPoolExecutor() as executor:
    pass


# if scores_per_test:
#       self._database.register_program(new_function, island_id, scores_per_test)
#     elif island_id is not None:
#       self._database._register_failure(island_id)
