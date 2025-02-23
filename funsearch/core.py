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

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
import time
from concurrent import futures
from typing import TYPE_CHECKING

import llm

from funsearch import code_manipulation
from funsearch.evaluator import Evaluator
from funsearch.sampler import LLM, Sampler
from funsearch.sandbox import ExternalProcessSandbox

if TYPE_CHECKING:
  from funsearch.programs_database import ProgramsDatabase


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
# So let's just ask them for the id directly and be a bit inefficient upfront. This might
# make errors uglier, though.
def run(database: "ProgramsDatabase", llm_name: str, log_path: Path, iterations: int = -1) -> None:
  """Launches a FunSearch experiment in parallel using threads."""
  database.print_status()

  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")

  # Stores (program, island_id, version_generated, index) per LLM-call
  llm_responses: queue.Queue[tuple[str, int, int, int]] = queue.Queue()
  # The maximum size of the llm_responses queue.
  # Increasing this might make the program a little faster (because the queue is
  # less likely to be empty), but it also means that function-improvements are
  # only transferred to the LLM every `max_stored_responses` iterations, which
  # degrades the quality of the algorithm.
  max_stored_responses = 20

  # Keep track of how many llm requests you made, to not
  # exceed `iterations` (TODO: rename parameter, also on callsites of `run`)
  # and to pass to the llm-prompting to make logging to files safe
  llm_prompt_index = 0
  llm_prompt_index_lock = threading.Lock()

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
      while llm_responses.qsize() >= max_stored_responses and not stop_event.is_set():
        time.sleep(0.1)

      # Perform the web request and enqueue the result
      prompt = database.get_prompt()
      # TODO: Is this not-blocking?
      sample = llm.draw_sample(prompt.code, current_index)
      llm_responses.put((sample, prompt.island_id, prompt.version_generated, current_index))

    logging.info("LLM response worker stopped.")

  def analysation_dispatcher(stop_event: threading.Event, executor: futures.ProcessPoolExecutor) -> None:
    """Dispatcher thread that pulls web results from the queue and submits CPU tasks to the process pool."""
    # TODO: What a waste of an evaluator and sandbox. Can't we have one per thread?
    evaluator = Evaluator(
      ExternalProcessSandbox(log_path),
      database.template,
      database.function_to_evolve,
      database.function_to_run,
      database.inputs,
    )
    while not stop_event.is_set():
      try:
        sample, island_id, version_generated, current_index = llm_responses.get(timeout=0.1)
      except queue.Empty:
        with llm_prompt_index_lock:
          # If `iterations` is reached, we can exit now
          # TODO: Do we really need to check llm_responses.empty() here again?
          if iterations != -1 and llm_prompt_index >= iterations and llm_responses.empty():
            break
        continue

      future = executor.submit(evaluator.analyse, sample, version_generated, current_index)

      # TODO: How much work is it to update the database after every successful function-call?
      # If it's a problem, we could instead implement another producer-consumer structure.
      def on_complete(
        future: futures.Future[tuple[code_manipulation.Function, dict[float | str, float]]],
        island_id: int = island_id,
      ) -> None:
        # See https://docs.python-guide.org/writing/gotchas/#late-binding-closures for why
        # we have island_id as an optional argument here.
        new_function, scores_per_test = future.result()
        if scores_per_test:
          database.register_program(new_function, island_id, scores_per_test)
        elif island_id is not None:
          database.register_failure(island_id)

      future.add_done_callback(on_complete)

    logging.info("Analysation-dispatcher exiting.")

  stop_event = threading.Event()

  # TODO: Consider passing `max_workers=os.cpu_count()` to ProcessPoolExecutor.
  # This might help because the cpu-heavy task involves a subprocess-call itself.
  database.print_status()
  with futures.ProcessPoolExecutor() as executor:
    # Start web request worker threads.
    num_llm_workers = 5
    llm_threads: list[threading.Thread] = []
    for _ in range(num_llm_workers):
      model = llm.get_model(llm_name)
      t = threading.Thread(target=llm_response_worker, args=(stop_event, LLM(model, log_path)))
      t.start()
      llm_threads.append(t)

    # Start the dispatcher thread.
    dispatcher_thread = threading.Thread(target=analysation_dispatcher, args=(stop_event, executor))
    dispatcher_thread.start()
    try:
      # Wait for web request workers to finish (if M is finite, they will eventually stop)
      for t in llm_threads:
        t.join(timeout=10)
        database.print_status()

      # Wait until the llm_responses queue is empty (i.e. all llm-requests have been dispatched)
      while not llm_responses.empty():
        time.sleep(0.1)

      # Signal the dispatcher, database updater, and adjuster threads to stop
      stop_event.set()

      dispatcher_thread.join()

    except KeyboardInterrupt:
      logging.info("KeyboardInterrupt received, shutting down.")
      stop_event.set()
      for t in llm_threads:
        t.join()
      dispatcher_thread.join()

  database.print_status()
  database.backup()
