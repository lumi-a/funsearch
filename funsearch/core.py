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
import os
import queue
import threading
import time
from typing import TYPE_CHECKING

import llm

from funsearch.evaluator import Evaluator
from funsearch.sampler import LLM
from funsearch.sandbox import ExternalProcessSandbox

if TYPE_CHECKING:
  from pathlib import Path

  from funsearch.programs_database import ProgramsDatabase


class IterationManager:
  """Keeps track of how many iterations we've run so far."""

  def __init__(self, max_iterations: int) -> None:
    """Keeps track of how many iterations we've run so far."""
    self._max_iterations = max_iterations
    # Keep track of how many llm requests you made, to not
    # exceed `max_iterations` (TODO: rename parameter, also on callsites of `run`)
    # and to pass to the llm-prompting to make logging to files safe
    self._index = 0
    self._index_lock = threading.Lock()

  def get_next_index(self) -> int | None:
    """Returns the next index, or None if we're done."""
    with self._index_lock:
      if self._max_iterations != -1 and self._index >= self._max_iterations:
        return None
      index = self._index
      self._index += 1
    return index

  def is_done(self) -> bool:
    return self._max_iterations != -1 and self._index >= self._max_iterations


# TODO: Instead of giving each Database a lock, consider writing a DbManager class
# that implements the locking outside of the Database-class.


# We pass in llm_name because there doesn't seem to be a good way of getting the class of
# a model from its string. You could do:
# |  a = get_model(llm_name)
# |  b = a.__class__(a.model_id)
# but it feels like we're lying to the caller there, who passes an instance of Model.
# We could ask the caller to pass a class, but then we'd *also* need them to ask for the id.
# So let's just ask them for the id directly and be a bit inefficient upfront. This might
# make errors uglier, though.
def run(database: ProgramsDatabase, llm_name: str, log_path: Path, iterations: int = -1) -> None:
  """Launches a FunSearch experiment in parallel using threads."""
  database.print_status()

  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")

  # Stores (program, island_id, version_generated, index) per LLM-call
  # None is a sentinel-value to signal to the analysation-workers that
  # no more values will be added.
  llm_responses: queue.Queue[None | tuple[str, int, int, int]] = queue.Queue()
  # Increasing the semaphore-value might make the program a little faster
  # (because the queue is less likely to be empty when an analyser tries to
  # get a sample from it), but it also means that function-improvements are
  # only transferred to the LLM ~every `value` iterations, which
  # degrades the quality of the algorithm.
  llm_responses_slots = threading.Semaphore(32)

  def llm_response_worker(iteration_manager: IterationManager, stop_event: threading.Event, llm: LLM) -> None:
    """Worker thread that continuously makes web requests as long as we haven't reached `iterations`.

    Waits if the output queue has size >= dynamic_max_queue_size.
    """
    while not (stop_event.is_set() or iteration_manager.is_done()):
      # Acquire a queue slot
      while not stop_event.is_set():
        if llm_responses_slots.acquire(timeout=0.1):
          break
      if stop_event.is_set():
        break

      current_index = iteration_manager.get_next_index()
      if current_index is None:
        break

      prompt = database.get_prompt()
      sample = llm.draw_sample(prompt.code, current_index)
      llm_responses.put((sample, prompt.island_id, prompt.version_generated, current_index))

  def analysation_dispatcher(stop_event: threading.Event) -> None:
    """Dispatcher thread that pulls web results from the queue and analyses the results."""
    evaluator = Evaluator(
      ExternalProcessSandbox(log_path),
      database.template,
      database.function_to_evolve,
      database.function_to_run,
      database.inputs,
    )
    while not stop_event.is_set():
      try:
        queue_item = llm_responses.get(timeout=0.1)
      except queue.Empty:
        continue

      if queue_item is None:
        break

      sample, island_id, version_generated, current_index = queue_item

      new_function, scores_per_test = evaluator.analyse(sample, version_generated, current_index)

      if scores_per_test:
        database.register_program(new_function, island_id, scores_per_test)
      elif island_id is not None:
        database.register_failure(island_id)

      llm_responses_slots.release()

  def database_printer(stop_event: threading.Event) -> None:
    while True:
      for _ in range(10):
        time.sleep(1)
        if stop_event.is_set():
          return
      database.print_status()

  stop_event = threading.Event()
  iteration_manager = IterationManager(iterations)

  # Start web request worker threads.
  num_llm_workers = os.cpu_count() * 2
  llm_threads: list[threading.Thread] = [
    threading.Thread(
      target=llm_response_worker, args=(iteration_manager, stop_event, LLM(llm.get_model(llm_name), log_path))
    )
    for _ in range(num_llm_workers)
  ]
  for t in llm_threads:
    t.start()

  # Start analysation dispatcher threads
  num_dispatcher_workers = os.cpu_count()
  dispatcher_threads: list[threading.Thread] = [
    threading.Thread(target=analysation_dispatcher, args=(stop_event,)) for _ in range(num_dispatcher_workers)
  ]
  for t in dispatcher_threads:
    t.start()

  db_printer_thread = threading.Thread(target=database_printer, args=(stop_event,))
  db_printer_thread.start()

  try:
    # Wait for web request workers to finish (if iterations is finite, they will eventually stop)
    while any(t.is_alive() for t in llm_threads):
      for t in llm_threads:
        t.join(timeout=0.1)

    for _ in range(num_dispatcher_workers):
      llm_responses.put(None)

    # Wait for dispatcher threads to finish now
    while any(t.is_alive() for t in dispatcher_threads):
      for t in dispatcher_threads:
        t.join(timeout=0.1)

    # Signal the dispatcher, database updater, and adjuster threads to stop.
    # This should be redundant, anyway.
    stop_event.set()

  except KeyboardInterrupt:
    print("Stopping threads...")  # noqa: T201
    stop_event.set()

    print("Waiting for requests to finish...")  # noqa: T201
    while any(t.is_alive() for t in llm_threads):
      for t in llm_threads:
        t.join(timeout=0.1)

    print(f"Analysing {llm_responses.qsize()} remaining responses...")  # noqa: T201
    # TODO: The KeyboardInterrupt is forwarded to the subprocesses, so they always fail here.
    while any(t.is_alive() for t in dispatcher_threads):
      for t in dispatcher_threads:
        t.join(timeout=0.1)
  finally:
    db_printer_thread.join()
    database.print_status()
    database.backup()
