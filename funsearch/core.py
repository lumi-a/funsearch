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

import os
import queue
import threading
import time
from typing import TYPE_CHECKING

from mistralai import Mistral

from funsearch.sampler import LLM

if TYPE_CHECKING:
    from pathlib import Path

    from funsearch.programs_database import ProgramsDatabase


class IterationManager:
    """Manages iteration indices, batching, and the response queue.

    Ensures that every draw_samples call receives a full batch (except the last batch)
    and that the total queued samples do not exceed max_cached_samples.
    """

    def __init__(self, max_iterations: int, batch_size: int, max_cached_samples: int) -> None:
        self._max_iterations = max_iterations  # Total iterations (-1 for unbounded)
        if batch_size > max_cached_samples:
            msg = f"max_cached_samples ({max_cached_samples}) must be at least batch_size ({batch_size})"
            raise ValueError(msg)
        self._batch_size = batch_size  # Fixed batch size for each call
        self._max_cached_samples = max_cached_samples  # Maximum samples allowed in the queue
        self._index = 0  # Next iteration index
        self._pending_count = 0  # Count of samples currently enqueued (pending)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        # Internal queue holding results, each item is a tuple (sample, island_id, version_generated, index)
        self._queue: queue.Queue[tuple[str, int, int, int]] = queue.Queue()

    def reserve_batch(self) -> list[int]:
        """Wait until there's room for an entire batch, then reserve batch_size slots.

        Returns a list of indices reserved for this batch.
        If there are fewer iterations remaining, returns only those indices.
        """
        with self._cv:
            while self._pending_count + self._batch_size > self._max_cached_samples:
                self._cv.wait()
            # Reserve full batch slots.
            self._pending_count += self._batch_size

            # Determine the actual batch size based on remaining iterations.
            if self._max_iterations != -1:
                remaining = self._max_iterations - self._index
                actual_batch_size = min(self._batch_size, remaining)
            else:
                actual_batch_size = self._batch_size

            if actual_batch_size <= 0:
                # No work remains; release reserved tokens.
                self._pending_count -= self._batch_size
                self._cv.notify_all()
                return []

            indices = list(range(self._index, self._index + actual_batch_size))
            self._index += actual_batch_size

            # If the final batch is smaller, release the extra reserved slots.
            if actual_batch_size < self._batch_size:
                self._pending_count -= self._batch_size - actual_batch_size
                self._cv.notify_all()
            return indices

    def enqueue_batch(self, samples: list[tuple[int, str]], island_id: int, version_generated: int) -> None:
        """Enqueues each sample from a batch with its associated prompt information and unique index.

        The prompt_info is a tuple (island_id, version_generated).
        """
        for idx, sample in samples:
            self._queue.put((sample, island_id, version_generated, idx))

    def get_sample(self, timeout: float = 0.1) -> None | tuple[str, int, int, int]:
        """Attempt to retrieve a sample from the queue. Return the sample tuple if available, or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def release_sample(self) -> None:
        """Called after a sample is processed, reducing the pending count and notifying waiting threads."""
        with self._cv:
            self._pending_count -= 1
            self._cv.notify_all()

    def is_done(self) -> bool:
        """Returns True if no more iterations need to be processed."""
        with self._lock:
            return (
                self._max_iterations != -1
                and self._index >= self._max_iterations
                and self._pending_count == 0
                and self._queue.empty()
            )


# We pass in llm_name because there doesn't seem to be a good way of getting the class of
# a model from its string. You could do:
# |  a = get_model(llm_name)
# |  b = a.__class__(a.model_id)
# but it feels like we're lying to the caller there, who passes an instance of Model.
# We could ask the caller to pass a class, but then we'd *also* need them to ask for the id.
# So let's just ask them for the id directly and be a bit inefficient upfront. This might
# make errors uglier, though.
def run(
    database: ProgramsDatabase, output_path: Path, timestamp: int, num_samples: int = -1, num_samples_per_call: int = 4
) -> None:
    """Launches a FunSearch experiment in parallel using threads."""
    database.print_status()

    log_path = output_path / database._config.problem_name / str(timestamp)
    backup_dir = output_path / "backups"
    log_path.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    max_cached_samples = 24  # TODO: Allow configuring this as a parameter
    stop_event = threading.Event()
    iteration_manager = IterationManager(num_samples, num_samples_per_call, max_cached_samples)
    database_lock = threading.Lock()

    def llm_response_worker(iteration_manager: IterationManager, stop_event: threading.Event, llm: LLM) -> None:
        """Worker thread that continuously makes web requests as long as we haven't reached `iterations`.

        Waits if the output queue has size >= dynamic_max_queue_size.
        """
        while not stop_event.is_set():
            indices = iteration_manager.reserve_batch()
            if not indices:
                break  # No more work.
            with database_lock:
                prompt = database.get_prompt()
            samples: list[tuple[int, str]] = llm.draw_samples(indices, prompt.code)
            iteration_manager.enqueue_batch(samples, prompt.island_id, prompt.version_generated)

    def analysation_dispatcher(stop_event: threading.Event) -> None:
        """Dispatcher thread that pulls web results from the queue and analyses the results."""
        with database_lock:
            evaluator = database.construct_evaluator(log_path)
        while True:
            maybe_sample = iteration_manager.get_sample(timeout=0.1)
            if maybe_sample is None:
                # Does the empty queue mean all iterations have been processed?
                if iteration_manager.is_done() or (stop_event.is_set() and iteration_manager._pending_count == 0):
                    break
                # Otherwise, try getting another sample.
                continue

            sample, island_id, version_generated, current_index = maybe_sample
            new_function, scores_per_test = evaluator.analyse(sample, version_generated, current_index)

            with database_lock:
                if scores_per_test:
                    database.register_program_in_island(new_function, island_id, scores_per_test)
                elif island_id is not None:
                    database.register_failure(island_id)

            iteration_manager.release_sample()

            with database_lock:
                # TODO: Add as parameter
                if current_index % 500 == 0:
                    backup_file = backup_dir / f"{database._config.problem_name}_{timestamp}_{current_index}.pickle"
                    database.backup(backup_file)

    def database_printer(stop_event: threading.Event) -> None:
        while True:
            # Wait 10 seconds
            for _ in range(20):
                time.sleep(0.5)
                if stop_event.is_set():
                    return
            with database_lock:
                print(iteration_manager._queue.qsize(), iteration_manager._pending_count, end=" | ")
                database.print_status()

    # Start web request worker threads.
    num_llm_workers = max_cached_samples // num_samples_per_call
    llm_threads: list[threading.Thread] = [
        threading.Thread(
            target=llm_response_worker,
            args=(iteration_manager, stop_event, LLM(Mistral(api_key=os.environ["MISTRAL_API_KEY"]), log_path)),
        )
        for _ in range(num_llm_workers)
    ]
    for t in llm_threads:
        t.start()

    # Start analysation dispatcher threads
    num_dispatcher_workers = os.cpu_count() or 8
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

        print(f"Waiting for {iteration_manager._pending_count} requests to finish...")  # noqa: T201
        while any(t.is_alive() for t in llm_threads):
            for t in llm_threads:
                t.join(timeout=0.1)

        print(f"Analysing {iteration_manager._queue.qsize()} remaining responses...")  # noqa: T201
        # TODO: The KeyboardInterrupt is forwarded to the subprocesses, so they always fail here.
        while any(t.is_alive() for t in dispatcher_threads):
            for t in dispatcher_threads:
                t.join(timeout=0.1)
    finally:
        db_printer_thread.join()
        # Shouldn't be necessary to acquire this lock anymore, but just to be safe:
        with database_lock:
            database.print_status()
            backup_file = backup_dir / f"{database._config.problem_name}_{timestamp}.pickle"
            database.backup(backup_file)
            print(f"Database backed up to: {backup_file}")
