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
import threading

from funsearch import code_manipulation
from funsearch.programs_database import ProgramsDatabase
from funsearch.sampler import Sampler


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
    code_manipulation.yield_decorated(specification, "funsearch", "run")
  )
  if len(run_functions) != 1:
    raise ValueError("Expected 1 function decorated with `@funsearch.run`.")
  evolve_functions = list(
    code_manipulation.yield_decorated(specification, "funsearch", "evolve")
  )
  if len(evolve_functions) != 1:
    raise ValueError("Expected 1 function decorated with `@funsearch.evolve`.")
  return evolve_functions[0], run_functions[0]


def sampler_runner(sampler: Sampler, iterations: int):
  try:
    if iterations < 0:
      while True:
        sampler.sample()
    else:
      for _ in range(iterations):
        sampler.sample()
  except KeyboardInterrupt:
    logging.info("Keyboard interrupt in sampler thread.")


def run(samplers: list[Sampler], database: ProgramsDatabase, iterations: int = -1):
  """Launches a FunSearch experiment in parallel using threads."""
  threads = []

  for sampler in samplers:
    t = threading.Thread(target=sampler_runner, args=(sampler, iterations))
    t.daemon = True
    t.start()
    threads.append(t)

  try:
    # If iterations is finite, wait for all threads to finish.
    # Otherwise, just keep the main thread alive.
    if iterations > 0:
      for t in threads:
        t.join()
    else:
      while True:
        t.join(timeout=1)

  except KeyboardInterrupt:
    logging.info("Keyboard interrupt. Stopping all sampler threads.")
  finally:
    database.backup()
