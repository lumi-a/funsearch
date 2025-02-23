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
from typing import TYPE_CHECKING

from funsearch import code_manipulation
from funsearch.sampler import Sampler

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


def run(samplers: list[Sampler], database: "ProgramsDatabase", iterations: int = -1) -> None:
  """Launches a FunSearch experiment in parallel using threads."""
  database.print_status()

  llm_responses = queue.Queue()
  analysation_results = queue.Queue()

  # Keep track of how many llm requests you made, to not
  # exceed `iterations` (TODO: rename parameter, also on callsites of `run`)
  # and to pass to the llm-prompting to make logging to files safe
  llm_prompt_index = 0
  llm_prompt_index_lock = threading.Lock()

  # The maximum size of the llm_responses queue
  dynamic_max_queue_size = 30
  dynamic_max_queue_lock = threading.Lock()
