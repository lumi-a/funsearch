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

"""Class for sampling new programs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from funsearch.evaluator import Evaluator

if TYPE_CHECKING:
  import pathlib
  from collections.abc import Sequence

  import llm

  from funsearch import programs_database


# TODO: This is currently unused, but I feel like we should use it again.
def reformat_to_two_spaces(code: str) -> str:
  # Regular expression to match leading spaces at the beginning of each line
  pattern = r"^\s+"

  # Function to replace leading spaces with two spaces per indentation level
  def replace_with_two_spaces(match: re.Match[str]) -> str:
    space_count = len(match.group(0))
    return " " * (2 * (space_count // 4))  # Assumes original indentation was 4 spaces

  # Split the code into lines, reformat each line, and join back into a single string
  reformatted_lines = [re.sub(pattern, replace_with_two_spaces, line) for line in code.splitlines()]
  return "\n".join(reformatted_lines)


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, model: llm.Model, log_path: pathlib.Path) -> None:
    """Initialize a new LLM."""
    self._model = model
    self._log_path = log_path

  def draw_sample(self, prompt: str, index: int) -> str:
    """Draw a sample from the language model, given a prompt.

    The index is used for logging and must be unique across threads.
    """
    # TODO: We could provide a temperature here, see
    # https://llm.datasette.io/en/stable/python-api.html#model-options
    try:
      output_text = self._model.prompt(prompt).text()
    except Exception as e:
      print("LLM call failed:", e)  # noqa: T201
      output_text = ""

    # TODO: Move this elsewhere. Being able to see the whole
    # llm-response in the logs is useful.
    match = re.search(r"(```(python|))(.*?)```", output_text, re.DOTALL)
    response = match.group(3) if match else output_text

    self._log(prompt, response, index)

    return response

  def _log(self, prompt: str, response: str, index: int) -> None:
    """Log prompt and response to file.

    The index must be unique across thread.
    """
    if self._log_path:
      with (self._log_path / f"prompt_{index}.log").open("a") as f:
        f.write(prompt)
      with (self._log_path / f"response_{index}.log").open("a") as f:
        f.write(str(response))


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
    self, database: programs_database.ProgramsDatabase, evaluators: Sequence[evaluator.Evaluator], model: LLM
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = model

  def sample(self) -> None:
    """Continuously gets prompts, samples programs, sends them for analysis."""
    prompt = self._database.get_prompt()
    samples = self._llm.draw_samples(prompt.code)

    # This loop can be executed in parallel on remote evaluator machines.
    for sample in samples:
      chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
      chosen_evaluator.analyse(sample, prompt.island_id, prompt.version_generated)


# if scores_per_test:
#       self._database.register_program(new_function, island_id, scores_per_test)
#     elif island_id is not None:
#       self._database._register_failure(island_id)
