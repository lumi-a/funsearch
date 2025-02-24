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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import pathlib

  import llm


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
    # Keep sampling until we get a response
    while True:
      try:
        output_text = self._model.prompt(prompt).text()
        break
      except Exception as e:
        print("Retrying LLM call after", e)  # noqa: T201

    self._log(prompt, output_text, index)

    return output_text

  def _log(self, prompt: str, response: str, index: int) -> None:
    """Log prompt and response to file.

    The index must be unique across thread.
    """
    if self._log_path:
      call_data_folder = self._log_path / f"{index}"
      call_data_folder.mkdir(exist_ok=True)
      with (call_data_folder / "prompt.log").open("a") as f:
        f.write(prompt)
      with (call_data_folder / "response.log").open("a") as f:
        f.write(str(response))
