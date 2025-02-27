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

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

    import openai


class LLM:
    """Language model that predicts continuation of provided source code."""

    input_tokens: int = 0
    output_tokens: int = 0
    lock: threading.Lock = threading.Lock()

    def __init__(self, model: openai.OpenAI, log_path: pathlib.Path) -> None:
        """Initialize a new LLM."""
        self._model = model
        self._log_path = log_path

    def draw_samples(self, indices: list[int], prompt: str) -> list[tuple[int, str]]:
        """Draw num_samples samples from the language model, given a prompt.

        The indices are used for logging and must be unique across threads.
        You'll want to draw several samples to decrease billing, because input-token are
        only billed once per sample-run.
        """
        # Keep sampling until we get a response
        while True:
            try:
                response = self._model.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "developer",
                            "content": "You are a helpful coding assistant who only responds with code "
                            "and no markdown-formatting.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    n=len(indices),
                )
                with self.lock:
                    if response.usage is not None:
                        self.input_tokens += response.usage.prompt_tokens
                        self.output_tokens += response.usage.completion_tokens
                print(len(indices) == len(response.choices))
                outputs = list(zip(indices, [choice.message.content or "" for choice in response.choices]))
                break
            except Exception as e:
                print("Retrying LLM call after error:", e)  # noqa: T201

        self._log(outputs, prompt)

        return outputs

    def _log(self, outputs: list[tuple[int, str]], prompt: str) -> None:
        """Log prompt and response to file.

        The index must be unique across thread.
        """
        if self._log_path:
            for index, response in outputs:
                call_data_folder = self._log_path / f"{index}"
                call_data_folder.mkdir(exist_ok=True)
                # Note that we log the prompt for every index, even
                # though their prompts are identical.
                with (call_data_folder / "prompt.log").open("a") as f:
                    f.write(prompt)
                with (call_data_folder / "response.log").open("a") as f:
                    f.write(str(response))
