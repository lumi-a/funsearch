"""Sandboxes for code-execution."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import cloudpickle

CONTAINER_MAIN = (Path(__file__).parent / "container" / "main.py").absolute()

IMAGE_NAME = "funsearch_sandbox"


def _compile_code(program: str) -> dict:
  namespace = {}

  parsed_code = ast.parse(program)
  compiled_code = compile(parsed_code, filename="<ast>", mode="exec")
  exec(compiled_code, namespace)
  return namespace


class ExternalProcessSandbox:
  """Sandbox that executes the code in a separate Python process in the same host.

  Note: This does not provide real safety and should be only used in an environment where the host process is
  in some kind of safe sandbox itself (i.e., a container).
  This kind of sandbox merely makes it more probable that single invalid call does not break the whole
  funsearch algorithm. It might be easier to set up and thus nice environment to tune the prompts and other
  code.
  """

  def __init__(self, base_path: Path, timeout_secs: int = 30) -> None:
    self.output_path = Path(base_path) / f"sandbox{self.id}"
    self.timeout_secs = timeout_secs

    self.input_path = self.output_path / "inputs"
    for p in [self.output_path, self.input_path]:
      if not p.exists():
        p.mkdir(parents=True)

  def _exec(self, call_data_path: Path, input_path: Path, error_file_path: Path) -> int:
    """Execute python-code in a separate process.

    - The main.py shall execute the LLM generated method from program.pickle file providing
      input.pickle as the input for the method.
    - main.py writes the output of the method into output.pickle.
    Everything except the /workspace folder will be read-only so that the environment remains good
    for future runs.
    """
    import subprocess

    program_path = call_data_path / "program.pickle"
    output_file = call_data_path / "output.pickle"

    cmd = [
      "uv",
      "run",
      "python",
      str(CONTAINER_MAIN),
      str(program_path),
      str(input_path),
      str(output_file),
    ]

    logging.debug(f"Executing {cmd}")
    timeout = int(self.timeout_secs)
    try:
      result: subprocess.CompletedProcess = subprocess.run(  # noqa: S603
        cmd, timeout=timeout, stderr=error_file_path, check=False
      )
    except subprocess.TimeoutExpired:
      logging.debug(f"Command timed out after {timeout} seconds")
      return 1
    except Exception as e:  # noqa: BLE001
      logging.debug(f"Command failed with error: {e}")
      return 1
    else:
      return result.returncode

  def run(self, program: str, function_to_run: str, input: str | float, index: int) -> float | None:
    """Executes the code in a separate process.

    Returns the output of the code if it was successful and could be converted to a float,
    None otherwise.
    """
    call_data_folder = (self.output_path / f"call-{index}").absolute()
    if not call_data_folder.exists():
      call_data_folder.mkdir()

    input_hash = hash(input)  # Good enough
    input_path = (self.input_path / f"{input_hash}.pickle").absolute()

    if not input_path.exists():
      with Path(input_path).open("wb") as f:
        cloudpickle.dump(input, f)

    try:
      namespace = _compile_code(program)

      with (call_data_folder / "program.pickle").open("wb+") as f:
        cloudpickle.dump(namespace[function_to_run], f)

      error_file = self.output_path / f"stderr-{index}.log"

      return_code = self._exec(call_data_folder, input_path, error_file)

      if return_code != 0:
        return False

      with (call_data_folder / "output.pickle").open("rb") as f:
        out = cloudpickle.load(f)
        try:
          return float(out)
        except ValueError:
          return None

    except Exception as e:
      logging.debug(f"Could not execute code: {e}")

    return None, False
