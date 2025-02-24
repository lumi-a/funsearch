"""Sandboxes for code-execution."""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import cloudpickle

CONTAINER_MAIN = (Path(__file__).parent / "container" / "main.py").absolute()

IMAGE_NAME = "funsearch_sandbox"


def _compile_code(program: str) -> dict:
  namespace = {}

  parsed_code = ast.parse(program)
  compiled_code = compile(parsed_code, filename="<ast>", mode="exec")
  exec(compiled_code, namespace)  # noqa: S102
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
    """Create a new sandbox that logs to `base_path` and runs a sample for at most `timeout_secs`."""
    self.output_path = Path(base_path) / "sandbox"
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
      sys.executable,
      str(CONTAINER_MAIN),
      str(program_path),
      str(input_path),
      str(output_file),
    ]

    logging.debug(f"Executing {cmd}")
    try:
      result: subprocess.CompletedProcess = subprocess.run(  # noqa: S603
        cmd, timeout=self.timeout_secs, stderr=error_file_path, check=False
      )
    except subprocess.TimeoutExpired:
      logging.debug(f"Command timed out after {self.timeout_secs} seconds")
      return 1
    except Exception as e:  # noqa: BLE001
      logging.debug(f"Command failed with error: {e}")
      return 1
    else:
      return result.returncode

  def run(self, program: str, function_to_run: str, input_to_run: str | float, index: int) -> float | None:
    """Executes the code in a separate process.

    Returns the output of the code if it was successful and could be converted to a float,
    None otherwise.
    """
    call_data_folder = (self.output_path / f"call_{index}").absolute()
    if not call_data_folder.exists():
      call_data_folder.mkdir()

    input_hash = hash(input_to_run)  # Good enough
    input_path = (self.input_path / f"{input_hash}.pickle").absolute()

    if not input_path.exists():
      with Path(input_path).open("wb") as f:
        cloudpickle.dump(input_to_run, f)

    try:
      namespace = _compile_code(program)

      with (call_data_folder / "program.pickle").open("wb+") as f:
        cloudpickle.dump(namespace[function_to_run], f)

      with (call_data_folder / f"stderr_{index}.log").open("w") as error_file:
        return_code = self._exec(call_data_folder, input_path, error_file)

      if return_code != 0:
        return None

      with (call_data_folder / "output.pickle").open("rb") as f:
        out = cloudpickle.load(f)
        try:
          return float(out)
        except ValueError:
          return None

    except Exception as e:
      logging.debug(f"Could not execute code: {e}")

    return None
