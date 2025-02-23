"""Sandboxes for code-execution."""

import ast
import logging
from pathlib import Path
from typing import Any

import cloudpickle

CONTAINER_MAIN = (Path(__file__).parent / "container" / "container_main.py").absolute()

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

  # TODO: Add type to input
  def run(self, program: str, function_to_run: str, input, index: int) -> tuple[Any, bool]:
    call_data_folder = (self.output_path / f"call-{index}").absolute()
    if not call_data_folder.exists():
      call_data_folder.mkdir()

    input_hash = hash(input)  # Good enough
    input_path = (self.input_path / f"{input_hash}.pickle").absolute()

    if not input_path.exists():
      with Path(input_path).open("wb") as f:
        cloudpickle.dump(input, f)

    try:
      namespace = DummySandbox.compile_code(program)

      program_file = (call_data_folder / "program.pickle").absolute()
      with open(program_file, "wb+") as f:
        cloudpickle.dump(namespace[function_to_run], f)

      error_file = self.output_path / f"stderr-{index}.log"

      retcode = self._exec(call_data_folder, input_path, error_file)

      if retcode != 0:
        self._save_diagnostics(program, call_data_folder)
        return None, False

      output_file = call_data_folder / "output.pickle"
      with open(output_file, "rb") as f:
        out = cloudpickle.load(f)
        return out, True
    except Exception as e:
      logging.debug(f"Could not execute code: {e}")
    self._save_diagnostics(program, call_data_folder)
    return None, False

  @staticmethod
  def _save_diagnostics(program: str, output_path: Path) -> None:
    filepath = output_path / "program.py"
    logging.debug(f"Writing program to {filepath}")
    with open(filepath, "w+") as f:
      f.write(program)


print(4)
if __name__ == "__main__":
  print(3)
  print(
    _compile_code("""
import random

def x(y):
  print(f"Received {y}")
  return y + 1

print(1)
""")
  )
