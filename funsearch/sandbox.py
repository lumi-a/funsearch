import ast
import logging
import os
import pathlib
import sys
from typing import Any

import cloudpickle

CONTAINER_MAIN = (pathlib.Path(__file__).parent / "container" / "container_main.py").absolute()


class ExternalProcessSandbox:
  """Sandbox that executes the code in a separate Python process in the same host.

  Note: This does not provide real safety and should be only used in an environment where the host process is
  in some kind of safe sandbox itself (i.e., a container).
  This kind of sandbox merely makes it more probable that single invalid call does not break the whole
  funsearch algorithm. It might be easier to set up and thus nice environment to tune the prompts and other
  code.
  """

  def __init__(self, base_path: pathlib.Path, timeout_secs: int = 30, python_path: str = "python") -> None:
    super().__init__()

    self.output_path = pathlib.Path(base_path) / f"sandbox{self.id}"
    self.timeout_secs = timeout_secs
    self.python_path = python_path
    self.call_count = 0

    self.input_path = self.output_path / "inputs"
    for p in [self.output_path, self.input_path]:
      if not p.exists():
        p.mkdir(parents=True)

  def _exec(self, call_data_path: pathlib.Path, input_path: pathlib.Path, error_file_path: pathlib.Path):
    """Use podman/docker to execute python in a container.
    - The main.py shall execute the LLM generated method from prog.pickle file providing
      input.pickle as the input for the method.
    - main.py writes the output of the method into output.pickle.
    Everything except the /workspace folder will be read-only so that the environment remains good
    for future runs.
    """
    import subprocess

    prog_path = call_data_path / "prog.pickle"
    output_file = call_data_path / "output.pickle"
    cmd = f"{self.python_path} {CONTAINER_MAIN} {prog_path} {input_path} {output_file}  2> {error_file_path}"

    logging.debug(f"Executing {cmd}")
    timeout = int(self.timeout_secs)
    try:
      result = subprocess.run(cmd, timeout=timeout, shell=True, check=False)
      return result.returncode
    except subprocess.TimeoutExpired:
      logging.debug(f"Command timed out after {timeout} seconds")
      return 1
    except Exception as e:
      logging.debug(f"Command failed with error: {e}")
      return 1

  def run(self, program: str, function_to_run: str, test_input, timeout_seconds: int) -> tuple[Any, bool]:
    call_data_folder = (self.output_path / f"call{self.call_count}").absolute()
    if not call_data_folder.exists():
      call_data_folder.mkdir()

    input_hash = hash(test_input)
    input_path = (self.input_path / f"{input_hash}.pickle").absolute()

    if not input_path.exists():
      with open(input_path, "wb") as f:
        cloudpickle.dump(test_input, f)
    try:
      namespace = DummySandbox.compile_code(program)

      prog_file = (call_data_folder / "prog.pickle").absolute()
      with open(prog_file, "wb+") as f:
        cloudpickle.dump(namespace[function_to_run], f)

      error_file = self.output_path / f"stderr_{self.call_count}.log"

      retcode = self._exec(call_data_folder, input_path, error_file)
      self.call_count += 1

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
  def _save_diagnostics(program: str, output_path: pathlib.Path) -> None:
    filepath = output_path / "program.py"
    logging.debug(f"Writing program to {filepath}")
    with open(filepath, "w+") as f:
      f.write(program)
