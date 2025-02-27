"""Sandboxes for code-execution."""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from typing import Sequence, Any

import cloudpickle

CONTAINER_MAIN = (Path(__file__).parent / "container" / "main.py").absolute()


def _compile_code(program: str) -> dict:
    namespace: dict[str, Any] = {}

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

    def __init__(self, base_path: Path, timeout_secs: float = 30.0) -> None:
        """Create a new sandbox that logs to `base_path` and runs a sample for at most `timeout_secs`."""
        base_path.mkdir(exist_ok=True)
        self.base_path = Path(base_path)
        self.timeout_secs: float = float(timeout_secs)

    def _exec(self, program_path: Path, output_path: Path, input_path: Path, error_file_path: Path) -> int:
        """Execute python-code in a separate process.

        - The main.py shall execute the LLM generated method from program.pickle file providing
          input.pickle as the input for the method.
        - main.py writes the output of the method into output.pickle.
        Everything except the /workspace folder will be read-only so that the environment remains good
        for future runs.
        """
        import subprocess

        cmd: Sequence[str] = [
            sys.executable,
            str(CONTAINER_MAIN),
            str(program_path),
            str(input_path),
            str(output_path),
        ]

        logging.debug(f"Executing {cmd}")
        try:
            with error_file_path.open("w") as error_file:
                result: subprocess.CompletedProcess = subprocess.run(  # noqa: S603
                    args=cmd, timeout=self.timeout_secs, stderr=error_file, check=False
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
        call_data_folder = (self.base_path / f"{index}").absolute()
        call_data_folder.mkdir(exist_ok=True)

        input_hash = hash(input_to_run)  # Good enough
        input_path = (call_data_folder / f"input-{input_hash}.pickle").absolute()

        if not input_path.exists():
            with Path(input_path).open("wb") as f:
                cloudpickle.dump(input_to_run, f)

        try:
            namespace = _compile_code(program)

            output_path = call_data_folder / "output.pickle"
            program_path = call_data_folder / "program.pickle"
            with program_path.open("wb+") as f:
                cloudpickle.dump(namespace[function_to_run], f)

            error_file_path = call_data_folder / "stderr.log"
            return_code = self._exec(program_path, output_path, input_path, error_file_path)

            if return_code != 0:
                return None

            with output_path.open("rb") as f:
                out = cloudpickle.load(f)
                try:
                    return float(out)
                except ValueError:
                    return None

        except Exception as e:
            logging.debug(f"Could not execute code: {e}")

        return None
