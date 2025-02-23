"""This file will be used as an executable script by the ContainerSandbox and ExternalProcessSandbox."""

import pickle
import sys
from pathlib import Path


def main(program_file: str, input_file: str, output_file: str) -> None:
  """Execute function from cloudpickle file with input data and write the output data to another file."""
  with Path(input_file).open("rb") as input_f:
    input_data = pickle.load(input_f)
    with Path(program_file).open("rb") as program_f:
      func = pickle.load(program_f)
      result = func(input_data)
      with Path(output_file).open("wb") as output_f:
        pickle.dump(result, output_f)


if __name__ == "__main__":
  if len(sys.argv) != 4:
    sys.exit(-1)
  main(sys.argv[1], sys.argv[2], sys.argv[3])
