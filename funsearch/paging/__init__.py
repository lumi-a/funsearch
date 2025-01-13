from typing import Iterator
import os


def read_accesses() -> Iterator[int]:
  with open(os.path.join(os.path.dirname(__file__), "accesses.txt")) as f:
    for line in f:
      yield int(line.strip())
