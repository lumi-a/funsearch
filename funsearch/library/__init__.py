import os
from collections.abc import Iterator


def read_accesses() -> Iterator[int]:
  with open(os.path.join(os.path.dirname(__file__), "accesses-random-exponential.txt")) as f:
    for line in f:
      yield int(line.strip())
