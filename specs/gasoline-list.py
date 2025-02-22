"""Find sets for which the rounding algorithm on the gasoline-problem has a poor (high) approximation-ratio.

The gasoline-problem find a permutation of the xs and ys (lists of positive integers) such that maximum of the
differences of prefix-sums is as small as possible, i.e. maximum_(m,n) zs[n]-zs[m] is as as small as possible,
where zs[n] = xs[0] - ys[0] + xs[1] - ys[1] + ... + xs[n//2] - (ys[n] if n is odd else 0).

As such, the problem is invariant under a permutation of the xs and ys.

On every iteration, improve gasoline_v1 over the gasoline_vX methods from previous iterations.
The methods must be deterministic.
Make only small code-changes.
"""

import funsearch
from funsearch.gasoline.iterative_rounding import SlotOrdered


@funsearch.run
def evaluate(n: int) -> float:
  """Returns the approximation-ratio of the gasoline problem."""
  xs, ys = gasoline(n)

  # Assert determinancy
  if (xs, ys) != gasoline(n):
    return 0

  xs = [max(0, min(2**31 - 1, x)) for x in xs]
  ys = [max(0, min(2**31 - 1, y)) for y in ys]

  return SlotOrdered().approximation_ratio(xs, ys)


@funsearch.evolve
def gasoline(n: int) -> tuple[list[int], list[int]]:
  """Return a new gasoline-problem, specified by the list of x-values and y-values.
  The integers will be clamped to [0, 2**31 - 1].
  """
  xs = []
  for i in range(1, n):
    u = int(2**n * (1 - 2 ** (-i)))
    xs.extend([u for _ in range(2**i)])
  xs.extend([int(2**n) for _ in range(2**n - 1)])
  xs.append(0)
  u = int(2**n * (1 - 2 ** (-n)))
  ys = [0 for _ in range(len(xs))]
  return xs, ys
