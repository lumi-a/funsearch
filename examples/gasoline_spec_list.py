"""Finds sets for which the iterative rounding algorithm on the gasoline-problem has a poor (high) approximation-ratio.

The gasoline-problem find a permutation of the xs and ys (lists of positive integers) such that maximum of the differences of prefix-sums is as small as possible, i.e. maximum_(m,n) zs[n]-zs[m] is as as small as possible, where zs[n] = xs[0] - ys[0] + xs[1] - ys[1] + ... + xs[n//2] - (ys[n] if n is odd else 0).

As such, the problem is invariant under a permutation of the xs and ys.

On every iteration, improve gasoline_v1 over the gasoline_vX methods from previous iterations.
Make only small code-changes. Do not use np.random.
"""

from typing import List
import funsearch
from funsearch.gasoline.iterative_rounding import SlotOrdered


@funsearch.run
def evaluate(_: int) -> float:
  """Returns the approximation-ratio of the gasoline problem"""
  xs, ys = gasoline()

  # Check determinancy
  if (xs, ys) != gasoline():
    return 0

  if any(x < 0 for x in xs) or any(y < 0 for y in ys):
    return 0

  ratio = SlotOrdered().approximation_ratio(xs, ys)
  return ratio


@funsearch.evolve
def gasoline() -> tuple[List[int], List[int]]:
  """Return a new gasoline-problem, specified by the list of x-values and y-values.
  The integers are always non-negative.
  """
  k = 4
  xs, ys = [], []
  for i in range(1, k):
    u = int(2**k * (1 - 2 ** (-i)))
    xs.extend([u for _ in range(2**i)])
    ys.extend([u for _ in range(2**i)])
  xs.extend([int(2**k) for _ in range(2**k - 1)])
  xs.append(0)
  u = int(2**k * (1 - 2 ** (-k)))
  ys.extend([u for _ in range(2**k)])
  return xs, ys
