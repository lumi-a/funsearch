"""Finds sets for which the iterative rounding algorithm on the gasoline-problem has a poor (high) approximation-ratio.

The gasoline-problem find a permutation of the xs and ys (lists of positive integers) such that maximum of the differences of prefix-sums is as small as possible, i.e. maximum_(m,n) zs[n]-zs[m] is as as small as possible, where zs[n] = xs[0] - ys[0] + xs[1] - ys[1] + ... + xs[n//2] - (ys[n] if n is odd else 0).

As such, the problem is invariant under a permutation of the xs and ys.

On every iteration, improve gasoline_v1 over the gasoline_vX methods from previous iterations.
Make only small code-changes. Do not use np.random.
"""

from typing import List
import numpy as np
import funsearch
from funsearch.gasoline.iterative_rounding import SlotOrdered

@funsearch.run
def evaluate(n: int) -> float:
  """Returns the approximation-ratio of the gasoline problem"""
  xs, ys = gasoline(n)
  if any(x < 0 for x in xs) or any(y < 0 for y in ys):
    return 0
  ratio = SlotOrdered().approximation_ratio(xs, ys)
  return ratio

@funsearch.evolve
def gasoline(n: int) -> tuple[List[int], List[int]]:
  """Returns a gasoline-problem specified by the list of x-values and y-values,
  with poor approximation-ratio.
  n is the length of the x-values and y-values.
  """
  xs = [int((i**2)/np.pi) for i in range(n)]
  ys = [int((i*5)/np.e) for i in range(n)]
  return xs, ys