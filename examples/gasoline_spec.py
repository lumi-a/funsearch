"""Finds sets for which the iterative rounding algorithm on the gasoline-problem has a poor (high) approximation-ratio.

On every iteration, improve gasoline_v1 over the gasoline_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
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