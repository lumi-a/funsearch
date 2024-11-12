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
  ratio = SlotOrdered().approximation_ratio(xs, ys)
  return ratio

@funsearch.evolve
def gasoline(n: int) -> tuple[List[float], List[float]]:
  """Returns a gasoline-problem specified by the list of x-values and y-values,
  with poor approximation-ratio.
  n is the length of the x-values and y-values.
  """
  xs = [np.sqrt(i) for i in range(n)]
  ys = [i/3 + i**2 % np.pi for i in range(n)]
  return xs, ys
