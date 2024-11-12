"""Finds sets for which the iterative rounding algorithm on the gasoline-problem has a poor (high) approximation-ratio.

On every iteration, improve gasoline_v1 over the gasoline_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""

from typing import List
import funsearch
from funsearch.gasoline.generalised_instance import GeneralisedInstance
from funsearch.gasoline.iterative_rounding import SlotOrdered
import gurobipy as gp

@funsearch.run
def evaluate(n: int) -> float:
  """Returns the approximation-ratio of the gasoline problem"""
  xs, ys = gasoline(n)

  n = len(xs)
  if len(ys) < n - 1:
      print(f"<*> len(ys) < n-1")
      return 0
  ys = ys[: n - 1]
  difference = sum(xs) - sum(ys)
  ys.append(difference)
  instance = GeneralisedInstance()
  instance.n = n
  instance.k = 1
  instance.x = gp.tuplelist(((x,) for x in xs))
  instance.y = gp.tuplelist(((y,) for y in ys))
  instance.init_model()

  opt = instance.solve()
  if opt <= 0:
      print(f"<x> opt <=0")
      return 0
  _, val = SlotOrdered().run(instance)
  ratio = val / opt
  print(f"</> {ratio}")
  return ratio


@funsearch.evolve
def gasoline(n: int) -> tuple[List[float], List[float]]:
  """Returns a gasoline-problem specified by the list of x-values and y-values,
  with poor approximation-ratio.
  n is the length of the x-values and y-values.
  """
  xs = [i/3 + i%2 for i in range(n)]
  ys = [i/5 + i%7 for i in range(n)]
  return xs, ys
