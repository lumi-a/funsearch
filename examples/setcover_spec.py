"""Finds instances for which the greedy algorithm on set-cover has a poor approximation-ratio.

On every iteration, improve setcover_instance_v1 over the setcover_instance_vX methods from previous iterations. Do not use np.random.
"""

from typing import List
import numpy as np
import funsearch
from gurobipy import Model, GRB


@funsearch.run
def evaluate(_: int) -> float:
    """Returns the greedy approximation-ratio of the set-cover instance."""
    instance = bad_setcover()
    return greedy_ratio(instance)


def greedy_ratio(sets: list[set[int]]) -> float:
    # Take union over all sets
    universe = set.union(*sets)

    # Greedy solution
    uncovered = universe.copy()
    greedy_solution = []
    while uncovered:
        best_set = max(sets, key=lambda s: len(uncovered & s))
        greedy_solution.append(best_set)
        uncovered -= best_set
    greedy_cost = len(greedy_solution)

    # Exact solution using Gurobi
    model = Model("SetCover")
    model.setParam("OutputFlag", 0)  # Suppress output
    model.setParam("LogToConsole", 0)
    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(len(sets))]
    model.update()
    for element in universe:
        model.addConstr(
            sum(x[i] for i, s in enumerate(sets) if element in s) >= 1,
            f"cover_{element}",
        )
    model.setObjective(sum(x), GRB.MINIMIZE)
    print(f"start {len(sets)}")
    model.optimize()
    print("stop")

    if model.status != GRB.OPTIMAL:
        raise RuntimeError("Gurobi did not find an optimal solution")
    optimal_cost = model.objVal

    approximation_ratio = greedy_cost / optimal_cost
    return approximation_ratio


@funsearch.evolve
def bad_setcover() -> list[set[int]]:
    """Return an instance of set-cover (a list of sets of integers) which
    has a poor greedy approximation-ratio.
    """
    universe = set(range(1, 14 + 1))
    return [{x} for x in universe]
