"""Finds sets for which the iterative rounding algorithm on the gasoline-problem has a poor (high) approximation-ratio.

On every iteration, improve gasoline_v1 over the gasoline_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""

from typing import List
import funsearch

from gurobipy import GRB
from math import inf
from random import choices
from typing import Self
import abc
import gurobipy as gp


class Instance:
    def __init__(self) -> None:
        self.n: int = 0
        self.x: gp.tuplelist[int] = None
        self.y: gp.tuplelist[int] = None
        self.model: MyModel = None
        self.solutions: list[tuple[list[int], str, float]] = []

    def init_model(self, name: str = "Model"):
        self.model = MyModel()
        self.model.gurobi_model.ModelName = name
        self.model.initialize(self)

    def solve(self) -> float:
        val = self.model.solve()
        self.opt_permut = self.model.get_permut()
        return val

    def add_solution(self, permut: list[int], label: str, val: float) -> None:
        self.solutions.append((permut, label, val))

    def compute_cumulative(self, permut) -> list[float]:
        c = 0
        res = [0]
        for idx, p in enumerate(permut):
            c += self.x[p]
            res.append(c)
            c -= self.y[idx]
            res.append(c)
        return res

    def add_noise(self, intensity: int):
        valid = False
        while not valid:
            noise_x = [0 for _ in self.x]
            noise_y = [0 for _ in self.y]
            for _ in range(intensity):
                # Choice of indices to change
                rand_x = choices(range(self.n))[0]
                rand_y = choices(range(self.n))[0]
                sign = choices([-1, 1])[0]  # Sign of the change
                noise_x[rand_x] += sign
                noise_y[rand_y] += sign

            new_x = [self.x[i] + noise_x[i] for i in range(self.n)]
            new_y = [self.y[i] + noise_y[i] for i in range(self.n)]

            # Check for validity of the instance
            valid = True
            for i in range(self.n):
                if new_x[i] < 0 or new_y[i] < 0:
                    valid = False
                    break
        self.x = new_x
        self.y = new_y


class MyModel:
    def __init__(self) -> None:
        self.gurobi_model: gp.Model = gp.Model()

    def _init_vars(self, inst: Instance) -> None:
        self.alpha = self.gurobi_model.addVar(
            vtype=GRB.INTEGER, name="alpha", lb=-float("inf"), ub=float("inf")
        )
        self.beta = self.gurobi_model.addVar(
            vtype=GRB.INTEGER, name="beta", lb=-float("inf"), ub=float("inf")
        )
        l = list(range(inst.n))
        self.z = self.gurobi_model.addVars(l, l, vtype=GRB.BINARY, name="z")
        self.n = inst.n

    def __init_constrs(self, inst: Instance) -> None:
        self.gurobi_model.addConstrs((self.z.sum("*", j) == 1 for j in range(inst.n)))
        self.gurobi_model.addConstrs((self.z.sum(i, "*") == 1 for i in range(inst.n)))

        # prefix smaller than Beta
        self.gurobi_model.addConstrs(
            (
                gp.quicksum(
                    inst.x[i] * self.z[i, j] for i in range(inst.n) for j in range(k)
                )
                - gp.quicksum(inst.y[j] for j in range(0, k - 1))
                <= self.beta
                for k in range(1, inst.n + 1)
            )
        )

        # # prefix greater than Alpha
        self.gurobi_model.addConstrs(
            (
                gp.quicksum(
                    inst.x[i] * self.z[i, j] for i in range(inst.n) for j in range(k)
                )
                - gp.quicksum(inst.y[j] for j in range(0, k))
                >= self.alpha
                for k in range(1, inst.n + 1)
            )
        )

    def initialize(self, inst: Instance) -> None:
        self._init_vars(inst)
        self.__init_constrs(inst)
        self.gurobi_model.setObjective(self.beta - self.alpha, GRB.MINIMIZE)
        self.gurobi_model.setParam("OutputFlag", False)

    def relax(self) -> Self:
        relaxed_model = MyModel()
        self.gurobi_model.update()
        relaxed_model.gurobi_model = self.gurobi_model.relax()
        relaxed_model.n = self.n
        return relaxed_model

    def add_fixed_value_const(self, value_id: int, slot_id: int) -> gp.Constr:
        return self.gurobi_model.addConstr(self.z[value_id, slot_id] == 1.0)

    def delete_constr(self, c: gp.Constr) -> None:
        self.gurobi_model.remove(c)

    def solve(self) -> float:
        m = self.gurobi_model
        m.optimize()
        # self.display_results()
        if m.Status != GRB.OPTIMAL:
            return float("inf")
        return m.ObjVal

    def get_permut(self) -> list[int]:
        n = self.n
        res = [i for i in range(n)]
        for i in range(n):
            for j in range(n):
                if abs(self.z[i, j].X - 1.0) <= 0.000001:
                    res[j] = i
                    continue
        return res

    def display_results(self):
        m = self.gurobi_model
        print("___ Resuts ___\n")
        print("Obj: %g" % m.ObjVal)
        n = self.n
        vals = [[m.getVarByName(f"z[{i},{j}]").X for j in range(n)] for i in range(n)]
        for l in vals:
            print(*[f"{elem:.2f}" for elem in l], end="\n")

    def display_constrs(self) -> None:
        self.gurobi_model.write("model.lp")


class GeneralisedInstance:
    def __init__(self) -> None:
        self.n: int = 0
        self.k = 0
        self.x: gp.tuplelist[tuple[int]] = None
        self.y: gp.tuplelist[tuple[int]] = None
        self.model: MyGeneralisedModel = None
        self.solutions: list[tuple[list[int], str, float]] = []

    def init_model(self, name: str = "Model"):
        self.model = MyGeneralisedModel(self.k)
        self.model.gurobi_model.ModelName = name
        self.model.initialize(self)

    def solve(self) -> float:
        val = self.model.solve()
        self.opt_permut = self.model.get_permut()
        return val


class MyGeneralisedModel:
    def __init__(self, k: int) -> None:
        self.gurobi_model: gp.Model = gp.Model()
        self.k = k
        self.alpha = []
        self.beta = []

    def _init_vars(self, inst: GeneralisedInstance) -> None:
        for i in range(self.k):
            self.alpha.append(
                self.gurobi_model.addVar(
                    vtype=GRB.INTEGER,
                    name=f"alpha_{i}",
                    lb=-float("inf"),
                    ub=float("inf"),
                )
            )
            self.beta.append(
                self.gurobi_model.addVar(
                    vtype=GRB.INTEGER,
                    name=f"beta_{i}",
                    lb=-float("inf"),
                    ub=float("inf"),
                )
            )
        l = list(range(inst.n))
        self.z = self.gurobi_model.addVars(l, l, vtype=GRB.BINARY, name="z")
        self.n = inst.n

    def __init_constrs(self, inst: GeneralisedInstance) -> None:
        self.gurobi_model.addConstrs((self.z.sum("*", j) == 1 for j in range(inst.n)))
        self.gurobi_model.addConstrs((self.z.sum(i, "*") == 1 for i in range(inst.n)))

        for l in range(self.k):
            # prefix smaller than Beta
            self.gurobi_model.addConstrs(
                (
                    gp.quicksum(
                        inst.x[i][l] * self.z[i, j]
                        for i in range(inst.n)
                        for j in range(k)
                    )
                    - gp.quicksum(inst.y[j][l] for j in range(0, k - 1))
                    <= self.beta[l]
                    for k in range(1, inst.n + 1)
                )
            )
            # prefix greater than Alpha
            self.gurobi_model.addConstrs(
                (
                    gp.quicksum(
                        inst.x[i][l] * self.z[i, j]
                        for i in range(inst.n)
                        for j in range(k)
                    )
                    - gp.quicksum(inst.y[j][l] for j in range(0, k))
                    >= self.alpha[l]
                    for k in range(1, inst.n + 1)
                )
            )

    def initialize(self, inst: GeneralisedInstance) -> None:
        self._init_vars(inst)
        self.__init_constrs(inst)
        self.gurobi_model.setObjective(
            gp.quicksum(self.beta[l] - self.alpha[l] for l in range(self.k)),
            GRB.MINIMIZE,
        )
        self.gurobi_model.setParam("OutputFlag", False)

    def relax(self) -> Self:
        relaxed_model = MyGeneralisedModel(self.k)
        self.gurobi_model.update()
        relaxed_model.gurobi_model = self.gurobi_model.relax()
        relaxed_model.n = self.n
        return relaxed_model

    def add_fixed_value_const(self, value_id: int, slot_id: int) -> gp.Constr:
        return self.gurobi_model.addConstr(self.z[value_id, slot_id] == 1.0)

    def delete_constr(self, c: gp.Constr) -> None:
        self.gurobi_model.remove(c)

    def solve(self) -> float:
        m = self.gurobi_model
        m.optimize()
        # self.display_results()
        if m.Status != GRB.OPTIMAL:
            return float("inf")
        return m.ObjVal

    def get_permut(self) -> list[int]:
        n = self.n
        res = [i for i in range(n)]
        for i in range(n):
            for j in range(n):
                if abs(self.z[i, j].X - 1.0) <= 0.000001:
                    res[j] = i
                    continue
        return res


class Result:
    def __init__(self, instance: Instance, label: str) -> None:
        self.n = instance.n
        self.permut = [0 for i in range(self.n)]
        self.label: str = label

    def fix_value(self, index: int, value: int) -> None:
        self.permut[index] = value


class IterativeAlgo(abc.ABC):
    def __init__(self) -> None:
        self.available: list[bool]

    @abc.abstractmethod
    def run(self, instance: Instance) -> tuple[Result, float]:
        pass


class ValueOrdered(IterativeAlgo):
    def __init__(self) -> None:
        super().__init__()

    def run(self, instance: Instance) -> tuple[Result, float]:
        self.available = [True for _ in range(instance.n)]
        res = Result(instance, "Value ordered")
        for i in range(instance.n):
            opt_value = inf
            opt_idx = -1
            m = instance.model
            for j in (j for j, b in enumerate(self.available) if b):
                fixed_constr = m.add_fixed_value_const(i, j)
                new_m = m.relax()
                new_m.solve()
                if new_m.gurobi_model.Status == GRB.OPTIMAL:
                    value = new_m.get_obj_value()
                    if value < opt_value:
                        opt_value = value
                        opt_idx = j
                m.delete_constr(fixed_constr)
            m.add_fixed_value_const(i, opt_idx)
            self.available[opt_idx] = False
            res.fix_value(i, opt_idx)
        return res, opt_value


class SlotOrdered(IterativeAlgo):
    def __init__(self) -> None:
        super().__init__()

    def run(self, instance: Instance) -> tuple[Result, float]:
        self.available = [True for _ in range(instance.n)]
        res = Result(instance, "Slot ordered")
        for j in range(instance.n):
            opt_value = inf
            opt_idx = -1
            m = instance.model
            values = []
            for i in (i for i, b in enumerate(self.available) if b):
                fixed_constr = m.add_fixed_value_const(i, j)
                new_m = m.relax()
                value = round(new_m.solve(), 1)
                values.append(value)
                if value < opt_value:
                    opt_value = value
                    opt_idx = i
                m.delete_constr(fixed_constr)
            m.add_fixed_value_const(opt_idx, j)
            final_value = m.solve()
            self.available[opt_idx] = False
            res.fix_value(j, opt_idx)
        return res, final_value


@funsearch.run
def evaluate(n: int) -> int:
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
def gasoline(n: int) -> tuple[List[int], List[int]]:
  """Returns a gasoline-problem specified by the list of x-values and y-values,
  with poor approximation-ratio.
  n is the length of the x-values and y-values.
  """
  xs = [i/3 + i%2 for i in range(n)]
  ys = [i/5 + i%7 for i in range(n)]
  return xs, ys
