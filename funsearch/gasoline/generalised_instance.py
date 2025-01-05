import numpy.random as rand
from random import choices
import gurobipy as gp
import numpy as np
import itertools as it

from gurobipy import GRB
from typing import Self


class GeneralisedInstance:
    def __init__(self) -> None:
        self.n: int = 0
        self.k = 0
        self.x: gp.tuplelist[tuple[int]] = None
        self.y: gp.tuplelist[tuple[int]] = None
        self.model: MyGenealisedModel = None
        self.solutions: list[tuple[list[int], str, float]] = []

    def __str__(self) -> str:
        return self.x.__str__() + "\n" + self.y.__str__()

    def init_model(self, name: str = "Model"):
        self.model = MyGenealisedModel(self.k)
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
            noise_x = [[0 for _ in range(self.k)] for _ in self.x]
            noise_y = [[0 for _ in range(self.k)] for _ in self.y]
            for _ in range(intensity):
                rand_x = [choices(range(self.n))[0] for _ in range(self.k)]
                rand_y = [choices(range(self.n))[0] for _ in range(self.k)]
                coord = choices(range(self.k))[0]
                sign = choices([-1, 1])[0]
                for pos_x, pos_y in zip(rand_x, rand_y):
                    noise_x[pos_x][coord] += sign
                    noise_y[pos_y][coord] += sign

            new_x = [
                tuple(self.x[i][l] + noise_x[i][l] for l in range(self.k))
                for i in range(self.n)
            ]
            new_y = [
                tuple(self.y[i][l] + noise_y[i][l] for l in range(self.k))
                for i in range(self.n)
            ]

            valid = True
            if any(
                new_x[i][l] < 0 or new_y[i][l] < 0
                for l in range(self.k)
                for i in range(self.n)
            ):
                valid = False

        self.x = gp.tuplelist(new_x)
        self.y = gp.tuplelist(new_y)

    def copy(self):
        inst = GeneralisedInstance()
        inst.x = gp.tuplelist([e for e in self.x])
        inst.y = gp.tuplelist([e for e in self.y])
        inst.n = len(inst.x)
        inst.k = self.k
        return inst


class MyGenealisedModel:
    def __init__(self, k: int) -> None:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            self.gurobi_model: gp.Model = gp.Model(env=env)
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
        relaxed_model = MyGenealisedModel(self.k)
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


def _generate_tab(n: int, k: int, min: int, max: int) -> gp.tuplelist[tuple[int]]:
    l = gp.tuplelist()
    for i in range(n):
        l.append(tuple(rand.randint(min, max) for _ in range(k)))
    return l


def generate_instance(n: int, k: int, min: int, max: int) -> GeneralisedInstance:
    return generate_instance_distinct(n, k, min, max, min, max)


def generate_instance_distinct(
    n: int, k: int, x_min: int, x_max: int, y_min: int, y_max: int
) -> GeneralisedInstance:
    inst = GeneralisedInstance()
    valid = False
    while not valid:
        inst.n = n
        inst.k = k
        inst.x = _generate_tab(n, k, x_min, x_max)
        inst.y = _generate_tab(n - 1, k, y_min, y_max)
        diff = tuple(
            sum([inst.x[i][l] for i in range(n)])
            - sum([inst.y[i][l] for i in range(n - 1)])
            for l in range(k)
        )
        if all(diff[l] >= y_min and diff[l] < y_max for l in range(k)):
            inst.y.append(diff)
            valid = True
    return inst


# def genrate_lb_instance(k : int) -> BigInstance:
#     inst = BigInstance()
#     inst.x = list(it.chain.from_iterable([[2**k-2**(k-i) for _ in range(2**i)] for i in range(1, k)]))
#     inst.x.extend([2**k for _ in range(2**k-1)] + [0])
#     inst.y = list(it.chain.from_iterable([[2**k-2**(k-i) for _ in range(2**i)] for i in range(1, k+1)]))
#     inst.n = len(inst.x)
#     return inst
