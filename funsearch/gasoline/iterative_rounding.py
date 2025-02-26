import abc
from math import inf

from gurobipy import GRB, tuplelist

import funsearch.gasoline.instance as ins


class Result:
    def __init__(self, instance: ins.Instance, label: str) -> None:
        self.n = instance.n
        self.permut = [0 for i in range(self.n)]
        self.label: str = label

    def fix_value(self, index: int, value: int) -> None:
        self.permut[index] = value

    def __str__(self) -> str:
        s = "Results : \n\n"
        for i in range(self.n):
            s += f"x_{i} : {self.permut[i]}\n"
        return s


class IterativeAlgo(abc.ABC):
    def __init__(self) -> None:
        self.available: list[bool]

    @abc.abstractmethod
    def run(self, instance: ins.Instance) -> tuple[Result, float]:
        pass

    def approximation_ratio(self, xs: list[int], ys: list[int]) -> float:
        n = len(xs)
        if len(ys) < n - 1:
            return 0
        ys = ys[: n - 1]
        difference = sum(xs) - sum(ys)
        ys.append(difference)
        instance = ins.Instance()
        instance.n = n
        instance.k = 1
        instance.x = tuplelist(x for x in xs)
        instance.y = tuplelist(y for y in ys)
        instance.init_model()

        opt = instance.solve()
        if opt <= 0:
            return 0
        _, val = self.run(instance)
        return val / opt


class ValueOrdered(IterativeAlgo):
    def __init__(self) -> None:
        super().__init__()

    def run(self, instance: ins.Instance) -> tuple[Result, float]:
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

    def run(self, instance: ins.Instance) -> tuple[Result, float]:
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
