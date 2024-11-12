import numpy.random as rand
from random import choices
import gurobipy as gp
import numpy as np
import itertools as it

from gurobipy import GRB
from typing import Self

class Instance :
    def __init__(self) -> None:
        self.n : int = 0
        self.x : gp.tuplelist[int] = None
        self.y : gp.tuplelist[int] = None
        self.model : MyModel = None
        self.solutions : list[tuple[list[int], str, float]] = []

    def __str__(self) -> str:
        return self.x.__str__() + "\n" + self.y.__str__()
    
    """
        Used to initialise the model. Used once before any computation.
    """
    def init_model(self, name : str = "Model") :
        self.model = MyModel()
        self.model.gurobi_model.ModelName = name
        self.model.initialize(self)

    """
        Used to solve optimally the model using Gurobi
    """
    def solve(self) -> float :
        val = self.model.solve()
        self.opt_permut = self.model.get_permut()
        return val

    def add_solution(self, permut : list[int], label : str, val : float) -> None :
        self.solutions.append((permut, label, val))
    
    """
        Computes the prefix sums of the solution associated with the 
        permutation given as a parameter.
    """
    def compute_cumulative(self, permut) -> list[float]:
        c = 0
        res = [0]
        for idx, p in enumerate(permut) :
            c += self.x[p]
            res.append(c)
            c-= self.y[idx]
            res.append(c)
        return res
    
    """
        Auxilary method used for generating random instances.
    """
    def add_noise(self, intensity : int) :
        valid = False
        while not valid :
            noise_x = [0 for _ in self.x]
            noise_y = [0 for _ in self.y]
            for _ in range(intensity) :
                # Choice of indices to change
                rand_x = choices(range(self.n))[0]
                rand_y = choices(range(self.n))[0]
                sign = choices([-1, 1])[0] # Sign of the change
                noise_x[rand_x] += sign
                noise_y[rand_y] += sign
            
            new_x = [self.x[i] + noise_x[i] for i in range(self.n)]
            new_y = [self.y[i] + noise_y[i] for i in range(self.n)]

            # Check for validity of the instance
            valid = True
            for i in range(self.n) :
                if new_x[i] < 0 or new_y[i] < 0 :
                    valid = False
                    break
        self.x = new_x
        self.y = new_y
            

    def copy(self) :
        inst = Instance()
        inst.x = [e for e in self.x]
        inst.y = [e for e in self.y]
        inst.n = len(inst.x)
        return inst

class MyModel :
    def __init__(self) -> None:
        self.gurobi_model : gp.Model = gp.Model()

    def _init_vars(self, inst : Instance) -> None:
        self.alpha = self.gurobi_model.addVar(vtype = GRB.INTEGER, name="alpha", lb = -float('inf'), ub = float('inf'))
        self.beta = self.gurobi_model.addVar(vtype = GRB.INTEGER, name="beta", lb = -float('inf'), ub = float('inf'))
        l = list(range(inst.n))
        self.z = self.gurobi_model.addVars(l, l, vtype=GRB.BINARY, name="z")
        self.n = inst.n

    def __init_constrs(self, inst : Instance) -> None :
        self.gurobi_model.addConstrs( 
            (self.z.sum('*', j) == 1 for j in range(inst.n)))
        self.gurobi_model.addConstrs(
            (self.z.sum(i, '*') == 1 for i in range(inst.n)))

        # prefix smaller than Beta
        self.gurobi_model.addConstrs(
            (gp.quicksum(inst.x[i] * self.z[i,j] for i in range(inst.n) for j in range(k))
            - gp.quicksum(inst.y[j] for j in range(0, k-1)) <= self.beta for k in range(1, inst.n + 1)))

        # # prefix greater than Alpha
        self.gurobi_model.addConstrs(
            (gp.quicksum(inst.x[i] * self.z[i,j] for i in range(inst.n) for j in range(k))
            - gp.quicksum(inst.y[j] for j in range(0, k)) >= self.alpha for k in range(1, inst.n + 1)))

    def initialize(self, inst : Instance) -> None :
        self._init_vars(inst)
        self.__init_constrs(inst)
        self.gurobi_model.setObjective(self.beta - self.alpha, GRB.MINIMIZE)
        self.gurobi_model.setParam('OutputFlag', False)

    def relax(self) -> Self :
        relaxed_model = MyModel()
        self.gurobi_model.update()
        relaxed_model.gurobi_model = self.gurobi_model.relax()
        relaxed_model.n = self.n
        return relaxed_model
    
    def add_fixed_value_const(self, value_id : int, slot_id : int) -> gp.Constr :
        return self.gurobi_model.addConstr(self.z[value_id, slot_id] == 1.)
    
    def delete_constr(self, c : gp.Constr) -> None :
        self.gurobi_model.remove(c)
    
    def solve(self) -> float :
        m = self.gurobi_model
        m.optimize()
        #self.display_results()
        if m.Status != GRB.OPTIMAL :
            return float('inf')
        return m.ObjVal
    
    def get_permut(self) -> list[int] :
        n = self.n
        res = [i for i in range(n)]
        for i in range(n) :
            for j in range(n) :
                if abs(self.z[i,j].X - 1.) <= 0.000001 :
                    res[j] = i
                    continue
        return res

    def display_results(self):
        m = self.gurobi_model
        print("___ Resuts ___\n")
        print('Obj: %g' % m.ObjVal)
        n = self.n
        vals = [[m.getVarByName(f"z[{i},{j}]").X for j in range(n)]for i in range(n)]
        for l in vals :
            print(*[f"{elem:.2f}" for elem in l], end='\n')

    def display_constrs(self) -> None :
        self.gurobi_model.write("model.lp")
        
def _generate_tab(n : int, min : int, max : int) -> gp.tuplelist[int] :
    l = gp.tuplelist()
    for i in range(n):
        l.append(rand.randint(min, max))
    return l

def generate_instance(n : int, min : int, max : int) -> Instance :
    return generate_instance_distinct(n, min, max, min, max)

def generate_instance_distinct(n : int, x_min : int, x_max : int, y_min : int, y_max : int) -> Instance :
    inst = Instance()
    valid = False
    while not valid : 
        inst.n = n   
        inst.x = _generate_tab(n, x_min, x_max)
        inst.y = _generate_tab(n - 1, y_min, y_max)
        diff = sum(inst.x) - sum(inst.y)
        if diff >= y_min and diff < y_max :
            inst.y.append(diff)
            valid = True
    return inst

def genrate_lb_instance(k : int) -> Instance:
    inst = Instance()
    inst.x = list(it.chain.from_iterable([[2**k-2**(k-i) for _ in range(2**i)] for i in range(1, k)]))
    inst.x.extend([2**k for _ in range(2**k-1)] + [0])
    inst.y = list(it.chain.from_iterable([[2**k-2**(k-i) for _ in range(2**i)] for i in range(1, k+1)]))
    inst.n = len(inst.x)
    return inst