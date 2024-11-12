import gurobipy as gp
import numpy as np
from funsearch.gasoline.iterative_rounding import SlotOrdered

SIZE = 5000
k = 15

n = 10
xs = [np.sqrt(i) for i in range(n)]
ys = [i/3 + i**2%np.pi for i in range(n)]
apx = SlotOrdered().approximation_ratio(xs, ys)
print(apx)
