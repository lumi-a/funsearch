import numpy as np

from funsearch.gasoline.iterative_rounding import SlotOrdered

SIZE = 5000
k = 15

n = 10

tests = []
for _i in range(100):
    xs = list(np.random.randint(1, 20, size=5))
    ys = list(np.random.randint(1, 20, size=5))
    apx = SlotOrdered().approximation_ratio(xs, ys)
    tests.append((xs, ys, apx))
print(max(tests, key=lambda x: x[2]))  # noqa: T201
