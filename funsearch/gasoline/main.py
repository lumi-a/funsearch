import gurobipy as gp
import funsearch.gasoline.iterative_rounding as ir
import funsearch.gasoline.generalised_instance as ins

SIZE = 5000
k = 15

for d in range(2, 1, -1) :
    instance = ins.generate_instance(10, d, 0, 50)
    instance.init_model()

    opt = instance.solve()
    res, val = ir.SlotOrdered().run(instance)
    ratio = val/opt

    print("x : ", instance.x)
    print("y : ", instance.y)
    print('Optimal solution : ', gp.tuplelist([instance.x[i] for i in instance.opt_permut]))
    print("Approximated solution : ", gp.tuplelist([instance.x[i] for i in res.permut]))
    print(f"ratio = {ratio}")
