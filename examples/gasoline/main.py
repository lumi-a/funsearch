import iterative_rounding as ir
import gurobipy as gp
import generalised_instance as ins

from alive_progress import alive_bar

SIZE = 5000
k = 15

for d in range(2, 1, -1) :
    instance = ins.generate_instance(10, d, 0, 50)
    instance.init_model()

    opt = instance.solve()
    res, val = ir.SlotOrdered().run(instance)
    ratio = val/opt

    with alive_bar(SIZE) as bar :
        for i in range(SIZE) :
            new_inst = instance.copy()
            new_inst.add_noise(k)
            new_inst.init_model()
            current_opt = new_inst.solve()
            current_res, current_val = ir.SlotOrdered().run(new_inst)
            if current_val / current_opt > ratio :
                opt = current_opt
                ratio = current_val / current_opt
                instance = new_inst
            bar()

    print("x : ", instance.x)
    print("y : ", instance.y)
    print('Optimal solution : ', gp.tuplelist([instance.x[i] for i in instance.opt_permut]))
    print("Approximated solution : ", gp.tuplelist([instance.x[i] for i in res.permut]))
    print(f"ratio = {ratio}")
