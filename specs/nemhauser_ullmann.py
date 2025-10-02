"""I'm trying to find an instance of the knapsack-problem where the pareto-set is significantly smaller than the pareto-set of one of its subinstances.

Instances are generated via the `get_instance` function and then evaluated. I have tried the following implementations for `get_instance` so far. Please write another one that is similar and has the same signature, but has some lines altered.
"""

import funsearch


@funsearch.run
def evaluate(_: int) -> float:
    instance = get_instance()
    assert instance == get_instance()  # Assert determinancy
    return evaluate_instance([(max(0, int(weight)), max(0, int(profit))) for (weight, profit) in instance])


def evaluate_instance(instance: list[tuple[int, int]]) -> float:
    """Returns the ratio between sizes of the pareto-set and sub-pareto-sets of the instance.

    Weights and profits must be non-negative.
    """
    assert all(weight >= 0 and profit >= 0 for (weight, profit) in instance), "weights and profits must be non-negative"

    type KnapsackDigest = tuple[int, int]  # WeightSum, ProfitSum

    def add_item(p: list[KnapsackDigest], next_item: tuple[int, int]) -> list[KnapsackDigest]:
        (next_weight, next_profit) = (max(0, next_item[0]), max(0, next_item[1]))

        p_plus_i = [(weight + next_weight, profit + next_profit) for (weight, profit) in p]

        q = []
        ix, plus_ix = 0, 0
        while ix < len(p) and plus_ix < len(p_plus_i):
            if (p[ix][0], -p[ix][1]) < (p_plus_i[plus_ix][0], -p_plus_i[plus_ix][1]):
                q.append(p[ix])
                ix += 1
            else:
                q.append(p_plus_i[plus_ix])
                plus_ix += 1
        q.extend(p[ix:])
        q.extend(p_plus_i[plus_ix:])

        new_p = []
        max_profit_so_far = -1
        weight_of_previous_max_profit = -1

        for weight, profit in q:
            if profit > max_profit_so_far:
                weight_of_previous_max_profit = weight
                new_p.append((weight, profit))
            elif profit == max_profit_so_far and weight == weight_of_previous_max_profit:
                new_p.append((weight, profit))

            max_profit_so_far = max(max_profit_so_far, profit)
        return new_p

    # This is a list instead of a set so that we can track individual pareto-sets
    p: list[KnapsackDigest] = [(0, 0)]

    max_sub_size = 0
    max_ratio = 0

    for next_item in instance:
        p = add_item(p, next_item)
        p_size = len(p)
        max_sub_size = max(max_sub_size, p_size)

        if p_size > 0:
            max_ratio = max(max_ratio, max_sub_size / p_size)

    return max_ratio


@funsearch.evolve
def get_instance() -> list[tuple[int, int]]:
    """Return an instance, specified by the list of (weight, profit) pairs.

    Weights and profits must be non-negative integers.
    """
    items = [(4, 4)] * 2 + [(2, 1), (1, 2), (2, 2)]
    return items
