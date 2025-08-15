"""I'm trying to find instances of the bin-packing problem where, if the input is shuffled, the best-fit online-heuristic performs poorly in expectation. All bins have capacity 1.0.

To generate instances with poor scores, I have tried the following functions so far. Please write another one that is similar and has the same signature, but has some lines altered.
"""

import math
import funsearch


@funsearch.run
def evaluate(_: int) -> float:
    """Returns the estimated score of the instance."""
    instance = get_items()
    # Assert determinancy
    if instance != get_items() or len(instance) == 0:
        return 0.0
    instance = [min(1.0, item) for item in instance if item > 0]
    return evaluate_instance(instance)


def evaluate_instance(instance: list[float]) -> float:
    """Return the estimated score of an instance.

    The items must be floats between 0 and 1.
    """
    assert all(0 <= item <= 1 for item in instance)

    import random

    def optimum_value(instance: list[float]) -> int:
        from collections import Counter
        from collections.abc import Iterator
        from dataclasses import dataclass
        from queue import PriorityQueue

        type Bin = frozenset[float, int]

        @dataclass(order=True, frozen=True, kw_only=True)
        class Node:
            number_of_bins: int
            next_node_index: int
            bins: Bin

        def neighbors(node: Node) -> Iterator[Node]:
            updated_index = node.next_node_index + 1
            item = instance[node.next_node_index]
            old_bins = Counter(dict(node.bins))
            for old_level in old_bins:
                new_level = old_level + item
                # Avoid floating-point rounding issues
                if new_level > 1 + 1e-9:
                    continue

                new_bins = old_bins.copy()
                new_bins += Counter([new_level])
                new_bins -= Counter([old_level])
                yield Node(
                    number_of_bins=node.number_of_bins,
                    next_node_index=updated_index,
                    bins=frozenset(Counter(new_bins).items()),
                )

            new_singleton = old_bins.copy()
            new_singleton += Counter([item])
            yield Node(
                number_of_bins=node.number_of_bins + 1,
                next_node_index=updated_index,
                bins=frozenset(Counter(new_singleton).items()),
            )

        start_node = Node(next_node_index=0, number_of_bins=0, bins=frozenset())
        pqueue: PriorityQueue[Node] = PriorityQueue()
        pqueue.put(start_node)
        seen: set[Bin] = {start_node}

        while not pqueue.empty():
            node = pqueue.get()
            if node.next_node_index == len(instance):
                return node.number_of_bins
            for neighbor in neighbors(node):
                if neighbor.bins in seen:
                    continue
                seen.add(neighbor.bins)
                pqueue.put(neighbor)

        return math.inf

    def best_fit(instance: list[float]) -> int:
        bins = []  # There are more efficient datastructures.
        for item in instance:
            best_bin_ix = None
            best_bin_size = -1
            for i, old_bin in enumerate(bins):
                new_bin = old_bin + item
                # Avoid floating-point rounding issues
                if new_bin <= 1 + 1e-9 and new_bin > best_bin_size:
                    best_bin_ix = i
                    best_bin_size = new_bin

            if best_bin_ix is not None:
                bins[best_bin_ix] += item
            else:
                bins.append(item)

        return len(bins)

    total = 0
    samples = 10_000
    for _ in range(samples):
        random.shuffle(instance)
        total += best_fit(instance)
    mean = total / samples
    return mean / optimum_value(instance)


@funsearch.evolve
def get_items() -> list[float]:
    """Return a new bin-packing-instance, specified by the list of items.

    The items must be floats between 0 and 1.
    """
    items = [0.4, 0.5, 0.6]
    return items
