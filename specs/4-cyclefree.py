"""Find undirected graphs with many edges that don't contain a 4-cycle.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small code-changes.
"""

import funsearch


@funsearch.run
def evaluate(total_vertex_count: int) -> float:
    """Returns the number of edges in an undirected graph on `total_vertex_count` vertices that has no 4-cycle."""
    return len(solve(total_vertex_count))


def solve(total_vertex_count: int) -> set[tuple[int, int]]:
    """Returns a large graph on `total_vertex_count` vertices without 4-cycles.

    This incrementally adds edges in order of their priority, skipping edges that would create a 4-cycle.
    """
    edge_to_priority = {
        (vertex_v, vertex_w): priority(vertex_v, vertex_w, total_vertex_count)
        for vertex_v in range(total_vertex_count)
        for vertex_w in range(vertex_v + 1, total_vertex_count)
    }
    vertex_to_neighbors = {vertex_v: set() for vertex_v in range(total_vertex_count)}

    for (vertex_v, vertex_w), _ in sorted(edge_to_priority.items(), key=lambda x: -x[1]):
        # Does adding the edge {vertex_v, vertex_w} add a 4-cycle?
        has_4_cycle = False
        for neighbor_u in vertex_to_neighbors[vertex_v]:
            if not vertex_to_neighbors[neighbor_u].isdisjoint(vertex_to_neighbors[vertex_w]):
                has_4_cycle = True
                break
        if has_4_cycle:
            continue

        # There's no 4-cycle, so insert the edge
        vertex_to_neighbors[vertex_v].add(vertex_w)
        vertex_to_neighbors[vertex_w].add(vertex_v)

    edges = {
        (vertex_v, vertex_w)
        for vertex_v in range(total_vertex_count)
        for vertex_w in vertex_to_neighbors[vertex_v]
        if vertex_w > vertex_v
    }
    return edges


@funsearch.evolve
def priority(vertex_v: int, vertex_w: int, total_vertex_count: int) -> float:
    """Returns the priority with which we want to add the undirected edge {vertex_v, vertex_w} to the graph, where vertex_v < vertex_w.
    `total_vertex_count` is the number of vertices in the graph.
    """
    return 0.0
