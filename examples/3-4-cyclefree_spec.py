"""Find undirected graphs with many edges that don't contain a 3-cycle or 4-cycle.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small code-changes. Do not use np.random.
"""

import funsearch

@funsearch.run
def evaluate(n: int) -> float:
  """Returns the number of edges in an undirected graph on `n` vertices having no 3-cycles or 4-cycles."""
  edges = solve(n)
  return len(edges)

def solve(n: int) -> set[tuple[int, int]]:
  """Returns a large graph on `n` vertices without 3-cycles or 4-cycles."""

  priorities = {(vertex_v, vertex_w): priority(vertex_v, vertex_w, n) for vertex_v in range(n) for vertex_w in range(vertex_v + 1, n)}
  neighbors = {vertex_v: set() for vertex_v in range(n)}

  for (vertex_v, vertex_w), _ in sorted(priorities.items(), key=lambda x: -x[1]):
    # Does adding the edge {vertex_v, vertex_w} add a 3-cycle?
    if not neighbors[vertex_v].isdisjoint(neighbors[vertex_w]):
      break

    # Does adding the edge {vertex_v, vertex_w} add a 4-cycle?
    has_4_cycle = False
    for neighbor_u in neighbors[vertex_v]:
      if not neighbors[neighbor_u].isdisjoint(neighbors[vertex_w]):
        has_4_cycle = True
        break
    if has_4_cycle:
      break

    # Insert edge
    neighbors[vertex_v].add(vertex_w)
    neighbors[vertex_w].add(vertex_v)
  
  edges = set((vertex_v, vertex_w) for vertex_v in range(n) for vertex_w in neighbors[vertex_v] if vertex_w > vertex_v)
  return edges

@funsearch.evolve
def priority(vertex_v: int, vertex_w: int, n: int) -> float:
  """Returns the priority with which we want to add the undirected edge {vertex_v, vertex_w} to the graph. `n` is the number of vertices in the graph.
  """
  return 0.0