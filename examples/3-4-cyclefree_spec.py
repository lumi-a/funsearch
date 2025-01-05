"""Find undirected graphs with many edges that don't contain a 3-cycle or 4-cycle.

On every iteration, improve cyclefree_v1 over the cyclefree_vX methods from previous iterations.
Make only small code-changes. Do not use np.random.
"""

import funsearch
import numpy as np

@funsearch.run
def evaluate(total_vertex_count: int) -> float:
  """Returns the average number of edges in an undirected graph on `total_vertex_count` vertices that has no 3-cycles or 4-cycles."""
  val = len(solve(total_vertex_count))
  print(val)
  return val

def solve(total_vertex_count: int) -> set[tuple[int, int]]:
  """Returns a large graph on `total_vertex_count` vertices without 3-cycles or 4-cycles."""

  graph = cyclefree(total_vertex_count)
  vertex_to_neighbors = {v: set(w for w in range(total_vertex_count) if (graph[v, w]==1 or graph[w, v]==1) and v!=w) for v in range(total_vertex_count)}

  for (vertex_v, vertex_w) in [(v,w) for v in range(total_vertex_count) for w in vertex_to_neighbors[v] if w > v]:
    # Is the edge {vertex_v, vertex_w} contained in a 3-cycle?
    if not vertex_to_neighbors[vertex_v].isdisjoint(vertex_to_neighbors[vertex_w]):
      vertex_to_neighbors[vertex_v].remove(vertex_w)
      vertex_to_neighbors[vertex_w].remove(vertex_v)
      continue

    # Is the edge {vertex_v, vertex_w} contained in a 4-cycle?
    for neighbor_u in vertex_to_neighbors[vertex_v]:
      if neighbor_u != vertex_w and not vertex_to_neighbors[neighbor_u].isdisjoint(vertex_to_neighbors[vertex_w]):
        vertex_to_neighbors[vertex_v].remove(vertex_w)
        vertex_to_neighbors[vertex_w].remove(vertex_v)
        break

  edges = set((vertex_v, vertex_w) for vertex_v in range(total_vertex_count) for vertex_w in vertex_to_neighbors[vertex_v] if vertex_w > vertex_v)
  return edges

@funsearch.evolve
def cyclefree(total_vertex_count: int) -> np.ndarray:
  """Return the adjacency matrix of an undirected graph on `total_vertex_count` vertices that has many edges but contains no 3-cycles or 4-cycles.
  The matrix should only have entries 0 and 1.
  """
  matrix = np.zeros((total_vertex_count, total_vertex_count))
  return matrix