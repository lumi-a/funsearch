"""Find undirected graphs with many edges that don't contain a 3-cycle or 4-cycle.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small code-changes. Do not use np.random.
"""

import funsearch


@funsearch.run
def evaluate(_total_vertex_count: int) -> float:
  """Returns the number of edges in an undirected graph on `total_vertex_count` vertices having no 3-cycles or 4-cycles."""

  BEST_KNOWN = {150: 749, 151: 756, 152: 763, 153: 770, 154: 777, 155: 784, 156: 791, 157: 798, 158: 805, 159: 812, 160: 819, 161: 826, 162: 834, 163: 841, 164: 849, 165: 856, 166: 863, 167: 871, 168: 878, 169: 886, 170: 893, 171: 901, 172: 909, 173: 917, 174: 925, 175: 933, 176: 941, 177: 948, 178: 956, 179: 963, 180: 971, 181: 979, 182: 986, 183: 994, 184: 1001, 185: 1009, 186: 1017, 187: 1025, 188: 1033, 189: 1041, 190: 1049, 191: 1057, 192: 1065, 193: 1073, 194: 1081, 195: 1089, 196: 1097, 197: 1105, 198: 1113, 199: 1121, 200: 1129}  # fmt: skip
  fraction = max(len(solve(x)) / y for x, y in BEST_KNOWN.items())
  return fraction


def solve(total_vertex_count: int) -> set[tuple[int, int]]:
  """Returns a large graph on `total_vertex_count` vertices without 3-cycles or 4-cycles."""

  edge_to_priority = {
    (vertex_v, vertex_w): priority(vertex_v, vertex_w, total_vertex_count)
    for vertex_v in range(total_vertex_count)
    for vertex_w in range(vertex_v + 1, total_vertex_count)
  }
  vertex_to_neighbors = {vertex_v: set() for vertex_v in range(total_vertex_count)}

  for (vertex_v, vertex_w), _ in sorted(edge_to_priority.items(), key=lambda x: -x[1]):
    # Does adding the edge {vertex_v, vertex_w} add a 3-cycle?
    if not vertex_to_neighbors[vertex_v].isdisjoint(vertex_to_neighbors[vertex_w]):
      continue

    # Does adding the edge {vertex_v, vertex_w} add a 4-cycle?
    has_4_cycle = False
    for neighbor_u in vertex_to_neighbors[vertex_v]:
      if not vertex_to_neighbors[neighbor_u].isdisjoint(vertex_to_neighbors[vertex_w]):
        has_4_cycle = True
        break
    if has_4_cycle:
      continue

    # There's no 3-cycle or 4-cycle, so insert the edge
    vertex_to_neighbors[vertex_v].add(vertex_w)
    vertex_to_neighbors[vertex_w].add(vertex_v)

  edges = set(
    (vertex_v, vertex_w)
    for vertex_v in range(total_vertex_count)
    for vertex_w in vertex_to_neighbors[vertex_v]
    if vertex_w > vertex_v
  )
  return edges


@funsearch.evolve
def priority(vertex_v: int, vertex_w: int, total_vertex_count: int) -> float:
  """Returns the priority with which we want to add the undirected edge {vertex_v, vertex_w} to the graph, where vertex_v < vertex_w.
  `total_vertex_count` is the number of vertices in the graph.
  """
  return 0.0
