"""Find good online-heuristics for the paging-problem.

On every iteration, improve replace_v1 over the replace_vX methods from previous iterations.
Make only small code-changes.
"""

from typing import List
import funsearch
from funsearch.paging import read_accesses
from collections import deque


@funsearch.run
def evaluate(_: int) -> float:
  """Returns the (negative) number of page-faults of the paging-problem."""

  num_pages = 30
  pages_list = []
  history = deque(maxlen=1000)
  page_faults = 0

  for page in read_accesses():
    if page not in pages_list:
      if len(pages_list) < num_pages:
        pages_list.append(page)
      else:
        index = max(0, min(replace(history, pages_list, page), num_pages - 1))
        pages_list[index] = page
        page_faults += 1

    history.append(page)

  return -page_faults


@funsearch.evolve
def replace(access_history: List[int], pages: List[int], new_page: int) -> int:
  """Given a list of `pages` and a `new_page` not currently in `pages`, return
  the index of the page that `new_page` should replace in `pages`.
  `access_history` is the list of the past 1000 accesses, but excluding `new_page`.
  """
  return 0
