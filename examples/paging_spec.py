"""Find good online-heuristics for the paging-problem.

On every iteration, improve replace_v1 over the replace_vX methods from previous iterations.
Make only small code-changes.
"""

from typing import List
import funsearch
from numpy import random


@funsearch.run
def evaluate(_: int) -> float:
  """Returns the (negative) number of page-faults of the paging-problem."""

  accesses = random.randint(1, 100, size=10000)
  num_pages = 30
  pages = []
  page_faults = 0

  for i, page in enumerate(accesses):
    if page not in pages:
      if len(pages) < num_pages:
        pages.append(page)
      else:
        index = replace(accesses[:i], pages, page)
        pages[index] = page
        page_faults += 1

  return -page_faults


@funsearch.evolve
def replace(access_history: List[int], pages: List[int], new_page: int) -> int:
  """Given a list of `pages` and a `new_page` not currently in `pages`, return
  the index of the page that `new_page` should replace in `pages`.
  `access_history` is the list of all page-accesses up to this point, but excluding `new_page`.
  """
  return 0
