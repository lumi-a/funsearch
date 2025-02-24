"""Find a good online-heuristic for the library-problem.

In the library-problem, we consider a physical library with a limited number of bookshelves.
Readers come in and ask to see a book. If we currently have it on display, the reader can simply
read it and then leave. When we don't have it on display, we need to order it, and put it on
display, which incurs a cost for us. If the bookshelves are already full, we must throw out another
book.

Readers arrive sequentially, and each reader arrives only after the previous reader has left and we
(if necessary) ordered and displayed the book the previous reader requested.
Our goal is to minimise the number of books we need to order, i.e. maximize the negative number of orders.

We need to make decisions on-line, meaning we only know the current readers' request, and the
requests of a limited number of readers before them. Books are idealised as integers.
"""

from collections import deque
import random

import funsearch
from funsearch.library import read_accesses

type Book = int


@funsearch.run
def evaluate() -> float:
  """Returns the (negative) number of orders the library had to make."""
  shelf_space = 30
  bookshelf = set()
  past_reader_requests = deque(maxlen=2048)
  book_orders = 0

  for book in read_accesses():
    if book not in bookshelf:
      book_orders += 1
      if len(bookshelf) < shelf_space:
        bookshelf.add(book)
      else:
        book_to_replace = replace(past_reader_requests.copy(), bookshelf.copy(), book)
        if book_to_replace not in bookshelf:
          # Remove the book with the smallest id:
          book_to_replace = min(bookshelf)
        bookshelf.remove(book_to_replace)
        bookshelf.add(book)

    past_reader_requests.append(book)

  return -book_orders


@funsearch.evolve
def replace(past_reader_requests: deque[Book], bookshelf: set[Book], book_order: Book) -> Book:
  """Given our past reader requests (from oldest to newest), the current bookshelf, and the book
  that we'll order, return the book on our bookshelf that should be replaced by the ordered book.
  """
  return min(bookshelf)
