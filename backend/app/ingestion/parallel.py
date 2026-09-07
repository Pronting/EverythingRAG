"""Bounded, ordered thread work: no shared writes and no unbounded future queue."""
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ordered_parallel_map(fn: Callable[[T], R], items: Iterable[T], workers: int) -> Iterator[R]:
    if workers <= 1:
        yield from map(fn, items)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="knowledge-prepare") as pool:
        pending = deque()
        try:
            for item in islice(iterator, workers):
                pending.append(pool.submit(fn, item))
            while pending:
                result = pending.popleft().result()
                for item in islice(iterator, 1):
                    pending.append(pool.submit(fn, item))
                yield result
        finally:
            for future in pending:
                future.cancel()
