import itertools
import time
import math
import random
from typing import TypeVar, Sequence, Iterable, List

T = TypeVar('T')


def random_permutations(
    items: Sequence[T],
    count: int,
    r: int,
    faster_heuristic: bool = True
) -> Iterable[Sequence[T]]:
    """Samples `count` random unique permutations of size `r`.
    Tries to avoid instantiating all permutations."""
    if len(items) < r:
        raise ValueError(f"Number of items is less than r of {r}")
    max_perms = num_of_valid_permutations_no_replacement(num_items=len(items), r=r)
    if count > max_perms:
        raise ValueError(f"Trying to sample {count} permuations of {r} from {len(items)}")
    if faster_heuristic and (count >= max_perms / 3 or max_perms < 1_000):
        # If we are returning a large enough fraction of the total permutations it is
        # probably faster to just generate them all rather than trying to randomly sample
        # individually. This is an arbitrary and non-benchmarked/optimized heuristic.
        yield from random.sample(list(itertools.permutations(items, r=r)), k=count)
        return
    # Go through and randomly sample permutations keeping track of the ones we add
    # If we are only choosing like 5 our 1 million possible permutations, this will be
    # way more efficient.
    potential_idxs = list(range(len(items)))
    out_perm_inds = set()
    num_yielded = 0
    while num_yielded < count:
        # TODO it would be faster to generate batches of these with numpy
        inds = tuple(random.sample(potential_idxs, k=r))
        if inds in out_perm_inds:
            # We already yielded this permutation
            continue
        out_perm_inds.add(inds)
        yield [items[i] for i in inds]
        num_yielded += 1


def random_permutations_sets(
    item_sets: Sequence[Sequence[T]],
    count: int,
    r: int,
) -> Iterable[Sequence[T]]:
    """Samples random permutations of items. However, the items are divided into
    an arbitrary number of sets. Items in the same set will not be together
    in permutations"""
    if len(item_sets) < r:
        raise ValueError("Number of sets must be at lest r")
    num_yielded = 0
    non_empty_sets = [s for s in item_sets if len(s) != 0]
    if len(non_empty_sets) < r:
        return
    while num_yielded < count:
        sets = random.sample(non_empty_sets, k=r)
        yield tuple(random.choice(items) for items in sets)
    raise NotImplemented


def num_of_valid_permutations_no_replacement(num_items: int, r: int) -> int:
    return math.factorial(num_items) // math.factorial(num_items - r)


def _benchmark_random_permuations():
    items = list(range(1000))
    start = time.time()
    random_permutations(items, count=200, r=2)
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    _benchmark_random_permuations()
