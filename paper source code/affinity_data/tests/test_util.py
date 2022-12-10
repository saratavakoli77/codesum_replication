from affinity_data.util import *
import pytest


@pytest.mark.parametrize("fast_heuristic", (True, False))
def test_random_permutations(fast_heuristic):
    options = [0, 1, 2, 3]
    for i in range(100):
        res = list(random_permutations(options, 2, 2, faster_heuristic=fast_heuristic))
        print(res)
        assert len(res) == 2
        assert res[0] != res[1]
        for a, b in res:
            assert a in options
            assert b in options
            assert a != b


def test_random_permutations2():
    with pytest.raises(ValueError):
        list(random_permutations([1, 2, 3, 4], count=100, r=2))
