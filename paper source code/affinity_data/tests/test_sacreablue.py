"""A misnomer module. Tests the file that is currently called sacrebleu_testing"""
import itertools
from unittest.mock import MagicMock

from affinity_data.data_representations import DataScrape, ScrappedProject, ScrappedClass, \
    ScrappedMethod
from affinity_data.analyze_affinity import get_intraclass_pairs, run_sacrebleu_intraclass, \
    get_intraproject_pairs, run_all_groups

toy_data = DataScrape(
    projects=[
        ScrappedProject("java", MagicMock(), [
            ScrappedClass(MagicMock(), [
                ScrappedMethod("foo", MagicMock(), "hello there"),
                ScrappedMethod("bar", MagicMock(), "hello world"),
            ], MagicMock(), MagicMock())
        ], [], [])
    ]
)


toy_data_2 = DataScrape(
    projects=[
        ScrappedProject("java", MagicMock(), [
            ScrappedClass(MagicMock(), [
                ScrappedMethod("foo", MagicMock(), "hello there"),
                ScrappedMethod("bar", MagicMock(), "hello world"),
                ScrappedMethod("bar", MagicMock(), "world hello"),
            ], MagicMock(), MagicMock())
        ], [], [])
    ]
)


def test_get_intraclass_pairs():
    a = get_intraclass_pairs(toy_data)
    assert len(a) == 2
    assert len(a[0]) == 2


def test_intra_bleu():
    a = run_sacrebleu_intraclass(toy_data)
    print(a)


def test_get_intraproject():
    a = get_intraproject_pairs(toy_data, sample_fraction=1)
    assert len(a) == 2
    assert len(a[0]) == 2


def test_get_intraproject2():
    refs, sys = get_intraproject_pairs(toy_data_2, sample_fraction=1)
    assert len(refs) == len(sys) == 3


def test_run_all():
    a = run_all_groups(toy_data_2, sample_count=3)
