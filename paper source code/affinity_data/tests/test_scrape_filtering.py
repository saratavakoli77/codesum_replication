from unittest.mock import MagicMock

from affinity_data.data_representations import ScrappedClass, ScrappedProject
from affinity_data.scrape_filtering import *

toy_data_overload = DataScrape(
    projects=[
        ScrappedProject("java", MagicMock(), [
            ScrappedClass(MagicMock(), [
                ScrappedMethod("foo", MagicMock(), "hello there"),
                ScrappedMethod("bar", MagicMock(), "hello world"),
                ScrappedMethod("foo", MagicMock(), "goodbye"),
            ], MagicMock(), MagicMock())
        ], [], [])
    ]
)


def test_filter_overload():
    r = datascrape_no_overload(toy_data_overload)
    assert len(list(r.iter_comments())) == 2
    assert len(list(toy_data_overload.iter_comments())) == 3
