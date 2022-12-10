"""Data classes for representing data scrapped from projects"""
import itertools

try:
   import cPickle as pickle
except:
   import pickle

from dataclasses import dataclass
import bz2
from typing import List, Iterable, Union, Callable, Tuple

from pathlib import Path
import copy


@dataclass(frozen=False)
class ScrappedMethod:
    method_name: str
    method_class: 'ScrappedClass' = None
    comment: str = None
    #method_body: str = None

    @property
    def class_name(self):
        if self.method_class is None:
            return None
        return self.method_class.class_name

    def __repr__(self):
        return f"{self.class_name}::{self.method_name}"


@dataclass(frozen=False)
class ScrappedClass:
    project: 'ScrappedProject'
    methods: List[ScrappedMethod]
    file_name: str
    class_name: str
    """Assuming one class per file where class matches file name"""

    def all_comments(self) -> List[str]:
        return [method.comment for method in self.methods if method.comment is not None]

    def filter(self, method_filter: Callable[[ScrappedMethod], bool]) -> 'ScrappedClass':
        """A new version with only methods passing a filter function"""
        new_version = ScrappedClass(
            None,
            [copy.copy(method) for method in self.methods if method_filter(method)],
            self.file_name,
            self.class_name
        )
        for method in new_version.methods:
            method.method_class = new_version
        return new_version


@dataclass(frozen=False)
class FailedScrappedClass:
    project: 'ScrappedProject'
    file_name: str
    class_name: str
    exception: 'UnexpectedUnparseableSourceError'


@dataclass(frozen=False)
class IgnoredClass:
    project: 'ScrappedProject'
    file_name: str
    class_name: str


ClassLike = Union[ScrappedClass, FailedScrappedClass, IgnoredClass]


@dataclass(frozen=True)
class ScrappedProject:
    lang: str
    project_name: str
    classes: List['ScrappedClass']
    failed_classes: List['FailedScrappedClass']
    ignored_classes: List['IgnoredClass']

    def iter_methods(self) -> Iterable[ScrappedMethod]:
        for cls in self.classes:
            for method in cls.methods:
                yield method

    def all_comments(self) -> Iterable[str]:
        for cls in self.classes:
            yield from cls.all_comments()

    def add_class_like(self, class_like: ClassLike) -> None:
        if isinstance(class_like, ScrappedClass):
            self.classes.append(class_like)
            return
        elif isinstance(class_like, FailedScrappedClass):
            self.failed_classes.append(class_like)
            return
        elif isinstance(class_like, IgnoredClass):
            self.ignored_classes.append(class_like)
            return
        raise ValueError("Unknown", class_like)

    def filter(self, method_filter: Callable[[ScrappedMethod], bool]) -> 'ScrappedProject':
        """A new version with only methods passing a filter function"""
        new_version = ScrappedProject(
            self.lang,
            self.project_name,
            [cls.filter(method_filter) for cls in self.classes],
            [copy.copy(cls) for cls in self.failed_classes],
            [copy.copy(cls) for cls in self.ignored_classes]
        )
        for clses in (new_version.classes, new_version.failed_classes, new_version.ignored_classes):
            for cls in clses:
                cls.project = new_version
        return new_version


@dataclass(frozen=True)
class DataScrape:
    projects: List[ScrappedProject]
    project_pairs_inds: List[Tuple[int, int]] = None

    def iter_methods(self) -> Iterable[ScrappedMethod]:
        for proj in self.projects:
            yield from proj.iter_methods()

    def iter_comments(self):
        yield from (
            method.comment
            for method in self.iter_methods()
            if method.comment is not None
        )

    def serialize(self, out_path: Path):
        if str(out_path).endswith("bz2"):
            with bz2.BZ2File(str(out_path), 'w') as fp:
                pickle.dump(self, fp)
        else:
            pickle.dump(self, out_path.open('wb'))

    def paired_projects(self) -> Iterable[Tuple[ScrappedProject, ScrappedProject]]:
        for p1, p2 in self.project_pairs_inds:
            yield self.projects[p1], self.projects[p2]

    @classmethod
    def from_serialization(cls, load_path: Path) -> 'DataScrape':
        if str(load_path).endswith("bz2"):
            with bz2.BZ2File(str(load_path), 'rb') as fp:
                load = pickle.load(fp)
        else:
            load = pickle.load(load_path.open('rb'))
        assert isinstance(load, DataScrape)
        return load

    def filter(self, method_filter: Callable[[ScrappedMethod], bool]) -> 'DataScrape':
        """A new version with only methods passing a filter function"""
        return DataScrape(
            projects=[proj.filter(method_filter) for proj in self.projects]
        )



# We define our own set of errors of things that might go wrong during parsing.
# We do this rather than reusing the exceptions in javalang because more things can go
# wrong then just those exceptions. Also, we end up not defining them as actual
# python exceptions so that we have better control with serializing them.
class UnexpectedUnparseableSourceError():
    def __init__(self, message: str = None):
        self.message = message


class JavaWeirdAttributeError(UnexpectedUnparseableSourceError):
    pass


class Java9ModuleError(UnexpectedUnparseableSourceError):
    pass


class JavaBaseError(UnexpectedUnparseableSourceError):
    pass


class JavaRecursionLimit(UnexpectedUnparseableSourceError):
    pass


class JavaSyntaxError(UnexpectedUnparseableSourceError):
    pass

class FileReadError(UnexpectedUnparseableSourceError):
    pass

