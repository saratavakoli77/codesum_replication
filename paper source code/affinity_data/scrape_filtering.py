import random
from collections import defaultdict
from typing import Sequence, List

from affinity_data.data_representations import DataScrape, ScrappedMethod


def datascrape_no_overload(data: DataScrape) -> DataScrape:
    """
    :param data: The old dataset
    :return: A version of dataset with only one of the with only one of the overloaded
        methods. Which overloaded method to take is chosen randomly.
    """
    new_version = data.filter(lambda _: True)  # Just make exact clone
    for proj in new_version.projects:
        for cls in proj.classes:
            cls.methods = _sample_methods_by_overload(cls.methods)
    return new_version


def datascrape_no_getter_setters(data: DataScrape) -> DataScrape:
    return data.filter(method_is_not_getter_setter)


def method_is_not_getter_setter(method: ScrappedMethod) -> bool:
    method_name = method.method_name
    looks_like_a_getter_setter = (
        len(method_name) >= 3 + 1  # "get" and then something
        and (method_name.lower().startswith("get") or method_name.lower().startswith("set"))
        and (method_name[3].isupper() or method_name[3] == "_")
    )
    return not looks_like_a_getter_setter


def _sample_methods_by_overload(methods: Sequence[ScrappedMethod]) -> List[ScrappedMethod]:
    if len(methods) == 0:
        return []
    # Group methods with same name together
    method_by_name = defaultdict(list)
    for method in methods:
        method_by_name[method.method_name].append(method)
    # Return one from each method name
    return [
        methods[0] if len(methods) == 1 else random.sample(methods, k=1)[0]
        for methods in method_by_name.values()
    ]
