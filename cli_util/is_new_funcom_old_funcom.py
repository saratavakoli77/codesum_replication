import itertools
import random

from collections import Counter

from data_handling.data_hardcode import get_dataset_examples, valid_splits
import pprint


def _every_example(dataset: str, only_splits = None):
    docs = itertools.chain.from_iterable(
        get_dataset_examples(dataset, split)
        for split in (only_splits if only_splits is not None else valid_splits[dataset])
    )
    yield from (
        (doc.code_words, doc.comment)
        for doc in docs
    )


def main():
    each_split_as_a_sequence_by_itself = ((s,) for s in valid_splits["newfuncom"])
    dif_str = None
    for split in (valid_splits['newfuncom'], *each_split_as_a_sequence_by_itself):
        print(f"Only {split}:")
        #print("  load old")
        old_funcom = Counter(_every_example("leclair", only_splits=split))
        #print("  load new")
        new_funcom = Counter(_every_example("newfuncom", only_splits=split))
        print(f"  Examples in old_funcom {sum(old_funcom.values())}. With {len(old_funcom)} unique")
        print(f"  Examples in new_funcom {sum(new_funcom.values())}. With {len(new_funcom)} unique")
        diff = sum((new_funcom - old_funcom).values())
        print("  Set difference (new_funcom - old_funcom):", diff)
        if diff == 0 and dif_str is None:
            dif_str = "Some of the examples that are in old but not in new\n"
            dif_str += pprint.pformat(
                random.sample((old_funcom - new_funcom).keys(), k=5),
                width=60
            )
    if dif_str:
        print(dif_str)


if __name__ == "__main__":
    main()
