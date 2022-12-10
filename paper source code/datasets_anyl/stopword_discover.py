import random
from collections import Counter
from typing import Iterable, Sequence

import clize

from data_handling.data_hardcode import get_dataset_examples
from data_handling.data_proc import Document
from data_handling.token_proc_nl import tokenize_german


def count_words(docs: Iterable[Document], sample_count = 1000):
    docs: Sequence[Document] = random.sample(list(docs), sample_count)
    word_counter = Counter()
    doc_occur_counter = Counter()
    for doc in docs:
        #words = doc.code_words.split()
        words = tokenize_german(doc.code_words, remove_stop_words=False)
        word_counter.update(words)
        doc_occur_counter.update(set(words))
    print(word_counter.most_common(100))
    print(doc_occur_counter.most_common(100))


def main(
):
    for dataset in ("nl",):#all_datasets:
        print(dataset)
        loader = get_dataset_examples(dataset, split="train")
        count_words(loader)


if __name__ == "__main__":
    clize.run(main)
