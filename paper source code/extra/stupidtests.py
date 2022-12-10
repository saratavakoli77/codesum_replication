"""How well can you do just always returning the same thing?"""
from collections import Counter

import nltk

from data_handling.data_hardcode import get_dataset_examples
from data_handling.eval_funcs import save_and_eval_results
from statistics import mean


if __name__ == "__main__":
    datasets = "docstring"
    data = list(get_dataset_examples(datasets, "train"))
    unigrams = Counter()
    doc_unigrams = Counter()
    bigrams = Counter()
    doc_bigrams = Counter()
    trigrams = Counter()
    tetragrams = Counter()
    toks_by_doc = [d.comment.lower().split() for d in data]
    for toks in toks_by_doc:
        unigrams.update(toks)
        doc_unigrams.update(set(toks))
        bigrams.update(nltk.bigrams(toks))
        doc_bigrams.update(set(nltk.bigrams(toks)))
        trigrams.update(nltk.trigrams(toks))
        tetragrams.update(nltk.ngrams(toks, n=4))
    print(unigrams.most_common(40))
    print(bigrams.most_common(20))
    print(sum(bigrams.values()))
    print(trigrams.most_common(20))
    print(tetragrams.most_common(20))

    print("Tok len", mean(len(t) for t in toks_by_doc))
    print("doc set")
    print(doc_unigrams.most_common(20))
    print(doc_bigrams.most_common(20))

    hypoths = [
        "how to get in c # ?"
        for d in data
    ]
    #for d in data:
    #    print(d.comment)
    save_and_eval_results(data, hypoths, datasets, "stupid")