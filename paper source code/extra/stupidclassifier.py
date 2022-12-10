"""Can basic classifiers distinguish between certain phrases are in the comment?"""
from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from data_handling.data_hardcode import get_dataset_examples
from data_handling.data_proc import Document
from data_handling.token_proc import tokenize_and_proc

from extra.util import Pmf


def has_csharp(s: str):
    return 1 if "how to" in s else 0
    matches_csharp = "c #" in s
    matches_asp = "asp . net" in s
    return {
        (True, False): 1,
        (False, True): 2,
        (True, True): 3,
    }.get((matches_csharp, matches_asp), 0)
    #if "string" in s:
    #    return 1
    #else:
    #    return 0


def get_docs():
    doc_sets = [
        list(get_dataset_examples("codenn", split))
        for split in ("train", "val")
    ]
    filtered_doc_sets = []
    for docs in doc_sets:
        ys = [has_csharp(d.comment.lower()) for d in docs]
        docs, ys = zip(*[(d, y) for d, y in zip(docs, ys) if y is not None])
        filtered_doc_sets.append((docs, ys))
    return filtered_doc_sets


def get_features(train_docs, val_docs):
    def get_tok_strs(docs: List[Document]):
        return [" ".join(tokenize_and_proc(d.code_words)) for d in docs]
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0)
    train_toks = get_tok_strs(train_docs)
    val_toks = get_tok_strs(val_docs)
    features_train = vectorizer.fit_transform(train_toks)
    features_val = vectorizer.transform(val_toks)
    return features_train, features_val


if __name__ == "__main__":
    (train_docs, train_ys), (val_docs, val_ys) = get_docs()
    train_features, val_features = get_features(train_docs, val_docs)

    model = RandomForestClassifier(n_estimators=100, n_jobs=32, max_depth=5)
    model.fit(train_features, train_ys)

    for split, feats, ys in [("train", train_features, train_ys), ("val", val_features, val_ys)]:
        print("scoring", split)
        score = model.score(feats, ys)
        print(f"score {split} {score}")
        baseline = Pmf(ys)
        baseline.normalize()
        print("basline", baseline)

