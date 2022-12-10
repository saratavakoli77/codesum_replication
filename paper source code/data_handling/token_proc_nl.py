import itertools
from typing import List
from somajo import SoMaJo

_german_tokenizer = None


def tokenize_german(
    in_str: str,
    lowercase: bool = True,
    remove_stop_words: bool = True
) -> List[str]:
    global _german_tokenizer
    if _german_tokenizer is None:
        _german_tokenizer = SoMaJo("de_CMC", split_camel_case=True, split_sentences=False)
    tokens = list(_german_tokenizer.tokenize_text([in_str]))
    tokens = [
        tok.text
        for tok in tokens[0]
    ]
    if lowercase:
        tokens = [t.lower() for t in tokens]
    if remove_stop_words:
        tokens = [t for t in tokens if t not in _german_stop_words]
    return tokens


_german_stop_words = {
    ".", ",", "die", 'und', "der", "in", "das", "den", "von", "zu", "ist",
    "sie", "mit", "für", "auf"
}


if __name__ == "__main__":
    from data_handling.data_hardcode import get_dataset_examples

    for doc in itertools.islice(get_dataset_examples("nl", "test"), 20):
        print(doc.code_words)
        toks = tokenize_german(doc.code_words)
        print(toks)
        print(len(toks))
