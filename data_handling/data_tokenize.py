from itertools import islice
from typing import List
import sacrebleu

from data_handling.data_hardcode import is_case_insensitive_eval, relevant_datasets, \
    get_dataset_examples
from data_handling.token_proc_nl import tokenize_german


def get_tokens_for_dataset_comment(comment: str, dataset: str) -> List[str]:
    if dataset == "nl":
        toks = sacrebleu.tokenize_13a(comment).split(" ")
    else:
        toks = comment.split(" ")
    if is_case_insensitive_eval[dataset]:
        toks = [t.lower() for t in toks]
    return toks


def get_tokenized_str_for_dataset_comment(comment: str, dataset: str) -> str:
    """Space seperated string of comment tokens"""
    return " ".join(get_tokens_for_dataset_comment(comment, dataset))


def get_tokens_for_dataset_code(code_words: str, dataset: str) -> List[str]:
    if dataset == "nl":
        return tokenize_german(code_words, lowercase=True, remove_stop_words=True)
    return code_words.split()


def get_tokenized_str_for_dataset_code(code_words: str, dataset: str) -> str:
    """Space seperated string of code_words"""
    return " ".join(get_tokens_for_dataset_code(code_words, dataset))


if __name__ == "__main__":
    for dataset in relevant_datasets:
        print("--- DATASET", dataset)
        for example in islice(get_dataset_examples(dataset, "train"), 10):
            print("comment", get_tokens_for_dataset_comment(example.comment, dataset))
            print("code", get_tokens_for_dataset_code(example.code_words, dataset))
