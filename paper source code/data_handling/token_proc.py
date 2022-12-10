"""Tokenization specifically as input into IR model for extra processing"""
import itertools
from typing import List
import re
import nltk

from data_handling.token_proc_nl import tokenize_german


def tokenize_and_proc(
    in_str: str, dataset: str, limit_to: int = None
) -> List[str]:
    toks = tokenize_german(in_str, lowercase=True, remove_stop_words=True)
    if dataset == "nl":
        toks = [t for t in toks if not _is_bad_solr_word(t)]
        return toks[:min(len(toks), limit_to or 9e9)]
    else:
        return proc_tokens(toks, limit_to=limit_to)


_camel_case_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')


def camel_case_split(identifier):
    # https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
    matches = _camel_case_regex.finditer(identifier)
    return [m.group(0) for m in matches]


def split_identifier(identifier: str):
    out = []
    for underscore_split in identifier.split("_"):
        out.extend(camel_case_split(underscore_split))
    return out


def _is_bad_solr_word(word):
    return word in ("[", ']', '"', "'", "(", ")", "\\", "/", ',') \
           or '"' in word or "\\" in word


def _is_potential_stop_word(word):
    # These were derived from running stopword_discover.py and looking for
    # words that appeared in a majority of documents accross the datasets
    return word in ("public", ".", ";", "=", "{", "}", "(", ")", "_")


def tokenize_non_alphanum(in_str: str, ignore: List[str] = None) -> List[str]:
    if in_str.isalnum():
        return [in_str]
    ignore = ignore or []
    toks = []
    cur_tok = []
    is_alpha = in_str[0].isalnum()
    for c in in_str:
        if c.isalnum() or c in ignore:
            cur_tok.append(c)
        else:
            toks.append("".join(cur_tok))
            toks.append(c)
            cur_tok = []
    if cur_tok:
        toks.append("".join(cur_tok))
    return toks


def proc_tokens(
    original_toks: List[str],
    filter_stopwords: bool = True,
    limit_to: int = None,
    convert_long_toks_to_bigrams: bool = False
) -> List[str]:
    # Subword split
    new_toks = []
    for tok in original_toks:
        if _is_bad_solr_word(tok):
            continue
        if filter_stopwords and _is_potential_stop_word(tok):
            continue
        new_toks.append(tok)
        if tok in ("CODE_INTEGER", "CODE_STRING", "DUMMY_PERIOD"):
            # Codenn dataset uses these special ids. Don't split them
            continue
        camelCaseToks = split_identifier(tok)
        if len(camelCaseToks) == 1:
            continue
        # Help distinguish the split toks
        camelCaseToks = [f"$${t}$$" for t in camelCaseToks]
        new_toks.extend(camelCaseToks)
        if convert_long_toks_to_bigrams:
            # For really long words we want to split into bigrams of the camelCaseToks
            # This has the advantage of upweighting more complete matches.
            # For example "ApacheWebServerCard" and "ApacheWebClientCard" would
            # be both broken into (apache, web, server, card) and (apache, web, client, card)
            # in adition to (apache&web, web&server, server&card) (apache&web, web&client, client&card)
            # However a disadvantage is since this is split into so many tokens, a full match
            # ends of counting for quite a lot and it adds complexity to our "simple"
            # baseline system.
            camel_case_bigrams = []
            if len(camelCaseToks) > 3:
                # only care if more than 3 because otherwise we already
                # captured it with the whole token.
                camel_case_bigrams.extend(["&".join(bg) for bg in nltk.bigrams(camelCaseToks)])
            new_toks.extend(camel_case_bigrams)

    # Add extra n grams for phrase matching bonus
    #trigram_toks = [
    #    f"$${t1}&{t2}&{t3}$$"
    #    for t1, t2, t3 in nltk.ngrams([
    #        t for t in original_toks
    #        if not _is_bad_solr_word(t) and not _is_potential_stop_word(t)],
    #        n=3
    #    )
    #]
    #new_toks.extend(trigram_toks)

    if limit_to:
        new_toks = new_toks[:min(len(new_toks), limit_to)]

    # lowercase
    new_toks = (t.lower() for t in new_toks)

    # We want the tf-idf counts of our manually created tokens to be different
    # from the tokens in the raw field. So we will add filler characters before
    # to make them different
    new_toks = ["{" + t + "}" for t in new_toks]

    return list(new_toks)


if __name__ == "__main__":
    from data_handling.data_hardcode import get_dataset_examples, all_datasets

    for dataset in all_datasets:
        print(f"--- {dataset} ---")
        for doc in itertools.islice(get_dataset_examples(dataset, "train"), 20):
            print(doc.code_words)
            toks = tokenize_and_proc(doc.code_words)
            print(toks)
            print(len(toks))
