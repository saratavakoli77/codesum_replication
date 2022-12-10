import itertools
import statistics
import enum
from collections import defaultdict
from pathlib import Path
import random
from tqdm import tqdm
import sacrebleu
from typing import List, Iterable, Callable, NewType, Tuple, Sequence
from affinity_data.data_representations import DataScrape, ScrappedProject
from affinity_data.method_testing import text_is_english, remove_unwanted_characters_from_comment, \
    is_unicode_encodable
from affinity_data.scrape_filtering import datascrape_no_overload, datascrape_no_getter_setters
from affinity_data.util import random_permutations, num_of_valid_permutations_no_replacement
from data_handling.eval_funcs import save_and_eval_results, eval_bleu_m2, \
    EVAL_METRIC_DATASET_AND_NAME
import pandas as pd
from tabulate import tabulate
import clize
import numpy as np


def apply_basic_comment_list_filters(comments_list: Iterable[str]) -> Iterable[str]:
    comments_list = filter(is_unicode_encodable, comments_list)
    comments_list = filter(text_is_english, comments_list)
    comments_list = map(remove_unwanted_characters_from_comment, comments_list)
    comments_list = map(lambda s: s.strip(), comments_list)
    comments_list = filter(lambda s: len(s) > 0, comments_list)
    return comments_list


def apply_filter_trivial_comments(comments_list: Iterable[str]) -> Iterable[str]:
    return filter(lambda s: len(s) >= 25, comments_list)


CommentFilterCallable = NewType('CommentFilterCallable ',
                                Callable[[Iterable[str]], Iterable[str]])
DatasetFilterCallable = NewType('DatasetFilterCallable ',
                                Callable[[DataScrape], DataScrape])


def get_intraclass_pairs(
    data: DataScrape,
    comment_filter: CommentFilterCallable = apply_basic_comment_list_filters
) -> Tuple[Sequence[str], Sequence[str]]:
    refs, sys = [], []
    for proj in tqdm(data.projects):
        for cls in proj.classes:
            comments_list = cls.all_comments()
            comments_list = comment_filter(comments_list)
            comments_list = list(comments_list)

            new_refs, new_sys = sample_class_comments(comments_list)
            refs.extend(new_refs)
            sys.extend(new_sys)
    return refs, sys


def get_interproject_pairs(
    data: DataScrape,
    sample_count: int,
    comment_filter: CommentFilterCallable = apply_basic_comment_list_filters
):
    all_comments = list(comment_filter(data.iter_comments()))
    refs, sys = zip(*random_permutations(
        all_comments,
        count=sample_count,
        r=2
    ))
    return refs, sys


def get_intraproject_pairs(
    data: DataScrape,
    sample_fraction: float = 0.05,
    comment_filter: CommentFilterCallable = apply_basic_comment_list_filters
):
    """
    :param comment_filter:
    :param data: Data scrape to load from
    :param sample_fraction: Getting all permutations of comments form a project
        can be massive number. Instead for each project we sample the number of
        permutations equal to `sample_fraction` * the number of comments in the
        project. Projects with more comments/methods will be more represented.
    :return:
    """
    assert 0 < sample_fraction <= 1

    def _get_pairs_from_project(project: ScrappedProject):
        all_comments = list(comment_filter(project.all_comments()))
        if len(all_comments) < 2:
            return []
        num_samples = int(len(all_comments) * sample_fraction)
        num_samples = max(1, num_samples)
        yield from random_permutations(
            all_comments,
            count=num_samples,
            r=2
        )

    refs, sys = zip(*  # unzip all the permutation pairs
        itertools.chain.from_iterable(  # Chain together the pairs from every project
            _get_pairs_from_project(project)
            for project in data.projects
        )
    )
    return refs, sys


def get_analagousproj_pairs(
    data: DataScrape,
    sample_fraction: float = 0.05,
    comment_filter: CommentFilterCallable = apply_basic_comment_list_filters
):
    """Pairs where the each are from different projects but the two projects
    are 'analogous'"""
    for p1, p2 in data.paired_projects():
        comments1, comments2 = list(p1.all_comments()), list(p2.all_comments())
        num_samples = int(statistics.mean((len(comments1), len(comments2)))*sample_fraction)
        num_samples = max(1, num_samples)



def run_sacrebleu_intraclass(
    data: DataScrape
):
    refs, sys = get_intraclass_pairs(data)
    refs = [refs]
    bleu = sacrebleu.corpus_bleu(sys, refs)
    print("BLEU: ", bleu)
    return bleu


def eval_all_metrics(
    refs_tokenized: List[str], preds_tokenized: List[str]
):
    results = {}
    done_funcs = set()
    for dataset_name, func_name in EVAL_METRIC_DATASET_AND_NAME.items():
        if dataset_name == "PLOT":
            continue  # Hack
        if func_name in done_funcs:
            continue  # Some datasets share a function
        bleu = save_and_eval_results(
            refs_tokenized,
            preds_tokenized,
            dataset=dataset_name,
            save_postfix="affinity",
            pretokenized=True,
        )
        results[dataset_name] = bleu
        done_funcs.add(func_name)
    # A bit of a hack to throw in the extra one for the plots
    results["PLOT"] = eval_bleu_m2(refs_tokenized, preds_tokenized)
    return results


DEFAULT_TOKENIZE = sacrebleu.tokenize_13a


def tokenize_all(refs, preds, func=DEFAULT_TOKENIZE):
    def proc(strs):
        return list(map(func, strs))
    return proc(refs), proc(preds)


AffinityGroupExamplesIterable = Iterable[
    Tuple[str, Tuple[List[str], List[str]]]
]


def get_all_groups(
    data,
    sample_count: int,
    comment_filter=apply_basic_comment_list_filters,
) -> AffinityGroupExamplesIterable:
    """
    For each affinity group gets the examples. Yielded as a name and the pairs of strings"""
    for name, make_pairs_func in [
        ("interproject", lambda: get_interproject_pairs(
            data, sample_count=sample_count, comment_filter=comment_filter)),
        ("intraproject", lambda: get_intraproject_pairs(
            data, sample_fraction=0.1, comment_filter=comment_filter)),
        ("intraclass", lambda: get_intraclass_pairs(
            data, comment_filter=comment_filter)),
    ]:
        print(f"Get {name}")
        pairs = list(zip(*make_pairs_func()))
        if len(pairs) > sample_count:
            pairs = random.sample(pairs, k=min(sample_count, len(pairs)))
        refs, sys = zip(*pairs)
        yield name, (refs, sys)
    #yield "SimilarAPI", get_similar_api_ref_preds()


def print_all_metrics_on_group(
    group_data: AffinityGroupExamplesIterable
):
    affinity_group_results = {}
    for name, (refs, sys) in group_data:
        print(f"Eval {name}")
        refs_tokenized, sys_tokenized = tokenize_all(refs, sys)
        #print(list(zip(refs_tokenized, sys_tokenized))[:10])
        bleu_by_dataset = eval_all_metrics(refs_tokenized, sys_tokenized)
        affinity_group_results[name] = bleu_by_dataset
    make_table_from_results(affinity_group_results)


def make_table_from_results(affinity_group_results):
    cols = defaultdict(list)
    only_one = True
    for group_i, (group_name, vals) in enumerate(affinity_group_results.items()):
        if only_one and group_name != "intraclass":
            continue
        for score_i, (dataset_name, score) in enumerate(vals.items()):
            func_name = EVAL_METRIC_DATASET_AND_NAME[dataset_name]
            if group_i == 0 or only_one:
                cols['Function'].append(func_name)
            cols[group_name].append(f"{score:.2f}")
    print("TABLE")
    print(tabulate(cols, tablefmt="latex_booktabs", headers="keys"))


def run_sacrebleu_interproject(
    data: DataScrape
):
    print("getting pairs")
    refs, sys = get_interproject_pairs(data, sample_count=100)
    print("finding bleu")
    refs = [refs]
    bleu = sacrebleu.corpus_bleu(sys, refs)
    print("BLEU: ", bleu)
    return bleu


def run_sacrebleu_intraproject(
    data: DataScrape
):
    print("getting pairs")
    refs, sys = get_intraproject_pairs(data)
    print("finding bleu")
    refs = [refs]
    bleu = sacrebleu.corpus_bleu(sys, refs)
    print("BLEU: ", bleu)
    return bleu


def sample_class_comments(comments_list: List[str], max_samples: int = 6):
    """Samples random pairs from a set of comments aware of the count"""
    if len(comments_list) <= 1:
        return [], []
    else:
        refs, sys = zip(*random_permutations(
            comments_list,
            count=min(
                max_samples,
                num_of_valid_permutations_no_replacement(len(comments_list), r=2)
            ),
            r=2
        ))
        return refs, sys


class FilterNames(enum.Enum):
    NoFilter: str = "No Filter"
    NoOverload: str = "No Overload"
    NoOverloadNoGetters: str = "No Overload, No getters/setters"
    AllFilts: str = "No Overload, No getter/setters. No Trivial (>= 25 chars)"

    def __cmp__(self, other):
        if not isinstance(other, FilterNames):
            raise NotImplemented()
        return str(self).__cmp__(str(other))


def get_all_data_filters(
    data,
    sample_count: int
) -> Iterable[Tuple[str, AffinityGroupExamplesIterable]]:
    for filters_names, data_filters, comment_filters in (
            (FilterNames.NoFilter, None, (apply_basic_comment_list_filters,)),
            #(FilterNames.NoOverload, (datascrape_no_overload,), (apply_basic_comment_list_filters,)),
            #(
            #        FilterNames.NoOverloadNoGetters,
            #        (datascrape_no_overload, datascrape_no_getter_setters),
            #        (apply_basic_comment_list_filters,)
            #),
            (
                    FilterNames.AllFilts,
                    (datascrape_no_overload, datascrape_no_getter_setters),
                    (apply_basic_comment_list_filters, apply_filter_trivial_comments)
            ),
    ):
        print(f"-- {filters_names} --")
        if data_filters:
            for data_filter in data_filters:
                filt_data = data_filter(data)
        else:
            filt_data = data

        def comment_filter(comments):
            if comment_filters:
                # Chain the filters together
                for f in comment_filters:
                    comments = f(comments)
            return comments

        yield filters_names, get_all_groups(filt_data, sample_count, comment_filter)


def get_filtered_and_grouped_dataframe(data, sample_count: int) -> pd.DataFrame:
    df_data = []
    for filter_name, groups in get_all_data_filters(data, sample_count):
        for group_name, (refs, preds) in groups:
            for ref, pred in zip(refs, preds):
                df_data.append({
                    "filter_name": filter_name,
                    "group_name": group_name,
                    "ref": DEFAULT_TOKENIZE(ref),
                    "pred": DEFAULT_TOKENIZE(pred)
                })
    return pd.DataFrame(df_data)


def run_all_filters(
    data,
    sample_count: int
):
    for filt_name, group_data in get_all_data_filters(data, sample_count):
        print(f"FILTER {filt_name}")
        print_all_metrics_on_group(group_data)


def load_data_as_df(
    data_scrape_path: Path,
    invalidate_cache: bool = False,
    sample_size: int = 5000
) -> pd.DataFrame:
    df_cache_file = data_scrape_path.parent / (data_scrape_path.name + f".pd_cache{sample_size}")
    compression = "bz2"
    if df_cache_file.exists() and not invalidate_cache:
        # Cache hit
        return pd.read_pickle(str(df_cache_file), compression=compression)
    data = DataScrape.from_serialization(data_scrape_path)
    df = get_filtered_and_grouped_dataframe(data, sample_count=sample_size)
    df.to_pickle(str(df_cache_file), compression=compression)
    return df


def main(
    *,
    sample_count: int = 5000,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    load_path = Path("./data/affinity-data/affinity_fromfile1000-n.pkl.bz2").expanduser()
    print(f"load data from '{load_path}'")
    data = DataScrape.from_serialization(
        load_path
    )
    #run_sacrebleu_intraclass(data)
    #run_sacrebleu_interproject(data)
    #run_sacrebleu_intraproject(data)
    print("sample_count", sample_count)
    run_all_filters(data, sample_count=sample_count)


if __name__ == "__main__":
    clize.run(main)
