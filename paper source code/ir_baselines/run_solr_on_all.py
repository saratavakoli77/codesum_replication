"""Runs the solr IR baseline on all the datasets"""
from pathlib import Path
import clize
from typing import Tuple
import cli_util.dataset_to_csv
import ir_baselines.runsolr
from data_handling.data_hardcode import relevant_datasets, convert_dataset_name_to_paper_name
from tabulate import tabulate
import math
from data_handling.eval_funcs import EVAL_METRIC_DATASET_AND_NAME
from ir_baselines.ir_index_manage import prepare_dataset, add_to_solr, export_dataset_to_csv
from ir_baselines.multifold_deepcom import run_on_all_hybrid_deepcom_folds

cur_file = Path(__file__).parent.absolute()


def main(
    *,
    num_to_sample: int = None,
    split: str = "test",
):
    assert split in ("val", "test")
    all_results = run_ir_on_all(num_to_sample, split)
    print_results_table(all_results, split)


def run_ir_on_all(
    num_to_sample: int,
    split: str,
):
    dataset_to_score = {}
    for dataset in relevant_datasets:
        print("")
        print("------------------------------------------------------------------")
        print(f"---------- {convert_dataset_name_to_paper_name(dataset)} ({dataset})")
        print("------------------------------------------------------------------")
        print("")
        if dataset == "hybridDeepCom":
            # hybridDeepCom is special special since has many folds
            assert split == "test", "deepcom only has test splits for folds"
            run_on_all_hybrid_deepcom_folds(pickle_results=False)
        else:
            prepare_dataset(dataset)
            print("Run Solr")
            dataset_to_score[dataset] = ir_baselines.runsolr.main(
                dataset=dataset,
                num_to_sample=num_to_sample,
                split=split,
                search_oracle_peak_count=1
            )
    return dataset_to_score


def print_results_table(all_results, split):
    print("-------------------")
    print(f"All IR Results on '{split}' split")
    print(tabulate(
        [
            [  # One row of the table
                convert_dataset_name_to_paper_name(dataset),
                f"{score:.2f}",
                EVAL_METRIC_DATASET_AND_NAME[dataset]
            ]
            for dataset, score in all_results.items()
        ],
        headers=['Dataset', "Score", "Score Method"]
    ))


if __name__ == "__main__":
    clize.run(main)
