import pickle

from data_handling.data_hardcode import get_dataset_examples, valid_splits, get_split_name_for_fold, relevant_datasets
from cli_util.dataset_to_csv import CSV_ROOT
from tqdm import tqdm
from ir_baselines.ir_index_manage import prepare_dataset, run_ir, add_to_solr
import multiprocessing
import datetime
import cli_util
import shutil
import statistics
import time


SAVE_FILE = "hdc_folds.pkl"

def main():
    run_on_all_hybrid_deepcom_folds()
    #_analyize_bootstrap_outs()
    #make_all_csvs()


dataset = "hybridDeepCom"


def do_write(fold_num):
    print(f"FOLD {fold_num} {datetime.datetime.now()}")
    split_name = get_split_name_for_fold("train", fold_num)
    included_splits = (split_name,)
    print(f"Make CSV {included_splits}")
    cli_util.dataset_to_csv.main(
        dataset=dataset,
        included_splits=",".join(included_splits),
        tag=str(fold_num),
        progress_bar_pos=fold_num - 1
    )


def make_all_csvs():
    with multiprocessing.Pool(10) as pool:
        pool.map(do_write, range(1, 11))


def pre_prepare(fold_num):
    print("Copy file to make thing happy")
    shutil.copy((CSV_ROOT / f"{dataset}-{fold_num}.csv"), (CSV_ROOT / f"{dataset}.csv"))
    print("Add solr")
    add_to_solr(dataset)


def run_on_all_hybrid_deepcom_folds(pickle_results: bool = True):
    dataset = "hybridDeepCom"
    outs = []
    for fold_num in range(1, 10):
        print(f"FOLD {fold_num} {datetime.datetime.now()}")
        split_name = get_split_name_for_fold("train", fold_num)
        prepare_dataset(dataset, included_splits=(split_name,))
        #pre_prepare(fold_num)
        for i in tqdm(range(10), desc='chill'):
            time.sleep(1)
        print("Run IR")
        outs.append(run_ir(
            dataset,
            num_to_sample=None,
            split=get_split_name_for_fold("test", fold_num)
        ))
        print(f"!! Result !! {fold_num}")
        print(outs[-1])
        print("so far...")
        _print_result_data(outs)
    print("DONE")
    print(outs)
    mean_val = _print_result_data(outs)
    if pickle_results:
        with open(SAVE_FILE, "wb") as fp:
            pickle.dump(outs, fp)
    return mean_val


def _print_result_data(data):
    bleus = [d['bleu'] for d in data]
    print("Min", min(bleus))
    print("Max", max(bleus))
    mean = statistics.mean(bleus)
    print("Mean", mean)
    print("Median", statistics.median(bleus))
    return mean


def _analyize_bootstrap_outs():
    with open(SAVE_FILE, "rb") as fp:
        data = pickle.load(fp)
    _print_result_data(data)


def _count_number_of_examples_per_dataset():
    for dataset in relevant_datasets:
        print(dataset)
        total_examples = 0
        for split in valid_splits[dataset]:
            examples = get_dataset_examples(dataset, split)
            num_examples_in_this_split = len(list(examples))
            print("split", num_examples_in_this_split)
            total_examples += num_examples_in_this_split
        print(f"Total examples {total_examples}")


if __name__ == "__main__":
    #_count_number_of_examples_per_dataset()
    main()
