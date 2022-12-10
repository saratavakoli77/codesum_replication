"""
A converter over from the dataloaders to dump files for the plotting code. Not
ideal interfacing, but will work until can be refactored
"""
from data_handling.data_hardcode import relevant_datasets, get_dataset_examples_all_main_splits, \
    convert_dataset_name_to_paper_name
import clize
import random
from pathlib import Path
from typing import List
from tqdm import tqdm

from data_handling.data_tokenize import get_tokenized_str_for_dataset_code, get_tokenized_str_for_dataset_comment
cur_file = Path(__file__).parent.absolute()


def save_univariate_data(comments: List[str], dataset: str):
    # TODO change to env variable
    dataset_friendly = convert_dataset_name_to_paper_name(dataset)
    save_path = cur_file / f"../data/plot-data/univariate/{dataset_friendly}_comments.txt"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_path.write_text(
        "\n".join(comments)
    )


def get_bivariate_files(dataset: str):
    # TODO change to env variable
    dataset_friendly = convert_dataset_name_to_paper_name(dataset)
    save_root = cur_file / f"../data/plot-data/bivariate/{dataset_friendly}"
    save_root.mkdir(exist_ok=True, parents=True)
    return (save_root / "code.txt"), (save_root / "comments.txt")


def save_bivariate_data(code: List[str], comments: List[str], dataset: str):
    code_f, comment_f = get_bivariate_files(dataset)
    code_f.write_text("\n".join(code))
    code_f.write_text("\n".join(comments))


def main(
    univariate_sample_size: int = 5000
):
    random.seed(42)
    for dataset in relevant_datasets:
        print("Doing", dataset)
        data = list(tqdm(get_dataset_examples_all_main_splits(dataset)))
        ###
        print("Shuffling")
        data = random.shuffle()
        ###
        code_tokenized = [
            get_tokenized_str_for_dataset_code(d.code_words, dataset)
            for d in tqdm(data, desc="tokenize code")
        ]
        comment_tokenized = [
            get_tokenized_str_for_dataset_comment(d.comment, dataset)
            for d in tqdm(data, desc="tokenize comments")
        ]
        print("  Save univariate")
        save_univariate_data(random.sample(comment_tokenized, k=univariate_sample_size), dataset)
        print("  Save bivariate")
        save_bivariate_data(code_tokenized, comment_tokenized, dataset)


if __name__ == "__main__":
    clize.run(main)
