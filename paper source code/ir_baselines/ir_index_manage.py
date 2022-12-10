import pickle
import json
import shutil
import subprocess
from datetime import datetime
import clize
from pathlib import Path
from typing import Tuple
import cli_util.dataset_to_csv
import ir_baselines.runsolr
import time

cur_file = Path(__file__).parent.absolute()
solr_stuff = cur_file / "./solrstuff" 
starttime = str(datetime.now()) 

def get_git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def export_dataset_to_csv(
    dataset: str,
    included_splits: Tuple[str, ...] = ("train",),
):
    assert solr_stuff.exists()
    print(f"Make CSV {dataset} {included_splits}")
    cli_util.dataset_to_csv.main(
        dataset=dataset,
        included_splits=",".join(included_splits)
    )


def prepare_dataset(
    dataset: str,
    included_splits: Tuple[str, ...] = ("train",),
):
    assert solr_stuff.exists()
    print(f"---- Prepare {dataset}")
    export_dataset_to_csv(dataset, included_splits=included_splits)
    print("Exported. Sleep 5.")
    time.sleep(5)
    print(f"Copy to Solr")
    add_to_solr(dataset)


def add_to_solr(dataset: str) -> None:
    assert solr_stuff.exists()
    return_code = subprocess.call([
        str(solr_stuff / "setupcore.sh"), 
        dataset, 
        cli_util.dataset_to_csv.CSV_ROOT
    ])
    if return_code != 0:
        raise ValueError(f"Non-zero {return_code} return code for add_to_solr")


def run_ir(dataset: str, num_to_sample: int = None, split: str = None):
    num_to_sample = num_to_sample or int(9e9)
    split = split or "val"
    return {
        "bleu": ir_baselines.runsolr.main(
            dataset=dataset,
            num_to_sample=num_to_sample,
            split=split
        ),
        "num_to_sample": num_to_sample,
        "time": starttime
    }

