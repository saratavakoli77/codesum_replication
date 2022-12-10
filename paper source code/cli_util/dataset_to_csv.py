"""A generic version of all the others to convert datasets to a csv."""
import csv
from tqdm import tqdm
import itertools
from pathlib import Path
from dataclasses import asdict
from typing import Iterable, Optional

import clize

from data_handling.data_hardcode import get_dataset_examples, DATASETS_ROOT
from data_handling.data_proc import Document
from extra.util import TOKENIZED_CODE_KEY
from data_handling.token_proc import tokenize_and_proc

CSV_ROOT: Path = DATASETS_ROOT / "../csv_files"


def main(
    *,
    dataset: str = "muse",
    override_out_file: str = None,
    tag: str = None,
    override_root: str = None,
    included_splits: Optional[str] = "train",  # Should be comma separated list to make clize happy
    progress_bar_pos: Optional[int] = 0
):
    cur_file = Path(__file__).parent.absolute()
    out_file = Path(override_out_file or str(
        CSV_ROOT / f"{dataset}{'-' + tag if tag else ''}.csv"))
    out_file.parent.mkdir(exist_ok=True, parents=True)
    included_splits = included_splits.split(",") if included_splits else ("train", "test", "val")
    write_docs_to_csv(
        data=itertools.chain.from_iterable((
            # We are going to chain together iterator for every split
            get_dataset_examples(dataset, split, override_root=override_root)
            for split in included_splits
        )),
        out_path=out_file,
        dataset=dataset,
        progress_bar_pos=progress_bar_pos
    )


def write_docs_to_csv(
    data: Iterable[Document],
    out_path: Path,
    dataset: str = None,
    include_extra_tokenizations: bool = True,
    progress_bar_pos: int = 0,
):
    with out_path.open("w", buffering=1024*1024) as out_file:
        field_names = list(Document.__dataclass_fields__.keys())
        if include_extra_tokenizations:
            field_names.append(TOKENIZED_CODE_KEY)
        writer = csv.DictWriter(
            out_file,
            fieldnames=field_names
        )
        writer.writeheader()
        for doc in tqdm(
            data,
            desc=f"Write {dataset} to {out_path}",
            position=progress_bar_pos,
            leave=False
        ):
            doc_dict = asdict(doc)
            if include_extra_tokenizations:
                doc_dict[TOKENIZED_CODE_KEY] = " ".join(
                    tokenize_and_proc(doc.code_words, dataset))
            writer.writerow(doc_dict)


if __name__ == "__main__":
    clize.run(main)

