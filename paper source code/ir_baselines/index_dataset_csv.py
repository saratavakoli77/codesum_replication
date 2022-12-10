import clize
from typing import Generic, TypeVar

from data_handling.data_hardcode import relevant_datasets_friendly_names, \
    convert_paper_name_to_dataset_name
from extra.clize_util import one_of
from ir_baselines.ir_index_manage import export_dataset_to_csv, add_to_solr


def main(
    *,
    dataset: one_of(*relevant_datasets_friendly_names) = "CodeNN",
):
    """
    Index a dataset csv in solr.
    """
    # Map to non-paper name
    dataset_real_name = convert_paper_name_to_dataset_name(dataset)
    add_to_solr(dataset_real_name)


if __name__ == "__main__":
    clize.run(main)
