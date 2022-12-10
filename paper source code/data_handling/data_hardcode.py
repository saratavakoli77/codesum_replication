"""Hard coded loaders for the datasets. This assumes that the datasets
exist at the env variable CODE_COM_DATASETS or a default data/datsets directory
in the project root."""
from pathlib import Path
from typing import Iterable, Dict, Tuple

from data_handling.data_proc import get_all_example_pairs, Document
from extra.util import flatten_list
import os

# Dataset Properties

# TODO: right now the dataset names are kinda stringly typed which is
#   probably not ideal but works for now.
is_case_insensitive_eval = {
    "leclair": True,
    "newfuncom": True,
    "codenn": True,
    #"deepcom": False,
    "hybridDeepCom": True,
    "nl": True,  # TODO verify this right
    "docstring": False,
    "muse": True,
    "ncsPy": True,
    "ncsJava": True,
    "PLOT": False
}
all_datasets = tuple(is_case_insensitive_eval.keys())
relevant_datasets = [
    d for d in all_datasets
    if d not in ("muse", "PLOT")  # Exclude the irrelevant ones.
]

cur_file = Path(__file__).parent.absolute()
_default_datsets_path = cur_file / "../data/datasets"
_env_var_name = 'CODE_COM_DATASETS'
DATASETS_ROOT = Path(
    os.environ.get(_env_var_name, _default_datsets_path)
)

if not DATASETS_ROOT.exists():
    raise ValueError(
        f"No data at expected location {DATASETS_ROOT}. Please make sure the datasets"
        f" are available there or set the {_env_var_name} environment variable"
    )


# Because of the poor design decision to stringly type the dataset names
# have diverged from the ones we use in the paper. As temp hack here is
# list of this mapping
# TODO: fix ^
_dataset_to_name_map = {
    "leclair": "FunCom1",
    "newfuncom": "FunCom2",
    "codenn": "CodeNN",
    #"deepcom": "DeepCom1",
    "hybridDeepCom": "DeepCom2",
    "nl": "NL",
    "docstring": "Docstring1",
    "muse": "Muse",
    "ncsPy": "Docstring2",
    "ncsJava": "DeepCom1",
}
_paper_name_to_name_map = {v: k for k, v in _dataset_to_name_map.items()}


def convert_dataset_name_to_paper_name(dataset: str):
    return _dataset_to_name_map[dataset]


def convert_paper_name_to_dataset_name(dataset: str):
    return _paper_name_to_name_map[dataset]

relevant_datasets_friendly_names = [
    convert_dataset_name_to_paper_name(n) for n in relevant_datasets
]

# Splits

def get_split_name_for_fold(split: str, fold_number: int) -> str:
    """Get the split name for datsets which are packaged with multiple folds
    like in deepcom."""
    assert split in ("train", "val", "test")
    assert fold_number > 0, "fold numbers are 1 indexed to conform to convention" \
                            " set by deepcom"
    return f"{split}_fold-{fold_number}"


def parse_split_fold_name(split_name: str) -> Tuple[str, int]:
    """The inverse of `get_split_name_for_fold`"""
    split = split_name.split("_")[0]
    fold_num = int(split_name.split("-")[1])
    return split, fold_num


# Map dataset to all the valid splits of that dataset
valid_splits: Dict[str, Tuple[str, ...]] = {  # The default
    k: ("train", "val", "test")
    for k in is_case_insensitive_eval.keys()
}
valid_splits['muse'] = tuple([
    f"{split}_{sub}"
    for split in ("train", "val", "test")
    for sub in ("py", "c", "java")
])
valid_splits['hybridDeepCom'] += tuple(flatten_list([
    # in addition to the normal 'RQ1' train, test, val splits,
    # hybrid deepcom also provides 10 folds
    (get_split_name_for_fold("train", fold), get_split_name_for_fold("test", fold))
    for fold in range(1, 10)
]))
valid_splits['nl'] = ("train", "test")


def get_dataset_examples(
    dataset_name: str,
    split: str,
    override_root: str = None
) -> Iterable[Document]:
    assert split in valid_splits[dataset_name], f"Bad split {split}"
    assert dataset_name in all_datasets
    if dataset_name == "deepcom":
        raise ValueError(
            "The name 'deepcom' is now deprecated. This is because what we origionally "
            "thought was DeepCom was actually closer to hybridDeepCom. Now 'ncsJava' "
            "represents the original DeepCom."
        )
    dataset_function = {
        "leclair": _get_leclair_examples,
        "newfuncom": _get_newfuncom_examples,
        "codenn": _get_codenn_examples,
        "deepcom": _get_deepcom_examples,
        "hybridDeepCom": _get_hybrid_deepcom_examples,
        "nl": _get_nl_examples,
        "docstring": _get_docstring_examples,
        "muse": _get_muse_examples, # NOTE: muse dataset not used in final paper
        "ncsPy": _get_ncs_py_examples,
        "ncsJava": _get_ncs_java_examples,
    }[dataset_name]
    return dataset_function(split, override_root)


def get_dataset_examples_all_main_splits(
    datset_name: str,
    override_root: str = None
) -> Iterable[Document]:
    for split in ("train", "val", "test"):
        if split not in valid_splits[datset_name]:
            continue
        yield from get_dataset_examples(datset_name, split, override_root)


def _get_leclair_examples(
    split: str,
    override_root: str
) -> Iterable[Document]: 
    """Old funcom from https://s3.us-east-2.amazonaws.com/icse2018/index.html"""
    data_root = Path(override_root or DATASETS_ROOT / "funcom/data/standard/output").expanduser()
    return get_all_example_pairs(
        data_root / f"dats.{split}",
        data_root / f"coms.{split}",
        split,
        should_remove_special_tags=True
    )


def _get_newfuncom_examples(
    split: str,
    override_root: str
):
    """From http://leclair.tech/data/funcom/#tokdatal"""
    data_root = Path(override_root or DATASETS_ROOT / "new-funcom/funcom_tokenized").expanduser()
    if split == "val":
        split = "valid"  # Files use 'valid' instead of 'val'
    return get_all_example_pairs(
        data_root / f"{split}/functions.{split}",
        data_root / f"{split}/comments.{split}",
        split,
        should_remove_special_tags=False,
        has_id=True,
        id_is_tsv=True,
    )


def _get_codenn_examples(
    split: str,
    override_root: str
):
    in_path = Path(override_root or DATASETS_ROOT / "codenn/proc1").expanduser()
    return get_all_example_pairs(
        in_path / f"codenn_{split}_code.txt",
        in_path / f"codenn_{split}_comments.txt",
        split,
        has_id=False,
        should_remove_special_tags=False,
        dataset_name="codenn"
    )


def _get_deepcom_examples(
    split: str,
    override_root: str
):
    in_path = Path(override_root or DATASETS_ROOT / "deepcom").expanduser()
    return get_all_example_pairs(
        in_path / f"deepcom_code.{split}",
        in_path / f"deepcom_comments.{split}",
        split,
        has_id=False,
        should_remove_special_tags=False
    )


def _get_hybrid_deepcom_examples(
    split: str,
    override_root: str
):
    in_path = Path(override_root or DATASETS_ROOT / "newdeepcom").expanduser()
    in_path /= "Dataset"

    def _get_pairs(code_word_file, comment_file):
        return get_all_example_pairs(
            code_word_file=code_word_file,
            comment_file=comment_file,
            split_name=split,
            has_id=False,
            should_remove_special_tags=False
        )

    if "fold" in split:
        in_path /= "data_RQ4"
        subsplit, fold_num = parse_split_fold_name(split)
        assert subsplit in ("train", "test")
        in_path /= f"fold_{fold_num}/{subsplit}"
        return _get_pairs(
            code_word_file=in_path /f"{subsplit}.token.code",
            comment_file=in_path / f"{subsplit}.token.nl",
        )
    else:  # Not the fold version so use the RQ1 version
        assert split in ("train", "val", "test")
        if split == "val":
            split = "valid"
        in_path /= f"data_RQ1/{split}"
        return _get_pairs(
            code_word_file=in_path / f"{split}.token.code",
            comment_file=in_path / f"{split}.token.nl",
        )


def _get_nl_examples(
    split: str,
    override_root: str
):
    in_path = Path(override_root or DATASETS_ROOT / "nl").expanduser()
    if split == 'val':
        split = "test"  # Only have a test set?
    suffix = "_shuffled_ids" if split == "train" else "_ids"
    return get_all_example_pairs(
        in_path / f"{split}{suffix}.de",
        in_path / f"{split}{suffix}.en",
        split,
        has_id=True,
        should_remove_special_tags=False
    )


def _get_docstring_examples(
    split: str,
    override_root: str
):
    if split == "val":
        split = "valid"  # Files use 'valid' instead of 'val'
    in_path = Path(override_root or DATASETS_ROOT / "code-docstring-corpus/parallel-corpus").expanduser()
    clean_suffix = ".clean" if split == "train" else ""
    return get_all_example_pairs(
        in_path / f"data_ps.declbodies2desc.{split}.bpe{clean_suffix}.db",
        in_path / f"data_ps.declbodies2desc.{split}.bpe{clean_suffix}.d",
        split,
        has_id=False,
        should_remove_special_tags=False
    )


def get_tagged_dataset(dataset: str, tag: str) -> str:
    """Tags can be a dataset under a certain condition, like a fold"""
    # Not sure what solr and stuff can support, so
    return f"datasetTT{tag}"


def _get_muse_examples(
    split: str,
    override_root: str
):
    in_path = Path(
        override_root or DATASETS_ROOT / "code_and_comments/code_and_comments/proc_files"
    ).expanduser()
    return get_all_example_pairs(
        in_path / f"muse_{split}.code.txt",
        in_path / f"muse_{split}.comment.txt",
        split,
        has_id=False,
        should_remove_special_tags=False
    )


def _get_ncs_py_examples(
    split: str,
    override_root: str
):
    in_path = Path(
        override_root or DATASETS_ROOT / "NeuralCodeSum/data/python-method"
    ).expanduser()
    if split == 'val':
        split = "valid"
    return get_all_example_pairs(
        code_word_file=in_path / f"{split}/code.original_subtoken",
        comment_file=in_path / f"{split}/javadoc.original",
        split_name=split,
        has_id=False,
        should_remove_special_tags=False,
    )


def _get_ncs_java_examples(
    split: str,
    override_root: str
):
    in_path = Path(
        override_root or DATASETS_ROOT / "NeuralCodeSum/data/tlcodesum"
    ).expanduser()
    if split == 'val':
        split = "dev"
    return get_all_example_pairs(
        code_word_file=in_path / f"{split}/code.original_subtoken",
        comment_file=in_path / f"{split}/javadoc.original",
        split_name=split,
        has_id=False,
        should_remove_special_tags=False,
    )
