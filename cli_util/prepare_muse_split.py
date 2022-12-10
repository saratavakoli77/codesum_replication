"""
Do the train, val, test splits of muse. Please run prepare_muse first.
"""
import csv
import re
from collections import Counter
from pathlib import Path
import random

import clize
from tqdm import tqdm
from typing import List, Callable

from data_handling.token_proc import camel_case_split, tokenize_non_alphanum


def main(
    *,
    data_path: str = "~/data/code_and_comments/code_and_comments/rawexamples_filt1.csv",
):
    data_path = Path(data_path).expanduser()
    print("Reading data")
    with data_path.open("rb") as fp:
        data = csv.DictReader(
            (line.decode().replace('\0', '') for line in fp), delimiter=",")
        #data = list(itertools.islice(data, 10000))
        data = list(data)
    print("Shuffle.")
    random.shuffle(data)
    # There is supposed to be 50,000 from each language kind.
    # We are going to shuffle and take the first 50k of each file type.
    test_files = {
        (".java",): [], (".py",): [], (".c", ".cpp", ".h", ".hpp", ".c++"): []
    }
    non_test_files = []
    num_weird_files = 0
    weird_file_exts = Counter()
    num_examples_each_test = 10
    for d in tqdm(data, desc="pulling test examples"):
        code, comment, file = d.values()
        # Find which file extension it is
        for exts, examples in test_files.items():
            if any(file.endswith(ext) for ext in exts):
                if len(examples) < num_examples_each_test:
                    examples.append(d)
                else:
                    non_test_files.append(d)
                break
        else:
            num_weird_files += 1
            weird_file_exts.update([file.split(".")[-1]] if "." in file else "NO_DOT")
            non_test_files.append(d)
    print("num weird", num_weird_files)
    print(weird_file_exts)
    print("Shuffle.")
    random.shuffle(non_test_files)
    fraction_in_val = 0.2
    idx_at_val = int(len(non_test_files) * fraction_in_val)
    val, train = non_test_files[:idx_at_val], non_test_files[idx_at_val:]
    del non_test_files

    # Write out the files
    def write_examples(out_examples, out_fn: str):
        dir_root = (data_path.parent / "proc_files")
        (dir_root / (out_fn + ".code.txt")).write_text(
            "\n".join(
                ' '.join(tokenize_code(hack_escape_backslash(ex['code'])))
                for i, ex in enumerate(out_examples)
            )
        )
        (dir_root / (out_fn + ".comment.txt")).write_text(
            "\n".join(
                ' '.join(tokenize_comment(hack_escape_backslash(ex['comment'])))
                for i, ex in enumerate(out_examples)
            )
        )
    print("Writing out train")
    write_examples(train, "muse_train")
    print("Writing out val")
    write_examples(val, "muse_val")
    print("Writing out test")
    for exts, examples in test_files.items():
        write_examples(examples, f"muse_test_{exts[0][1:]}")


_whitespace_split_regex = re.compile(r'\s+')


def apply_to_all_toks(toks: List[str], func: Callable[[str], List[str]]) -> List[str]:
    new_tokens = []
    for tok in toks:
        new_tokens.extend(func(tok))
    return new_tokens


def re_unescape(str_with_escapes: str):
    # https://stackoverflow.com/questions/43662474/reversing-pythons-re-escape
    return re.sub(r'\\(.)', r'\1', str_with_escapes)


def hack_escape_backslash(in_str: str) -> str:
    return in_str.replace("\\", "BCKSL")



def tokenize_code(code: str) -> List[str]:
    # Adapted from https://github.com/sriniiyer/codenn/blob/
    #   0f7fbb8b298a84faf4a14a8f76199e39af685e4a/src/model/buildData.py
    # This is different than whatever MUSE did. This is fine because for
    # IR doesn't really matter.
    code = code.replace("\\", "BS")
    try:
        # It gets unhappy with bad escape sequences. So just hackily convert
        # any backslashes to another character
        tokens = _whitespace_split_regex.sub(code, ' ').split()
    except Exception as e:
        raise e
    tokens = apply_to_all_toks(tokens, lambda s: tokenize_non_alphanum(s, ignore=["_", "/", "*"]))
    return tokens


def split_ing_ly(tok: str) -> List[str]:
    """From the paper we are supposed to split words ending in 'ing' or 'ly'"""
    for ending in ("ing", "ly"):
        if tok.endswith(ending):
            return [tok[:-len(ending)], tok[-len(ending):]]
    return [tok]


def split_s(tok: str) -> List[str]:
    """The papers wants to split of the 's' from words in dictionary"""
    raise NotImplemented()


def tokenize_comment(comment):
    tokens = _whitespace_split_regex.sub(comment, ' ').split()
    tokens = apply_to_all_toks(tokens, tokenize_non_alphanum)
    tokens = apply_to_all_toks(tokens, camel_case_split)
    tokens = apply_to_all_toks(tokens, split_ing_ly)
    return tokens


if __name__ == "__main__":
    #tokenize_code("sdfsadf asdf we a")
    clize.run(main)

