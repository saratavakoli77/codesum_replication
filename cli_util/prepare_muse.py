'''
This code adapted from initialize.py of the MUSE dataset. This is available
https://muse-portal.net/datasets
'''
import csv
import json, sys
import os
import sqlite3
import time
from pathlib import Path
from tqdm import tqdm

def muse_pass_len(code, comment) -> bool:
    """
    Section 3.1.2
    There- fore, we exclude samples where the code has fewer than 8 characters...
    Similarly, we find that very long code samples are unlikely to be
    well-summarized by a single comment, so we exclude samples where the code
    contains more than 4,096 characters.
    """
    return 8 <= len(code) <= 4096


def muse_pass_ignore_phrases(code, comment):
    """From apendix
    Per Section
    3.1.2, we exclude code/comment pairs if the comment contains
    specific words or phrases.
    """
    return not any(
        test_phrase in comment
        for test_phrase in [
            "created by",
            "thanks to",
            "precondition",
            "copyright",
            "do not remove",
            "bug",
            "fix",
            "?",
            "->",
            ">>>",
            "(self,"
        ]
    )


def muse_pass_filter(code, comment):
    return muse_pass_len(code, comment) \
           and muse_pass_ignore_phrases(code, comment)


def get_first_sentence_of_comment(comment) -> str:
    """
    Criteria specified in 3.1.2 and apendix
    """
    # Is this inclusive or exclusive???
    # Going to assume it exclusive.
    sent_endings = [".", "\n \n", ":param", "@param", "@return", "@rtype"]
    def get_first_occur(find_str):
        try:
            return comment.index(find_str)
        except ValueError:
            # Fallback to the whole string
            return len(comment)
    sent_end_idx = min(get_first_occur(ending) for ending in sent_endings)
    return comment[:sent_end_idx]


def convert(
    out_path: str = "~/data/code_and_comments/code_and_comments/rawexamples_filt1.csv",
    #out_path: str = "/tmp/out.txt",
    filter: bool = True
):
    DELETE = False
    DATA_DIR = Path("~/data/code_and_comments/code_and_comments/data").expanduser()

    # Deduplicate any code/comment pairs
    dedupe_set = set()

    start = time.time()
    # walk through the data directory
    print("Walking projects...")
    pbar = tqdm(
        total=len(os.listdir(DATA_DIR))*2,
        mininterval=10
    )

    with Path(out_path).expanduser().open("w") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=("code", "comment", "filename")
        )
        writer.writeheader()
        i = 0
        pass_first_filter, pass_dup, pass_all_filt, pass_len_and_words = 0, 0, 0, 0
        for (dirpath, dirnames, filenames) in os.walk(DATA_DIR):
            for filename in filenames:
                # whenever you get to one of the project's code/comment data files
                if filename.endswith('.json'):
                    # open up the file, load the json, close the file.
                    fo = open(os.path.join(dirpath, filename), 'rb')
                    d = json.load(fo)
                    fo.close()
                    # get a list of [code,comment,filename] triplets from the json
                    filename_code_comment_triplets = d['contents']
                    # add them to the output file
                    for triplet in filename_code_comment_triplets:
                        filename, code, comment = triplet
                        # sometimes developers copy code/comments exactly from other
                        # code. This results in duplication, which we avoid here
                        codecomment_pair = hash(code + comment)
                        if codecomment_pair in dedupe_set:
                            continue
                        # otherwise store the data in the hash set
                        dedupe_set.add(codecomment_pair)
                        pass_dup += 1
                        # sometimes doxygen fails to extract the code or a comment,
                        # don't store these pairs in the database
                        if code == '' or comment == '':
                            continue
                        pass_first_filter += 1
                        # Run filters
                        if filter and not muse_pass_filter(code, comment):
                            continue
                        # Only include first sentence
                        if filter:
                            comment = get_first_sentence_of_comment(comment)
                        pass_len_and_words += 1
                        code = code.strip()
                        comment = comment.strip()
                        if filter and (len(comment) == 0 or len(code) == 0):
                            continue
                        pass_all_filt += 1
                        #print("---")
                        #print("CODE: ", code)
                        #print("len", len(code))
                        #print("comment: ", comment)
                        #print("fn: ", filename)
                        #i += 1
                        #if i == 50:
                        #    exit(1)
                        writer.writerow({
                            "code": code,
                            "comment": comment,
                            "filename": filename
                        })

            # progress bar update
            pbar.update(1)

    print(" ")
    print("Database populated")
    # Delete the json file from original project's derivatives folder.
    # Choosing to enable this option helps reduce the disk space
    # consumed by working with this dataset.
    if DELETE == True:
        print("Deleting raw files from data directory...")
        os.system('find ' + DATA_DIR + ' -name "*.json" -type f -delete')

    end = time.time()
    print("Completed in %d seconds" % (end - start))
    print(pass_dup, "pass_dup")
    print(pass_first_filter, "pass_first_filt")
    print(pass_len_and_words, "pass_len_and_words")
    print(pass_all_filt, "pass_all_filt")


def count_lines():
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            print("fail", maxInt)
            maxInt = int(maxInt / 10)
    with Path(
        "~/data/code_and_comments/code_and_comments/rawexamples_filt1.csv"
    ).expanduser().open("rb") as fp:
        #print("count", sum(1 for _ in csv.reader(fp)))
        data = csv.reader((line.decode().replace('\0', '') for line in fp), delimiter=",")
        count = 0
        for _ in tqdm(data):
            count += 1
    print("Found", count)


if __name__ == "__main__":
    convert()
    #count_lines()