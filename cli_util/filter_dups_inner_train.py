import clize
from pathlib import Path

from data_handling.data_hardcode import get_dataset_examples


def main(
    *,
    dataset: str = "deepcom",
    override_out_file_code: str = None,
    override_out_file_comments: str = None,
    override_root: str = None
):
    out_code_file = Path(
        override_out_file_code or f"../serv/{dataset}_train_unique_code.txt")
    out_comments_file = Path(
        override_out_file_comments or f"../serv/{dataset}_train_unique_comments.txt")
    all_docs = list(get_dataset_examples(
        dataset, "train", override_root=override_root))
    # We will convert the documents to a map between their
    # code and comments, and the actual doc. This will only keep
    # one pair for each unique code and comments
    train_pairs = {
        (doc.code_words, doc.comment): doc
       for doc in all_docs
    }
    print("ALL len", len(all_docs))
    print("Set len", len(train_pairs))
    dup_count = len(all_docs) - len(train_pairs)
    print(f"Dup count {dup_count} ({dup_count/len(all_docs)*100:.2f}%)")
    out_code_file.write_text("\n".join(
        doc.code_words for _, doc in train_pairs.items()
    ))
    out_comments_file.write_text("\n".join(
        doc.comment for _, doc in train_pairs.items()
    ))


if __name__ == "__main__":
    clize.run(main)
