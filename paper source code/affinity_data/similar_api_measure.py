from pathlib import Path
import csv



def load_examples():
    cur_file = Path(__file__).parent.absolute()
    pair_file = (cur_file / "manual_api_collection.csv")
    csv_data = csv.DictReader(pair_file.read_text().split("\n"))
    a_comments, b_comments = zip(*[
        (pair['comment_a'], pair['comment_b'])
        for pair in csv_data
    ])
    return a_comments, b_comments


def get_similar_api_ref_preds():
    a_comments, b_comments = load_examples()
    # It is not clear which set is the 'reference', so just do it both ways
    refs = a_comments + b_comments
    preds = b_comments + a_comments
    return refs, preds


def print_things():
    from affinity_data.analyze_affinity import eval_all_metrics, tokenize_all
    refs, preds = tokenize_all(*get_similar_api_ref_preds())
    print("COUNT", len(refs))
    print(eval_all_metrics(refs, preds))


if __name__ == "__main__":
    print_things()
