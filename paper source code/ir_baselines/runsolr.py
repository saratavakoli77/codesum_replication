import random
import clize
import multiprocessing
from functools import partial
from pathlib import Path
from attr import dataclass
from tqdm import tqdm
from typing import List, Iterable, Optional, Tuple
from nltk.translate.bleu_score import sentence_bleu
import pysolr
from data_handling.data_hardcode import get_dataset_examples, relevant_datasets, \
    relevant_datasets_friendly_names, convert_paper_name_to_dataset_name
from data_handling.data_proc import Document, remove_special_tags
from data_handling.eval_funcs import save_and_eval_results, corpus_bleu_datasets, \
    eval_bleu_1, eval_rouge_l, eval_meteor, eval_cider
import time
from data_handling.token_proc import tokenize_and_proc
from extra.util import TOKENIZED_CODE_KEY


def main(
    *,
    dataset: str = "Docstring1",
    num_to_sample: int = None,
    split: str = "test",
    search_oracle_peak_count: int = 1,
    thread_load: float = 0.25
) -> float:
    """
    Run the IR baseline.
    """
    if dataset not in relevant_datasets and dataset in relevant_datasets_friendly_names:
        print(f"Remapping the friendly name {dataset} to the internal dataset name")
        dataset = convert_paper_name_to_dataset_name(dataset)
    print("running on ", dataset, "split", split)
    searcher = DocSearcher(dataset, dataset)
    docs = get_dataset_examples(dataset, split)
    preds = predict_all(
        docs,
        searcher,
        num_to_sample,
        max(int(multiprocessing.cpu_count()*thread_load), 1),
        oracle_peek_count=search_oracle_peak_count
    )
    refs = [r.ref_doc for r, p in preds]
    pred_strings = [
        p.pred_raw if p.pred_raw else "null"
        for r, p in preds
    ]
    bleu = get_and_print_scores_for_strings(refs, pred_strings, dataset)
    return bleu


class DocSearcher:
    def __init__(self, core_name: str, dataset_name: str):
        self.core_name = core_name
        self.dataset = dataset_name
        pass
        #self.solr = pysolr.Solr('http://localhost:8983/solr/leclair', timeout=10)

    def search_code(self, code, split: str = "train", limit=1):
        solr = pysolr.Solr(f'http://localhost:8983/solr/{self.core_name}', timeout=60)
        words = tokenize_and_proc(code, self.core_name, limit_to=1000)
        qwords = [
            f'{TOKENIZED_CODE_KEY}:"{word}"'
            for word in words
        ]
        if not qwords:
            print("No query words", code)
            return []
        #query = f"split_name:{split} AND ({' '.join(qwords)})"
        # ^^ proper way maybe. Temp hack to handle folds. We only export the train split
        #    in the export_csv, so will only return things in training.
        query = f"({' '.join(qwords)})"
        for retry in range(3):
            try:
                results = solr.search(query, rows=limit)
                break
            except pysolr.SolrError as e:
                print("QUERY", query)
                print(e)
                if "Fail to connect to server" in str(e):
                    print("RETRY", retry)
                    time.sleep(5)
                else:
                    return []
        else:
            return []
        for result in results:
            yield parse_solr_result(result)


def parse_solr_result(result) -> Document:
    if 'comment' not in result:
        print("Weird result", result)
    return Document(
        comment=result['comment'][0] if 'comment' in result else "",
        code_words=result['code_words'][0] if 'code_words' in result else "",
        doc_id=result['doc_id'][0],
        split_name=result['split_name'][0]
    )


def predict_comment(
    code,
    searcher: DocSearcher,
    oracle_peek_count: int = 1,
    gt_comment: str = None
) -> Tuple[str, Document]:
    results = list(searcher.search_code(code, limit=oracle_peek_count))
    if oracle_peek_count > 1:
        assert gt_comment is not None
        raise NotImplemented
    if results:
        best_doc = results[0]
    else:
        print(f"NO RESULTS FOR {code}")
        return None, None
    return best_doc.comment, best_doc


def predict_comment_for_doc(doc: Document, searcher: DocSearcher, oracle_peek_count: int):
    return predict_comment(
        doc.code_words, searcher, oracle_peek_count, gt_comment=doc.comment
    )


@dataclass
class PredStuff:
    # A struct for just stuffing prediction stuff haphazardly
    pred_words: List[str]
    pred_raw: str
    pred_doc: Optional[Document]


@dataclass
class RefStuff:
    ref_words: List[List[str]]
    ref_doc: Optional[Document]


def predict_all(
    docs: Iterable[Document],
    searcher: DocSearcher,
    num_to_sample,
    num_procs: int = 1,
    log_output: bool = False,
    skip_exact_matches: bool = False,
    oracle_peek_count: int = 1
) -> List[Tuple[RefStuff, PredStuff]]:
    all_docs = list(docs)
    if num_to_sample and num_to_sample < len(all_docs):
        sampled_docs = random.sample(all_docs, num_to_sample)
    else:
        sampled_docs = all_docs

    import multiprocessing
    print("Num procs", num_procs)
    pool = multiprocessing.Pool(processes=num_procs)
    pred_partial = partial(predict_comment_for_doc, searcher=searcher,
                           oracle_peek_count=oracle_peek_count)
    preds = tqdm(pool.imap(pred_partial, sampled_docs), total=len(sampled_docs))
    exact_match_count = 0
    exact_match_diff_code_count = 0

    outs: List[Tuple[RefStuff, PredStuff]] = []

    for i, (doc, (pred, ret_doc)) in enumerate(zip(sampled_docs, preds)):
        if doc.code_words is None or doc.code_words == "":
            raise ValueError("No code for comment", doc.comment)
        if pred is None:
            pred = doc.code_words
        refs = [remove_special_tags(doc.comment.split())]
        pred_words = remove_special_tags(pred.split())
        if pred == doc.comment:
            exact_match_count += 1
            if doc.code_words != ret_doc.code_words:
                exact_match_diff_code_count += 1
            elif skip_exact_matches:
                continue
        # Periodically print results for debugging if logging is on
        if i % 500 == 0 and log_output:
            print("------")
            print("Iteration: ", i)
            print("PREDICT:")
            print(pred)
            print("ACTUAL:")
            print(doc.comment)
            print("BLUE", sentence_bleu(refs[0], pred_words))
            print("CUR Exact Match Count:", exact_match_count)
            print("CUR Exact Match But Diff Code:", exact_match_diff_code_count)
        # This is kind of sloppy, but just stuff results in a array for later
        outs.append((RefStuff(refs, doc), PredStuff(pred_words, pred, ret_doc)))
    if log_output:
        print("CUR Exact Match Count:", exact_match_count)
        print("CUR Exact Match But Diff Code:", exact_match_diff_code_count)
    return outs


def write_outputs(
    preds: List[Tuple[RefStuff, PredStuff]],
    ref_file: Path,
    pred_file: Path
):
    """This does it in the style that the codenn code expects.
    This is an id followed by a tab, followed by the string"""
    print(ref_file, pred_file)
    with ref_file.open("w") as rf, pred_file.open("w") as pf:
        rf.writelines(
            f"{r.ref_doc.doc_id}\t{r.ref_doc.comment}\n"
            for r, p in preds
        )
        pf.writelines(
            f"{r.ref_doc.doc_id}\t{p.pred_raw}\n"
            for r, p in preds
        )


def get_and_print_scores_for_strings(refs, pred_strings, dataset: str):
    bleu = save_and_eval_results(refs, pred_strings, dataset, "runsolr")
    print("BLEU", bleu)
    if dataset in corpus_bleu_datasets:
        print("Find BLEU 1")
        print(f"BLEU 1: {eval_bleu_1(dataset, refs, pred_strings):.1f}")
    if dataset == "docstring":
        # On these other metrics for docstring since have bleu on the other datasets
        print("Find ROUGE-L")
        print(f"ROUGE-L: {eval_rouge_l(dataset, refs, pred_strings)}")
        print("Find Meteor")
        print(f"METEOR: {eval_meteor(dataset, refs, pred_strings):.1f}")
        print("Find CIDEr")
        print(f"CIDEr: {eval_cider(dataset, refs, pred_strings):.1f}")
    return bleu


if __name__ == "__main__":
    clize.run(main)

