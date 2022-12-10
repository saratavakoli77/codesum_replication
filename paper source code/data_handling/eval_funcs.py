"""Each of the datsets use slightly different eval measures (like different
ways of calculating BLEU). This module tries to capture all of these and
provide unified api for evaling any of the datasets"""
import statistics
from pathlib import Path
from typing import List, Tuple, Union
import nltk
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import otherlib.from_deepcom.evaluate
import otherlib.from_codenn.bleu
from data_handling.data_hardcode import is_case_insensitive_eval
from data_handling.data_proc import Document
from data_handling.data_tokenize import get_tokenized_str_for_dataset_comment
from otherlib.from_moses.wrap_multibleu import moses_multibleu
from rouge import Rouge
import nltk.translate.meteor_score

try:
    import otherlib.pycocoevalcap.meteor.meteor
    import otherlib.pycocoevalcap.rouge.rouge
    import otherlib.pycocoevalcap.cider.cider
except ImportError:
    raise ImportError("Please make sure that pycocoevalcap is available/installed in the "
                      "otherlib library. This was done with something like: \n"
                      ">> git clone https://github.com/tylin/coco-caption.git\n"
                      ">> cd coco-caption\n"
                      ">> git checkout 3a9afb2682141a\n"
                      ">> ln -s coco-caption/pycocevalcap PROJECT_ROOT\n"
                      "Kinda hacky, I know...")
try:
    import otherlib.NeuralCodeSum.main.test
except ImportError:
    raise ImportError("Please install NeuralCodeSum so we have that eval func. See README and see "
                      "https://github.com/wasiahmad/NeuralCodeSum/tree/d563e584")

corpus_bleu_datasets = ("leclair", "newfuncom", "docstring")


EVAL_METRIC_DATASET_AND_NAME = {
    "newfuncom": "BLEU-FC",
    "leclair": "BLEU-FC",
    "codenn": "BLUE-CN",
    "hybridDeepCom": "BLEU-DC",
    "nl": "SacreBLEU",
    "docstring": "BLEU-Moses",
    "ncsPy": "BLEU-ncs",
    "ncsJava": "BLEU-ncs",
    "PLOT": "BLEU-M2"
}
"""A map to relate the datsetname to the eval func name used in the paper"""


def save_and_eval_results(
    refs: List[Union[Document, str]],
    hypotheses: List[str],
    dataset: str,
    save_postfix: str = None,
    pretokenized: bool = False,
) -> float:
    """Prints to std out eval results and returns a 'primary metric' (usually BLEU-4)

    :arg dataset. The name of the dataset.
    :arg save_postfix this is appened when writing the results to a file. This can
        help identifiy where these predictions came from (what model or method).
    :arg pretokenized: the inputs have already been converted to a space seperated
        string for each token
    """
    if pretokenized and isinstance(refs[0], Document):
        raise ValueError("Cannot have documents be pretokenized?")
    if dataset == "PLOT":
        raise NotImplemented()
    ref_strings, hypotheses = _prepare_strings(refs, hypotheses, dataset, pretokenized)
    ids = [d.doc_id for d in refs] if isinstance(refs[0], Document) else None
    ref_file, preds_file = save_outs(
        hypotheses, ref_strings, dataset, save_postfix, ids=ids)
    if dataset == "leclair" or dataset == "newfuncom":
        return eval_leclair(ref_strings, hypotheses)
    elif dataset == "deepcom" or dataset == "hybridDeepCom":
        return eval_deepcom(ref_strings, hypotheses)
    elif dataset == "codenn":
        return eval_codenn(ref_file, preds_file)
    elif dataset == "nl":
        return eval_nl(ref_strings, hypotheses)
    elif dataset == "docstring":
        return eval_docstring(ref_strings, hypotheses)
    elif dataset == "muse":
        return eval_muse(ref_strings, hypotheses)
    elif dataset == "ncsPy" or dataset == "ncsJava":
        return eval_ncs(ref_strings, hypotheses)
    else:
        raise ValueError(dataset)


def _prepare_strings(
    refs: List[Union[Document, str]],
    hypotheses: List[str],
    dataset: str,
    pretokenized: bool = False
) -> Tuple[List[str], List[str]]:
    should_lowercase = is_case_insensitive_eval[dataset]
    def maybe_lower(s: str):
        return s.lower() if should_lowercase else s
    ref_strings = [maybe_lower(r) for r in _get_comments_from_docs_or_strings(refs)]
    hypotheses = [maybe_lower(h) for h in hypotheses]
    if not pretokenized:
        ref_strings = [get_tokenized_str_for_dataset_comment(s, dataset) for s in ref_strings]
        hypotheses = [get_tokenized_str_for_dataset_comment(s, dataset) for s in hypotheses]
    return ref_strings, hypotheses


def _get_comments_from_docs_or_strings(docs_or_strings: List[Union[Document, str]]) -> List[str]:
    if not docs_or_strings:
        return docs_or_strings
    if isinstance(docs_or_strings[0], Document):
        return [d.comment for d in docs_or_strings]
    return docs_or_strings


def save_outs(
    preds: List[str],
    refs: List[str],
    dataset: str,
    save_postfix: str = None,
    ids: List[str] = None
):
    """Write outs in codenn form. This is two paired file, one example per line with
    the text and the id separated by a tab"""
    ref_file = _save_ref_file(save_postfix, dataset)
    pred_file = _save_pred_file(save_postfix, dataset)
    if ids is None:
        ids = list(range(len(preds)))
    assert len(ids) == len(preds) == len(refs)

    def make_str(vals: List[str]):
        return "\n".join([f"{idstr}\t{val}" for val, idstr in zip(vals, ids)])
    ref_file.write_text(make_str(refs))
    pred_file.write_text(make_str(preds))
    return ref_file, pred_file


def _save_root():
    cur_file = Path(__file__).parent.absolute()
    root = Path(cur_file / "../outputs")
    root.mkdir(exist_ok=True)
    return root


def _save_ref_file(save_postfix, dataset):
    return Path(_save_root() / f"{dataset}{'-' + save_postfix if save_postfix else ''}.refs")


def _save_pred_file(save_postfix, dataset):
    return Path(_save_root() / f"{dataset}{'-' + save_postfix if save_postfix else ''}.preds")


def load_saved_outs(
    dataset: str,
    save_postfix: str = None,
) -> Tuple[List[str], List[str]]:
    ref_file = _save_ref_file(save_postfix, dataset)
    pred_file = _save_pred_file(save_postfix, dataset)

    def get_text_with_id(text):
        comma_idx = text.index("\t")
        id, text = text[:comma_idx], text[comma_idx + 1:]
        return text.strip()

    refs, preds = tuple(
        [get_text_with_id(line) for line in file.read_text().split("\n")]
        for file in (ref_file, pred_file)
    )
    return preds, refs



def eval_leclair(refs: List[str], hypotheses: List[str]):
    """Same as funcom"""
    refs = [[r.split()] for r in refs]
    preds = [h.split() for h in hypotheses]
    # Adapted from https://github.com/mcmillco/funcom/blob/master/bleu.py
    Ba = nltk.translate.bleu_score.corpus_bleu(refs, preds)
    B1 = nltk.translate.bleu_score.corpus_bleu(refs, preds, weights=(1, 0, 0, 0))
    B2 = nltk.translate.bleu_score.corpus_bleu(refs, preds, weights=(0, 1, 0, 0))
    B3 = nltk.translate.bleu_score.corpus_bleu(refs, preds, weights=(0, 0, 1, 0))
    B4 = nltk.translate.bleu_score.corpus_bleu(refs, preds, weights=(0, 0, 0, 1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    ret = ''
    ret += ('BLEU for %s functions\n' % (len(preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))
    print(ret)
    return Ba


def eval_deepcom(refs: List[str], hypotheses: List[str]):
    # Deepcom eval doesn't work if have only one token
    # because the nltk smoothing method used divides by the log of the
    # length. When this is 1 it divides by 0. It's a hack, but for now
    # pad our hypotheses when only one token.
    hypotheses = [
        hyp if len(hyp.split()) > 1 else hyp + " pad"
        for hyp in hypotheses
    ]

    assert all(len(hyp.split()) > 1 for hyp in hypotheses)
    _, avg_score = otherlib.from_deepcom.evaluate.nltk_bleu(hypotheses, refs)
    return avg_score * 100


def eval_codenn(refs_file: Path, hypoth_file: Path):
    return otherlib.from_codenn.bleu.from_files(refs_file, hypoth_file)


def eval_nl(refs: List[str], hypotheses: List[str]) -> float:
    """Note the strings should already be space tokenizable"""
    refs = [refs]
    preds = hypotheses
    bleu_results = sacrebleu.corpus_bleu(
        ref_streams=refs, sys_stream=preds, tokenize="none")
    print(bleu_results)
    return bleu_results.score


def eval_muse(refs: List[str], hypotheses: List[str]) -> float:
    # Base this off how deepcom did things, but don't use smoothing since
    # MUSE claimed they don't use it
    #count = 0
    #total_score = 0.0
    #for hyp, ref in zip(hypotheses, refs):
    #    hyp = hyp.split()
    #    ref = ref.split()

    #    score = nltk.translate.bleu([ref], hyp)
    #    total_score += score
    #    count += 1

    #avg_score = total_score / count
    #return avg_score * 100

    #print("ref", refs[67], "hyp", hypotheses[67])
    return eval_deepcom(refs, hypotheses)


def eval_docstring(refs: List[str], hypotheses: List[str]):
    return moses_multibleu(
        [r.split() for r in refs],
        [h.split() for h in hypotheses]
    )


def eval_bleu_1(
    dataset: str,
    refs: List[Union[Document, str]],
    hypotheses: List[str],
):
    ref_strings, hypotheses = _prepare_strings(refs, hypotheses, dataset)
    if dataset not in corpus_bleu_datasets:
        raise NotImplemented("Non-corpus bleu datasets not currently supported")
    refs = [[r.split()] for r in ref_strings]
    preds = [h.split() for h in hypotheses]
    return nltk.translate.bleu_score.corpus_bleu(refs, preds, weights=(1, 0, 0, 0))*100


def eval_rouge_l(
    dataset: str,
    refs: List[Union[Document, str]],
    hypotheses: List[str],
):
    ref_strings, hypoth_strings = _prepare_strings(refs, hypotheses, dataset)
    rouge = Rouge()
    scores = rouge.get_scores(hypoth_strings, ref_strings, avg=True)
    first_score = scores['rouge-l']['f']*100
    ####
    #otherrouge = pycocoevalcap.rouge.rouge.Rouge()
    #ref_dict, hyp_dict = _make_resuls_dict(ref_strings, hypoth_strings)
    #otherscore, scores = otherrouge.compute_score(ref_dict, hyp_dict)
    #otherscore *= 100
    #assert first_score == otherscore, f"{first_score} {otherscore}"
    return first_score


def _make_resuls_dict(ref_strings, hypoth_strings):
    ref_dict = {
        i: [r] for i, r in enumerate(ref_strings)
    }
    hyp_dict = {
        i: [h] for i, h in enumerate(hypoth_strings)
    }
    return ref_dict, hyp_dict


def eval_meteor(
    dataset: str,
    refs: List[Union[Document, str]],
    hypotheses: List[str],
):
    ref_strings, hypoth_strings = _prepare_strings(refs, hypotheses, dataset)
    ref_dict, hyp_dict = _make_resuls_dict(ref_strings, hypoth_strings)
    meteor_class = otherlib.pycocoevalcap.meteor.meteor.Meteor()
    score, scores = meteor_class.compute_score(ref_dict, hyp_dict)
    return score*100
    #return nltk.translate.meteor_score.meteor_score(ref_strings, hypoth_strings)


def eval_cider(
    dataset: str,
    refs: List[Union[Document, str]],
    hypotheses: List[str],
):
    ref_strings, hypoth_strings = _prepare_strings(refs, hypotheses, dataset)
    ref_dict, hyp_dict = _make_resuls_dict(ref_strings, hypoth_strings)
    cider = otherlib.pycocoevalcap.cider.cider.Cider()
    score, scores = cider.compute_score(ref_dict, hyp_dict)
    return score*100


def eval_ncs(
    refs: List[str],
    hypotheses: List[str],
):
    """Eval funcs for https://github.com/wasiahmad/NeuralCodeSum"""
    ref_dict, hyp_dict = _make_resuls_dict(refs, hypotheses)
    metrics = otherlib.NeuralCodeSum.main.test.eval_accuracies(hyp_dict, ref_dict)
    bleu, rouge_l, meteor, precision, recal, f1, ind_bleu, ind_rogue = metrics
    print("NCS METRICS", metrics[:6])
    return bleu


def eval_bleu_m2(
    refs: List[str],
    hypotheses: List[str],
):
    assert len(refs) == len(hypotheses)
    return statistics.mean(
        sentence_bleu(
            references=[r.split()],
            hypothesis=h.split(),
            smoothing_function=SmoothingFunction().method2
        )
        for r, h in zip(refs, hypotheses)
    )*100


