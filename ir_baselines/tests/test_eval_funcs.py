from data_handling.eval_funcs import *


def test_save_load():
    preds = ["foobas sf", "qqss efa", "sdf wa q sdf aw"]
    refs = ["sAs sf", "Ba wqqss efa", "wa sa q sdf aw"]
    dataset = "testtoy"
    save_outs(preds, refs, dataset, None, None)
    restore_preds, restore_refs = load_saved_outs(dataset, None)
    assert restore_preds == preds
    assert refs == restore_refs


