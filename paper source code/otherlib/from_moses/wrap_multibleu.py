import subprocess
from pathlib import Path
import tempfile
from typing import List


def moses_multibleu(refs: List[List[str]], hypotheses: List[List[str]]):
    rf = Path("/tmp/moses-ref-file.txt")
    hf = Path("/tmp/moses-hyp-file.txt")

    def writelines(f: Path, data: List[List[str]]):
        f.write_text("\n".join([
            " ".join(ex) for ex in data
        ]))

    writelines(rf, refs)
    writelines(hf, hypotheses)

    cur_file = Path(__file__).parent.absolute()
    mcall = subprocess.check_output(f"{cur_file / 'multi-bleu.perl'} {rf} < {hf}", shell=True)
    mcall = mcall.decode()
    # Example output
    # BLEU = 0.00, 100.0/100.0/0.0/0.0 (BP=0.513, ratio=0.600, hyp_len=3, ref_len=5)
    assert mcall.startswith("BLEU = ")
    mcall = mcall.split(",")[0]
    assert mcall.startswith("BLEU = ")
    score = float(mcall.split(" ")[2])
    print(mcall)
    return score


if __name__ == "__main__":
    moses_multibleu(
        [["foo", 'bar'], ['a', 'b', 'c'], ['d', 'e', 'f']],
        [["foo"], ['a', 'c'], ['d']]
    )




