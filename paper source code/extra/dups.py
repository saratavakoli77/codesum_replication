from pathlib import Path
import statistics

def multifold():
    def parse(text):
        return [l.strip() for l in text.split("\n")]

    def read_both(path, split):
        return (
            parse((path / f"{split}/{split}.token.code").read_text()),
            parse((path / f"{split}/{split}.token.nl").read_text())
        )

    fracs = []

    for fold in range(1, 11):
        fold_path = Path(f"~/sanity/data_RQ4/fold_{fold}").expanduser()
        #print("load train")
        train_code, train_nl = read_both(fold_path, "train")
        #print("load test")
        test_code, test_nl = read_both(fold_path, "test")
        #print("make set ")
        train_code_set = set(zip(train_code, train_nl))
        #print("getting overlap")
        overlap = [c for c in zip(test_code, test_nl) if c in train_code_set]
        print(len(overlap))
        print(len(test_code))
        frac = len(overlap) / len(test_code)
        print(fold, "Overlap frac", frac)
        fracs.append(frac)
    print("mean", statistics.mean(fracs))

    print("hello world")

if __name__ == "__main__":
    multifold()
