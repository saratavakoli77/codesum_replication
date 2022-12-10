import itertools
from data_handling.data_hardcode import get_dataset_examples

if __name__ == "__main__":
    data = get_dataset_examples("codenn", "train")
    for d in itertools.islice(data, 10):
        print(d)
