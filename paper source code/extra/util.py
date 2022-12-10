import hashlib
import pickle
from collections import Counter
from typing import Sequence


def sha256(value) -> str:
    """Hashes a python object into a 256-bit hex string"""
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


class Pmf(Counter):
    # From https://www.dataquest.io/blog/python-counter-class/
    """A Counter with probabilities."""

    def normalize(self):
        """Normalizes the PMF so the probabilities add to 1."""
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total

    def __add__(self, other):
        """Adds two distributions.

        The result is the distribution of sums of values from the
        two distributions.

        other: Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for key1, prob1 in self.items():
            for key2, prob2 in other.items():
                pmf[key1 + key2] += prob1 * prob2
        return pmf

    def __hash__(self):
        """Returns an integer hash value."""
        return id(self)

    def __eq__(self, other):
        return self is other

    def render(self):
        """Returns values and their probabilities, suitable for plotting."""
        return zip(*sorted(self.items()))


TOKENIZED_CODE_KEY = "tokenized_code"


def get_git_sha():
    # https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    import subprocess
    return subprocess.check_output(["git", "describe"]).strip()


def flatten_list(l: Sequence[Sequence]) -> Sequence:
    return [item for sublist in l for item in sublist]