import torch
from torch.utils.data import DataLoader, IterableDataset
import random
import numpy as np
import math
from constants import device
from scipy.stats import truncnorm

class CopyTask(IterableDataset):
    def __init__(self, max_sequence, n_classes):
        self._max_sequence = max_sequence
        self._n_classes = n_classes
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Make some local copies
        max_seq = self._max_sequence
        n_classes = self._n_classes

        # Generate the random sequence
        n = torch.randint(max_seq//4, (max_seq-1)//2, tuple())
        random_sequence = (torch.rand(n)*n_classes).long() + 1

        # Generate the input, target and loss mask
        x = torch.zeros(max_seq, dtype=torch.long)
        y = torch.zeros(max_seq, dtype=torch.long)
        mask = torch.zeros(max_seq)
        x[:n] = random_sequence
        x[n+1:2*n+1] = random_sequence
        y[:-1] = x[1:]
        mask[n-1:2*n] = 1
        return x, y, mask

class CountTask(IterableDataset):
    def __init__(self, max_sequence):
        self.max_sequence = max_sequence

    def __iter__(self):
        return self

    def __next__(self):
        tok_a = 0
        tok_b = 1
        # Make some local copies
        lengths = random.choices(range(1,12), weights=(10, 6, 4, 3, 1, 1, 1, 1, 1, 1, 1), k=10)
        x = []
        for length in lengths:
            if len(x) + length*2 <= self.max_sequence:
                x += [tok_a]*length + [tok_b]*length
            else:
                x += [tok_a]*(self.max_sequence - len(x))
                break
        x += [tok_a]*(self.max_sequence - len(x))
        y = x[1:] + [tok_a]
        mask = [1 if x[i] == tok_b else 0 for i in range(len(x))]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.float64)
        return x, y, mask

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    # From https://stackoverflow.com/questions/36894191/
    # how-to-get-a-normal-distribution-within-a-range-in-numpy/44308018
    return truncnorm( (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
def normal_gen(max_depth):
    assert max_depth >= 1 and max_depth % 1 == 0
    gen = get_truncated_normal(mean=0, sd=max_depth/2, low=-max_depth, upp=max_depth)
    while True:
        v = math.floor(abs(gen.rvs())) + 1
        while v > max_depth:
            v = math.floor(abs(gen.rvs())) + 1
        yield v # outputs a number from 1 to max_depth, inclusive

class CountTaskWithEOS(IterableDataset):
    @staticmethod
    def get_seq(n, tok_a, tok_b, tok_EOS, max_sequence=None):
        if max_sequence is None:
            max_sequence = 2 * n + 1
        assert 2 * n + 1 <= max_sequence
        x = []
        for _ in range(n):
            x.append(tok_a)
        for _ in range(n):
            x.append(tok_b)
        for _ in range(max_sequence - len(x)):
            x.append(tok_EOS)
        y = x[1:]
        y.append(tok_EOS)
        mask = [1 if x[i] == tok_b else 0 for i in range(len(x))]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.float64)
        return x, y, mask

    def __init__(self, max_sequence, max_depth=12): # max_depth inclusive
        self.max_sequence = max_sequence
        self.max_depth = max_depth
        self.gen = normal_gen(min(max_depth, (max_sequence - 1) // 2))

    def __iter__(self):
        return self

    def __next__(self):
        tok_EOS = 0
        tok_a = 1
        tok_b = 2
        return CountTaskWithEOS.get_seq(next(self.gen), tok_a, tok_b, tok_EOS, self.max_sequence)

class SubjectVerbAgreement(IterableDataset):

    @staticmethod
    def get_seq(n, max_sequence=None, plurality_array=None):
        tok_EOS = 0
        tok_a_s = 1
        tok_a_p = 2
        tok_b_s = 3
        tok_b_p = 4

        is_singular = []
        assert plurality_array is None or len(plurality_array) == n
        for i in range(n):
            if plurality_array is not None:
                is_singular.append(plurality_array[i])
                continue
            if random.random() < 0.5:
                is_singular.append(True)
            else:
                is_singular.append(False)

        if max_sequence is None:
            max_sequence = 2 * n + 1
        assert 2 * n + 1 <= max_sequence
        x = []
        for i in range(n):
            x.append(tok_a_s if is_singular[i] else tok_a_p)
        for i in range(n):
            x.append(tok_b_s if is_singular[n-1-i] else tok_b_p)
        for _ in range(max_sequence - len(x)):
            x.append(tok_EOS)
        y = x[1:]
        y.append(tok_EOS)
        mask = [1 if (x[i] == tok_b_s or x[i] == tok_b_p) else 0 for i in range(len(x))]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.float64)
        return x, y, mask

    def __init__(self, max_sequence, max_depth=12): # max_depth inclusive
        self.max_sequence = max_sequence
        self.max_depth = max_depth
        self.gen = normal_gen(min(max_depth, (max_sequence - 1) // 2))

    def __iter__(self):
        return self

    def __next__(self):
        return SubjectVerbAgreement.get_seq(next(self.gen), self.max_sequence)


if __name__ == "__main__":
    max_depth = 3
    ds = SubjectVerbAgreement(2*max_depth+1, max_depth)
    print(ds.__next__())
    pass
