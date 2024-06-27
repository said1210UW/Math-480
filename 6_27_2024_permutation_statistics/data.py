import csv
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch import nn
import torch


def permutation_to_tensor(permutation):
    n = len(permutation)
    return F.one_hot(torch.tensor([c - 1 for c in permutation]), n).float().flatten()


class PermutationDataset(Dataset):
    def __init__(self, n, label_name):
        self.data = []
        self.n = n
        self.label_counts = []
        with open(f"data/permutations_{n}.csv", "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                permutation = eval(row["permutation"])
                label = eval(row[label_name])
                while label >= len(self.label_counts):
                    self.label_counts.append(0)
                self.data.append((permutation_to_tensor(permutation), label))
                self.label_counts[label] += 1

    def class_weights(self):
        return torch.tensor(
            [1 / self.label_counts[i] for i in range(len(self.label_counts))]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
