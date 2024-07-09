import csv
import torch
from torch.utils.data import Dataset
from torch import nn

CLS_TOKEN = 2  # Classification token
PAD_TOKEN = 3  # Padding token
MAX_LEN = 32


def parenthesization_to_tensor(parenthesization):
    """
    Converts a given parenthesization string into a tensor representation.

    Args:
        parenthesization (str): The parenthesization string to be converted.

    Returns:
        torch.Tensor: The tensor representation of the parenthesization. The tensor has shape (MAX_LEN,) and contains
        integers representing the corresponding parenthesization character. The CLS_TOKEN is added at the beginning
        of the tensor, followed by the binary representation of the parenthesization string. The PAD_TOKEN is added
        at the end of the tensor to pad the tensor to a length of MAX_LEN.
    """
    pad_length = MAX_LEN - len(parenthesization) - 1
    return torch.tensor(
        [CLS_TOKEN]
        + [0 if c == "(" else 1 for c in parenthesization]
        + [PAD_TOKEN] * pad_length
    )


def padding_mask(src):
    """
    Returns a boolean mask indicating which elements in the input tensor `src` are equal to the `PAD_TOKEN`.

    Parameters:
        src (torch.Tensor): The input tensor of shape (batch_size, seq_len).

    Returns:
        torch.Tensor: A boolean mask of shape (batch_size, seq_len) indicating which elements in `src` are equal to `PAD_TOKEN`.
    """
    return src == PAD_TOKEN


class ParenthesizationDataset(Dataset):
    def __init__(self, type="training"):
        self.data = []
        filename = f"data/{type}.csv"
        with open(filename, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                self.data.append(
                    (
                        parenthesization_to_tensor(row["parenthesization"]),
                        int(row["valid"]),
                    )
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
