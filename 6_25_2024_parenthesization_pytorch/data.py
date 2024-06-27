import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch import nn

device = "cpu"

def parenthesization_to_tensor(parenthesization):
    """
    Convert a parenthesization string to a pytorch tensor representation.

    Args:
        parenthesization (str): The parenthesization string to convert.

    Returns:
        torch.Tensor: The tensor representation of the parenthesization.
        The tensor has shape (4*n), where 2*n is the length of the parenthesization string.
        Each element in the tensor is either 0 or 1, representing whether the corresponding
        parenthesization character is "(" or ")".
    """ 


    # Create  our List to store the parenthesization results
    intial_ParenList = []
    # Traverse through this string and add our intial parenthesis list
    for char in parenthesization:
        if char == "(":
            intial_ParenList.append(0)
        elif char == ")":
            intial_ParenList.append(1)
    secondary_ParenList = []
    
    # Traverse through this list and add our secondary parenthesis list
    for value in intial_ParenList:
        if value == 0:
            secondary_ParenList.append([1,0])
        elif value == 1:
            secondary_ParenList.append([0,1])
            
    # Flatten our Secondary List 
    Flattened_ParenList = []
    for sublist in secondary_ParenList:
        for item in sublist:
            Flattened_ParenList.append(item)
        
    return torch.tensor(Flattened_ParenList, dtype=torch.int8)


class ParenthesizationDataset(Dataset):
    def __init__(self, n):
        self.data = []
        filename = f"data/parenthesizations_{n}.csv"
        with open(filename, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                self.data.append((parenthesization_to_tensor(row["parenthesization"]), int(row["valid"])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ParenthesizationModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(2*2*n, 2)

    def forward(self, x):
        return self.fc(x)
