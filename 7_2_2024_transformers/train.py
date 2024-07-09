import torch
from data import padding_mask


def train_one_epoch(training_loader, model, loss_fn, optimizer):
    """
    Trains the model for one epoch using the given training data loader, model, loss function, and optimizer.
    Args:
        training_loader (torch.utils.data.DataLoader): The data loader for the training data.
        model (torch.nn.Module): The model to be trained.
        loss_fn (torch.nn.modules.loss._Loss): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
    Returns:
        float: The average loss computed over the entire epoch.
    """
    total_loss = 0
    total_items = 0

    for data in training_loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs, mask=padding_mask(inputs))
        loss = loss_fn(torch.select(outputs, 1, 0), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_items += inputs.size(0)

    return total_loss / total_items


def compute_validation_loss(validation_loader, model, loss_fn):
    """
    Computes the validation loss for the given validation data loader using the provided model and loss function.

    Args:
        validation_loader (torch.utils.data.DataLoader): The data loader for the validation data.
        model (torch.nn.Module): The model for which the validation loss is computed.
        loss_fn (torch.nn.modules.loss._Loss): The loss function used to compute the validation loss.

    Returns:
        float: The average validation loss computed over the entire validation dataset.
    """
    with torch.no_grad():
        total_loss = 0
        total_items = 0

        for data in validation_loader:
            inputs, labels = data
            outputs = model(inputs, mask=padding_mask(inputs))
            loss = loss_fn(torch.select(outputs, 1, 0), labels)
            total_loss += loss.item()
            total_items += inputs.size(0)

        return total_loss / total_items
