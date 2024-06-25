import torch
def train_one_epoch(training_loader, model, loss_fn, optimizer):
    """
    Trains the model for one epoch using the given training data loader, model, loss function, and optimizer.
    
    Args:
        training_loader (torch.utils.data.DataLoader): The data loader for the training data.
        model (torch.nn.Module): The model to be trained.
        loss_fn (torch.nn.loss._Loss): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        
    Returns:
        float: The total loss computed over the entire epoch.
    """
    # TODO: Use https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop as a reference.
    pass

def evaluate_model(model, test_dataset):
    """
    Evaluates the model using the provided test dataset and returns the confusion matrix.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_dataset (torch.utils.data.Dataset): The dataset used for evaluation.

    Returns:
        list: A 2x2 confusion matrix where rows represent true labels and columns represent predicted labels.
    """
    model.eval()

    with torch.no_grad():
        confusion_matrix = [[0, 0], [0, 0]]
        # TODO
        pass
