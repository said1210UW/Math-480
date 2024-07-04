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
    model.train()
    total_loss = 0

    for batch in training_loader:
        # Data inputs and targets
        inputs, labels = batch

        # Zero the gradients
        optimizer.zero_grad()
        
        # Prediction y-hat
        outputs = model(inputs)

        #Loss computation and its gradient
        loss = loss_fn(outputs, labels)
        loss.backward()

        #Adjust weights
        optimizer.step()

        #Gather Data
        total_loss += loss.item()
   
    return total_loss

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
        for data in test_dataset:
            input, label = data
            outputs = model(input)
            confusion_matrix[label][torch.argmax(outputs)] += 1
            
    return confusion_matrix
