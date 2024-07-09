import matplotlib.pyplot as plt
import torch

from data import PAD_TOKEN, padding_mask


def plot_linear_layer(layer,):
    """
    Plots a heatmap of the weights for a linear layer.

    Parameters:
        layer (nn.Linear): The Linear layer for which the weights are to be visualized.
    """
    # TODO
    
    # Get the weights of the layer
    weights = layer.weight.detach().numpy()

    plt.imshow(weights, cmap="bwr")
    


def incorrect_predictions(model, dataloader):
    """
    Given a model and a dataloader, this function evaluates the model by predicting the labels for each input in the dataloader.
    It keeps track of incorrect predictions and returns a list of inputs that were incorrectly predicted for each label.

    Args:
        model (torch.nn.Module): The model used for prediction.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.

    Returns:
        List[List[List[int]]]: A list of incorrect predictions for each label. The list contains two sublists, one for each label.
            Each sublist contains a list of inputs that were incorrectly predicted for that label.
            Each input is represented as a list of integers.
    """
    model.eval()

    with torch.no_grad():
        incorrect_predictions = [[], []]
        # TODO
        for inputs, labels in dataloader:
            inputs =inputs.to(model.device)
            labels = labels.to(model.device)
        
            output = model(inputs)
            predicted_labels = torch.argmax(output, dim=1)

            for i in range(len(labels)):
                if labels[i] != predicted_labels[i]:
                    incorrect_predictions[labels[i].item()].append(inputs[i].tolist())
    return incorrect_predictions


def token_contributions(model, single_input):
    """
    Calculates the contributions of each token in the single_input sequence to each class in the model's
    predicted output. The contribution of a single token is calculated as the difference between the
    model output with the given input and the model output with the single token changed to the other
    parenthesis.

    Args:
        model (torch.nn.Module): The model used for prediction.
        single_input (torch.Tensor): The input sequence for which token contributions are calculated.

    Returns:
        List[float]: A list of contributions of each token to the model's output.
    """
    output = model(single_input, mask=padding_mask(single_input))

    result = []
    # TODO
    return result

def activations(model, dataloader):
    """
    Returns the frequency of each hidden feature's activation in the feedforward layer of the model
    over all inputs in the dataloader.

    Args:
        model (torch.nn.Module): The model used for prediction.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.

    Returns:
        List[int]: A list of frequencies for each hidden feature in the feedforward layer of the model.
    """
    result = []
    # TODO
    return result
