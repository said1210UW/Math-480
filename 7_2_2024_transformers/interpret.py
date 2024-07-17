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
    weights = layer.weight.data.cpu().numpy()

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap="bwr")
    plt.colorbar(label = "Weight Value")
    
    # Add labels and title
    plt.title('Heatmap of Weights in Linear Layer')
    plt.xlabel('Input Features')
    plt.ylabel('Output Features')

    plt.show()
  


def incorrect_predictions(model, dataloader,):
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

    device = next(model.parameters()).device

    with torch.no_grad():
        incorrect_predictions = [[], []]

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get Model predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Index of incorrect predictions
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    input_list = inputs[i].tolist()
                    incorrect_predictions.append(input_list)
        
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
    mask = padding_mask(single_input)

    # Get the original model output
    output = model(single_input, mask=mask)
    original_prediction = torch.softmax(output, dim=1)

    result = []
    sequence_length = single_input.size(1)

    for i in range(sequence_length):
        modified_input = single_input.clone()

        modified_input[:, i] = PAD_TOKEN

        # Get the model output with the modified input
        modified_output = model(modified_input, mask = padding_mask(modified_input))
        modified_prediction = torch.softmax(modified_output, dim=1)

        # Calculate the result as the difference between original and modified predictions
        result_item = original_prediction - modified_prediction
        result.append(result_item.detach().cpu().numpy())

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
    model.eval()
    # Register a forward hook to capture the activations
    feedforward_layer = model.feedforward  # Adjust this to your model's architecture
    hook = feedforward_layer.register_forward_hook(
        lambda module, input, output: result.append(output.detach().cpu().numpy())
    )
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(next(model.parameters()).device)
            model(inputs)
    hook.remove()

    # Count activations
    num_features = result[0].size(1)  # Assuming the shape is (batch_size, num_features)
    frequencies = [0] * num_features

    for activations in result:
        activated_indices = torch.argmax(activations, dim=1)
        for index in activated_indices:
            frequencies[index.item()] += 1

    return result
