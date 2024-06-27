import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class PermutationModel(nn.Module):
    def __init__(self, n, layers):
        """
        Initializes a PermutationModel object with the given `n` and a list of layer sizes `layers`.

        Parameters:
            n (int): The number of elements in each permutation.
            layers (List[int]): A list of integers representing the number of nodes in each layer of the model.
        """
        super().__init__()
        self.n = n
        self.layers = nn.ModuleList()
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(nn.Linear(n**2, layers[0]))
            else:
                self.layers.append(nn.Linear(layers[i - 1], layers[i]))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

    def plot_heatmap_feature(
        self,
        layer,
        index,
    ):
        """
        Plots a heatmap of the weights for a specific layer and index.

        Parameters:
            layer (int): The layer for which the weights are to be visualized.
            index (int): The index (i.e. row) of the weights to be visualized.
        """
        weights = self.layers[layer].weight
        maxVal = max(abs(weights.min()), abs(weights.max()))

        # Print the first layer specially, since permutations are one-hot encoded.
        if layer == 0:
            n = self.n
            fig, axes = plt.subplots(1, n, figsize=(n, 1))
            for i in range(n):
                axes[i].imshow(
                    weights[index][i * n : (i + 1) * n].unsqueeze(0).detach().numpy(),
                    cmap="bwr",
                    vmin=-maxVal,
                    vmax=maxVal,
                )
                axes[i].yaxis.set_visible(False)
                axes[i].set_xticks(
                    [x for x in range(n)], [str(x + 1) for x in range(n)]
                )
                axes[i].set_title(f"{i+1}")
            fig.suptitle(f"Row {index}")
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(weights.shape[1], 1))
            ax.imshow(
                weights[index].unsqueeze(0).detach().numpy(),
                cmap="bwr",
                vmin=-maxVal,
                vmax=maxVal,
            )
            ax.yaxis.set_visible(False)
            ax.set_xticks(
                [x for x in range(weights.shape[0])],
                [str(x + 1) for x in range(weights.shape[0])],
            )
            fig.suptitle(f"Row {index}")
            plt.tight_layout()
            plt.show()

    def plot_connections(self):
        """
        Plots the connections between features in the model.

        This function generates a plot that visualizes the connections between features in the model.
        It uses the weights in each layer to determine the thickness and colors of the edges. Thickness
        corresponds to the magnitude of the weight with appropriate normalization per layer and the
        color is red if the weight is positive and blue if the weight is negative.

        Parameters:
            None
        """
        n = self.n
        width = max([n**2] + [layer.weight.shape[0] for layer in self.layers])
        max_val = max(
            [
                max(abs(layer.weight.min()), abs(layer.weight.max()))
                for layer in self.layers
            ]
        )
        xspace = 1
        yspace = width / 3
        radius = 0.1

        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot edges
        for i in range(len(self.layers)):
            for j in range(self.layers[i].weight.shape[0]):
                for k in range(self.layers[i].weight.shape[1]):
                    x = [
                        (k - (self.layers[i].weight.shape[1] - 1) / 2) * xspace,
                        (j - (self.layers[i].weight.shape[0] - 1) / 2) * xspace,
                    ]
                    y = [
                        yspace * (i - len(self.layers) / 2),
                        yspace * (i + 1 - len(self.layers) / 2),
                    ]
                    val = self.layers[i].weight[j][k].item()
                    color = "red" if val > 0 else "blue"
                    plt.plot(
                        x,
                        y,
                        color=color,
                        linestyle="-",
                        linewidth=abs(val) / max_val,
                        zorder=1,
                    )

        # Plot nodes
        rows = [self.n**2] + [layer.weight.shape[0] for layer in self.layers]
        for i in range(len(rows)):
            for j in range(rows[i]):
                circle = plt.Circle(
                    (
                        (j - (rows[i] - 1) / 2) * xspace,
                        (i - (len(rows) - 1) / 2) * yspace,
                    ),
                    radius,
                    color="black",
                    zorder=2,
                )
                ax.add_patch(circle)

        lim = max(width / 2 * xspace + 2 * radius, yspace + 2 * radius)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.title("Model Feature Connections")
        ax.axis("off")
        plt.show()
