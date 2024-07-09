import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        """
        Initializes the PositionalEncoding object with the given `d_model` parameter.

        Parameters:
            d_model (int): The dimensionality of the embedding.
        """
        super().__init__()
        max_len = 256
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if len(x.shape) == 2:
            return x + self.pe.squeeze(0)[: x.size(0)]
        else:
            return x + self.pe[:, : x.size(1)]


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activations = []

    def _ff_block(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        self.activations = x > 0
        return self.dropout2(self.linear2(self.dropout(x)))


class Model(nn.Module):
    def __init__(self, n_vocab, d_model, nhead, num_layers):
        """
        Initializes the Model object with the specified parameters.

        Parameters:
            n_vocab (int): The size of the vocabulary.
            d_model (int): The dimensionality of the embedding.
            nhead (int): The number of heads in the encoder.
            num_layers (int): The number of nn.TransformerEncoderLayer in nn.TransformerEncoder.
        """
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(n_vocab, d_model)
        self.positional = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            CustomTransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True, dropout=0.0
            ),
            num_layers=num_layers,
        )
        self.proj = nn.Linear(d_model, 2)
        self.activations = None

    def forward(self, x, mask=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask matrix, used to mask out padding from attention.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.embed(x) / math.sqrt(self.d_model)
        x = self.positional(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        self.activations = self.encoder.layers[0].activations
        x = self.proj(x)
        return x
