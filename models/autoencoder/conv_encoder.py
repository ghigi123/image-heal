import torch.nn as nn
from models.autoencoder.nn_utils import convolution


class Encoder(nn.Module):
    # Implement autoencoder's decoder
    def __init__(self, in_channels, layers):
        # Layers : [(layer_channels, kernel_size), ...]
        super().__init__()

        layers = [(in_channels,)] + layers

        self.encoder = nn.Sequential(
            *(convolution(layers[i][0], *layers[i + 1]) for i in range(len(layers) - 1))
        )

    def forward(self, x):
        return self.encoder(x)
