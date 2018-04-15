import torch.nn as nn

from autoencoder.models.nn_utils import deconvolution


class Decoder(nn.Module):
    def __init__(self, in_channels, layers):
        super().__init__()

        layers = [(in_channels, )] + layers

        self.decoder = nn.Sequential(
            *(deconvolution(layers[i][0], *layers[i + 1]) for i in range(len(layers) - 1))
        )

    def forward(self, x):
        return self.decoder(x)