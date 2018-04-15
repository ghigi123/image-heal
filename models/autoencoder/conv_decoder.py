import torch.nn as nn
from models.autoencoder.nn_utils import deconvolution


class Decoder(nn.Module):
    def __init__(self, in_channels, layers):
        super().__init__()

        layers = [(in_channels, )] + layers

        self.decoder = nn.Sequential(
            *(deconvolution(layers[i][0], *layers[i + 1]) for i in range(len(layers) - 2)),
            nn.Sequential(
                nn.ConvTranspose2d(layers[-2][0], layers[-1][0], layers[-1][1], stride=2, padding=(layers[-1][1] - 2) // 2),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        return self.decoder(x)