import torch.nn as nn

from autoencoder.models.nn_utils import convolution


class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.bottle_neck = nn.Sequential(
            # channel wise
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),
            nn.ReLU(True),
            nn.BatchNorm2d(in_channels),
            convolution(in_channels, in_channels, 3, reduction=1)
        )

    def forward(self, x):
        return self.bottle_neck(x)


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    x = Variable(torch.rand((1, 3, 64, 64)))
    print(x)
    bn = Bottleneck(3)
    print(bn(x))