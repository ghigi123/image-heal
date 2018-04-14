import torch.nn as nn
from models.autoencoder.conv_decoder import Decoder
from models.autoencoder.conv_encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, input_channels=3, encoder_layers=None, decoder_layers=None, bn_size=4):
        # For layers the tuple represent (channels, kernel size)
        if encoder_layers is None:
            encoder_layers = [(32, 5), (64, 3), (128, 3), (256, 3)]

        if decoder_layers is None:
            decoder_layers = [(256, 4), (128, 4), (64, 4), (3, 4)]

        super().__init__()

        self.encoder = Encoder(input_channels, encoder_layers)

        encoder_output_channels = encoder_layers[-1][0]

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(encoder_output_channels, encoder_output_channels * bn_size, 1, groups=encoder_output_channels),
            nn.ConvTranspose2d(encoder_output_channels * bn_size, encoder_output_channels, 1, groups=encoder_output_channels),
            nn.ReLU(True)
        )

        self.decoder = Decoder(encoder_output_channels, decoder_layers)

        # self.nn = nn.Sequential(
        #     self.encoder,
        #     self.bottle_neck,
        #     self.decoder
        # )

    def forward(self, input):
        out = self.encoder(input)
        out = self.bottle_neck(out)
        out = self.decoder(out)
        return out

    def to_tune(self):
        return self


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    x = Variable(torch.rand((1, 3, 64, 64)))
    print(x)
    ae = AutoEncoder()
    print(ae(x))