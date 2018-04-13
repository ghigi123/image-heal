import torch.nn as nn


def convolution(input_channels, output_channels, kernel_size, reduction=2):
    pad = (kernel_size - reduction) // 2
    if (kernel_size - reduction) % 2 == 0:
        padl, padr = pad, pad
    else:
        padl, padr = pad, pad + 1

    return nn.Sequential(
        # assertion that reduction | input_size -> input_size / reduction = output_size int
        nn.ReflectionPad2d((padl, padr, padl, padr)),
        nn.Conv2d(input_channels, output_channels, kernel_size, reduction),
        nn.ReLU(True),
        nn.BatchNorm2d(output_channels),
    )


def deconvolution(input_channels, output_channels, kernel_size, augment=2):
    pad = (kernel_size - augment) // 2
    if (kernel_size - augment) % 2 == 0:
        padl, padr = pad, pad
    else:
        padl, padr = pad, pad + 1

    return nn.Sequential(
        # assertion that reduction | input_size -> input_size / reduction = output_size int
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=augment, padding=pad),
        nn.ReLU(True),
        nn.BatchNorm2d(output_channels),
    )


class AutoEncoder(nn.Module):
    def __init__(self, input_size=(3, 64), encoder_layers_params=None, decoder_layers_params=None, bn_size=1024):
        # For layers the tuple represent (channels, kernel size)

        super(AutoEncoder, self).__init__()

        if encoder_layers_params is None:
            encoder_layers_params = [(32, 5), (64, 3), (128, 3), (256, 3)]

        if decoder_layers_params is None:
            decoder_layers_params = [(256, 4), (128, 4), (64, 4), (3, 4)]

        n_convolutions = len(encoder_layers_params)

        encoder_layers_params = [input_size] + encoder_layers_params

        self.encoder = nn.Sequential(
            *(convolution(encoder_layers_params[i][0], *encoder_layers_params[i + 1]) for i in range(n_convolutions))
        )

        encoder_output_channels = encoder_layers_params[-1][0]

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(encoder_output_channels, encoder_output_channels * bn_size, 1, groups=encoder_output_channels),
            nn.ConvTranspose2d(encoder_output_channels * bn_size, encoder_output_channels, 1, groups=encoder_output_channels),
            nn.ReLU(True)
        )

        n_deconvolutions = len(decoder_layers_params)

        decoder_layers_params = [(encoder_output_channels, -1)] + decoder_layers_params

        self.decoder = nn.Sequential(
            *(deconvolution(decoder_layers_params[i][0], *decoder_layers_params[i + 1]) for i in range(n_deconvolutions))
        )

        # self.hr = nn.Sequential(
        #     nn.Conv2d(3, 16, 5, stride=1, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 3, 5, stride=1, padding=2),
        #     nn.Tanh()
        # )

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