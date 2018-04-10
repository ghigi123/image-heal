import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # IN: 3 * 128 * 128 -> 49152
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            # 16 * 64 * 64 -> 65536
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            # 16 * 32 * 32 -> 16384
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            # 8 * 16 * 16 -> 2048
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
            # CODE: 8 * 8 * 8 -> 512
            # 1.04 %
        )

        self.decoder = nn.Sequential(
            # CODE: 8 * 8 * 8 -> 512
            # 1.04 %
            nn.ConvTranspose2d(8, 16, 2, stride=2),
            # 16 * 16 * 16 -> 4096
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            # 8 * 32 * 32 -> 8192
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, 2, stride=2),
            # 8 * 64 * 64 -> 32768
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2),
            # OUT: 3 * 128 * 128 -> 49152
            nn.Tanh()
        )

    def forward(self, input):
        code = self.encoder(input)
        output = self.decoder(code)
        return output


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    x = Variable(torch.rand((1, 3, 128, 128)))
    print(x)
    ae = Autoencoder()
    print(ae(x))