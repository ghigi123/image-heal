import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # IN: 3 * 128 * 128 -> 49152
            nn.Conv2d(3, 32, 5, stride=2, padding=3),
            nn.BatchNorm2d(32),
            # 32 * 64 * 64 -> 131072
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            # 32 * 32 * 32 -> 32768
            nn.Conv2d(32, 16, 4, stride=2, padding=2),
            nn.BatchNorm2d(16),
            # 16 * 16 * 16 -> 4096
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
            # CODE: 16 * 8 * 8 -> 1024
            # 2.08 %
        )

        self.decoder = nn.Sequential(
            # CODE: 16 * 8 * 8 -> 1024
            # 2.08 %
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            # 32 * 16 * 16 -> 8192
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            # 16 * 32 * 32 -> 16284
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            # 8 * 64 * 64 -> 32768
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),
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