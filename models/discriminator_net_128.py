import torch.nn as nn

class DiscriminatorNet128(nn.Module):
    def __init__(self, n_features = 64):
        super(DiscriminatorNet128, self).__init__()
        self.main = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(3, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # n_features x 64 x 64
            nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # n_features * 2 x 32 x 32
            nn.Conv2d(n_features * 2, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # n_features * 4 x 16 x 16
            nn.Conv2d(n_features * 4, n_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # n_features * 8 x 8 x 8
            nn.Conv2d(n_features * 8, n_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # n_features * 16 x 4 x 4
            nn.Conv2d(n_features * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1,1).squeeze(1)
    
    def to_tune(self):
        return self.main