from torchvision import models
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# let us customize a bit our alexnet

class CustomAlexNet(models.AlexNet):

    def __init__(self, pretrained = True):
        super(CustomAlexNet, self).__init__()
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

        # disable training for feature detection (be carefull not to give every parameteres to optimize to the optimizer)
        for parameter in self.features.parameters():
            parameter.requires_grad = False

        # custom classifier
        self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 3 * 3, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 1), # output is just true or false
                    nn.Sigmoid()
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3)
        x = self.classifier(x)
        return x.squeeze()

    # parameters to tune (excluding features here)
    def to_tune(self):
        return self.classifier