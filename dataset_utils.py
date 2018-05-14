from torchvision.datasets import ImageFolder
import torch
from operator import itemgetter
from functools import partial

class ListView:
    def __init__(self, l, view_func, container):
        self._list = l
        self._view_func = view_func
        self._container = container

    def __getitem__(self, item):
        return self._container(self._view_func(element) for element in self._list[item])


class ImageFolderBatchable(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.paths = ListView(self.samples, itemgetter(0), list)
        self.images = ListView(self.samples, lambda sample: self.transform(self.loader(sample[0])), lambda seq: torch.stack(list(seq), dim=0))
