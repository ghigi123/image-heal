from torchvision.datasets import ImageFolder
import torch
from operator import itemgetter
from collections import Iterable


class ListView:
    def __init__(self, l, view_func, container):
        self._list = l
        self._view_func = view_func
        self._container = container

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._container(self._view_func(element) for element in self._list[item])
        if isinstance(item, Iterable):
            return self._container(self._view_func(self._list[idx]) for idx in item)
        return self._container([self._view_func(self._list[item])])


class ImageFolderBatchable(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.paths = ListView(self.samples, itemgetter(0), list)
        self.images = ListView(self.samples, lambda sample: self.transform(self.loader(sample[0])), lambda seq: torch.stack(list(seq), dim=0))

    def batches(self, n=64):
        for i in range(0, len(self.samples), n):
            yield self.paths[i: i+n], self.images[i: i+n]

