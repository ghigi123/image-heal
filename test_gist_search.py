from gist import build_gist
from utils import parse_args, dataset_loaders
import torch
from time import time
import numpy as np
import torchvision.utils as vutils


if __name__ == '__main__':
    last_dataset_idx = 10000
    args = parse_args()
    im_size = 3, args.image_size, args.image_size

    gist = build_gist(im_size, kernel_size=7)

    t0 = time()
    dataset, train_loader, test_loader = dataset_loaders(args)

    imgs, targets = zip(*[dataset[i] for i in range(last_dataset_idx)])
    print('Loading images', time() - t0)
    t0 = time()

    gists = gist(torch.stack(imgs, 0))
    print(gists.size())



    def search(image):
        im_gist = gist(torch.stack([image], 0))
        return (gists - im_gist).norm(2, 0).sort()[1]

    print('Building gists', time() - t0)

    for i in range(20):
        t0 = time()

        new_idx = last_dataset_idx + i
        new_image, _ = dataset[new_idx]
        idxs = search(new_image)

        print('Searching nearest', time() - t0)

        vutils.save_image(torch.stack([new_image] + [dataset[idx][0] for idx in idxs[:23]], 0),
                            '%s/test_%s.png' % (args.output_dir, i),
                            normalize=True)