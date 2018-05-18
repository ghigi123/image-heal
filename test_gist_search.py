from gist import build_gist
from utils import parse_args, dataset_loaders
import torch
from time import time
import os
import torchvision.utils as vutils
from vector_store import VectorWriter, VectorReader
import random as rd


def build_descriptor_database(data, descriptor, descriptor_size, filename):
    with VectorWriter(filename, descriptor_size) as vw:
        for batch_paths, batch_imgs in data:
            vw.write_batch(descriptor(batch_imgs), batch_paths)


if __name__ == '__main__':
    last_dataset_idx = 10000
    args = parse_args()
    im_size = 3, args.image_size, args.image_size

    gist = build_gist(im_size, kernel_size=7)

    t0 = time()
    dataset, train_loader, test_loader = dataset_loaders(args)

    paths, imgs = next(dataset.batches(500))

    print('Loading images', time() - t0)
    t0 = time()

    filename = f'out/{args.data_path.split("/")[-1]}_gists'

    if not os.path.exists(f'{filename}.db.binary'):
        build_descriptor_database(dataset.batches(100), gist, 512, filename)
    else:
        print('Not computing gist db')

    with VectorReader(filename) as vr:

        def search(image):
            im_gist = gist(torch.stack([image], 0))
            return (vr[:] - im_gist).norm(2, 1).sort()[1]

        print('Building gists', time() - t0)

        for i in range(20):
            t0 = time()

            new_idx = rd.randint(0, len(vr) - 1)
            new_image, _ = dataset[new_idx]
            idxs = search(new_image)

            print('Searching nearest', time() - t0)

            vutils.save_image(torch.stack([new_image] + [dataset[idx][0] for idx in idxs[:23]], 0),
                                '%s/test_%s.png' % (args.output_dir, i),
                                normalize=True)