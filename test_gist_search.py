from gist import build_gist
from utils import parse_args, dataset_loaders, resolve
import torch
from time import time
import os
import torchvision.utils as vutils
from vector_store import VectorWriter, VectorReader
import random as rd
import datasketch
from struct import pack


def build_descriptor_database(data, descriptor, descriptor_size, filename):
    with VectorWriter(filename, descriptor_size) as vw:
        k = 0
        for batch_paths, batch_imgs in data:
            k += 1
            vw.write_batch(descriptor(batch_imgs), batch_paths)
            if k % 5 == 0:
                print(f'\r{k}', end='')
        print(f'\r', end='')


if __name__ == '__main__':
    last_dataset_idx = 10000
    args = parse_args()
    im_size = 3, args.image_size, args.image_size

    gist = build_gist(im_size)

    t0 = time()
    dataset, train_loader, test_loader = dataset_loaders(args)

    paths, imgs = next(dataset.batches(500))

    print('Loading images', time() - t0)
    t0 = time()

    filename = resolve(f'out/{args.data_path.split("/")[-1]}_gists_{gist.features_number}')
    print(filename)

    if not os.path.exists(f'{filename}.db.binary'):
        build_descriptor_database(dataset.batches(100), gist, gist.features_number, filename)
    else:
        print('Not computing gist db')

    with VectorReader(filename) as vr:

        def search(image, n):
            im_gist = gist(torch.stack([image], 0))
            return (vr[:int(len(vr) * 0.8)] - im_gist).norm(2, 1).topk(n, largest=False)[1]

        print('Building gists', time() - t0)

        if args.naive:
            print('Naive search')
            for i in range(20):
                t0 = time()

                new_idx = rd.randint(int(len(vr) * 0.8), len(vr) - 1)
                new_image, _ = dataset[new_idx]
                idxs = search(new_image, 23)

                print('Searching nearest', time() - t0)

                vutils.save_image(dataset.images[[new_idx] + list(idxs)],
                                    '%s/%s_test_naive.png' % (args.output_dir, i),
                                    normalize=True)
        if args.lsh:
            print('Building LSH')
            t0 = time()
            lsh = datasketch.MinHashLSH(num_perm=128, threshold=1)

            for i, vector_hash in enumerate(vr.hashes()):
                m = datasketch.MinHash(num_perm=128)
                m.update(vector_hash)
                lsh.insert(i, m)
                if i > len(vr) * 0.8:
                    break

            print('Building LSH Forest took', time() - t0)
            print('LSH search')
            for i in range(20):
                t0 = time()

                new_idx = rd.randint(int(len(vr) * 0.8), len(vr) - 1)
                new_image = dataset.images[new_idx]
                m = datasketch.MinHash(num_perm=128)
                m.update(pack(f'{vr._vector_size}f', *gist(new_image)[0]))
                idxs = lsh.query(m)

                print('Searching nearest', time() - t0)

                print(new_idx)
                print(idxs)

                vutils.save_image(torch.stack(new_image + [dataset[idx][0] for idx in idxs[:23]], 0),
                                    '%s/%s_test_lsh.png' % (args.output_dir, i),
                                    normalize=True)
