from gist import build_gist, get_masked_areas, build_gabor_conv
from utils import parse_args, dataset_loaders, resolve
import torch
from time import time
import os
import torchvision.utils as vutils
from vector_store import VectorWriter, VectorReader
import random as rd
from struct import pack
from shutil import copyfile
from prox_masking import best_translation, get_prox_op


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

        def search(image, n=200, blanks=None):
            im_gist_diff = (vr[:int(len(vr) * 0.8)] - gist(image))

            if blanks is not None:
                for i, j in blanks:
                    start = (i * 4 + j) * gist.filters_number
                    im_gist_diff[:, start:start + gist.filters_number] = 0

            return im_gist_diff.norm(2, 1).topk(n, largest=False)[1]

        print('Building gists', time() - t0)

        if args.naive:
            print('Naive search')
            for i in range(20):
                t0 = time()

                searched_image_idx = rd.randint(int(len(vr) * 0.8), len(vr) - 1)
                searched_image = dataset.images[searched_image_idx]
                mask = torch.ones(searched_image.size()[1:])
                mask[:, :im_size[1] // 4, :] = 0
                assert get_masked_areas(mask) == [(0, 0), (0, 1), (0, 2), (0, 3)]

                found_image_idxs = search(searched_image, blanks=get_masked_areas(mask))
                found_images = dataset.images[found_image_idxs[:23]]
                paths = dataset.paths[[searched_image_idx] + list(found_image_idxs[:40])]

                # vutils.save_image(mask, 'default_mask.jpg')
                # for i, path in enumerate(paths):
                #     fn = os.path.split(path)[-1]
                #     copyfile(path, os.path.join('./out/ex/', f'{i}_{fn}'))

                print('Searching nearest', time() - t0)

                vutils.save_image(torch.cat([searched_image, found_images], dim=0),
                                '%s/%s_test_naive.png' % (args.output_dir, i),
                                normalize=True)

                get_prox_mask = get_prox_op(im_size)
                gabor_conv = build_gabor_conv(im_size)
                prox_mask = get_prox_mask(mask)

                for found_image in found_images:
                    best_t, (mi, mj) = best_translation(
                        searched_image[0],
                        found_image,
                        prox_mask,
                        image_transform=gabor_conv,
                        punition=lambda x,y: 5 * (x + y)
                    )

                    print(mi, mj)


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

                searched_image_idx = rd.randint(int(len(vr) * 0.8), len(vr) - 1)
                new_image = dataset.images[searched_image_idx]
                m = datasketch.MinHash(num_perm=128)
                m.update(pack(f'{vr._vector_size}f', *gist(new_image)[0]))
                found_image_idxs = lsh.query(m)

                print('Searching nearest', time() - t0)

                print(searched_image_idx)
                print(found_image_idxs)

                vutils.save_image(torch.stack(new_image + [dataset[idx][0] for idx in found_image_idxs[:23]], 0),
                                    '%s/%s_test_lsh.png' % (args.output_dir, i),
                                  normalize=True)
