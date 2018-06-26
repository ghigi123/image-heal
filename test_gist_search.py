from gist import build_gist, get_masked_areas, build_gabor_conv
from utils import parse_args, dataset_loaders, resolve
import torch
from time import time
import os
import torchvision.utils as vutils
from vector_store import VectorWriter, VectorReader
import random as rd
from prox_masking import best_translation, get_prox_op, dumb_seamcut
from img_grad import compute_image_grad
import heapq


def load_bar(it):
    total = len(it)
    for i, item in enumerate(it):
        print(f'\r{100 * (i + 1) / total:.2f}%', end='')
        yield item
    print()

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

        
        print('Naive search')
        for i in range(20):
            t0 = time()

            searched_image_idx = rd.randint(int(len(vr) * 0.8), len(vr) - 1)
            searched_image = dataset.images[searched_image_idx]
            mask = torch.ones(searched_image.size()[1:])
            mask[:, :im_size[1] // 5, :im_size[1] // 2] = 0

            assert get_masked_areas(mask) == [(0, 0), (0, 1)]

            found_image_idxs = search(searched_image, blanks=get_masked_areas(mask))
            found_images = dataset.images[found_image_idxs]
            paths = dataset.paths[[searched_image_idx] + list(found_image_idxs[:40])]

            vutils.save_image(mask, 'default_mask.jpg')

            print('Searching nearest', time() - t0)

            vutils.save_image(torch.cat([searched_image, found_images[:23]], dim=0),
                            '%s/%s_test_naive.png' % (args.output_dir, i),
                            normalize=True)

            get_prox_mask = get_prox_op(im_size)
            gabor_conv = build_gabor_conv(im_size)
            prox_mask = get_prox_mask(mask)

            patched_images = []

            for found_image in load_bar(found_images):
                best_t, (mi, mj) = best_translation(
                    searched_image[0],
                    found_image,
                    prox_mask,
                    image_transform=gabor_conv,
                    punition=lambda x, y: 5 * (x + y),
                    result_scoring=lambda image: int(compute_image_grad(dumb_seamcut(searched_image[0], mask, image)))
                )

                patched = dumb_seamcut(searched_image[0], mask, best_t)
                patched_images.append(patched)

            top23 = heapq.nlargest(23, patched_images, key=lambda image: compute_image_grad(image))

            vutils.save_image([searched_image[0]] + top23,
                            '%s/%s_test_naive_patched.png' % (args.output_dir, i),
                            normalize=True)
