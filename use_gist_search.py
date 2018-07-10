from gist import build_gist, get_masked_areas, build_gabor_conv
from utils import parse_args, dataset_loaders, resolve
import torch
from time import time
import os
import torchvision.utils as vutils
from vector_store import VectorWriter, VectorReader
import random as rd
from prox_masking import best_translation, get_prox_op, dumb_seamcut, blending_seamcut
from img_grad import compute_image_grad
import heapq
from segmentation import segment_quickshift, show
import numpy as np

def load_bar(it, load_bar_size=30, total=None):
    if total is None:
        total = len(it)
    update_step = 10 ** -3
    last_update_ratio = 0
    t_launch = time()
    for loading_idx, item in enumerate(it):
        ratio = (loading_idx + 1) / total

        if ratio > last_update_ratio + update_step:
            load_bar_ratio = int(load_bar_size * ratio)
            t_spent = time() - t_launch
            loaded = '\u2593' * load_bar_ratio
            not_loaded = '\u2591' * (load_bar_size - load_bar_ratio)
            load_bar_drawing = f'|{loaded}{not_loaded}|'
            estimations = f' -- Spent:{t_spent:.1f}s Left:{t_spent * (1 - ratio):.1f}s' if t_spent > 10 else ''
            print(f'\r {load_bar_drawing} {100 * ratio:.1f}% {estimations}', end='')
            last_update_ratio = ratio
        yield item

    print('\r')


def build_descriptor_database(data, n_iterations, descriptor, descriptor_size, filename):
    print('Start building db')
    with VectorWriter(filename, descriptor_size) as vw:
        print('VectorWriter Initiated')
        for batch_paths, batch_imgs in load_bar(data, total=n_iterations):
            vw.write_batch(descriptor(batch_imgs), batch_paths)


if __name__ == '__main__':
    last_dataset_idx = 10000
    ratio_of_known_images = 0.95
    args = parse_args()
    im_size = 3, args.image_size, args.image_size

    gist = build_gist(im_size, scales=(0.5, 0.81, 0.90, 1))
    get_prox_mask = get_prox_op(im_size)
    gabor_conv = build_gabor_conv(im_size)

    print('Loading Images')
    t0 = time()
    dataset, train_loader, test_loader = dataset_loaders(args)

    print('Loading images', time() - t0)
    t0 = time()

    filename = resolve(f'out/{args.data_path.split("/")[-1]}_gists_{gist.features_number}')
    print(filename)

    if not os.path.exists(f'{filename}.db.binary'):
        n_batch = 50
        build_descriptor_database(dataset.batches(n_batch), len(dataset) // n_batch, gist, gist.features_number, filename)
    else:
        print('Not computing gist db')

    with VectorReader(filename) as vr:

        def search(image, n=200, blanks=None):
            im_gist_diff = (vr[:int(len(vr) * ratio_of_known_images)] - gist(image))

            if blanks is not None:
                for i, j in blanks:
                    start = (i * 4 + j) * gist.filters_number
                    im_gist_diff[:, start:start + gist.filters_number] = 0

            return im_gist_diff.norm(2, 1).topk(n, largest=False)[1]

        print('Building gists', time() - t0)


        print('Naive search')
        for i in range(20):
            t0 = time()

            # Get Random Image from []
            searched_image_idx = rd.randint(int(len(vr) * ratio_of_known_images), len(vr) - 1)
            searched_image = dataset.images[searched_image_idx]

            if args.mask_type == 'center':
                mask = torch.ones(searched_image.size()[1:])
                chan, height, width = searched_image.size()[1:]
                cx, cy = height // 2, width // 2
                dx, dy = height // 6, width // 6

                mask[:, cx - dx: cx + dx, cy - dy: cy + dy] = 0

            elif args.mask_type == 'auto':
                segmentation = torch.Tensor(segment_quickshift(searched_image[0]))
                top_segmentation_labels = dict(zip(*np.unique(segmentation, return_counts=True)))
                sorted_labels = sorted(top_segmentation_labels, key=lambda key: top_segmentation_labels[key],
                                       reverse=True)
                chosen_label = sorted_labels[2] if len(sorted_labels) > 2 else sorted_labels[-1]

                segmented = (segmentation != float(chosen_label)).type(torch.FloatTensor).repeat(3, 1, 1)
                mask = segmented
            else:
                raise ValueError(f'mask_type {args.mask_type} is not valid (Available : center, auto)')

            print('Masked area', get_masked_areas(mask))

            found_image_idxs = search(searched_image, blanks=get_masked_areas(mask))
            found_images = dataset.images[found_image_idxs]
            paths = dataset.paths[[searched_image_idx] + list(found_image_idxs[:40])]

            vutils.save_image(mask, 'default_mask.jpg')

            print('Searching nearest', time() - t0)

            vutils.save_image(torch.cat([searched_image, found_images[:23]], dim=0),
                            '%s/%s_test_naive.png' % (args.output_dir, i),
                            normalize=True)


            prox_mask = get_prox_mask(mask)
            blending_mask = prox_mask + 1 - mask
            vutils.save_image(blending_mask, 'blending_mask.jpg')

            patched_images = []

            t0 = time()

            for found_image in load_bar(found_images):
                best_t, (mi, mj) = best_translation(
                    searched_image[0],
                    found_image,
                    prox_mask,
                    image_transform=gabor_conv,
                    punition=lambda x, y: 3 * (x + y),
                    result_scoring=lambda image: float(compute_image_grad(blending_seamcut(searched_image[0], blending_mask, image)))
                )

                patched = blending_seamcut(searched_image[0], blending_mask, best_t)
                patched_images.append(patched)

            top22 = heapq.nlargest(22, patched_images, key=lambda image: compute_image_grad(image))

            print('Computing best patched images', time() - t0)

            vutils.save_image([searched_image[0], mask] + top22,
                            '%s/%s_test_naive_patched.png' % (args.output_dir, i),
                            normalize=True)
