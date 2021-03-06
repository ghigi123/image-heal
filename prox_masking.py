import torch
from torch import nn
SPATIAL_RESOLUTION = 4

def get_prox_op(image_size, kernel_size=25):
    input_channels, width, height = image_size

    kernel = torch.ones((input_channels, input_channels, kernel_size, kernel_size))
    n_tiles = kernel_size ** 2
    kernel[:, :, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = - n_tiles
    max_value = (n_tiles - 1) * input_channels / 2  # half of the kernel is full

    pad = (kernel_size - 1) // 2
    if (kernel_size - 1) % 2 == 0:
        padl, padr = pad, pad
    else:
        padl, padr = pad, pad + 1

    _prox_conv = nn.Conv2d(input_channels, input_channels, kernel_size, groups=1,
                              bias=False)

    _prox_conv.weight.data = kernel
    _prox_conv.weight.requires_grad = False

    threshold = kernel_size / 5

    _prox_conv_padded = nn.Sequential(
        nn.ReflectionPad2d((padl, padr, padl, padr)),
        _prox_conv
    )

    def get_prox(mask):
        inverted_mask = 1 - mask.unsqueeze(0)
        limit = _prox_conv_padded(inverted_mask)[0]
        limit[limit <= threshold] = 0
        limit[limit > threshold] /= max_value
        limit[limit > 1] = 1
        return limit

    return get_prox



def _mse(searched_image, found_translations):
    return (searched_image - found_translations).pow(2).sum(2).sum(2).sum(2)


def mse_on_proximity(origin_image, translations, proximity_mask):
    searched_image = origin_image.clone()
    found_translations = translations.clone()

    non_prox_idxs = proximity_mask[0] != 1
    searched_image[:, non_prox_idxs] = 0
    found_translations[:, :, :, non_prox_idxs] = 0

    return _mse(searched_image, found_translations)


def build_translated(image, max_i, max_j):
    chan, height, width = image.size()
    boundary_image = torch.zeros_like(image)
    for i in range(height):
        for j in range(width):
            if i >= j:
                boundary_image[:, i, j] = image[:, i, 0]
            else:
                boundary_image[:, i, j] = image[:, 0, j]

    images = boundary_image.unsqueeze(0).unsqueeze(1).repeat(max_i, max_j, 1, 1, 1)
    for i in range(max_i):
        for j in range(max_j):
            images[i, j, :, i:, j:] = image[:, :height-i, :width-j]
    return images


def best_translation(searched_image, found_image, proximity_mask, image_transform=None, punition=None, result_scoring=None):
    if image_transform is None:
        image_transform = lambda x: x

    if punition is None:
        punition = lambda i, j: 0

    if result_scoring is None:
        result_scoring = lambda image: 0


    input_channels, width, height = searched_image.size()

    block_width, block_height = width // SPATIAL_RESOLUTION, height // SPATIAL_RESOLUTION
    width_range, height_range = block_width // 4, block_height // 4

    assert searched_image.size() == found_image.size()

    translations = build_translated(found_image, height_range, width_range)

    mse = mse_on_proximity(
        image_transform(searched_image.unsqueeze(0))[0],
        image_transform(translations.view(-1, *searched_image.size())).view(*translations.size()[:2], -1, *translations.size()[3:]),
        proximity_mask
    )

    min_i, min_j = min(((i, j) for i in range(height_range) for j in range(width_range)), key=lambda coo: mse[coo] + punition(*coo) + result_scoring(translations[coo]))
    return translations[min_i, min_j], (min_i, min_j)


def dumb_seamcut(orig_scene, mask, match):
    patched_scene = orig_scene.clone()
    patched_scene[mask == 0] = match[mask == 0]
    return patched_scene


def blending_seamcut(orig_scene, blending_mask, match):
    return orig_scene + (match - orig_scene) * blending_mask


if __name__ == '__main__':
    from torchvision import transforms, utils as vutils
    from PIL import Image

    mask = Image.open('masking/input1_mask.jpg')

    preprocess = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])

    mask_tensor = preprocess(mask)

    get_prox = get_prox_op((3, 128, 128))

    limit = get_prox(torch.stack([mask_tensor], 0))
    vutils.save_image(limit, 'masking/limit.jpg')

    fake_image = torch.Tensor([[[i + j + k for j in range(20)] for i in range(20)] for k in range(3)])
    found_image = fake_image + 5

    fake_image = fake_image / fake_image.max()
    found_image = found_image / found_image.max()

    fake_mask = torch.Tensor([[[i + j > 5 for j in range(20)] for i in range(20)] for k in range(3)])

    best_image, (mi, mj) = best_translation(fake_image, found_image, fake_mask)

    print(best_image)

    print(dumb_seamcut(fake_image, best_image, fake_image))
    print(mi, mj)

    from gist import build_gabor_conv

    best, (mi, mj) = best_translation(
        fake_image,
        found_image,
        fake_mask,
        image_transform=build_gabor_conv(fake_image.size()),
        punition=lambda x, y: 5 * (x + y)
    )
    print(mi, mj)
