import torch
from torch import nn
from itertools import product

SPATIAL_RESOLUTION = 4


def get_prox_op(image_size, kernel_size=25):
    input_channels, width, height = image_size

    kernel = torch.ones((input_channels, input_channels, kernel_size, kernel_size))
    kernel[:, :, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = - kernel_size ** 2

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
        inverted_mask = 1 - mask
        limit = _prox_conv_padded(inverted_mask)
        limit[limit <= threshold] = 0
        limit[limit > threshold] = 1
        return limit

    return get_prox


def _mse(searched_images, found_image):
    return (searched_images - found_image).pow(2).sum(3).argmin()


def build_ssd(searched_images):
    n_searched_images, input_channels, width, height = searched_images.size()
    block_width, block_height = width // SPATIAL_RESOLUTION, height // SPATIAL_RESOLUTION
    translated_images = torch.stack([torch.stack([searched_images[:, :, i:, j:] -  n for i in range(block_height)]) for j in range(block_width)]).permute(2, 0, 1, 3, 4, 5)

    def best_translation(found_image):
        return _mse(translated_images, found_image)

    return best_translation

def best_translation(searched_image, found_image, prox_mask):
    input_channels, width, height = searched_image.size()
    prox_idxs = prox_mask == 1
    print('idxs', prox_idxs)

    block_width, block_height = width // SPATIAL_RESOLUTION, height // SPATIAL_RESOLUTION
    assert searched_image.size() == found_image.size()
    print(torch.stack([torch.stack([(searched_image[:, i:i+block_height, j:j+block_width] - found_image[:, width-i-block_height:width-i, height-j-block_width:height-j]).pow(2).mean() for i in range(block_height)]) for j in range(block_width)]).argmin())


if __name__ == '__main__':
    from torchvision import transforms, utils as vutils
    from PIL import Image

    mask = Image.open('input1_mask.jpg')

    preprocess = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])

    mask_tensor = preprocess(mask)

    get_prox = get_prox_op((3, 128, 128))
    limit = get_prox(torch.stack([mask_tensor], 0))
    vutils.save_image(limit, 'limit.jpg')
    
    from gist import get_masked_areas

    print(get_masked_areas(mask_tensor))
    print(best_translation(mask_tensor, mask_tensor, limit))
    t = torch.Tensor([0, 1, 0, 0])
    print(t == 0)

    print(t)
    n = 2
    print(list(product(range(- n, n + 1), repeat=2)))