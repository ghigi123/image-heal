import torch
from torch import nn


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