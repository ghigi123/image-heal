import torch.nn as nn
import torch
from skimage.filters import gabor_kernel
import numpy as np


def build_gist(image_size, scales=4, orientations=8, kernel_size=5):
    input_channels, width, height = image_size

    if width % 4 !=0 or height % 4 != 0:
        raise ValueError('Image width or height are not divisible by 4.')

    gabor_filters = [gabor_kernel((scale + 1) * 0.1, orientation / orientations * np.pi).real
                     for scale in range(scales)
                     for orientation in range(orientations)]

    kernels = []
    for filter in gabor_filters:
        w, h = filter.shape
        kernels.append(filter[(w - kernel_size) // 2:(w + kernel_size) // 2, (h - kernel_size) // 2:(h + kernel_size) // 2])

    kernels = np.array([kernels[:] for _ in range(input_channels)]).swapaxes(0, 1)

    gist_conv = nn.Conv2d(input_channels, kernels.shape[0] * kernels.shape[1], (kernel_size, kernel_size), groups=1, bias=False)
    gist_conv.weight.data = torch.Tensor(kernels)

    pad = (kernel_size - 1) // 2
    if (kernel_size - 1) % 2 == 0:
        padl, padr = pad, pad
    else:
        padl, padr = pad, pad + 1

    pooling_step = (width // 4, height // 4)

    gist = nn.Sequential(
        nn.ReflectionPad2d((padl, padr, padl, padr)),
        gist_conv,
        nn.AvgPool2d(pooling_step, stride=pooling_step)
    )

    return gist


if __name__ == '__main__':
    im_size = 3, 32, 32
    gist = build_gist(im_size)
    tensor = torch.Tensor(np.ones((2, *im_size)))
    tensor[1] *= 2
    tensor[:, 1] *= 2
    tensor[:, 2] *= 3
    tensor[:, :, 16:, 16:] *= 0
    print(tensor)
    print(gist(tensor).size())