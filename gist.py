import torch.nn as nn
import torch
from skimage.filters import gabor_kernel
import numpy as np

SPATIAL_RESOLUTION = 4

def build_gist(image_size, scales=(.81, .90, 1), orientations=8, kernel_size=None):
    input_channels, width, height = image_size

    if width % SPATIAL_RESOLUTION !=0 or height % SPATIAL_RESOLUTION != 0:
        raise ValueError(f'Image width or height are not divisible by {SPATIAL_RESOLUTION}.')

    gabor_filters = [gabor_kernel(scale, orientation / orientations * np.pi).real
                     for scale in scales
                     for orientation in range(orientations)]

    if kernel_size is None:
        kernel_size = min(filter.shape[0] for filter in gabor_filters)
    kernels = []
    for filter in gabor_filters:
        w, h = filter.shape
        if w < kernel_size or h < kernel_size:
            raise ValueError('Some gabor filters are smaller than the kernel size')
        kernels.append(filter[(w - kernel_size) // 2:(w + kernel_size) // 2, (h - kernel_size) // 2:(h + kernel_size) // 2])

    kernels = np.stack([kernels[:] for _ in range(input_channels)], axis=1)

    gist_conv = nn.Conv2d(input_channels, kernels.shape[0] * kernels.shape[1], (kernel_size, kernel_size), groups=1, bias=False)

    gist_conv.weight.data = torch.Tensor(kernels)

    pad = (kernel_size - 1) // 2
    if (kernel_size - 1) % 2 == 0:
        padl, padr = pad, pad
    else:
        padl, padr = pad, pad + 1

    pooling_step = (width // SPATIAL_RESOLUTION, height // SPATIAL_RESOLUTION)

    _gist = nn.Sequential(
        nn.ReflectionPad2d((padl, padr, padl, padr)),
        gist_conv,
        nn.AvgPool2d(pooling_step, stride=pooling_step)
    )

    def gist(tensor):
        return _gist(tensor).view((tensor.size()[0], -1))

    gist.filters_number = len(gabor_filters)
    gist.features_number = len(gabor_filters) * SPATIAL_RESOLUTION ** 2

    return gist


if __name__ == '__main__':
    im_size = 3, 32, 32
    gist = build_gist(im_size, kernel_size=7)
    tensor = torch.Tensor(np.ones((2, *im_size)))
    tensor[1] *= 2
    tensor[:, 1] *= 2
    tensor[:, 2] *= 3
    tensor[:, :, 16:, 16:] *= 0
    print(tensor)
    print(gist(tensor))
