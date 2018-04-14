import torch.nn as nn


def convolution(input_channels, output_channels, kernel_size, reduction=2):
    pad = (kernel_size - reduction) // 2
    if (kernel_size - reduction) % 2 == 0:
        padl, padr = pad, pad
    else:
        padl, padr = pad, pad + 1

    return nn.Sequential(
        # assertion that reduction | input_size -> input_size / reduction = output_size int
        nn.ReflectionPad2d((padl, padr, padl, padr)),
        nn.Conv2d(input_channels, output_channels, kernel_size, reduction),
        nn.ReLU(True),
        nn.BatchNorm2d(output_channels),
    )


def deconvolution(input_channels, output_channels, kernel_size, augment=2):
    pad = (kernel_size - augment) // 2
    if (kernel_size - augment) % 2 == 0:
        padl, padr = pad, pad
    else:
        padl, padr = pad, pad + 1

    return nn.Sequential(
        # assertion that reduction | input_size -> input_size / reduction = output_size int
        # TODO: nn.ReflectionPad2d((padl, padr, padl, padr)) in transpose
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=augment, padding=pad),
        nn.ReLU(True),
        nn.BatchNorm2d(output_channels),
    )


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    x = Variable(torch.rand((1, 3, 64, 64)))
    print(x)
    deconv = deconvolution(3, 32, 5)
    print(deconv(x))

