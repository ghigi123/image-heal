import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import random as rd
import os
import math


def parse_args():
    # Parse settings from the CLI
    parser = argparse.ArgumentParser(description='Image inpainting with pytorch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--mode', default='train',
                        help='train or complete')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--output-dir', default='./out', help='folder to output images and model checkpoints')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-path', default='data/faces/raw',
                        help='relative path to a folder containing a folder containing images to learn from')
    parser.add_argument('--image-size', type=int, default=64, help='image will be resized to this size')
    parser.add_argument('--method', default='context-encoder', help='which method to use (context-encoder or dcgan)')
    parser.add_argument('--discriminator-model-name', default='discriminator')
    parser.add_argument('--generator-model-name', default = 'generator')
    parser.add_argument('--mask-size', type=int, default = 32, help="size of the mask used for inpainting")

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.data_path = os.path.abspath(args.data_path)

    return args


def weights_init(m):
    # Init weights in a more effective manner
    classname = m.__class__.__name__
    if 'Conv' in classname:
        m.weight.data.normal_(0.0, 0.02)
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif 'Linear' in classname:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class SubsetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def dataset_loaders(args, train_frac = 0.99):
    # Load the different datasets, resize images and transform them into tensors
    dataset = datasets.ImageFolder(args.data_path, transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor()
    ]))

    print(f'{len(dataset)} samples found')

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Make the train and test sets loader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False, sampler=torch.utils.data.sampler.SubsetRandomSampler(range(math.floor(len(dataset) * train_frac))), **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False, sampler=SubsetSampler(range(math.floor(len(dataset) * train_frac), len(dataset))), **kwargs)
    return dataset, train_loader, test_loader


def build_mask(im_size, width, height, position='center'):
    # Build a mask to apply on images
    im_width, im_height = im_size
    mask = torch.ByteTensor(im_width, im_height).zero_()
    if position == 'center':
        cx, cy = im_width / 2, im_height / 2
    elif position == 'random':
        cx, cy = rd.randint(0, im_width - 1), rd.randint(0, im_height - 1)
    elif isinstance(position, tuple):
        cx, cy = position
    else:
        raise ValueError('Position is not valid')

    mask[max(int(cx - width / 2), 0): min(int(cx + width / 2), im_width - 1), max(int(cy - height / 2), 0): min(int(cy + height / 2), im_height -1)] = 1

    return Variable(mask, requires_grad=False)


if __name__ == '__main__':
    images = torch.Tensor(1, 3, 64, 64).random_()
    print(images)
    mask = build_mask(images, 11, 7, position='random')
    print(mask.sum())
    print(images * mask)
