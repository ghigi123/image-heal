import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, transforms
import torchvision.utils as vutils

# import custom models
from models.generator_net_64 import GeneratorNet64
from models.discriminator_net_64 import DiscriminatorNet64
from models.custom_alex_net import CustomAlexNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--output-dir', default='./out', help='folder to output images and model checkpoints')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset', default='coco', help='which dataset to use (coco/folder)')
parser.add_argument('--data-path', default='data/faces/raw', help='relative path to a folder containing a folder containing images to learn from')
parser.add_argument('--coco-path', default='data/coco/train2017', help='relative path to a folder containing a folder containing images to learn from')
parser.add_argument('--coco-annotations-path', default='data/coco/annotations/stuff_train2017.json', help='relative path to a folder containing a folder containing images to learn from')
parser.add_argument('--image-size', type=int, default=64, help='image will be resized to this size')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print("Using CUDA")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_path = os.path.abspath(args.data_path)

# Load dataset
print(f'Loading dataset {args.dataset}')
if args.dataset == 'folder':
    dataset = datasets.ImageFolder(data_path, transforms.Compose([
            transforms.Resize([args.image_size,args.image_size]), 
            transforms.ToTensor()
        ]))
elif args.dataset == 'coco':
    print(f'Loading dataset from :')
    print(f'  - images : {args.coco_path}')
    print(f'  - annotations : {args.coco_annotations_path}')
    dataset = datasets.coco.CocoCaptions(args.coco_path, args.coco_annotations_path, transforms.Compose([
            transforms.Resize([args.image_size,args.image_size]), 
            transforms.ToTensor()
        ]))

print(f'{len(dataset)} samples found')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

# Creating networks

print('Creating networks')

generator = GeneratorNet64()
generator.to_tune().apply(weights_init)
print(generator)

discriminator = DiscriminatorNet64()
discriminator.to_tune().apply(weights_init)
print(discriminator)

# Launch training

def train():

    criterion = nn.BCELoss()

    image_tensor = torch.FloatTensor(args.batch_size, 3, 128, 128)
    image_noise = torch.FloatTensor(args.batch_size, 3, 128, 128)
    noise = torch.FloatTensor(args.batch_size, 100, 1, 1)
    fixed_noise = torch.FloatTensor(args.batch_size, 100, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(args.batch_size)
    real_label = 1
    fake_label = 0

    if args.cuda:
        discriminator.cuda()
        generator.cuda()
        criterion.cuda()
        image_tensor = image_tensor.cuda()
        image_noise = image_noise.cuda()
        label = label.cuda()
        noise = noise.cuda()
        fixed_noise = fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)

    generator_optimizer = optim.Adam(generator.to_tune().parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.to_tune().parameters(), lr=0.0002, betas=(0.5, 0.999))

    total_samples = args.epochs * len(train_loader) / 2

    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            ### update discriminator

            ip = len(train_loader) * epoch + i

            # first real sample
            discriminator.to_tune().zero_grad()
            image, _ = data
            if args.cuda:
                image = image.cuda()
            image_tensor.resize_as_(image).copy_(image)
            label.resize_(image.size(0)).fill_(real_label)

            image_noise.resize_as_(image).normal_(0.0, 1.0)
            if ip < total_samples:
                image_tensor = image_tensor * (0.9 + 0.1 * ip / total_samples) + image_noise * (0.1 - 0.1 * ip/total_samples)
            image_var = Variable(image_tensor)
            label_var = Variable(label)

            output = discriminator(image_var)

            err_discriminator_real = criterion(output, label_var)
            err_discriminator_real.backward()

            d_x = output.data.mean()

            # second fake sample

            noise.resize_(image.size(0), 100, 1, 1).normal_(0,1)
            noise_var = Variable(noise)
            fake = generator(noise_var)
            label.fill_(fake_label)
            label_var = Variable(label)

            output = discriminator(fake.detach())
            err_discriminator_fake = criterion(output, label_var)
            err_discriminator_fake.backward()

            d_g_z1 = output.data.mean()
            err_discriminator = err_discriminator_fake + err_discriminator_real

            discriminator_optimizer.step()

            ### update generator

            generator.to_tune().zero_grad()

            label.fill_(real_label)
            label_var = Variable(label)
            output = discriminator(fake)
            err_generator = criterion(output, label_var)
            err_generator.backward()

            d_g_z2 = output.data.mean()
            generator_optimizer.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, args.epochs, i, len(train_loader),
                 err_discriminator.data[0], err_generator.data[0], d_x, d_g_z1, d_g_z2))
            if i % 100 == 0:
                vutils.save_image(image,
                        '%s/real_samples.png' % args.output_dir,
                        normalize=True)
                fake = generator(fixed_noise)
                vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % (args.output_dir, epoch),
                        normalize=True)

    torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (args.output_dir, epoch))
    torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (args.output_dir, epoch))

train()