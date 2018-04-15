import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision.utils as vutils

# import custom models
from models.generator_net_128 import GeneratorNet128
from models.generator_net_64 import GeneratorNet64
from models.discriminator_net_128 import DiscriminatorNet128
from models.discriminator_net_64 import DiscriminatorNet64
from models.custom_alex_net import CustomAlexNet
from models.conv_autoencoder import AutoEncoder

from utils import parse_args, weights_init, dataset_loaders
args = parse_args()

os.makedirs('%s' % args.output_dir, exist_ok=True)

# Load dataset
dataset, train_loader, test_loader = dataset_loaders(args)


def train_autoencoder_gan():
    criterion = nn.BCELoss()
    l2_criterion = nn.MSELoss()

    mask_w = 60
    mask_h = 60

    image_tensor = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
    mask = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
    label = torch.FloatTensor(args.batch_size)

    real_label = 1
    fake_label = 0

    if args.cuda:
        discriminator.cuda()
        generator.cuda()
        criterion.cuda()
        image_tensor = image_tensor.cuda()
        label = label.cuda()
        mask = mask.cuda()

    generator_optimizer = optim.Adam(generator.to_tune().parameters(), lr=0.002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.to_tune().parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            ### update discriminator

            # on a real sample

            discriminator.to_tune().zero_grad()
            image, _ = data
            if args.cuda:
                image = image.cuda()
            image_tensor.resize_as_(image).copy_(image)
            mask.resize_as_(image).fill_(1.)
            mask[:, :, (args.image_size - mask_w) // 2:(args.image_size + mask_w) // 2,
            (args.image_size - mask_h) // 2:(args.image_size + mask_h) // 2] = 0.
            label.resize_(image.size(0)).fill_(real_label)

            image_var = Variable(image_tensor)
            label_var = Variable(label)

            output = discriminator(image_var)

            err_discriminator_real = criterion(output, label_var)
            err_discriminator_real.backward()

            d_x = output.data.mean()

            # on a fake sample

            masked_image_var = Variable(image_tensor * mask + (1-mask)*image_tensor.mean())
            fake = generator(masked_image_var)

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

            err_l2 = l2_criterion(fake, image_var) * 0.999
            err_l2.backward(retain_graph=True)

            label.fill_(real_label)
            label_var = Variable(label)
            output = discriminator(fake)

            err_adv = criterion(output, label_var) * 0.001
            err_adv.backward()

            err_generator = err_l2 + err_adv

            d_g_z2 = output.data.mean()
            generator_optimizer.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epochs, i, len(train_loader),
                     err_discriminator.data[0], err_generator.data[0], d_x, d_g_z1, d_g_z2))
            if i % 100 == 0:
                vutils.save_image(image,
                                  '%s/real_samples.png' % args.output_dir,
                                  normalize=True)
                fake = generator(masked_image_var)
                vutils.save_image(fake.data,
                                  '%s/fake_samples_epoch_%03d.png' % (args.output_dir, epoch),
                                  normalize=True)
                masked = image_tensor * mask + fake.data.clamp(0,1) * (1-mask)
                vutils.save_image(masked,
                                  '%s/reconstructed_samples_epoch_%03d.png' % (args.output_dir, epoch),
                                  normalize=True)

    torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (args.output_dir, epoch))
    torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (args.output_dir, epoch))

def train():
    criterion = nn.BCELoss()

    image_tensor = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
    image_noise = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
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

    total_samples = args.epochs * len(train_loader) / 10

    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            ### update discriminator

            ip = len(train_loader) * epoch + i

            # on a real sample
            discriminator.to_tune().zero_grad()
            image, _ = data
            if args.cuda:
                image = image.cuda()
            image_tensor.resize_as_(image).copy_(image)
            label.resize_(image.size(0)).fill_(real_label)

            # image_noise.resize_as_(image).normal_(0.0, 1.0)
            # if ip < total_samples:
            #    image_tensor = image_tensor * (0.5 + 0.5 * ip / total_samples) + image_noise * (0.5 - 0.5 * ip/total_samples)

            image_var = Variable(image_tensor)
            label_var = Variable(label)

            output = discriminator(image_var)

            err_discriminator_real = criterion(output, label_var)
            err_discriminator_real.backward()

            d_x = output.data.mean()

            # on a fake sample

            noise.resize_(image.size(0), 100, 1, 1).normal_(0, 1)
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


def complete():
    n_samples = 100
    lbd = 0.1
    image_to_complete = dataset[50][0]
    image_to_complete = image_to_complete.resize_(1, 3, args.image_size, args.image_size).repeat(n_samples, 1, 1, 1)
    mask_w = 26
    mask_h = 26
    centered_mask = torch.FloatTensor(n_samples, 3, args.image_size, args.image_size).fill_(1.)
    centered_mask[:, :, (args.image_size - mask_w) // 2:(args.image_size + mask_w) // 2,
    (args.image_size - mask_h) // 2:(args.image_size + mask_h) // 2] = 0.
    mask = centered_mask
    base_noise = torch.FloatTensor(n_samples, 100, 1, 1).normal_(0.0, 1.0)
    label = torch.FloatTensor(n_samples, 1).fill_(1.)

    contextual_loss = nn.SmoothL1Loss(reduce=False)
    perceptual_loss = nn.MSELoss(reduce=False)

    if args.cuda:
        contextual_loss.cuda()
        perceptual_loss.cuda()

        discriminator.cuda()
        generator.cuda()

        label = label.cuda()
        base_noise = base_noise.cuda()
        mask = mask.cuda()
        image_to_complete = image_to_complete.cuda()

    label_var = Variable(label)
    base_noise_var = Variable(base_noise)
    base_noise_var.requires_grad = True
    mask_var = Variable(mask)
    image_to_complete_var = Variable(image_to_complete)

    optimizer = optim.Adam([base_noise_var], lr=0.002, betas=(0.5, 0.999))

    for param in generator.parameters():
        param.requires_grad = False
    for param in discriminator.parameters():
        param.requires_grad = False

    n_iter = 300

    for i in range(n_iter):
        generated = generator(base_noise_var)

        ctx_loss = contextual_loss(generated * mask_var, image_to_complete_var * mask_var)
        ctx_loss = torch.mean(torch.mean(torch.mean(ctx_loss, 1), 1), 1)

        print(f'ctx_loss size : {ctx_loss.size()}')

        rating = discriminator(generated)
        pcpt_loss = perceptual_loss(rating, label_var)

        print(f'pcpt_loss size : {pcpt_loss.size()}')

        total_loss = ctx_loss + lbd * pcpt_loss

        print(f'total_loss size : {total_loss.size()}')

        total_loss.mean().backward()

        print('[%d/%d] Ctx_loss : %.4f, Pcpt_loss : %.4f, Total_loss : %.4f' % (
        i + 1, n_iter, ctx_loss.data[0], pcpt_loss.data[0], total_loss.data[0]))
        optimizer.step()

        reconstructed = mask_var * image_to_complete_var + (1.0 - mask_var) * generated

    _, idces = total_loss.data.min(0)

    print(idces)

    vutils.save_image(generated.data[idces], f'{args.output_dir}/generated.png')
    vutils.save_image(image_to_complete[idces], f'{args.output_dir}/source.png')
    vutils.save_image(reconstructed.data[idces], f'{args.output_dir}/reconstructed.png')


if __name__ == '__main__':
    # Creating networks
    print('Creating networks')

    generator = GeneratorNet64()
    discriminator = DiscriminatorNet64()

    if args.mode == 'train':
        generator.to_tune().apply(weights_init)
        discriminator.to_tune().apply(weights_init)
        print(generator)
        print(discriminator)
        train()
    elif args.mode == 'complete':
        epoch = 39
        generator.load_state_dict(torch.load(f"out/netG_epoch_{epoch}.pth"))
        discriminator.load_state_dict(torch.load(f"out/netD_epoch_{epoch}.pth"))
        print(generator)
        print(discriminator)
        complete()
    elif args.mode == 'cacestmoche':
        generator = AutoEncoder()
        discriminator = DiscriminatorNet64()

        generator.to_tune().apply(weights_init)
        discriminator.to_tune().apply(weights_init)
        print(generator)
        print(discriminator)
        train_autoencoder_gan()
