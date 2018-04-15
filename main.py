""""
Main module implementing the solution using a discrimator :
- DCGAN
- Context Encoder

Quick use :
- python3 main.py --method {context-encoder | dcgan} --mode {train | complete}
"""


import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import torchvision.utils as vutils

# import custom models
from models.generator_net import generator_net
from models.discriminator_net import discriminator_net
from models.custom_alex_net import CustomAlexNet
from models.conv_autoencoder import AutoEncoder

from utils import parse_args, weights_init, dataset_loaders
args = parse_args()

os.makedirs('%s' % args.output_dir, exist_ok=True)

# Load dataset
dataset, train_loader, test_loader = dataset_loaders(args)


def train_context_encoder():
    criterion = nn.BCELoss()
    l2_criterion = nn.MSELoss()

    mask_w = args.mask_size
    mask_h = args.mask_size

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

            # l2 loss

            err_l2 = l2_criterion(fake, image_var) * 0.999
            err_l2.backward(retain_graph=True)

            # adversarial loss

            label.fill_(real_label)
            label_var = Variable(label)
            output = discriminator(fake)

            err_adv = criterion(output, label_var) * 0.001
            err_adv.backward()

            # total loss
            
            err_generator = err_l2 + err_adv

            d_g_z2 = output.data.mean()
            generator_optimizer.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epochs, i, len(train_loader),
                     err_discriminator.data[0], err_generator.data[0], d_x, d_g_z1, d_g_z2))
            if i % 100 == 0:

                test_data, _ = [a for a in test_loader][0]
                if args.cuda:
                    test_data = test_data.cuda()
                image_tensor.resize_as_(test_data).copy_(test_data)
                mask.resize_as_(test_data).fill_(1.)
                mask[:, :, (args.image_size - mask_w) // 2:(args.image_size + mask_w) // 2,
                (args.image_size - mask_h) // 2:(args.image_size + mask_h) // 2] = 0.
                masked_image_var = Variable(image_tensor * mask + (1-mask)*image_tensor.mean())
                fake = generator(masked_image_var)
                masked = image_tensor * mask + fake.data.clamp(0,1) * (1-mask)

                #only ouput batchsize images
                vutils.save_image(test_data[0:args.batch_size],
                                    '%s/real_samples.png' % args.output_dir,
                                    normalize=True)
                vutils.save_image(fake.data[0:args.batch_size],
                                    '%s/fake_samples_epoch_%03d.png' % (args.output_dir, epoch),
                                    normalize=True)
                vutils.save_image(masked[0:args.batch_size],
                                    '%s/reconstructed_samples_epoch_%03d.png' % (args.output_dir, epoch),
                                    normalize=True)

    torch.save(generator.state_dict(), '%s/%s.pth' % (args.output_dir, args.generator_model_name))
    torch.save(discriminator.state_dict(), '%s/%s.pth' % (args.output_dir, args.discriminator_model_name))

def train_dcgan():
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

    generator_optimizer = optim.Adam(generator.to_tune().parameters(), lr=0.002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.to_tune().parameters(), lr=0.0002, betas=(0.5, 0.999))

    total_samples = args.epochs * len(train_loader) / 10

    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            ### update discriminator

            # on a real sample
            discriminator.to_tune().zero_grad()
            image, _ = data
            if args.cuda:
                image = image.cuda()
            image_tensor.resize_as_(image).copy_(image)
            label.resize_(image.size(0)).fill_(real_label)

            # This can be enabled to help discriminator to converge, mixing input with progressively dissapearing noise

            # ip = len(train_loader) * epoch + i
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
                vutils.save_image(fake.data.clamp(0,1),
                                  '%s/fake_samples_epoch_%03d.png' % (args.output_dir, epoch),
                                  normalize=True)

    torch.save(generator.state_dict(), '%s/%s.pth' % (args.output_dir, args.generator_model_name))
    torch.save(discriminator.state_dict(), '%s/%s.pth' % (args.output_dir, args.discriminator_model_name))


def complete_context_encoder():

    image_tensor = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
    mask = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
    
    mask_w = args.mask_size
    mask_h = args.mask_size

    if args.cuda:
        generator.cuda()
        image_tensor = image_tensor.cuda()
        mask = mask.cuda()

    test_data, _ = [a for a in test_loader][0]

    if args.cuda:
        test_data = test_data.cuda()
    image_tensor.resize_as_(test_data).copy_(test_data)
    mask.resize_as_(test_data).fill_(1.)
    mask[:, :, (args.image_size - mask_w) // 2:(args.image_size + mask_w) // 2,
    (args.image_size - mask_h) // 2:(args.image_size + mask_h) // 2] = 0.
    masked_image_var = Variable(image_tensor * mask + (1-mask)*image_tensor.mean())
    fake = generator(masked_image_var)
    masked = image_tensor * mask + fake.data.clamp(0,1) * (1-mask)

    #only ouput batchsize images
    vutils.save_image(test_data[0:args.batch_size],
                        '%s/real_samples.png' % args.output_dir,
                        normalize=True)
    vutils.save_image(fake.data[0:args.batch_size],
                        '%s/fake_samples_test.png' % (args.output_dir),
                        normalize=True)
    vutils.save_image(masked[0:args.batch_size],
                        '%s/reconstructed_samples_test.png' % (args.output_dir),
                        normalize=True)
    
    print('images written in %s' % (args.output_dir))


def complete_dcgan():
    # number of random z vector samples for every image
    n_samples = 10
    # to be parametrized
    # lambda to mix contextual and perceptual losses
    # to be parametrized
    lbd = 0.1

    batch_to_complete, _ = [a for a in test_loader][0]
    print(batch_to_complete)
    reconstructed_images = torch.FloatTensor(batch_to_complete.size())
    masked_images = torch.FloatTensor(batch_to_complete.size())

    # generate a whole batch
    for j in range(batch_to_complete.size(0)):
        image_to_complete = batch_to_complete[j]
        image_to_complete = image_to_complete.resize_(1, 3, args.image_size, args.image_size).repeat(n_samples, 1, 1, 1)

        mask_w = args.mask_size
        mask_h = args.mask_size

        centered_mask = torch.FloatTensor(n_samples, 3, args.image_size, args.image_size).fill_(1.)
        centered_mask[:, :, (args.image_size - mask_w) // 2:(args.image_size + mask_w) // 2,
        (args.image_size - mask_h) // 2:(args.image_size + mask_h) // 2] = 0.
        mask = centered_mask
        base_noise = torch.FloatTensor(n_samples, 100, 1, 1).normal_(0.0, 1.0)
        label = torch.FloatTensor(n_samples).fill_(1.)

        contextual_loss = nn.SmoothL1Loss(reduce=False)

        if args.cuda:
            contextual_loss.cuda()

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

        optimizer = optim.Adam([base_noise_var], lr=0.01, betas=(0.9, 0.999))

        for param in generator.parameters():
            param.requires_grad = False
        for param in discriminator.parameters():
            param.requires_grad = False

        # amount of optimzation steps
        # to be parametrized
        n_iter = 2000

        for i in range(n_iter):

            # contextual loss

            generated = generator(base_noise_var)

            ctx_loss = contextual_loss(generated * mask_var, image_to_complete_var * mask_var)

            ctx_loss = torch.mean(torch.mean(torch.mean(ctx_loss, 1), 1), 1)

            # perceptual loss

            rating = discriminator(generated)

            pcpt_loss = torch.pow(rating - label_var, 2)

            # total loss

            total_loss = ctx_loss + lbd * pcpt_loss

            total_loss.mean().backward()

            print('[%d/%d] Ctx_loss : %.4f, Pcpt_loss : %.4f, Total_loss : %.4f' % (
            i + 1, n_iter, ctx_loss.data[0], pcpt_loss.data[0], total_loss.data[0]))
            optimizer.step()

            reconstructed = mask_var * image_to_complete_var + (1.0 - mask_var) * generated.clamp(0,1)

        _, idces = total_loss.data.min(0)

        reconstructed_images[j] = reconstructed.data[idces]
        masked_images[j] = image_to_complete[idces]


    vutils.save_image(masked_images, f'{args.output_dir}/source.png')
    vutils.save_image(reconstructed_images, f'{args.output_dir}/reconstructed.png')


if __name__ == '__main__':
    # Creating networks
    print('Creating networks')

    if args.method == 'context-encoder':
        generator = AutoEncoder()
        discriminator = discriminator_net(args.image_size)
    elif args.method == 'dcgan':
        generator = generator_net(args.image_size)
        discriminator = discriminator_net(args.image_size)

    print(generator)
    print(discriminator)

    if args.mode == 'train':
        generator.to_tune().apply(weights_init)
        discriminator.to_tune().apply(weights_init)
    elif args.mode == 'complete':
        generator.load_state_dict(torch.load('%s/%s.pth' % (args.output_dir, args.generator_model_name)))
        discriminator.load_state_dict(torch.load('%s/%s.pth' % (args.output_dir, args.discriminator_model_name)))


    if args.method == 'context-encoder':
        if args.mode == 'train':
            train_context_encoder()
        elif args.mode == 'complete':
            complete_context_encoder()
    elif args.method == 'dcgan':
        if args.mode == 'train':
            train_dcgan()
        elif args.mode == 'complete':
            complete_dcgan()