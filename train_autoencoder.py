from models.conv_autoencoder import AutoEncoder

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import parse_args, dataset_loaders, weights_init, build_mask

args = parse_args()

# Load dataset
dataset, train_loader, test_loader = dataset_loaders(args)

autoencoder = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0002,
                             weight_decay=1e-5)

image_tensor = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
image_noise = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
noise = torch.FloatTensor(args.batch_size, 100, 1, 1)
fixed_noise = torch.FloatTensor(args.batch_size, 100, 1, 1).normal_(0, 1)
label = torch.FloatTensor(args.batch_size)
real_label = 1
fake_label = 0

if args.cuda:
    autoencoder.cuda()
    criterion.cuda()
    image_tensor = image_tensor.cuda()
    image_noise = image_noise.cuda()
    label = label.cuda()
    noise = noise.cuda()
    fixed_noise = fixed_noise.cuda()

autoencoder.apply(weights_init)

for epoch in range(args.epochs):
    for i, data in enumerate(train_loader):
        # Generate Mask
        mask = build_mask((64, 64), 26, 26, 'center')
        images, _ = data

        if args.cuda:
            images = images.cuda()

        image_tensor.resize_as_(images).copy_(images)
        image_var = Variable(image_tensor)

        # Forward propagation
        output = autoencoder(image_var.masked_fill(mask, 0))
        loss = criterion(output, image_var)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, args.epochs, i, len(train_loader), loss.data[0]))

        if i % 100 == 0:
            # Test and save example images
            autoencoder.eval()
            save_image(images, '%s/real_samples_ae.png' % args.output_dir, normalize=True)
            save_image(image_var.masked_fill(mask, 0).data, '%s/blanked_samples_ae.png' % args.output_dir, normalize=True)
            save_image(autoencoder(image_var).data,
                              '%s/fake_samples_epoch_%03d_ae.png' % (args.output_dir, epoch),
                              normalize=True)
            autoencoder.train()


# Save model
torch.save(autoencoder.state_dict(), f'./{args.output_dir}/conv_autoencoder.pth')