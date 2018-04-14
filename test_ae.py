from models.conv_autoencoder import AutoEncoder

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import parse_args, dataset_loaders, weights_init

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
        image, _ = data
        if args.cuda:
            image = image.cuda()
        image_tensor.resize_as_(image).copy_(image)
        image_var = Variable(image_tensor)
        # ===================forward=====================
        output = autoencoder(image_var)
        loss = criterion(output, image_var)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, args.epochs, i, len(train_loader), loss.data[0]))
        if i % 100 == 0:
            autoencoder.eval()
            save_image(image, '%s/real_samples_ae.png' % args.output_dir, normalize=True)
            save_image(autoencoder(image_var).data,
                              '%s/fake_samples_epoch_%03d_ae.png' % (args.output_dir, epoch),
                              normalize=True)
            autoencoder.train()

torch.save(autoencoder.state_dict(), './conv_autoencoder.pth')