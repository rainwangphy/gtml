# import argparse
# import os
import numpy as np

# import math
#
# import torchvision.transforms as transforms
# from torchvision.utils import save_image
#
# from torch.utils.data import DataLoader
# from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn

# import torch.nn.functional as F
import torch

# os.makedirs("images", exist_ok=True)
#
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--n_epochs", type=int, default=200, help="number of epochs of training"
# )
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument(
#     "--b1",
#     type=float,
#     default=0.5,
#     help="adam: decay of first order momentum of gradient",
# )
# parser.add_argument(
#     "--b2",
#     type=float,
#     default=0.999,
#     help="adam: decay of first order momentum of gradient",
# )
# parser.add_argument(
#     "--n_cpu",
#     type=int,
#     default=8,
#     help="number of cpu threads to use during batch generation",
# )
# parser.add_argument(
#     "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
# )
# parser.add_argument(
#     "--img_size", type=int, default=28, help="size of each image dimension"
# )
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument(
#     "--sample_interval", type=int, default=400, help="interval betwen image samples"
# )
# opt = parser.parse_args()
# print(opt)
#
# img_shape = (opt.channels, opt.img_size, opt.img_size)
#
cuda = True if torch.cuda.is_available() else False
# Loss function
adversarial_loss = torch.nn.BCELoss()


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args

        opt = args
        self.latent_dim = args.latent_dim
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        img_shape = self.img_shape
        # self.args = args

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img_shape = self.img_shape
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        opt = args
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        img_shape = self.img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class gan_generator:
    def __init__(self, args):
        self.args = args
        self.generator = Generator(args).to(args.device)

        opt = args
        self.g_opt = torch.optim.Adam(
            self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
        )

        self.train_steps = 0
        self.train_interval = 5

    def train(self, data, gan_discriminator, is_train=True):
        # print()
        self.generator.train()
        discriminator = gan_discriminator.discriminator
        if is_train:
            discriminator.train()
        else:
            discriminator.eval()
        # discriminator.train()
        if ((self.train_steps - 1) % self.train_interval == 0) or (
            self.train_steps == 0
        ):
            # discriminator.train()
            self.g_opt.zero_grad()
        g_loss = self.forward(data, discriminator)
        g_loss.backward()
        if self.train_steps % self.train_interval == 0:
            self.g_opt.step()
        self.train_steps += 1

    def eval(self, data, gan_discriminator):
        self.generator.eval()
        discriminator = gan_discriminator.discriminator
        discriminator.eval()
        g_loss = self.forward(data, discriminator)
        return {
            "g_loss": g_loss.detach().cpu().numpy(),
        }

    def forward(self, data, discriminator):
        args = self.args
        opt = args
        # args = self.args
        # img_shape = (opt.channels, opt.img_size, opt.img_size)
        # Generate a batch of images
        real_imgs = data["real_imgs"]
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        z = Variable(
            Tensor(np.random.normal(0, 1, (real_imgs.shape[0], opt.latent_dim)))
        ).to(self.args.device)
        fake_imgs = self.generator(z)
        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        fake_validity = discriminator(fake_imgs)

        valid = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        # fake = Variable(Tensor(real_imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # g_loss = -torch.mean(fake_validity)
        # print("iter g_loss: {}".format(g_loss))
        g_loss = adversarial_loss(discriminator(fake_imgs), valid)
        return g_loss


class gan_discriminator:
    def __init__(self, args):
        self.args = args
        self.discriminator = Discriminator(args).to(args.device)

        opt = args
        self.lambda_gp = args.lambda_gp
        self.d_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
        )

        self.train_steps = 0
        self.train_interval = 5

    def train(self, data, gan_generator, is_train=True):
        # print()
        # real_imgs = data['real_imgs']
        #
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.discriminator.train()
        generator = gan_generator.generator
        if is_train:
            generator.train()
        else:
            generator.eval()
        # generator.train()
        if ((self.train_steps - 1) % self.train_interval == 0) or (
            self.train_steps == 0
        ):
            self.d_opt.zero_grad()
        d_loss = self.forward(data, generator)
        d_loss.backward()
        if self.train_steps % self.train_interval == 0:
            self.d_opt.step()
        self.train_steps += 1

    def eval(self, data, gan_generator):
        generator = gan_generator.generator
        generator.eval()
        self.discriminator.eval()
        d_loss = self.forward(data, generator)

        return {
            "d_loss": d_loss.detach().cpu().numpy(),
        }

    def forward(self, data, generator):
        opt = self.args
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # Sample noise as generator input
        real_imgs = data["real_imgs"].to(self.args.device)
        z = Variable(
            Tensor(np.random.normal(0, 1, (real_imgs.shape[0], opt.latent_dim)))
        ).to(self.args.device)

        # Generate a batch of images
        fake_imgs = generator(z)
        valid = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(real_imgs.size(0), 1).fill_(0.0), requires_grad=False)
        real_loss = adversarial_loss(self.discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(self.discriminator(fake_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        # print("iter d_loss: {}".format(d_loss))
        return d_loss


# # Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()
#
# if cuda:
#     generator.cuda()
#     discriminator.cuda()
#     adversarial_loss.cuda()
#
# # Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [
#                 transforms.Resize(opt.img_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#             ]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )
#
# # Optimizers
# optimizer_G = torch.optim.Adam(
#     generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
# )
# optimizer_D = torch.optim.Adam(
#     discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
# )
#
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#
# # ----------
# #  Training
# # ----------
#
# for epoch in range(opt.n_epochs):
#     for i, (imgs, _) in enumerate(dataloader):
#
#         # Adversarial ground truths
#         valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
#         fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
#
#         # Configure input
#         real_imgs = Variable(imgs.type(Tensor))
#
#         # -----------------
#         #  Train Generator
#         # -----------------
#
#         optimizer_G.zero_grad()
#
#         # Sample noise as generator input
#         z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
#
#         # Generate a batch of images
#         gen_imgs = generator(z)
#
#         # Loss measures generator's ability to fool the discriminator
#         g_loss = adversarial_loss(discriminator(gen_imgs), valid)
#
#         g_loss.backward()
#         optimizer_G.step()
#
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#
#         optimizer_D.zero_grad()
#
#         # Measure discriminator's ability to classify real from generated samples
#         real_loss = adversarial_loss(discriminator(real_imgs), valid)
#         fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
#         d_loss = (real_loss + fake_loss) / 2
#
#         d_loss.backward()
#         optimizer_D.step()
#
#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#             % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#         )
#
#         batches_done = epoch * len(dataloader) + i
#         if batches_done % opt.sample_interval == 0:
#             save_image(
#                 gen_imgs.data[:25],
#                 "images/%d.png" % batches_done,
#                 nrow=5,
#                 normalize=True,
#             )
