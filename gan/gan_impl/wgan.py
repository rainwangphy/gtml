# import argparse
# import os
import numpy as np

# import math
# import sys
#
# import torchvision.transforms as transforms
# from torchvision.utils import save_image
#
# from torch.utils.data import DataLoader
# from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn

# import torch.nn.functional as F
import torch.autograd as autograd
import torch

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args

        opt = args
        self.latent_dim = args.latent_dim
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        img_shape = self.img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.args.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img_shape = self.img_shape
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
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
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class wgan_generator:
    def __init__(self, args):
        self.args = args
        self.generator = Generator(args).to(args.device)

        opt = args
        self.g_opt = torch.optim.Adam(
            self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
        )

        self.train_steps = 0
        self.train_interval = 5

    def train(self, data, wgan_discriminator, is_train=True):
        # print()
        self.generator.train()
        discriminator = wgan_discriminator.discriminator
        if is_train:
            discriminator.train()
        else:
            discriminator.eval()
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

    def eval(self, data, wgan_discriminator):
        self.generator.eval()
        discriminator = wgan_discriminator.discriminator
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
        g_loss = -torch.mean(fake_validity)
        # print("iter g_loss: {}".format(g_loss))
        return g_loss


class wgan_discriminator:
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

    def train(self, data, wgan_generator, is_train=True):
        # print()
        # real_imgs = data['real_imgs']
        #
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.discriminator.train()
        generator = wgan_generator.generator
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

    def eval(self, data, wgan_generator):
        generator = wgan_generator.generator
        generator.eval()
        self.discriminator.eval()
        d_loss = self.forward(data, generator)

        return {
            "d_loss": d_loss.detach().cpu().numpy(),
        }

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = Variable(
            Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False
        )
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

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

        # Real images
        real_validity = self.discriminator(real_imgs)
        # Fake images
        fake_validity = self.discriminator(fake_imgs)
        # Gradient penalty
        if self.lambda_gp > 0:
            gradient_penalty = self.compute_gradient_penalty(
                real_imgs.data, fake_imgs.data
            )
            # Adversarial loss
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + self.lambda_gp * gradient_penalty
            )
        else:
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                # + self.lambda_gp * gradient_penalty
            )
        # print("iter d_loss: {}".format(d_loss))
        return d_loss


# # Loss weight for gradient penalty
# lambda_gp = 10
#
# # Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()
#
# if cuda:
#     generator.cuda()
#     discriminator.cuda()

# Configure data loader
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

# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

# batches_done = 0
# for epoch in range(opt.n_epochs):
#     for i, (imgs, _) in enumerate(dataloader):
#
#         # Configure input
#         real_imgs = Variable(imgs.type(Tensor))
#
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#
#         optimizer_D.zero_grad()
#
#         # Sample noise as generator input
#         z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
#
#         # Generate a batch of images
#         fake_imgs = generator(z)
#
#         # Real images
#         real_validity = discriminator(real_imgs)
#         # Fake images
#         fake_validity = discriminator(fake_imgs)
#         # Gradient penalty
#         gradient_penalty = compute_gradient_penalty(
#             discriminator, real_imgs.data, fake_imgs.data
#         )
#         # Adversarial loss
#         d_loss = (
#                 -torch.mean(real_validity)
#                 + torch.mean(fake_validity)
#                 + lambda_gp * gradient_penalty
#         )
#
#         d_loss.backward()
#         optimizer_D.step()
#
#         optimizer_G.zero_grad()
#
#         # Train the generator every n_critic steps
#         if i % opt.n_critic == 0:
#
#             # -----------------
#             #  Train Generator
#             # -----------------
#
#             # Generate a batch of images
#             fake_imgs = generator(z)
#             # Loss measures generator's ability to fool the discriminator
#             # Train on fake images
#             fake_validity = discriminator(fake_imgs)
#             g_loss = -torch.mean(fake_validity)
#
#             g_loss.backward()
#             optimizer_G.step()
#
#             print(
#                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                 % (
#                     epoch,
#                     opt.n_epochs,
#                     i,
#                     len(dataloader),
#                     d_loss.item(),
#                     g_loss.item(),
#                 )
#             )
#
#             if batches_done % opt.sample_interval == 0:
#                 save_image(
#                     fake_imgs.data[:25],
#                     "images/%d.png" % batches_done,
#                     nrow=5,
#                     normalize=True,
#                 )
#
#             batches_done += opt.n_critic
