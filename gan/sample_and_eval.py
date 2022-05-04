import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch

# img_example = torch.rand([32, 3, 32, 32])
# vutils.save_image(
#     img_example.data[:25],
#     "epoch%d.png" % 10,
#     nrow=5,
#     normalize=True,
# )
#
# sample_size = 2000
from gan.gan_impl.wgan import Generator
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader


# torch.multiprocessing.set_start_method('spawn')


class GeneratorDataset(Dataset):
    def __init__(self, G):
        self.G = G
        self.z_dim = self.G.latent_dim
        self.device = next(self.G.parameters()).device

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        # z = torch.randn(1, self.z_dim).to(self.device)
        z = torch.tensor(
            np.random.normal(0, 1, (1, self.z_dim)), dtype=torch.float32
        ).to(self.device)
        return self.G(z)[0]


def eval_gan(generator):
    from gan.metrics.utils import (
        get_inception_score,
        get_fid,
    )

    dataset = GeneratorDataset(G=generator)
    loader = DataLoader(dataset, batch_size=50, num_workers=0)

    dict_score = {"inception_score": get_inception_score(loader, use_torch=True)}

    return dict_score


# parser = argparse.ArgumentParser()
#
# parser.add_argument("--max_loop", type=int, default=4)
# parser.add_argument("--solution", type=str, default="the solution for the meta game")
# parser.add_argument("--train_max_epoch", type=int, default=5)
# parser.add_argument("--eval_max_epoch", type=int, default=2)
# parser.add_argument("--device", type=str, default="cuda")
#
# parser.add_argument(
#     "--n_epochs", type=int, default=200, help="number of epochs of training"
# )
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument(
#     "--lambda_gp", type=float, default=10.0, help="gradient penalty for WGAN"
# )
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
# parser.add_argument("--channels", type=int, default=3, help="number of image channels")
# parser.add_argument(
#     "--n_critic",
#     type=int,
#     default=5,
#     help="number of training steps for discriminator per iter",
# )
# parser.add_argument(
#     "--clip_value",
#     type=float,
#     default=0.01,
#     help="lower and upper clip value for disc. weights",
# )
# parser.add_argument(
#     "--sample_interval", type=int, default=400, help="interval betwen image samples"
# )
# args = parser.parse_args()
# gen = Generator(args=args).to("cuda")
# gen.eval()
# # print(next(gen.parameters()).device)
#
# # z_np = np.random.normal(0, 1, (1, args.latent_dim))
# # print(z_np.shape)
# #
# # z = torch.randn(1, args.latent_dim).cuda()
# # print(z.shape)
# # imgs = gen(torch.tensor(z_np, dtype=torch.float32).cuda())
# # print(imgs.shape)
# dataset = GeneratorDataset(G=gen)
# loader = DataLoader(dataset, batch_size=50, num_workers=0)
# from gan.metrics.utils import (
#     get_inception_score,
#     get_fid,
# )
#
# print(get_inception_score(loader, use_torch=True))
