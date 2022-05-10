import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


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
        return (self.G(z).detach()[0] + torch.tensor(1.0)) / 2.0


def eval_gan(generator):
    from gan.metrics.utils import (
        get_inception_score,
        get_fid,
    )

    generator.eval()
    dataset = GeneratorDataset(G=generator)
    loader = DataLoader(dataset, batch_size=50, num_workers=0)

    dict_score = {"inception_score": get_inception_score(loader, use_torch=True)}

    return dict_score


class GeneratorListDataset(Dataset):
    def __init__(self, G_list, dis):
        self.G_list = G_list
        self.dis = dis
        self.z_dim = self.G_list[0].generator.latent_dim
        self.device = next(self.G_list[0].generator.parameters()).device

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        # z = torch.randn(1, self.z_dim).to(self.device)
        z = torch.tensor(
            np.random.normal(0, 1, (1, self.z_dim)), dtype=torch.float32
        ).to(self.device)
        gen_idx = np.random.choice(range(len(self.G_list)), p=self.dis)
        return (self.G_list[gen_idx].generator(z).detach()[0] + torch.tensor(1.0)) / 2.0


def eval_gan_list(generator_list, generator_distribution):
    from gan.metrics.utils import (
        get_inception_score,
        get_fid,
    )

    for generator in generator_list:
        generator.generator.eval()
    dataset = GeneratorListDataset(G_list=generator_list, dis=generator_distribution)
    loader = DataLoader(dataset, batch_size=50, num_workers=0)

    dict_score = {"inception_score": get_inception_score(loader, use_torch=True)}

    return dict_score


#
# import argparse
#
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
# from gan.gan_impl.wgan import Generator
#
# gen = Generator(args).to('cuda')
#
# print(eval_gan(generator=gen))
