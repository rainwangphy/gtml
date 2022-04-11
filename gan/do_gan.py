import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import numpy as np
import torch
import torch.utils.data
import tqdm
import os
import torch
from torch.utils.data import DataLoader
from gan import wgan

from torchvision import datasets, transforms
from meta_solvers.prd_solver import projected_replicator_dynamics


class do_gan:
    def __init__(self, args):
        self.args = args
        self.max_loop = args.max_loop
        self.solution = args.solution
        self.train_max_epoch = args.train_max_epoch
        self.eval_max_epoch = args.eval_max_epoch

        self.data = None
        # self.generator = None
        # self.discriminator = None
        self.generator_list = []
        self.discriminator_list = []

        self.meta_games = [
            np.array([[]], dtype=np.float32),
            np.array([[]], dtype=np.float32),
        ]

        self.meta_strategies = [
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

    def get_generator(self):
        return wgan.wgan_generator(args)

    def get_discriminator(self):
        return wgan.wgan_discriminator(args)

    def get_dataset(self):
        #########################
        # Load and preprocess data for model
        #########################
        os.makedirs("../../data/mnist", exist_ok=True)
        opt = self.args
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize(opt.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
        return dataloader

    def init(self):
        self.data = self.get_dataset()
        # Training
        generator = self.get_generator()
        discriminator = self.get_discriminator()

        # print("preprocessing of the generator")
        # generator.preprocessing_it(self.data)
        gen_loss = 0.0
        dis_loss = 0.0
        sum_idx = 0
        print("evaluating the current generator/discriminator")
        for _ in range(self.eval_max_epoch):
            for i, (imgs, _) in enumerate(self.data):
                sum_idx += 1
                data = {"real_imgs": imgs}
                g_loss = generator.eval(data, discriminator)
                gen_loss += g_loss["g_loss"]

                d_loss = discriminator.eval(data, generator)
                dis_loss += d_loss["d_loss"]
        gen_loss /= sum_idx
        dis_loss /= sum_idx

        self.generator_list.append(generator)
        self.discriminator_list.append(discriminator)

        r = len(self.generator_list)
        c = len(self.discriminator_list)
        self.meta_games = [
            np.full([r, c], fill_value=-gen_loss),
            np.full([r, c], fill_value=-dis_loss),
        ]
        self.meta_strategies = [np.array([1.0]), np.array([1.0])]
        print(self.meta_games)
        print(self.meta_strategies)

    def solve(self):
        dataloader = self.data
        for loop in range(self.max_loop):

            # Training
            generator = self.get_generator()
            discriminator = self.get_discriminator()
            # generator.preprocessing_it(
            #     dataloader=None, pre_generator=self.generator_list[-1]
            # )

            logger = tqdm.trange(
                self.args.train_max_epoch, desc=f"train the generator {loop}"
            )
            for epoch in logger:
                for i, (imgs, _) in enumerate(dataloader):
                    dis_idx = np.random.choice(
                        range(len(self.discriminator_list)), p=self.meta_strategies[1]
                    )
                    # data = {"X": X_mb, "T": T_mb}
                    data = {"real_imgs": imgs}
                    generator.train(data, self.discriminator_list[dis_idx])
            logger = tqdm.trange(
                self.args.train_max_epoch, desc=f"train the discriminator {loop}"
            )
            for epoch in logger:
                for i, (imgs, _) in enumerate(dataloader):
                    gen_idx = np.random.choice(
                        range(len(self.generator_list)), p=self.meta_strategies[0]
                    )
                    data = {"real_imgs": imgs}
                    discriminator.train(data, self.generator_list[gen_idx])

            # evaluation
            print("augment the game")
            self.generator_list.append(generator)
            self.discriminator_list.append(discriminator)
            r = len(self.generator_list)
            c = len(self.discriminator_list)
            meta_games = [
                np.full([r, c], fill_value=np.nan),
                np.full([r, c], fill_value=np.nan),
            ]
            (o_r, o_c) = self.meta_games[0].shape
            for i in [0, 1]:
                for t_r in range(o_r):
                    for t_c in range(o_c):
                        meta_games[i][t_r][t_c] = self.meta_games[i][t_r][t_c]
            for t_r in range(r):
                for t_c in range(c):
                    if np.isnan(meta_games[0][t_r][t_c]):
                        generator = self.generator_list[t_r]
                        discriminator = self.discriminator_list[t_c]
                        gen_loss = 0.0
                        dis_loss = 0.0
                        sum_idx = 0
                        for _ in range(self.eval_max_epoch):
                            for i, (imgs, _) in enumerate(self.data):
                                sum_idx += 1
                                data = {"real_imgs": imgs}
                                g_loss = generator.eval(data, discriminator)
                                gen_loss += g_loss["g_loss"]

                                d_loss = discriminator.eval(data, generator)
                                dis_loss += d_loss["d_loss"]
                        gen_loss /= sum_idx
                        dis_loss /= sum_idx
                        meta_games[0][t_r][t_c] = -gen_loss
                        meta_games[1][t_r][t_c] = -dis_loss

            self.meta_games = meta_games
            self.meta_strategies = projected_replicator_dynamics(self.meta_games)
            print(self.meta_games)
            print(self.meta_strategies)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_loop", type=int, default=4)
    parser.add_argument(
        "--solution", type=str, default="the solution for the meta game"
    )
    parser.add_argument("--train_max_epoch", type=int, default=5)
    parser.add_argument("--eval_max_epoch", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--n_epochs", type=int, default=200, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "--lambda_gp", type=float, default=10.0, help="gradient penalty for WGAN"
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--img_size", type=int, default=28, help="size of each image dimension"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="number of image channels"
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=5,
        help="number of training steps for discriminator per iter",
    )
    parser.add_argument(
        "--clip_value",
        type=float,
        default=0.01,
        help="lower and upper clip value for disc. weights",
    )
    parser.add_argument(
        "--sample_interval", type=int, default=400, help="interval betwen image samples"
    )
    opt = parser.parse_args()
    args = parser.parse_args()
    # print()
    do_time_gan = do_gan(args)
    do_time_gan.init()
    do_time_gan.solve()
