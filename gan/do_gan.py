import os
import sys

sys.path.append("../")
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import os.path as osp
import argparse
import numpy as np
import torch.utils.data
import tqdm

# import os
import torch
from gan.gan_impl import wgan, v_gan

from torchvision import datasets, transforms
from meta_solvers.prd_solver import projected_replicator_dynamics
from torch.utils.data import DataLoader


def setup_seed(seed=42):
    import numpy as np
    import torch
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def soft_update(online, target, tau=0.9):
    for param1, param2 in zip(target.parameters(), online.parameters()):
        param1.data *= 1.0 - tau
        param1.data += param2.data * tau


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

        self.result_dir = "./results"
        self.result_dict = {}

    def get_generator(self):
        args = self.args
        if args.gan_name == "wgan":
            return wgan.wgan_generator(args)
        else:
            return v_gan.gan_generator(args)

    def get_discriminator(self):
        args = self.args
        if args.gan_name == "wgan":
            return wgan.wgan_discriminator(args)
        else:
            return v_gan.gan_discriminator(args)

    def get_dataset(self):
        #########################
        # Load and preprocess data for model
        #########################
        # os.makedirs("../../data/mnist", exist_ok=True)
        # opt = self.args
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
        # dataset = ImageDataset(path_to_dir, exts=["png", "jpg"])
        args = self.args

        if args.dataset == "cifar10":
            input_size = args.img_size
            dataroot = "./data/cifar10"
            dataset = datasets.CIFAR10(
                root=dataroot,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            )
        else:
            args.img_size = 48
            input_size = args.img_size
            dataroot = "./data/stl10"
            # dataset = datasets.CIFAR10(
            #     root=dataroot,
            #     train=True,
            #     download=True,
            #     transform=transforms.Compose(
            #         [
            #             transforms.Resize((input_size, input_size)),
            #             transforms.ToTensor(),
            #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #         ]
            #     ),
            # )
            dataset = datasets.STL10(
                root=dataroot,
                split="unlabeled",
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
        return dataloader

    def gan_init(self, do_generator, do_discriminator):
        load = False
        if load:
            generator = torch.load("wgan_generator.pth")
            discriminator = torch.load("wgan_discriminator.pth")
            do_generator.generator = generator
            do_discriminator.discriminator = discriminator
        else:
            # generator = do_generator.generator
            # discriminator = do_discriminator.discriminator

            opt = self.args
            # lambda_gp = opt.lambda_gp
            # optimizer_G = torch.optim.Adam(
            #     generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
            # )
            # optimizer_D = torch.optim.Adam(
            #     discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
            # )

            # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

            dataloader = self.data
            for epoch in range(opt.n_epochs):
                for i, (imgs, _) in enumerate(dataloader):
                    data = {"real_imgs": imgs}
                    do_discriminator.train(data, do_generator, True)
                    if i % opt.n_critic == 0:
                        do_generator.train(data, do_discriminator, True)
            torch.save(do_generator.generator, "wgan_generator.pth")
            torch.save(do_discriminator.discriminator, "wgan_discriminator.pth")
            # # Configure input
            # # real_imgs = Variable(imgs.type(Tensor))
            # # Sample noise as generator input
            # real_imgs = imgs.to(self.args.device)
            # z = torch.tensor(
            #     np.random.normal(0, 1, (real_imgs.shape[0], opt.latent_dim)),
            #     dtype=torch.float32,
            # ).to(self.args.device)
            # # ---------------------
            # #  Train Discriminator
            # # ---------------------
            #
            # optimizer_D.zero_grad()

            # Sample noise as generator input
            # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # # Generate a batch of images
            # fake_imgs = generator(z)
            #
            # # Real images
            # real_validity = discriminator(real_imgs)
            # # Fake images
            # fake_validity = discriminator(fake_imgs)
            # # Gradient penalty
            # gradient_penalty = do_discriminator.compute_gradient_penalty(
            #     real_imgs.data, fake_imgs.data
            # )
            # # Adversarial loss
            # d_loss = (
            #     -torch.mean(real_validity)
            #     + torch.mean(fake_validity)
            #     + lambda_gp * gradient_penalty
            # )
            #
            # d_loss.backward()
            # # print("d loss: {}".format(d_loss))
            # optimizer_D.step()
            #
            # optimizer_G.zero_grad()
            #
            # # Train the generator every n_critic steps
            # if i % opt.n_critic == 0:
            #     # -----------------
            #     #  Train Generator
            #     # -----------------
            #
            #     # Generate a batch of images
            #     fake_imgs = generator(z)
            #     # Loss measures generator's ability to fool the discriminator
            #     # Train on fake images
            #     fake_validity = discriminator(fake_imgs)
            #     g_loss = -torch.mean(fake_validity)
            #
            #     g_loss.backward()
            #     # print("g loss: {}".format(g_loss))
            #     optimizer_G.step()

    def init(self):
        self.data = self.get_dataset()
        # Training
        generator = self.get_generator()
        discriminator = self.get_discriminator()

        do_gan_init = True
        if do_gan_init:
            self.gan_init(generator, discriminator)
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
        # self.eval_one_generator(idx=-1)
        self.eval_generator_list()

    def solve(self):
        dataloader = self.data
        for loop in range(self.max_loop):

            # Training
            generator = self.get_generator()
            discriminator = self.get_discriminator()

            use_soft_update = True
            if use_soft_update:
                soft_update(
                    online=self.generator_list[0].generator, target=generator.generator
                )
                soft_update(
                    online=self.discriminator_list[0].discriminator,
                    target=discriminator.discriminator,
                )

            # logger = tqdm.trange(
            #     int(self.args.train_max_epoch * (loop + 1)),
            #     desc=f"train the generator {loop}",
            # )
            for epoch in range(int(self.args.train_max_epoch * (loop + 1))):
                for i, (imgs, _) in enumerate(dataloader):
                    dis_idx = np.random.choice(
                        range(len(self.discriminator_list)), p=self.meta_strategies[1]
                    )
                    # data = {"X": X_mb, "T": T_mb}
                    data = {"real_imgs": imgs}
                    generator.train(data, self.discriminator_list[dis_idx])
            # logger = tqdm.trange(
            #     int(self.args.n_critic * self.args.train_max_epoch * (loop + 1)),
            #     desc=f"train the discriminator {loop}",
            # )
            for epoch in range(int(self.args.n_critic * self.args.train_max_epoch * (loop + 1))):
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
            if self.args.solution == "nash":
                self.meta_strategies = projected_replicator_dynamics(self.meta_games)
            else:
                self.meta_strategies = [
                    np.array([1.0 for _ in range(len(self.generator_list))])
                    / len(self.generator_list),
                    np.array([1.0 for _ in range(len(self.discriminator_list))])
                    / len(self.discriminator_list),
                ]
            print(self.meta_games)
            print(self.meta_strategies)
            # self.eval_one_generator(idx=-1)
            self.eval_generator_list()

    def final_eval(self):
        print("final evaluation of the generator")
        torch.save(self.generator_list, "generator.pth")
        for idx in range(len(self.generator_list)):
            self.eval_one_generator(idx)

    def eval_one_generator(self, idx):
        print("final evaluation of the generator")
        # torch.save(self.generator_list, "generator.pth")
        from gan.gan_eval import eval_gan

        dict_score = eval_gan(generator=self.generator_list[idx].generator)
        print(dict_score)

    def eval_generator_list(self):
        from gan.gan_eval import eval_gan_list

        dict_score = eval_gan_list(
            generator_list=self.generator_list,
            generator_distribution=self.meta_strategies[0],
        )
        print(dict_score)
        loop = len(self.generator_list)
        self.result_dict[loop] = {"score": dict_score}
        torch.save(
            self.result_dict,
            osp.join(
                self.result_dir,
                "seed_{}_dataset_{}_solution_{}".format(
                    self.args.seed, self.args.dataset, self.args.solution
                ),
            ),
        )

        models = {
            "generator": self.generator_list[-1],
            "discriminator": self.discriminator_list[-1],
        }
        torch.save(
            self.result_dict,
            osp.join(
                self.result_dir,
                "seed_{}_dataset_{}_solution_{}_model_{}".format(
                    self.args.seed, self.args.dataset, self.args.solution, loop
                ),
            ),
        )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_loop", type=int, default=5)
    parser.add_argument("--solution", type=str, default="nash")
    parser.add_argument("--train_max_epoch", type=int, default=100)
    parser.add_argument("--eval_max_epoch", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--gan_name", type=str, default="gan")
    parser.add_argument("--dataset", type=str, default="stl")
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "--lambda_gp", type=float, default=0, help="gradient penalty for WGAN"
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
        "--channels", type=int, default=3, help="number of image channels"
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=3,
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
    setup_seed(args.seed)
    # print()
    do_time_gan = do_gan(args)
    do_time_gan.init()
    do_time_gan.solve()
    do_time_gan.final_eval()
