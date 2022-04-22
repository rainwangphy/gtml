import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import numpy as np

# import torch
import torch.utils.data

# import torch.nn as nn
# import tqdm
# import os
import torch

# from torch.utils.data import DataLoader

# from gan import wgan
# from models.wrn import Wide_ResNet
from torchvision import datasets, transforms
from meta_solvers.prd_solver import projected_replicator_dynamics
from cv.classifiers import do_classifier
from cv.attacks.pgd_attack import pgd_attacker


class do_at:
    def __init__(self, args):
        self.args = args
        self.max_loop = args.max_loop
        self.solution = args.solution
        self.train_max_epoch = args.train_max_epoch
        self.eval_max_epoch = args.eval_max_epoch
        self.device = args.device
        self.train_data_loader = None

        # self.criterion = nn.CrossEntropyLoss()
        self.classifier_list = []
        self.attacker_list = []

        self.meta_games = [
            np.array([[]], dtype=np.float32),
            np.array([[]], dtype=np.float32),
        ]

        self.meta_strategies = [
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

    def get_classifier(self):
        # device = self.device
        return do_classifier(args=self.args)

    def get_attacker(self):
        config = {
            "predict": self.classifier_list,
            "predict_dis": self.meta_strategies[0],
            "clean_train_dataloader": self.train_data_loader,
            "device": self.device,
        }
        return pgd_attacker(config)

    def get_dataset(self):
        input_size = 32
        dataroot = "./data/cifar10"
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
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
            ),
            batch_size=256,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        return train_loader

    def init(self):
        # print("init")
        self.train_data_loader = self.get_dataset()
        classifier = self.get_classifier()
        attacker = self.get_attacker()
        # device = self.device
        # classifier.train()
        for epoch in range(self.train_max_epoch):
            # print(epoch)
            for i, (imgs, labels) in enumerate(attacker.perturbed_train_dataloader):
                # print(i)
                # imgs = imgs.to(device)
                # labels = labels.to(device)
                end_epoch = True if i == len(self.train_data_loader) else False
                classifier.train(imgs, labels, end_epoch)
            # accuracy = 0.0
            # for i, (imgs, labels) in enumerate(attacker.perturbed_train_dataloader):
            #     accuracy += classifier.eval(imgs, labels)
            # accuracy /= len(self.train_data_loader)
            # print("accuracy: {}".format(accuracy))
        accuracy = 0.0
        for i, (imgs, labels) in enumerate(attacker.perturbed_train_dataloader):
            accuracy += classifier.eval(imgs, labels)
        accuracy /= len(self.train_data_loader)
        print(accuracy)

        self.classifier_list.append(classifier)
        self.attacker_list.append(attacker)
        r = len(self.classifier_list)
        c = len(self.attacker_list)
        self.meta_games = [
            np.full([r, c], fill_value=accuracy),
            np.full([r, c], fill_value=-accuracy),
        ]
        self.meta_strategies = [np.array([1.0]), np.array([1.0])]
        print(self.meta_games)
        print(self.meta_strategies)

    def solve(self):
        for loop in range(self.max_loop):
            classifier = self.get_classifier()
            print("get attacker")
            attacker = self.get_attacker()
            print("train classifier")
            for _ in range(self.train_max_epoch):
                for i in range(len(self.train_data_loader)):
                    batch_idx = np.random.choice(len(self.train_data_loader))
                    attacker_idx = np.random.choice(
                        len(self.attacker_list), p=self.meta_strategies[1]
                    )
                    (imgs, labels) = self.attacker_list[
                        attacker_idx
                    ].perturbed_train_dataloader[batch_idx]
                    # for i, (imgs, labels) in enumerate(attacker.perturbed_train_dataloader):
                    end_epoch = True if i == len(self.train_data_loader) else False
                    classifier.train(imgs, labels, end_epoch)
            # accuracy = 0.0
            # for i, (imgs, labels) in enumerate(attacker.perturbed_train_dataloader):
            #     accuracy += classifier.eval(imgs, labels)
            # accuracy /= len(self.train_data_loader)
            # self.generator_list.append(generator)
            # self.discriminator_list.append(discriminator)

            self.classifier_list.append(classifier)
            self.attacker_list.append(attacker)
            r = len(self.classifier_list)
            c = len(self.attacker_list)
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
                        accuracy = 0.0
                        for i, (imgs, labels) in enumerate(
                            self.attacker_list[t_c].perturbed_train_dataloader
                        ):
                            accuracy += self.classifier_list[t_r].eval(imgs, labels)
                        accuracy /= len(self.train_data_loader)
                        # generator = self.generator_list[t_r]
                        # discriminator = self.discriminator_list[t_c]
                        # gen_loss = 0.0
                        # dis_loss = 0.0
                        # sum_idx = 0
                        # for _ in range(self.eval_max_epoch):
                        #     for i, (imgs, _) in enumerate(self.data):
                        #         sum_idx += 1
                        #         data = {"real_imgs": imgs}
                        #         g_loss = generator.eval(data, discriminator)
                        #         gen_loss += g_loss["g_loss"]
                        #
                        #         d_loss = discriminator.eval(data, generator)
                        #         dis_loss += d_loss["d_loss"]
                        # gen_loss /= sum_idx
                        # dis_loss /= sum_idx
                        meta_games[0][t_r][t_c] = accuracy
                        meta_games[1][t_r][t_c] = -accuracy

            self.meta_games = meta_games
            self.meta_strategies = projected_replicator_dynamics(self.meta_games)
            print(self.meta_games)
            print(self.meta_strategies)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_loop", type=int, default=10)
    parser.add_argument(
        "--solution", type=str, default="the solution for the meta game"
    )
    parser.add_argument("--train_max_epoch", type=int, default=50)
    parser.add_argument("--eval_max_epoch", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--milestones", default=[60, 120, 160])
    parser.add_argument("--gamma", type=float, default=0.2)
    args = parser.parse_args()
    # print()
    do_at_pgd = do_at(args)
    do_at_pgd.init()
    do_at_pgd.solve()
