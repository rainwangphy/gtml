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

parser = argparse.ArgumentParser()

parser.add_argument("--max_loop", type=int, default=10)
parser.add_argument("--solution", type=str, default="the solution for the meta game")
parser.add_argument("--train_max_epoch", type=int, default=50)
parser.add_argument("--eval_max_epoch", type=int, default=2)
parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=2e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--milestones", default=[60, 120, 160])
parser.add_argument("--gamma", type=float, default=0.2)
parser.add_argument("--nb_iter", type=int, default=5)
args = parser.parse_args()
# print()

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

from cv.models.wrn import Wide_ResNet
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from adv.cv.attacks.pgd_attack import LinfPGDAttack

device = args.device
# classifier = Wide_ResNet(16, 10, 0.3, 10).to(device)
do_predict = do_classifier(args)
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    do_predict.classifier.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
scheduler = lr_scheduler.MultiStepLR(
    optimizer,
    milestones=args.milestones,
    gamma=args.gamma,
    last_epoch=-1,
)

attack = LinfPGDAttack(predict=[do_predict], predict_dis=[1.0], nb_iter=args.nb_iter)

do_predict.classifier.train()
max_epochs = 50
for _ in range(max_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        print(i)
        imgs = imgs.to(device)
        labels = labels.to(device)
        perturbed_x = attack.perturb(imgs, labels)
        output_y = do_predict.classifier(perturbed_x)
        optimizer.zero_grad()
        loss = loss_fn(output_y, labels)
        loss.backward()
        # print(loss)
        optimizer.step()
    scheduler.step()

# test
do_predict.classifier.eval()
accuracy = 0.0
for (imgs, labels) in train_loader:
    imgs = imgs.to(device)
    labels = labels.to(device)
    perturbed_x = attack.perturb(imgs, labels)
    output_y = do_predict.classifier(perturbed_x)
    _, predicted = output_y.max(1)
    accuracy += predicted.eq(labels).sum().item() / len(labels)
    # print(accuracy)
print(accuracy / len(train_loader))
