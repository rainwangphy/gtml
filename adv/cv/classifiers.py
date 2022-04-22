from cv.models.wrn import Wide_ResNet
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


class do_classifier:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.classifier = Wide_ResNet(16, 10, 0.3, 10).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.classifier.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.args.milestones,
            gamma=self.args.gamma,
            last_epoch=-1,
        )

    def train(self, imgs, labels, end_epoch=False):
        # print()
        device = self.device
        self.classifier.train()
        imgs = imgs.to(device)
        labels = labels.to(device)
        y = self.classifier(imgs)
        self.optimizer.zero_grad()
        loss = self.loss_fn(y, labels)
        loss.backward()
        # print(loss)
        self.optimizer.step()
        if end_epoch:
            self.scheduler.step()

    def eval(self, imgs, labels):
        device = self.device
        self.classifier.eval()
        imgs = imgs.to(device)
        labels = labels.to(device)
        y = self.classifier(imgs)
        _, predicted = y.max(1)
        return predicted.eq(labels).sum().item() / len(labels)


# # from models.wrn import Wide_ResNet
# #
# import torch
# # from dataset.cifar_dataset import CIFAR10, CIFAR100
# # import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# input_size = 32
# dataroot = "./data/cifar10"
# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(
#         root=dataroot,
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [
#                 transforms.Resize((input_size, input_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ]
#         ),
#     ),
#     batch_size=256,
#     shuffle=True,
#     num_workers=0,
#     drop_last=True,
# )
#
# device = "cuda"
# classifier_1 = Wide_ResNet(16, 10, 0.3, 10).to(device)
# # classifier_2 = Wide_ResNet(16, 10, 0.3, 10).to(device)
# #
# # predict = [classifier_1, classifier_2]
# # predict_dis = [0.3, 0.7]
#
# # config = {}
#
# # config["predict"] = predict
# # config["predict_dis"] = predict_dis
# # config["device"] = device
# # config["clean_train_dataloader"] = train_loader
#
# # attacker = pgd_attacker(config)
# #
# # # print(attacker)
# #
# # attacker.perturb_train_dataloader()
# #
# # attack_list = LinfPGDAttack(predict, predict_dis)
#
# for (x, y) in train_loader:
#     print(x)
#     x = x.to(device)
#     y = y.to(device)
#     y_o = classifier_1(x)
#
#     _, predicted = y_o.max(1)
#     acc = predicted.eq(y).sum().item() / len(y)
#     print(acc)
#
#
#     # perturbed_x = attack_list.perturb(x, y)
#
#     # print(perturbed_x)
#     break
