import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from gan.metrics.utils import (
    ImageDataset,
    calculate_frechet_inception_distance,
    calculate_inception_score,
    get_inception_score,
    get_fid,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

path_to_dir = ""
# dataset = ImageDataset(path_to_dir, exts=["png", "jpg"])
input_size = 32
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
loader = DataLoader(dataset, batch_size=50, num_workers=4)

# loopper = iter(loader)
#
# images = next(loopper)
# print(images[0].shape)

print(get_inception_score(loader, use_torch=True))
print(get_fid(loader, fid_stats_path="./fid_stats/cifar10.train.npz", use_torch=True))
