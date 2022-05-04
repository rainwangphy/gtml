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
