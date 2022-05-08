import torch

imgs = torch.rand([2, 3, 3, 4])

print(imgs)

imgs = (imgs + 1.0) / 2.0
print(imgs)
