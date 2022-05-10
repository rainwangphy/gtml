import torch
import torch.nn as nn
import torch.nn.functional as F

# imgs = torch.rand([2, 3, 3, 4])
#
# print(imgs)
#
# imgs = (imgs + 1.0) / 2.0
# print(imgs)

mlp = nn.Linear(10, 20)

input_example = torch.rand([20, 10])

# print(mlp(input_example))

g_opt = torch.optim.Adam(mlp.parameters(), lr=0.001)

g_opt.zero_grad()
mlp.train()

# input_example = torch.rand([20, 10])

# print(mlp(input_example))
target = torch.rand([20, 20])

out_put = mlp(input_example)

loss = F.mse_loss(out_put, target, reduction="mean")

loss.backward()

mlp.train()

for para in mlp.parameters():
    print(para.grad)

input_example = torch.rand([20, 10])

# print(mlp(input_example))

out_put = mlp(input_example)

loss = F.mse_loss(out_put, out_put, reduction="mean")

loss.backward()
