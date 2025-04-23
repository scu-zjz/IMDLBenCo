import torch
a = None
b = torch.tensor([1, 2, 3])
print(torch.cat([b], dim=0))
print(torch.cat([a, b], dim=0))