from cv2 import detail_ImageFeatures
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


class BayerConv(nn.Module):
    def __init__(self, in_channel=3):
        super(BayerConv, self).__init__()
        self.BayarConv2D = nn.Conv2d(in_channel, 3, 5, 1, padding=2, bias=False)

        # Convert tensors to Parameters
        bayar_mask = np.ones(shape=(5, 5))
        bayar_mask[2, 2] = 0
        self.bayar_mask = nn.Parameter(torch.tensor(bayar_mask, dtype=torch.float32), requires_grad=False)

        bayar_final = np.zeros((5, 5))
        bayar_final[2, 2] = -1
        self.bayar_final = nn.Parameter(torch.tensor(bayar_final, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final

        return self.BayarConv2D(x)

if __name__ == '__main__':
    torch.manual_seed(42)
    img = torch.randn(2,3,256,256).to(0)
    model = BayerConv().to(0)
    print(model(img))