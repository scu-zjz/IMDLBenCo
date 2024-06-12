from cv2 import detail_ImageFeatures
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
#Pytorch
import torch
from torch import nn
import torch.nn.functional as F


class SRMConv2D(nn.Module):
    def __init__(self):
        super(SRMConv2D,self).__init__()
        q = [4, 12, 2] # coefficient of the kernels
        kernel1 = np.array([
            [0, 0, 0, 0, 0],
            [0,-1, 2,-1, 0],
            [0, 2,-4, 2, 0],
            [0,-1, 2,-1, 0],
            [0, 0, 0, 0, 0]
        ],dtype=np.float32)
        kernel2 = np.array([
            [-1, 2,-2, 2,-1],
            [2, -6, 8,-6, 2],
            [-2, 8,-12,8,-2],
            [2, -6, 8,-6, 2],
            [-1, 2,-2, 2,-1]          
        ],dtype=np.float32)
        kernel3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1,-2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],dtype=np.float32)
        zero_kernel = np.zeros_like(kernel3)
        # shape (3,9,5,5)
        weight = torch.tensor(np.array([ 
            [[kernel1 / q[0] if j == i else zero_kernel for j in range(3)] for i in range(3)],
            [[kernel2 / q[1] if j == i else zero_kernel for j in range(3)] for i in range(3)],
            [[kernel3 / q[2] if j == i else zero_kernel for j in range(3)] for i in range(3)],
        ]),dtype=torch.float32)
        weight = weight.reshape(-1,3,5,5)
        self.weight = torch.nn.Parameter(weight, requires_grad=False) 

    def forward(self, x):
        with torch.no_grad():
            return torch.nn.functional.conv2d(x, weight=self.weight, padding = 2)


