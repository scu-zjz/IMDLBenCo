import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import inspect
class SobelFilter(nn.Module):
    def __init__(self,in_chan=3, out_chan=1,norm = None ):
        super(SobelFilter, self).__init__()
        if norm is not None and (not inspect.isclass(norm) or not issubclass(norm, nn.Module)):
            raise ValueError("norm must be a class derived from nn.Module or None")
        
        filter_x = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).astype(np.float32)
        filter_y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ]).astype(np.float32)

        filter_x = filter_x.reshape((1, 1, 3, 3))
        filter_x = np.repeat(filter_x, in_chan, axis=1)
        filter_x = np.repeat(filter_x, out_chan, axis=0)

        filter_y = filter_y.reshape((1, 1, 3, 3))
        filter_y = np.repeat(filter_y, in_chan, axis=1)
        filter_y = np.repeat(filter_y, out_chan, axis=0)

        filter_x = torch.from_numpy(filter_x)
        filter_y = torch.from_numpy(filter_y)
        filter_x = nn.Parameter(filter_x, requires_grad=False)
        filter_y = nn.Parameter(filter_y, requires_grad=False)

        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        conv_x.weight = filter_x
        conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        conv_y.weight = filter_y
        norm1 = nn.Identity() if norm is None else norm(out_chan)
        norm2 = nn.Identity() if norm is None else norm(out_chan)

        self.sobel_x = nn.Sequential(conv_x, norm1)
        self.sobel_y = nn.Sequential(conv_y, norm2)


        self.sobel_kernel_x = sobel_kernel_x.expand(1, 3, 3, 3)
        self.sobel_kernel_y = sobel_kernel_y.expand(1, 3, 3, 3)

        self.conv_x = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1, bias=False)

        
        self.conv_x.weight = nn.Parameter(self.sobel_kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(self.sobel_kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        grad = torch.sqrt(grad_x**2 + grad_y**2)
        return grad

