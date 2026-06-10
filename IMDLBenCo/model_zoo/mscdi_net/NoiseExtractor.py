import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import kornia
from .utils import *

class NoiseExtractor(nn.Module):
    """直接在原始图像上提取噪声特征 - 精确输出目标尺寸"""
    def __init__(self):
        super().__init__()
        # 输入尺寸: 3x512x512
        
        # 高频残差提取模块 (5x5高通滤波)
        self.hpf_conv = nn.Conv2d(3, 32, 5, padding=2, bias=False)
        nn.init.constant_(self.hpf_conv.weight, -1/25)
        nn.init.constant_(self.hpf_conv.weight[:, :, 2, 2], 1 - 1/25)
        
        # 第一阶段: 64x128x128输出
        self.stage1 = nn.Sequential(
            # 下采样到256x256
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 512->256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 下采样到128x128
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 256->128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            NoiseResidualBlock(64),
            NoiseResidualBlock(64)
        )
        
        # 第二阶段: 128x64x64输出
        self.stage2 = nn.Sequential(
            # 下采样到64x64
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128->64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            NoiseResidualBlock(128),
            NoiseResidualBlock(128)
        )
    
    def forward(self, x):
        """
        输入: (B, 3, 512, 512)
        输出: 
          feat1: (B, 64, 128, 128)
          feat2: (B, 128, 64, 64)
        """
        # 提取高频残差 (原始图像噪声)
        hfr = self.hpf_conv(x)  # (B,32,512,512)
        
        # 第一阶段处理
        feat1 = self.stage1(hfr)  # (B,64,128,128)
        
        # 第二阶段处理
        feat2 = self.stage2(feat1)  # (B,128,64,64)
        
        return feat1, feat2

class NoiseResidualBlock(nn.Module):
    """噪声特征残差块 - 保留噪声模式"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)
    

class BayarConv2d(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.empty(1, in_channels, 5, 5))
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.1, 0.1)
        with torch.no_grad():
            # Bayar约束1：中心权重置零
            self.weight[:, :, 2, 2] = 0
            # Bayar约束2：权重和归零
            self.weight -= self.weight.mean(dim=(2,3), keepdim=True)
    
    def forward(self, x):
        # 输入: [b,c,h,w] 输出: [b,1,h,w]
        return F.conv2d(x, self.weight, self.bias, padding=2)

class BayarNoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bayar = BayarConv2d()
        self.downsample = nn.Sequential(
            ChannelShuffleDownsample( in_channels=1, out_channels=16, scale_factor=2),
            ChannelShuffleDownsample( in_channels=16, out_channels=64, scale_factor=2),
            ChannelShuffleDownsample( in_channels=64, out_channels=128, scale_factor=2),
            #ChannelShuffleDownsample( in_channels=128, out_channels=320, scale_factor=2),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2)
        )
        # self.downf2 = nn.Sequential(           
        #     ChannelPixelDownsample( in_channels=64, out_channels=128, scale_factor=2),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2)
        # )
        # self.downf2 = ChannelPixelDownsample( in_channels=128, out_channels=512, scale_factor=2)
        # self.down3 = nn.Conv2d(128,320,1)
        # self.down4 = nn.Conv2d(128,512,1)
    def forward(self, x):
        # 输入: [b,3,512,512]
        noise = self.bayar(x)  # [b,1,512,512]
        noise_f = self.downsample(noise)       
        return noise_f

class SRMNoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # SRM预处理层
        self.srm_conv = nn.Conv2d(3, 9, 5, padding=2)
        self._init_srm_weights()
        
        self.downsample = nn.Sequential(
            ChannelShuffleDownsample( in_channels=9, out_channels=16, scale_factor=2),
            ChannelShuffleDownsample( in_channels=16, out_channels=64, scale_factor=2),
            ChannelShuffleDownsample( in_channels=64, out_channels=128, scale_factor=2),
           # ChannelShuffleDownsample( in_channels=128, out_channels=320, scale_factor=2),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2)
        )

    def _get_srm_list(self) :
        # srm kernel 1
        srm1 = np.zeros([5,5]).astype('float32')
        srm1[1:-1,1:-1] = np.array([[-1, 2, -1],
                                    [2, -4, 2],
                                    [-1, 2, -1]] )
        srm1 /= 4.
        # srm kernel 2
        srm2 = np.array([[-1, 2, -2, 2, -1],
                        [2, -6, 8, -6, 2],
                        [-2, 8, -12, 8, -2],
                        [2, -6, 8, -6, 2],
                        [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.
        # srm kernel 3
        srm3 = np.zeros([5,5]).astype('float32')
        srm3[2,1:-1] = np.array([1,-2,1])
        srm3 /= 2.
        srm_list = [ srm1, srm2, srm3 ]
    
    
        srm_matrix = np.zeros([9, 3, 5, 5])
        for i in range(9):
            first_idx = i // 3
            second_idx = i % 3
            srm_matrix[i, second_idx] = srm_list[first_idx]
        srm_tensor = torch.tensor(srm_matrix, dtype=torch.float)
        return srm_tensor
    def _init_srm_weights(self):
        # SRM核初始化代码
        srm_kernels = self._get_srm_list() # 保持原_srm_list实现
        self.srm_conv.weight.data = torch.FloatTensor(srm_kernels)
        self.srm_conv.weight.requires_grad = False

    def forward(self, x):
        # 输入: [B,3,512,512]
        noise = self.srm_conv(x)  # [B,9,512,512]
        noise_f = self.downsample(noise)    
        return noise_f
