import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

"""Implementation borrowed from SegFormer.

https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py

Thanks for the open-source research environment.
"""

class MLP(nn.Module):
    """MLP module."""
    def __init__(self, input_dim = 256, output_dim = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim , output_dim)
    def forward(self, x:torch.Tensor):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.permute(0, 2, 1)
    
class PredictHead(nn.Module):
    def __init__(self, 
                feature_channels : list,
                embed_dim = 256,
                predict_channels : int = 1,
                norm : str = "BN"
                ) -> None:
        """
        We tested three different types of normalization in the decoder head, and they may yield different results due to dataset configurations and other factors.
        Some intuitive conclusions are as follows:
            - "LN" -> Layer norm : The fastest convergence, but poor generalization performance.
            - "BN" Batch norm : When include authentic images during training, set batchsize = 2 may have poor performance. But if you can train with larger batchsize (e.g. A40 with 48GB memory can train with batchsize = 4) It may performs better.
            - "IN" Instance norm : A form that can definitely converge, equivalent to a batchnorm with batchsize=1. When abnormal behavior is observed with BatchNorm, one can consider trying Instance Normalization. It's important to note that in this case, the settings should include setting track_running_stats and affine to True, rather than the default settings in PyTorch.
        """
        
        super().__init__()
        c1_in_channel, c2_in_channel, c3_in_channel, c4_in_channel, c5_in_channel = feature_channels
        assert len(feature_channels) == 5 , "feature_channels must be a list of 5 elements"
        # self.linear_c5 = MLP(input_dim = c5_in_channel, output_dim = embed_dim)
        # self.linear_c4 = MLP(input_dim = c4_in_channel, output_dim = embed_dim)
        # self.linear_c3 = MLP(input_dim = c3_in_channel, output_dim = embed_dim)
        # self.linear_c2 = MLP(input_dim = c2_in_channel, output_dim = embed_dim)
        # self.linear_c1 = MLP(input_dim = c1_in_channel, output_dim = embed_dim)

        self.linear_fuse = nn.Conv2d(
            in_channels= embed_dim * 5,
            out_channels= embed_dim,
            kernel_size= 1
        )
        
        assert norm in ["LN", "BN", "IN"], "Argument error when initialize the predict head : Norm argument should be one of the 'LN', 'BN' , 'IN', which represent Layer_norm, Batch_norm and Instance_norm"
        
        if norm == "LN":
            self.norm = LayerNorm(embed_dim)
        elif norm == "BN" :
            self.norm = nn.BatchNorm2d(embed_dim)
        else:
            self.norm = nn.InstanceNorm2d(embed_dim, track_running_stats=True, affine=True)

        self.dropout = nn.Dropout()
        
        self.linear_predict = nn.Conv2d(embed_dim, predict_channels, kernel_size= 1)
        
    def forward(self, x):
        c1, c2, c3, c4, c5 = x    # 1/4 1/8 1/16 1/32 1/64
        
        n, _ , h, w = c1.shape # Target size of all the features
        
        # _c1 = self.linear_c1(c1).reshape(shape=(n, -1, c1.shape[2], c1.shape[3]))
        
        _c1 =  F.interpolate(c1, size=(h, w), mode='bilinear', align_corners=False)
        
        # _c2 = self.linear_c2(c2).reshape(shape=(n, -1, c2.shape[2], c2.shape[3]))
        
        _c2 = F.interpolate(c2, size=(h, w), mode='bilinear', align_corners=False)
        
        # _c3 = self.linear_c3(c3).reshape(shape=(n, -1, c3.shape[2], c3.shape[3]))
        
        _c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
        
        # _c4 = self.linear_c4(c4).reshape(shape=(n, -1, c4.shape[2], c4.shape[3]))
        
        _c4 = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False) 
        
        # _c5 = self.linear_c5(c5).reshape(shape=(n, -1, c5.shape[2], c5.shape[3]))
        
        _c5 = F.interpolate(c5, size=(h, w), mode='bilinear', align_corners=False)
        
        _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4, _c5], dim=1))
        
        _c = self.norm(_c)
        
        x = self.dropout(_c)
        
        x = self.linear_predict(x)
        
        return x
