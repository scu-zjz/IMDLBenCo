import timm
import torch 
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import sys
import os
sys.path.append('.')

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .extractor.high_frequency_feature_extraction import HighDctFrequencyExtractor
from .extractor.low_frequency_feature_extraction import LowDctFrequencyExtractor
import math
from functools import partial
from IMDLBenCo.registry import MODELS

class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self,conv_pretrain=False):
        super(ConvNeXt, self).__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain:
            print("Load Convnext pretrain.")
            model = timm.create_model('convnext_tiny', pretrained=True)
            self.load_state_dict(model.state_dict())
        original_first_layer = self.stem[0]
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.stem[0] = new_first_layer

    def forward_features(self, x):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        x = self.norm_pre(x)
        return x , out
    def forward(self, image, mask=None, *args, **kwargs):

        feature,out = self.forward_features(image)

        return feature,out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.float()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



class UpsampleConcatConvSegformer(nn.Module):
    def __init__(self):
        super(UpsampleConcatConvSegformer, self).__init__()
        # 192到96的上采样，单次上采样
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # 384到96的上采样，两次上采样，逐步降低通道数
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        # 768到96的上采样，三次上采样，逐步降低通道数
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, inputs):
        # 上采样
        x1,x2,x3,x4 = inputs
        up2 = self.upsample1(x2)
        up3 = self.upsample2(x3)
        up4 = self.upsample3(x4)
        
        x = torch.cat([x1, up2, up3, up4], dim=1)
        return x



# class mit_b3(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b3, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
class MixVisionTransformer(nn.Module):
    def __init__(self,pretrain_path=None, img_size=512, patch_size=4, in_chans=3,embed_dims=[64, 128, 320, 512],num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        if pretrain_path is not None:
            print("Load segformer pretrain pth.")
            self.load_state_dict(torch.load(pretrain_path),
                                strict=False)
        original_first_layer = self.patch_embed1.proj
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
    
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.patch_embed1.proj = new_first_layer


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return x,outs

    def forward(self, x):
        x,outs = self.forward_features(x)
        return x,outs 


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleConcatConv(nn.Module):
    def __init__(self):
        super(UpsampleConcatConv, self).__init__()
        # 192到96的上采样，单次上采样
        self.upsamplec2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        self.upsamples2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # 384到96的上采样，两次上采样，逐步降低通道数
        self.upsamplec3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        )
        self.upsamples3 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        # 768到96的上采样，三次上采样，逐步降低通道数
        self.upsamplec4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        )
        self.upsamples4 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, inputs):
        # 上采样
        c1,c2,c3,c4,s1,s2,s3,s4 = inputs

        c2 = self.upsamplec2(c2)
        c3 = self.upsamplec3(c3)
        c4 = self.upsamplec4(c4)
        s2 = self.upsamples2(s2)
        s3 = self.upsamples3(s3)
        s4 = self.upsamples4(s4)
        
        # 拼接四个tensor
        x = torch.cat([c1,c2,c3,c4,s1,s2,s3,s4 ], dim=1)
        features = [c1,c2,c3,c4,s1,s2,s3,s4]
        # shortcut = x
        # x = x.permute(0, 2, 3, 1)
        # x = self.fc2(self.act(self.fc1(x)))
        # x = x.permute(0, 3, 1, 2)
        # x = x + shortcut
        # 1x1卷积
        return x, features

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
class ScoreNetwork(nn.Module):
    def __init__(self):
        super(ScoreNetwork, self).__init__()
        self.conv1 = nn.Conv2d(9, 192, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(LayerNorm2d(192),
                                    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(192, 768, kernel_size=1),
                                    nn.Conv2d(768, 192, kernel_size=1),
                                    nn.GELU())
        self.conv2 = nn.Conv2d(192, 8,  kernel_size=7, stride=2, padding=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        short_cut = x
        x = self.invert(x)
        x = short_cut + x
        x = self.conv2(x)
        x = x.float()
        x = self.softmax(x)
        return x


@MODELS.register_module()
class Mesorch(nn.Module):
    def __init__(self, seg_pretrain_path=None, conv_pretrain=False, if_predict_label=False):
        super(Mesorch, self).__init__()
        self.convnext = ConvNeXt(conv_pretrain)
        self.segformer = MixVisionTransformer(seg_pretrain_path)
        self.upsample = UpsampleConcatConv()
        self.low_dct = LowDctFrequencyExtractor()
        self.high_dct = HighDctFrequencyExtractor()
        # 使用1x1的卷积将4个192通道合并为1通道
        self.inverse = nn.ModuleList([nn.Conv2d(96, 1, 1) for _ in range(4)]+[nn.Conv2d(64, 1, 1) for _ in range(4)])
        self.gate = ScoreNetwork()
        # 最后调整到512x512大小
        self.resize = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        # self.if_predict_label = if_predict_label
        # if if_predict_label:
        #     self.pooled = nn.AdaptiveAvgPool2d(1)
        #     self.label_head = nn.Sequential(
        #         nn.Linear(2464, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, 1)
        #     )
    def forward_features(self, image, *args, **kwargs):
        high_freq = self.high_dct(image)
        low_freq = self.low_dct(image)
        input_high = torch.concat([image,high_freq],dim=1)
        input_low = torch.concat([image,low_freq],dim=1)
        input_all = torch.concat([image,high_freq,low_freq],dim=1)
        _,outs1 = self.convnext(input_high)
        _,outs2 = self.segformer(input_low)
        gate_outputs = self.gate(input_all)
        features = outs1 + outs2
        x, features = self.upsample(features)
        reduced = torch.cat([self.inverse[i](features[i]) for i in range(8)], dim=1)
        features = torch.sum(gate_outputs * reduced, dim=1,keepdim=True)
        return features
    
    def forward(self, image, mask, label, *args, **kwargs):
        features = self.forward_features(image)
        pred_label = None
        label_loss = torch.tensor(0.0)
        # if self.if_predict_label:
            # weighted_features = []
            # pooled_features = [self.pooled(f).squeeze(-1).squeeze(-1) for f in features_raw]  # List of [B, C]
            # pooled_gate = self.pooled(gate_outputs)
            # for i in range(8):
            #     b = pooled_gate.shape[0]
            #     w = pooled_gate[:, i].view(b, 1)  # reshape 成 (b, 1)
            #     f = pooled_features[i]  # 取出第 i 个特征
            #     weighted_f = f * w  # 自动广播乘法
            #     weighted_features.append(weighted_f)
            # pooled_features = torch.cat(weighted_features, dim=1)  # [B, total_dim]
            # pred_label = self.label_head(pooled_features).squeeze(-1)
            # label_loss = self.loss_fn(pred_label,label.float())
            # pred_label = torch.sigmoid(pred_label)
        # 调整大小到512x512
        pred_mask = self.resize(features)
        mask_loss = self.loss_fn(pred_mask, mask.float())
        loss = mask_loss + label_loss
        pred_mask = pred_mask.float()
        pred_mask = torch.sigmoid(pred_mask)
        
        output_dict = {
            # loss for backward
            "backward_loss": loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": pred_mask,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "predict_loss": loss,
                'predict_maks_loss': mask_loss,
                # 'predcit_label_loss': label_loss
            },

            "visual_image": {
                "pred_mask": pred_mask,
            }
            # -----------------------------------------
        }
        return output_dict

if __name__ == "__main__":
    print(MODELS)