# All rights reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_


from timm.models.layers import trunc_normal_, DropPath, to_2tuple

from IMDLBenCo.registry import MODELS

## -- gloal variables for layer scale
layer_scale = True
init_value = 1e-6
## ----------------------------------

class Multiple(nn.Module):
    def __init__(self, 
                 init_value = 1e-6,
                 embed_dim = 256,
                 predict_channels = 1,
                 norm_layer = partial(nn.LayerNorm, eps=1e-6) ):
        super(Multiple, self).__init__()
        self.gamma1 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma2 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma3 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma4 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma5 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma6 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        # self.drop_path = nn.Identity()
        self.norm = norm_layer(embed_dim)
        
        self.conv_layer1 = nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_layer2 = nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_layer3 = nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_layer4 = nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_last = nn.Conv2d(embed_dim, predict_channels, kernel_size= 1)
    def forward(self, x):
        c1, c2, c3, c4, c5, c6 = x
        
        c1 = self.conv_layer1(c1)
        c2 = self.conv_layer2(c2)
        c3 = self.conv_layer3(c3)
        c4 = self.conv_layer4(c4)
        b, c , h, w = c1.shape
        c5 = F.interpolate(c5, size=(h, w), mode='bilinear', align_corners=False)
        c6 = F.interpolate(c6, size=(h, w), mode='bilinear', align_corners=False)
        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2)
        c4 = c4.flatten(2).transpose(1, 2) 
        c5 = c5.flatten(2).transpose(1, 2)
        c6 = c6.flatten(2).transpose(1, 2)
        x = self.gamma1*c1 + self.gamma2*c2 + self.gamma3*c3 + self.gamma4*c4 + self.gamma5*c5 + self.gamma6*c6
        x= x.transpose(1, 2).reshape(b, c, h, w)
        x = (self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).contiguous()
        x = self.conv_last(x)
        return x


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

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

  
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

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def block(x,block_size):
    B,H,W,C = x.shape
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  
    Hp, Wp = H + pad_h, W + pad_w  
    x = x.reshape(B,Hp//block_size,block_size,Wp//block_size,block_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x, H, Hp, C

def unblock(x, Ho):
    B,H,W,win_H,win_W,C = x.shape
    x = x.permute(0,1,3,2,4,5).contiguous().reshape(B,H*win_H,W*win_W, C)
    Wp = Hp = H*win_H
    Wo = Ho
    if Hp > Ho or Wp > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x


def alter_sparse(x, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    assert x.shape[1]%sparse_size == 0 & x.shape[2]%sparse_size == 0, 'image size should be divisible by block_size'
    grid_size = x.shape[1]//sparse_size
    out, H, Hp, C = block(x, grid_size)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, C)
    out = out.permute(0, 3, 1, 2)
    return out, H, Hp, C   


def alter_unsparse(x, H, Hp, C, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, Hp//sparse_size, Hp//sparse_size, sparse_size, sparse_size, C)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = unblock(x, H)
    out = out.permute(0, 3, 1, 2)
    return out

class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, sparse_size=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls = layer_scale
        self.sparse_size = sparse_size
        if self.ls:
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x_befor = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape
        if self.ls:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)  
            x = x_befor + self.drop_path(self.gamma_1 * x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        else:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_befor + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x        



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
    
@MODELS.register_module()
class SparseViTBackbone(nn.Module):
    def __init__(self, layers=[5, 8, 20, 7], img_size=224, in_chans=3, s_blocks3=[8, 4, 2, 1], s_blocks4=[2, 1], embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=None, pretrained_path=None,
                 ):
        super().__init__()
        self.pretrained_path=pretrained_path
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        
        
        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(layers[0])])
        self.norm1=norm_layer(embed_dim[0])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+layers[0]], norm_layer=norm_layer)
            for i in range(layers[1])])
        self.norm2 = norm_layer(embed_dim[1])
       
        self.blocks3 = nn.ModuleList()
        for i in range(layers[2]):
            block =  SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], sparse_size=32//s_blocks3[i//5], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+layers[0]+layers[1]], norm_layer=norm_layer)
            self.blocks3.append(block)
        self.norm3 = norm_layer(embed_dim[2])
        self.blocks4 = nn.ModuleList()
        for i in range(layers[3]):
            block = SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], sparse_size=16//s_blocks4[i//4], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+layers[0]+layers[1]+layers[2]], norm_layer=norm_layer)
            self.blocks4.append(block)
        self.norm4 = norm_layer(embed_dim[3])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
            
        self._uniformer_init_weights()
        self.apply(self._init_weights)
        
       
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _uniformer_init_weights(self):
        if self.pretrained_path != None:
            state_dict = torch.load(self.pretrained_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict['model'].items():
                if k.startswith('backbone.'):
                    new_key = k[len('backbone.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v  
            self.load_state_dict(new_state_dict, strict=False)
            print('load pretrained weights from \'{}\'.'.format(self.pretrained_path))

    def forward_features(self, x):
        outputs = {}
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for index, blk in enumerate(self.blocks3):
            x = blk(x)
            if (index+1)%5==0 and (index+1)//5!=4:
                outputs.update({"third"+str((index+1)//5): x})
        x_out = self.norm3(x.permute(0, 2, 3, 1))
        outputs.update({"third": x_out.permute(0, 3, 1, 2).contiguous()})
        x = self.patch_embed4(x)
        for index, blk in enumerate(self.blocks4):
            x = blk(x)
            if (index+1)%4==0:
                outputs.update({"last"+str((index+1)//4): x})
        x_out = self.norm4(x.permute(0, 2, 3, 1))
        outputs.update({"last": x_out.permute(0, 3, 1, 2).contiguous()})
        return outputs

    def forward(self, x):
        outputs = self.forward_features(x)
        return outputs



@MODELS.register_module()
class SparseViT(nn.Module):
    def __init__(self, 
                 depth = [5, 8, 20, 7],
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 img_size=512,
                 s_blocks3=[8, 4, 2, 1],
                 s_blocks4=[2, 1],
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pretrained_path=None,
    ):
        super(SparseViT, self).__init__()
        self.img_size = img_size
        self.encoder_net = SparseViTBackbone(
            layers=depth,
            embed_dim=embed_dim,
            img_size= img_size,
            s_blocks3=s_blocks3,
            s_blocks4=s_blocks4,
            head_dim=head_dim,
            drop_path_rate=0.2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pretrained_path=pretrained_path,
        )
        self.lmu = Multiple(embed_dim=512)
        self.BCE_loss = nn.BCEWithLogitsLoss()
        # self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, image, mask, *args, **kwargs):
        image = self.encoder_net(image)
        feature_list = []
        for k, v in image.items():
            feature_list.append(v)
            
        image = self.lmu(feature_list)
        image = F.interpolate(image, size = (self.img_size, self.img_size), mode='bilinear', align_corners=False)
        predict_loss = self.BCE_loss(image, mask)
        image = torch.sigmoid(image)

        output_dict = {
            # loss for backward
            "backward_loss": predict_loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": image,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": None,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "predict_loss": predict_loss,
            },

            "visual_image": {
                "pred_mask": image,
            }
            # -----------------------------------------
        }


        return output_dict
