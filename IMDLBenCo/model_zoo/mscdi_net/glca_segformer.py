
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from mmseg.models.builder import BACKBONES
from functools import partial
import math
from .utils import *

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

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class GLCA_MixAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                attn_drop=0., proj_drop=0., sr_ratio=1, depth=1, glca=0, window_size=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.window_size = window_size
        self.depth = depth
        self.glca = glca
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            
        if self.glca > 0  and depth < self.glca :
            self.fusion = InteractiveFusion_C(dim, reduction = 4) 
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
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
                
    def _sparsify_attention(self, attn):

        if 0 < self.sparse_ratio < 1:
            top_k = max(1, int(attn.size(-1) * self.sparse_ratio))
            val, idx = torch.topk(attn, k=top_k, dim=-1)
            sparse_attn = torch.zeros_like(attn).scatter(-1, idx, val)
            return sparse_attn
        return attn
    
    def _window_partition(self, x, H, W, window_size):
        """

        Args:
            x: (B, H, W, C)
            window_size (int): 
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def _window_reverse(self, windows, window_size, H, W):
        """

        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): 
            H (int): 
            W (int): 
        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]  
        
        # global attention
        attn_global = (q @ k.transpose(-2, -1)) * self.scale
        attn_global = attn_global.softmax(dim=-1)
        #attn_global = self._sparsify_attention(attn_global)
        attn_global = self.attn_drop(attn_global)
        x_global = (attn_global @ v).transpose(1, 2).reshape(B, N, C)
        
        # window-based local attention
        if self.glca >0  and self.depth < self.glca:
            if self.window_size > 0:
                # to  (B, H, W, C)
                q_space = q.transpose(1, 2).reshape(B, H, W, self.dim)
                k_space = k.transpose(1, 2).reshape(B, H // self.sr_ratio, W // self.sr_ratio, self.dim)
                v_space = v.transpose(1, 2).reshape(B, H // self.sr_ratio, W // self.sr_ratio, self.dim)
                

                q_windows = self._window_partition(q_space, H, W, self.window_size)
                k_windows = self._window_partition(k_space, H // self.sr_ratio, W // self.sr_ratio, 
                                                self.window_size // self.sr_ratio)
                v_windows = self._window_partition(v_space, H // self.sr_ratio, W // self.sr_ratio, 
                                                self.window_size // self.sr_ratio)
                

                q_windows = q_windows.view(-1, self.window_size * self.window_size, self.dim)
                k_windows = k_windows.view(-1, (self.window_size // self.sr_ratio) * (self.window_size // self.sr_ratio), 
                                        self.dim)
                v_windows = v_windows.view(-1, (self.window_size // self.sr_ratio) * (self.window_size // self.sr_ratio), 
                                        self.dim)
                
                attn_local = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
                attn_local = attn_local.softmax(dim=-1)
                #attn_local = self._sparsify_attention(attn_local)
                attn_local = self.attn_drop(attn_local)
                

                x_local = (attn_local @ v_windows).view(-1, self.window_size, self.window_size, self.dim)
                x_local = self._window_reverse(x_local, self.window_size, H, W)
                x_local = x_local.reshape(B, H * W, self.dim).transpose(1, 2)
                x_local = x_local.reshape(B, N, C)

        # fusion
        if self.glca >0 and  self.depth < self.glca:
            x_global = x_global.permute(0,2,1).view(B,self.dim,H,W)
            x_local = x_local.permute(0,2,1).view(B,self.dim,H,W)
            x = self.fusion(x_global, x_local)
            x = x.permute(0,2,3,1).view(B,N,C).contiguous()
        else:
            x = x_global
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class GLCA_MixAttention_Serial(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                attn_drop=0., proj_drop=0., sr_ratio=1, depth=1, glca=12, window_size=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.glca = glca
        self.depth = depth
        self.window_size = window_size
        self.sr_ratio = sr_ratio
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        # if self.glca >0  and  self.depth < self.glca:
        #    self.fusion = InteractiveFusion_C(dim, reduction = 4) # 
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        
        # 处理空间缩减
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_)
            _, _, H_sr, W_sr = x_.shape
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()

        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()#2,B,heads,N,head_dim
            H_sr, W_sr = H, W

        
        k, v = kv[0], kv[1]
        

        attn_global = (q @ k.transpose(-2, -1)) * self.scale
        attn_global = attn_global.softmax(dim=-1)
        # if 0 < self.sparse_ratio < 1:
        #     top_k = max(1, int(attn_global.size(-1) * self.sparse_ratio))
        #     val, idx = torch.topk(attn_global, k=top_k, dim=-1)
        #     attn_global = torch.zeros_like(attn_global).scatter_(-1, idx, val)
        attn_global = self.attn_drop(attn_global)
        x_global = (attn_global @ v).transpose(1, 2).reshape(B, N, C)


        if self.glca >0  and  self.depth < self.glca:
            q=k=v=x_global.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            window_size = self.window_size
            #scaled_ws = window_size//self.sr_ratio 
            q_windows = q.view(B,-1,H,W,self.head_dim)
            q_windows = q.unfold(2,window_size,window_size)
            q_windows = q.unfold(3,window_size,window_size).contiguous()
           
            k_windows = k.view(B,-1,H,W,self.head_dim)
            k_windows = k_windows.unfold(2,window_size,window_size)
            k_windows = k_windows.unfold(3,window_size,window_size).contiguous()

            v_windows = v.view(B,-1,H,W,self.head_dim)
            v_windows = v_windows.unfold(2,window_size,window_size)
            v_windows = v_windows.unfold(3,window_size,window_size).contiguous()
            
            q_windows = q.view(B,-1,window_size*window_size, self.head_dim)
            k_windows = k.view(B,-1,window_size*window_size, self.head_dim)
            v_windows = v.view(B,-1,window_size*window_size, self.head_dim)

            attn_local = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
            attn_local = attn_local.softmax(dim=-1)

            attn_local = self.attn_drop(attn_local)
            #x_local = (attn_local @ v_windows).transpose(1, 2).reshape(B, N, C)
            x_local = attn_local @ v_windows
            x_local = x_local.view(B, -1, window_size, window_size, self.head_dim)
            x_local = x_local.permute(0, 1, 4, 2, 3).contiguous()
            x_local = x_local.view(B, self.num_heads, self.head_dim, H, W)
            x_local = x_local.permute(0, 1, 3, 4, 2).contiguous()
            x_local = x_local.view(B, N, C)
            x_global = x_global.permute(0,2,1).view(B,self.dim,H,W)
            x_local = x_local.permute(0,2,1).view(B,self.dim,H,W)
            x =  x_local
            #x = self.fusion(x_global, x_local)
            x = x.permute(0,2,3,1).view(B,N,C).contiguous()
        else:
           x = x_global

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class GLCA_MixAttention_old(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                attn_drop=0., proj_drop=0., sr_ratio=1, depth=1, glca=12, window_size=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.glca = glca
        self.depth = depth
        self.window_size = window_size
        self.sr_ratio = sr_ratio
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        if self.glca >0  and  self.depth < self.glca:
           self.fusion = InteractiveFusion_C(dim, reduction = 4) # 
        #self.fusion = InteractiveFusion_N(dim,sr_ratio=sr_ratio) #ChannelFusion(self.dim)
        #self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
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
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_)
            _, _, H_sr, W_sr = x_.shape
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()

        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()#2,B,heads,N,head_dim
            H_sr, W_sr = H, W

        
        k, v = kv[0], kv[1]
        

        attn_global = (q @ k.transpose(-2, -1)) * self.scale
        attn_global = attn_global.softmax(dim=-1)
        # if 0 < self.sparse_ratio < 1:
        #     top_k = max(1, int(attn_global.size(-1) * self.sparse_ratio))
        #     val, idx = torch.topk(attn_global, k=top_k, dim=-1)
        #     attn_global = torch.zeros_like(attn_global).scatter_(-1, idx, val)
        attn_global = self.attn_drop(attn_global)
        x_global = (attn_global @ v).transpose(1, 2).reshape(B, N, C)


        if self.glca >0  and  self.depth < self.glca:
            window_size = self.window_size
            scaled_ws = window_size//self.sr_ratio 
            q_windows = q.view(B,-1,H,W,self.head_dim)
            q_windows = q.unfold(2,window_size,window_size)
            q_windows = q.unfold(3,window_size,window_size).contiguous()
           
            k_windows = k.view(B,-1,H_sr,W_sr,self.head_dim)
            
            k_windows = k_windows.unfold(2,scaled_ws,scaled_ws)
            k_windows = k_windows.unfold(3,scaled_ws,scaled_ws).contiguous()

            v_windows = v.view(B,-1,H_sr,W_sr,self.head_dim)
            v_windows = v_windows.unfold(2,scaled_ws,scaled_ws)
            v_windows = v_windows.unfold(3,scaled_ws,scaled_ws).contiguous()
            
            q_windows = q.view(B,-1,window_size*window_size, self.head_dim)
            k_windows = k.view(B,-1,scaled_ws*scaled_ws, self.head_dim)
            v_windows = v.view(B,-1,scaled_ws*scaled_ws, self.head_dim)

            attn_local = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
            attn_local = attn_local.softmax(dim=-1)
            # if 0 < self.sparse_ratio < 1:   
            #     top_k = max(1, int(attn_local.size(-1) * self.sparse_ratio))
            #     val, idx = torch.topk(attn_local, k=top_k, dim=-1)
            #     attn_local = torch.zeros_like(attn_local).scatter_(-1, idx, val)

            attn_local = self.attn_drop(attn_local)
            #x_local = (attn_local @ v_windows).transpose(1, 2).reshape(B, N, C)
            x_local = attn_local @ v_windows
            x_local = x_local.view(B, -1, window_size, window_size, self.head_dim)
            x_local = x_local.permute(0, 1, 4, 2, 3).contiguous()
            x_local = x_local.view(B, self.num_heads, self.head_dim, H, W)
            x_local = x_local.permute(0, 1, 3, 4, 2).contiguous()
            x_local = x_local.view(B, N, C)

        
            x_global = x_global.permute(0,2,1).view(B,self.dim,H,W)
            x_local = x_local.permute(0,2,1).view(B,self.dim,H,W)
            x = self.fusion(x_global, x_local)
            x = x.permute(0,2,3,1).view(B,N,C).contiguous()

        else:
           x = x_global

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
       
    def window_partition(self, x, window_size, H, W):

        B, num_heads, _, _, C = x.shape
        x = x.view(B, num_heads, H // window_size, window_size, 
                  W // window_size, window_size, C)
        windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        windows = windows.view(-1, window_size*window_size, C)
        return windows
    
class SalientFusion(nn.Module):
    def __init__(self, dim, threshold=0.7):
        super().__init__()
        self.dim = dim
        self.global_weight = nn.Parameter(torch.ones(1))
        self.saliency_conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.threshold = threshold

    def forward(self, x_global, x_local,H,W):

        B,N,C = x_global.shape

        x_global = x_global.permute(0,2,1).view(B,self.dim,H,W)
        x_local = x_local.permute(0,2,1).view(B,self.dim,H,W)

        saliency_map = torch.sigmoid(self.saliency_conv(x_local))
        

        mask = (saliency_map > self.threshold).float()
        

        fused = self.global_weight.sigmoid() * x_global + \
                mask * (1-self.global_weight.sigmoid()) * x_local
        fused = fused.permute(0,2,3,1).view(B,N,C).contiguous()
        return fused
    
class ChannelFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim//4, 1),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(),
            nn.Conv2d(dim//4, dim*2, 1)
        )
        self.dim = dim
        
    def forward(self, x_global, x_local, H,W):
        B,N,C = x_global.shape

        x_global = x_global.permute(0,2,1).view(B,self.dim,H,W)
        x_local = x_local.permute(0,2,1).view(B,self.dim,H,W)
        g_w = x_global.softmax(dim=1)
        l_w = x_local.softmax(dim=1)
        mask = torch.sigmoid(self.mlp(g_w+l_w))
        g_mask, l_mask = mask.chunk(2, dim=1)
        fused = g_mask * x_global + l_mask * x_local
        fused = fused.permute(0,2,3,1).view(B,N,C).contiguous()
        return fused
    
class GateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gate = nn.Sequential(
            nn.Conv2d(dim*2, dim//2,1),
            nn.ReLU(),
            nn.Conv2d(dim//2, 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_global, x_local, H,W):
        B,N,C = x_global.shape
        x_global = x_global.permute(0,2,1).view(B,self.dim,H,W)
        x_local = x_local.permute(0,2,1).view(B,self.dim,H,W)
        gate_value = self.gate(torch.cat([x_global, x_local], dim=1))
        #print(gate_value.shape,x_global.shape)
        fused = gate_value[0,0:1] * x_global + gate_value[0,1:2]* x_local
        fused = fused.permute(0,2,3,1).view(B,N,C).contiguous()
        return fused
        
class GLCA_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., depth=1, sr_ratio=1,glca=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GLCA_MixAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, depth=depth, glca=glca)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
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
    def __init__(self, img_size=512, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class GLCA_Segformer(nn.Module):
    def __init__(self, pretrain_path='', glca=0, img_size=224, patch_size=16, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):#, sparse_ratios=[1,1,0.9,0.8]):
        super().__init__()
        self.pretrain_path = pretrain_path

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
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([GLCA_TransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, 
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[0],
            depth=i, glca=0)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([GLCA_TransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],  sr_ratio=sr_ratios[1],
            depth=i, glca=0)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([GLCA_TransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, 
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[2],
            depth=i, glca=glca)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([GLCA_TransformerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],   sr_ratio=sr_ratios[3],
            depth=i, glca=glca)
            for i in range(depths[3])])
        
        self.norm4 = norm_layer(embed_dims[3])
        
        self.apply(self._init_weights)
        missing_keys, unexpected_keys=self.load_state_dict(torch.load(pretrain_path), strict=False)

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
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs



class mit_b0(GLCA_Segformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



class mit_b1(GLCA_Segformer):
    def __init__(self, pretrain_path = '', glca=0, **kwargs):
        super(mit_b1, self).__init__(pretrain_path = pretrain_path ,glca=glca,
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(GLCA_Segformer):
    def __init__(self, pretrain_path = '', glca = 0, **kwargs):
        super(mit_b2, self).__init__(pretrain_path = pretrain_path, glca=glca,
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



class mit_b3(GLCA_Segformer):
    def __init__(self,  pretrain_path = '', glca = 0,  **kwargs):
        super(mit_b3, self).__init__(pretrain_path = pretrain_path ,glca=glca, img_size = 512,
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



class mit_b4(GLCA_Segformer):
    def __init__(self, pretrain_path = '',glca = 0, **kwargs):
        super(mit_b4,  self).__init__(pretrain_path = pretrain_path, glca=glca,  img_size = 512,
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(GLCA_Segformer):
    def __init__(self, pretrain_path = '',glca = 0, **kwargs):
        super(mit_b5, self).__init__(pretrain_path = pretrain_path, glca=glca, img_size = 512, 
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)