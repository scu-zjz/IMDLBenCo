import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class PixelShuffleUpsampleV2(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, group_size=4, use_scope=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.group_size = group_size
        assert in_channels >= group_size and in_channels % group_size == 0

        out_channels = out_channels#2 * group_size * scale_factor ** 2

        # 增强offset预测能力，3x3卷积+激活
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        nn.init.normal_(self.offset_conv[-1].weight, std=0.001)
        nn.init.constant_(self.offset_conv[-1].bias, 0.)

        self.use_scope = use_scope
        if use_scope:
            self.scope_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Sigmoid()  # 范围0~1做权重
            )
            nn.init.constant_(self.scope_conv[-2].weight, 0.)
            nn.init.constant_(self.scope_conv[-2].bias, 0.)

        self.register_buffer('initial_pos', self._initialize_position())

    def _initialize_position(self):
        # h = torch.arange((-self.scale_factor + 1) / 2, (self.scale_factor - 1) / 2 + 1) / self.scale_factor
        # grid_x, grid_y = torch.meshgrid(h, h, indexing='ij')  # 兼容新pytorch版本
        # pos = torch.stack((grid_x.flatten(), grid_y.flatten()))  # [2, scale*scale]
        # pos = pos.repeat(self.group_size, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, 2*G*S^2, 1, 1]
        h = torch.arange((-self.scale_factor + 1) / 2, (self.scale_factor - 1) / 2 + 1) / self.scale_factor
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.group_size, 1).reshape(1, -1, 1, 1)
       

    def sample_coordinates(self, x, offset, scope=None):
        B, _, H, W = offset.shape
        device = x.device

        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H, device=device).float() + 0.5
        coords_w = torch.arange(W, device=device).float() + 0.5
        coords_y, coords_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
        coords = torch.stack((coords_x, coords_y), dim=0).unsqueeze(0).unsqueeze(2)  # [1, 2, 1, H, W]
        coords = coords.expand(B, 2, offset.shape[2], H, W)

        normalizer = torch.tensor([W, H], device=device).view(1, 2, 1, 1, 1)
        normalized_coords = 2 * (coords + offset) / normalizer - 1  # [-1, 1]

        # PixelShuffle解码
        normalized_coords = normalized_coords.view(B, 2 * self.group_size * self.scale_factor**2, H, W)
        normalized_coords = F.pixel_shuffle(normalized_coords, self.scale_factor)  # [B, 2*G, H*s, W*s]

        normalized_coords = normalized_coords.view(B * self.group_size, 2, self.scale_factor * H, self.scale_factor * W)
        normalized_coords = normalized_coords.permute(0, 2, 3, 1).contiguous()

        # 采样
        x_group = x.view(B * self.group_size, -1, H, W)
        sampled = F.grid_sample(x_group, normalized_coords, mode='bilinear', padding_mode='border', align_corners=False)

        sampled = sampled.view(B, -1, self.scale_factor * H, self.scale_factor * W)

        if scope is not None:
            scope = scope.view(B, -1, self.scale_factor * H, self.scale_factor * W)
            # 用scope权重重新加权输出，增强采样结果的表达能力
            sampled = sampled * scope

        return sampled

    def forward(self, x):

        offset = self.offset_conv(x) * 0.25 + self.initial_pos
        scope = None
        if self.use_scope:
            scope = self.scope_conv(x)

        return self.sample_coordinates(x, offset, scope)
    


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels,  scale_factor=2, group_size=4, use_scope=False):
        super(PixelShuffleUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.group_size = group_size
        
        assert in_channels >= group_size and in_channels % group_size == 0

        self.out_channels = 2 * group_size * scale_factor ** 2
        #self.group_size = in_channels//(2 * scale_factor ** 2)
        #self.out_channels = self.out_channels//2
        self.offset_conv = nn.Conv2d(in_channels, self.out_channels, 1)
        normal_init(self.offset_conv, std=0.001)
        if use_scope:
            self.scope_conv = nn.Conv2d(in_channels, self.out_channels, 1)
            constant_init(self.scope_conv, val=0.)

        self.register_buffer('initial_pos', self._initialize_position())

    def _initialize_position(self):
        h = torch.arange((-self.scale_factor + 1) / 2, (self.scale_factor - 1) / 2 + 1) / self.scale_factor
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.group_size, 1).reshape(1, -1, 1, 1)

    def sample_coordinates(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)

        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.contiguous()
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale_factor).view(
            B, 2, -1, self.scale_factor * H, self.scale_factor * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
       
        return F.grid_sample(x.reshape(B * self.group_size, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale_factor * H, self.scale_factor * W)

    def forward(self, x):
  
        offset = self.offset_conv(x) * 0.25 + self.initial_pos
        return self.sample_coordinates(x, offset)
    
class ChannelShuffleDownsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.out_channels = out_channels
        self.in_channels = in_channels
        # 自动计算输出通道数
        if out_channels is None:
            self.need_projection = False
        else:
            self.need_projection = True
            self.proj = nn.Conv2d(
                in_channels  = in_channels*(scale_factor**2),  # 来自PixelUnshuffle的通道扩展
                out_channels = out_channels, 
                kernel_size  = 1
            )
            nn.init.kaiming_normal_(self.proj.weight, mode='fan_out')

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 空间维度下采样，通道维度扩展
        x = F.pixel_unshuffle(x, self.scale_factor)  # [B, C*(r^2), H/r, W/r]
        
        # 通道维度处理
        if self.need_projection:
            # 使用1x1卷积调整通道数
            x = self.proj(x)
            x = x.reshape(B, self.out_channels, H//self.scale_factor, W//self.scale_factor)
        else:
            # 自动通道重组
            x = x.view(B, C, self.scale_factor**2, H//self.scale_factor, W//self.scale_factor)
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(
                B, self.scale_factor**2 * C, H//self.scale_factor, W//self.scale_factor
            )
        
        # 通道混洗增强信息流动
        x = channel_shuffle(x, groups=self.scale_factor)
        return x

def channel_shuffle(x, groups=2):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(B, C, H, W)

class CrossAttention_N(nn.Module):
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
        sr_ratio = 1
        if(dim<320):
          sr_ratio = 4  
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, y,H,W):
        B,N,C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            y_ = y.permute(0, 2, 1).reshape(B, C, H, W)
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class InteractiveFusion_N(nn.Module):
    """daul attention """
    def __init__(self, dim, sr_ratio=1):
        super().__init__()

        # 交叉模态注意力
        self.cross_attn1 = CrossAttention_N(dim, sr_ratio)
        self.cross_attn2 = CrossAttention_N(dim, sr_ratio)
        
        # 动态特征融合
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(2*dim, dim//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim//4, 2, 3, padding=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x1, x2):
        # 双向注意力交互
        B,C,H,W = x1.shape
        N = H*W
        x1 = x1.reshape(B,C,N).permute(0, 2, 1)# B,C,H,W ->B,N,C
        x2 = x2.reshape(B,C,N).permute(0, 2, 1)
        x1 = self.cross_attn1(x1, x2, H, W )  
        x2 = self.cross_attn2(x2, x1, H, W) 
        x1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
        x2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 动态门控融合
        gates = self.fuse_gate(torch.cat([x1, x2], dim=1))
        fused = gates[:,0:1] * x1 + gates[:,1:2] * x2
        return fused
    
class CrossAttention_C(nn.Module):
    """ 跨模态注意力机制 """
    def __init__(self, channel,reduction=4):
        super().__init__()
        self.query = nn.Conv2d(channel, channel//reduction, 1,bias=False)
        self.key = nn.Conv2d(channel, channel//reduction, 1,bias=False)
        self.value = nn.Conv2d(channel, channel, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, source, guidance):
        # 计算注意力图
        B, C, H, W = source.shape
        # source =source.float()
        # guidance = guidance.float()
        #source,guidance = channel_normalize(source,guidance)
        Q = self.query(source).view(B, -1, H*W)  # [B, C', N]
        K = self.key(guidance).view(B, -1, H*W)  # [B, C', N]
        V = self.value(guidance).view(B, -1, H*W)
        
        # 注意力分数
        energy = torch.bmm(Q.permute(0,2,1), K)  # [B, N, N]
        attention = F.softmax(energy, dim=-1)
        

        out = torch.bmm(V, attention.permute(0,2,1))
        #return out.view(B, -1, H, W)  # [B, C, H, W]
        return self.gamma * out.view(B, -1, H, W) + source
                           
class CrossAttention_C_New(nn.Module):
    """ cross attention  """
    def __init__(self, channel,reduction=1, sr_ratio=1):
        super().__init__()
        self.dim = channel//reduction
        self.q = nn.Conv2d(channel, self.dim, 1,bias=False)
        self.kv = nn.Conv2d(channel, self.dim*2, 1,bias=False) 
        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.scale = self.dim ** -0.5

        # if channel < 320:
        #    sr_ratio =4  
        # elif channel <128:
        #    sr_ratio =8
        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(channel, channel, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(channel)

    def forward(self, source, guidance):
        # 计算注意力图
        B, C, H, W = source.shape
        # H_sr = H // self.sr_ratio
        # W_sr = W// self.sr_ratio
        
        # # 注意力分数
        Q = self.q(source).view(B, self.dim, -1)  # [B, D, N]
        
        # if self.sr_ratio > 1:  
        #     guidance_ = self.sr(guidance).reshape(B, C, -1).permute(0, 2, 1)  #B,N',C
        #     guidance_ = self.norm(guidance_).permute(0, 2, 1) #B,N',C->B,C，N'
        #     guidance_ = guidance_.reshape(B,C,H_sr,W_sr)
        #     KV = self.kv(guidance_).reshape(B, 2, self.dim,-1).permute(1,0,2,3)  # [B,2, D, N']
        # else:
        KV = self.kv(guidance).view(B, 2, self.dim, -1).permute(1,0,2,3)  # [B,2, D, N']
        K,V = KV[0],KV[1]

        # 注意力分数
        energy = torch.bmm(Q.permute(0,2,1), K) * self.scale # [B, N, N']
        attention = F.softmax(energy, dim=-1)
        
        out = torch.bmm(attention,V.permute(0,2,1)).permute(0,2,1).view(B, -1, H, W) 
        out = self.gamma * out.view(B, -1, H, W) + source
        #
        #return out.view(B, -1, H, W)  # [B, C, H, W]
        return out
    
class InteractiveFusion_C(nn.Module):
    """daul attention """
    def __init__(self, channel,reduction = 1):
        super().__init__()
        # 交叉模态注意力
        self.cross_att1 = CrossAttention_C(channel,reduction = reduction)
        self.cross_att2 = CrossAttention_C(channel,reduction = reduction)

        # 动态特征融合
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(2*channel, channel//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel//4, 2, 3, padding=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x1, x2):
        # 双向注意力交互
        att_x1= self.cross_att1(x1, x2)  
        att_x2 = self.cross_att2(x2, x1)     # 用空间特征增强频域
        
        # 动态门控融合
        gates = self.fuse_gate(torch.cat([att_x1, att_x2], dim=1))
        return gates[:,0:1] * att_x1 + gates[:,1:2] * att_x2
    
class CrossAttention(nn.Module):
    """ cross attention  """
    def __init__(self, channel,reduction=1):
        super().__init__()
        self.dim = channel//reduction
        self.query = nn.Conv2d(channel, self.dim, 1,bias=False)
        self.kv = nn.Conv2d(channel, self.dim*2, 1,bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = self.dim ** -0.5
        self.proj = nn.Conv2d(self.dim, channel, 1)
        
    def forward(self, source, guidance):
        # 计算注意力图
        B, C, H, W = source.shape
        
        # # 注意力分数
        Q = self.query(source).view(B, -1, H*W)  # [B, C', N]
        KV = self.kv(guidance).view(B, -1, self.dim, H*W).permute(1,0,2,3)  # [B, C', C']
        K,V = KV[0],KV[1]
          
        # 注意力分数
        energy = (Q@K.permute(0,2,1)) * self.scale # [B, C', C']
        attention = F.softmax(energy, dim=-1)
        
        out = (attention@V).view(B, -1, H, W) 
        out = self.proj(out)
        #return out.view(B, -1, H, W)  # [B, C, H, W]
        return self.gamma * out.view(B, -1, H, W) + source
    
class InteractiveFusion(nn.Module):
    """daul attention """
    def __init__(self, channel):
        super().__init__()
        # 交叉模态注意力
        self.spatial_att = CrossAttention(channel)
        self.freq_att = CrossAttention(channel)
        
        # 动态特征融合
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(2*channel, channel//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel//4, 2, 3, padding=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, rgb, freq):
        # 双向注意力交互
        att_rgb = self.spatial_att(rgb, freq)  
        att_freq = self.freq_att(freq, rgb)     # 用空间特征增强频域
        
        # 动态门控融合
        gates = self.fuse_gate(torch.cat([att_rgb, att_freq], dim=1))
        return gates[:,0:1] * att_rgb + gates[:,1:2] * att_freq
