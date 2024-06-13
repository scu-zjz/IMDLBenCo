from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout
from timm.models.vision_transformer import Block

def dct(x):
    N = x.size(-1)
    x_v = torch.cat([x, x.flip(dims=[-1])], dim=-1)
    x_v = torch.fft.fft(x_v, dim=-1)
    return torch.real(x_v[..., :N])

def idct(x):
    N = x.size(-1)
    x_v = torch.cat([x, x[..., 1:N - 1].flip(dims=[-1])], dim=-1)
    x_v = torch.fft.ifft(x_v, dim=-1)
    return torch.real(x_v[..., :N])

def dct_2d(x):
    x = dct(x)
    x = dct(x.transpose(-1, -2)).transpose(-1, -2)
    return x

def idct_2d(x):
    x = idct(x)
    x = idct(x.transpose(-1, -2)).transpose(-1, -2)
    return x

# High-Frequency Feature Extractor
class HighFrequencyFeatureExtractor(nn.Module):
    def __init__(self, alpha=0.5):
        super(HighFrequencyFeatureExtractor, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        x_dct = dct_2d(x)
        x_hp = x_dct * (x_dct.abs() > self.alpha)
        x_ifreq = idct_2d(x_hp)
        return x_ifreq

# Sinusoidal positional encoding
def positional_encoding(sequence_length, embedding_dim):
    position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
    pos_encoding = torch.zeros(sequence_length, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding

import torch
import torch.nn as nn
import timm

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 12,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor) -> torch.Tensor:

        B, N, C = q.shape
        # import pdb
        # pdb.set_trace()
        q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# Object Encoder
class ObjectEncoder(nn.Module):
    def __init__(
            self,
            dim: int =768,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            num_prototypes=768,
    ) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.object_prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, patch_embeddings) -> torch.Tensor:
        batch_size = patch_embeddings.shape[0]
        query = self.norm1(self.object_prototypes).unsqueeze(0).repeat(batch_size,1,  1) 
        k = self.norm1(patch_embeddings)
        object_features = query + self.attn(query, k, k)
        object_features = object_features + self.mlp(self.norm2(object_features))
        return object_features


class BCIM(nn.Module):
    def __init__(self, embedding_dim, window_size=3):
        super(BCIM, self).__init__()
        self.window_size = window_size
        self.embedding_dim = embedding_dim

    def forward(self, patch_embeddings):  
        # Calculate height and width from sequence length
        batch_size, sequence_length, channels = patch_embeddings.shape
        height = width = int((sequence_length / 2) ** 0.5)  # Assuming square shape and divided by 2 due to concatenation
        
        # Reshape patch embeddings to 2D feature map
        feature_map = patch_embeddings.permute(0, 2, 1).reshape(batch_size, 2 * channels, height, width)  # Shape: [batch_size, 2*embedding_dim, height, width]
        
        # Unfold feature map to get local patches
        unfolded = F.unfold(feature_map, kernel_size=self.window_size, padding=self.window_size // 2)
        unfolded = unfolded.reshape(batch_size, 2 * channels, self.window_size**2, height, width)  # Shape: [batch_size, 2*embedding_dim, window_size*window_size, height, width]

        # Compute cosine similarity between center pixel and its local window
        center_pixel = feature_map.unsqueeze(2)  # Shape: [batch_size, 2*embedding_dim, 1, height, width]
        cosine_similarity = F.cosine_similarity(center_pixel, unfolded, dim=1)  # Shape: [batch_size, window_size*window_size, height, width]

        # Average similarity within the window
        similarity_matrix = cosine_similarity.mean(dim=1, keepdim=True) / (self.window_size ** 2)  # Shape: [batch_size, 1, height, width]

        # Compute boundary-sensitive feature map
        boundary_sensitive_feature_map = feature_map * similarity_matrix  # Shape: [batch_size, 2*embedding_dim, height, width]

        # Serialize back to patch embeddings
        patch_embeddings = boundary_sensitive_feature_map.reshape(batch_size, -1, 2 * height * width).permute(0, 2, 1)  # Shape: [batch_size, 2* height*width,  embedding_dim]
        return patch_embeddings

# Patch Decoder
class PatchDecoder(nn.Module):
    def __init__(
                self,
                dim: int = 768,
                num_heads: int = 12,
                mlp_ratio: float = 4.,
                qkv_bias: bool = True,
                qk_norm: bool = False,
                proj_drop: float = 0.,
                attn_drop: float = 0.,
                init_values: Optional[float] = None,
                drop_path: float = 0.,
                act_layer: nn.Module = nn.GELU,
                norm_layer: nn.Module = nn.LayerNorm,
                mlp_layer: nn.Module = Mlp,
        ) -> None:
            super().__init__()
            self.norm1 = norm_layer(dim)
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )

            self.norm2 = norm_layer(dim)
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )

    def forward(self, patch_embeddings,object_features) -> torch.Tensor:
        patch_embeddings = self.norm1(patch_embeddings)
        object_features = self.norm1(object_features)
        output = patch_embeddings + self.attn(patch_embeddings, object_features, object_features)
        output = output + self.mlp(self.norm2(output))
        return output

class EncoderDecoder(nn.Module):
    def __init__(self, num_prototypes, embedding_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = ObjectEncoder(embedding_dim, num_prototypes=num_prototypes)
        self.decoder = PatchDecoder(embedding_dim)
        self.bcim = BCIM(embedding_dim)
    def forward(self, patch_embeddings):
        return self.bcim(self.decoder(patch_embeddings, self.encoder(patch_embeddings)))

import torchvision.models as models



sys.path.append('./modules')

from IMDLBenCo.registry import MODELS
@MODELS.register_module()
class ObjectFormer(nn.Module):
    def __init__(self,patch_size:int =16 ,input_size:int =224, num_prototypes:int =392, embedding_dim:int =768, num_layers:int =8, init_weight_path:str =None):
        super(ObjectFormer, self).__init__()
        self.input_size = input_size
        self.encoder_net_r = nn.Sequential(
            # nn.Conv2d(3, 3, kernel_size=33, stride=1, padding=0),
            nn.Conv2d(3, embedding_dim, kernel_size=patch_size, stride=patch_size)
        )
        self.encoder_net_q = nn.Sequential(
            # nn.Conv2d(3, 3, kernel_size=33, stride=1, padding=0),
            nn.Conv2d(3, embedding_dim, kernel_size=patch_size, stride=patch_size)
        )
        self.high_freq_extractor = HighFrequencyFeatureExtractor()

        self.encoder_decoders = nn.ModuleList([EncoderDecoder(num_prototypes, embedding_dim) for _ in range(num_layers)])

        self.localization = nn.Sequential(
            nn.Conv2d(1536, 768, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(384, 1, kernel_size=3, stride=1, padding=1),
            # nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=33, stride=1, padding=0)
        )

        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.pos_encoding = nn.Parameter(torch.randn(1, 2 * (input_size // patch_size) ** 2, embedding_dim) * .02)

        if init_weight_path is not None:
            self._load_pretrain(init_weight_path)
    def _load_pretrain(self,init_weight_path):
        state_dict = torch.load(init_weight_path)
        for i, block in enumerate(self.encoder_decoders):
            block.encoder.load_state_dict({k.replace(f'blocks.{i}.', ''): v for k, v in state_dict.items() if f'blocks.{i}.' in k}, strict = False)
            block.decoder.load_state_dict({k.replace(f'blocks.{i}.', ''): v for k, v in state_dict.items() if f'blocks.{i}.' in k}, strict = True)
        self.encoder_net_r[0].load_state_dict({k.replace('patch_embed.proj.','') : v for k, v in state_dict.items() if 'patch_embed' in k})
        self.encoder_net_q[0].load_state_dict({k.replace('patch_embed.proj.','') : v for k, v in state_dict.items() if 'patch_embed' in k})
        self.pos_encoding = nn.Parameter(state_dict['pos_embed'])
    def forward(self, image, mask, label, *args, **kwargs):        
        x_rgb = self.encoder_net_r(image)
        x_freq = self.high_freq_extractor(image)
        x_freq = self.encoder_net_q(x_freq)
        # import pdb
        # pdb.set_trace()
        # Generate patches and flatten
        batch_size, channels, height, width = x_rgb.shape
        L = height * width
        # print(L)
        x_rgb_patches = x_rgb.reshape(batch_size, channels, L).permute(0, 2, 1)  # Shape: [ batch_size, L, channels]
        x_freq_patches = x_freq.reshape(batch_size, channels, L).permute(0, 2, 1)  # Shape: [batch_size, L, channels]

        # Concatenate patches
        patch_embeddings = torch.cat([x_rgb_patches, x_freq_patches], dim=1)  # Shape: [batch_size, 2L, channels]

        # Add positional encoding

        patch_embeddings += self.pos_encoding # Shape: [batch_size, 2L, channels]

        for encoder_decoder in self.encoder_decoders:
            patch_embeddings = encoder_decoder(patch_embeddings)

        refined_patches = patch_embeddings.permute(0, 2, 1).reshape(batch_size, -1, height, width)  # Shape back to [batch_size, embedding_dim, height, width]


        mask_pred = F.interpolate(refined_patches, scale_factor=2, mode='bilinear', align_corners=False)
        for layer in self.localization:
            mask_pred = layer(mask_pred)
            mask_pred = F.interpolate(mask_pred, scale_factor=2, mode='bilinear', align_corners=False)
        # mask_pred = self.localization[-1](mask_pred)
        mask_loss = self.BCE_loss(mask_pred, mask)
        mask_pred = torch.sigmoid(mask_pred)

        label_loss = torch.tensor(0.0)
        lable_pred = None
        combined_loss =  mask_loss + label_loss
        output_dict = {
            # loss for backward
            "backward_loss": combined_loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": mask_pred,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": lable_pred,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "predict_loss": mask_loss,
                'lable_loss' : label_loss
            },

            "visual_image": {
                "pred_mask": mask_pred,
            }
            # -----------------------------------------
        }

        return output_dict