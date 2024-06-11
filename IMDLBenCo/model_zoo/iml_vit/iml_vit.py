from .windowed_attention_vit import ViT as window_attention_vit, SimpleFeaturePyramid, LastLevelMaxPool
from .decoderhead import PredictHead

import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from typing import List
import sys

sys.path.append('./modules')

from IMDLBenCo.registry import MODELS


@MODELS.register_module()
class IML_ViT(nn.Module):
    def __init__(
            self,
            # ViT backbone:
            input_size:int=1024,
            patch_size:int=16,
            embed_dim:int=768,
            vit_pretrain_path:str=None,  # wether to load pretrained weights
            # Simple_feature_pyramid_network:
            fpn_channels:int=256,
            fpn_scale_factors:List[int,]=(4.0, 2.0, 1.0, 0.5),
            # MLP embedding:
            mlp_embeding_dim:int=256,
            # Decoder head norm
            predict_head_norm:str="BN",
            # Edge loss:
            edge_lambda:int=20,
    ):
        """init iml_vit_model
        # TODO : add more args
        Args:
            input_size (int): size of the input image, defalut to 1024
            patch_size (int): patch size of Vision Transformer
            embed_dim (int): embedding dim for the ViT
            vit_pretrain_path (str): the path to initialize the model before start training
            fpn_channels (int): the number of embedding channels for simple feature pyraimd
            fpn_scale_factors(list(float, ...)) : the rescale factor for each SFPN layers.
            mlp_embedding dim: dim of mlp, i.e. decoder head
            predict_head_norm: the norm layer of predict head, need to select amoung 'BN', 'IN' and "LN"
                                We tested three different types of normalization in the decoder head, and they may yield different results due to dataset configurations and other factors.
                            Some intuitive conclusions are as follows:
                                - "LN" -> Layer norm : The fastest convergence, but poor generalization performance.
                                - "BN" Batch norm : When include authentic image during training, set batchsize = 2 may have poor performance. But if you can train with larger batchsize (e.g. A40 with 48GB memory can train with batchsize = 4) It may performs better.
                                - "IN" Instance norm : A form that can definitely converge, equivalent to a batchnorm with batchsize=1. When abnormal behavior is observed with BatchNorm, one can consider trying Instance Normalization. It's important to note that in this case, the settings should include setting track_running_stats and affine to True, rather than the default settings in PyTorch.
            edge_lambda(float) : the hyper-parameter for edge loss (lambda in our paper)
        """
        super(IML_ViT, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        # window attention vit
        self.encoder_net = window_attention_vit(
            img_size=input_size,
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[],
            use_rel_pos=True,
            out_feature="last_feat",
        )
        self.vit_pretrain_path = vit_pretrain_path

        # simple feature pyramid network
        self.featurePyramid_net = SimpleFeaturePyramid(
            in_feature_shape=(1, embed_dim, 256, 256),
            out_channels=fpn_channels,
            scale_factors=fpn_scale_factors,
            top_block=LastLevelMaxPool(),
            norm="LN",
        )
        # MLP predict head
        self.predict_head = PredictHead(
            feature_channels=[fpn_channels for i in range(5)],
            embed_dim=mlp_embeding_dim,
            norm=predict_head_norm  # important! may influence the results
        )
        # Edge loss hyper-parameters    
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.edge_lambda = edge_lambda

        self.apply(self._init_weights)
        self._mae_init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _mae_init_weights(self):
        # Load MAE pretrained weights for Window Attention ViT encoder
        if self.vit_pretrain_path != None:
            self.encoder_net.load_state_dict(
                torch.load(self.vit_pretrain_path, map_location='cpu')['model'],  # BEIT MAE
                strict=False
            )
            print('load pretrained weights from \'{}\'.'.format(self.vit_pretrain_path))

    def forward(self, image: torch.Tensor, mask, edge_mask, shape=None, *args, **kwargs):
        x = self.encoder_net(image)
        x = self.featurePyramid_net(x)
        feature_list = []
        for k, v in x.items():
            feature_list.append(v)
        x = self.predict_head(feature_list)

        # up-sample to 1024x1024
        mask_pred = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        # compute the loss
        predict_loss = self.BCE_loss(mask_pred, mask)

        edge_loss = F.binary_cross_entropy_with_logits(
            input=mask_pred,
            target=mask,
            weight=edge_mask
        ) * self.edge_lambda
        combined_loss = edge_loss + predict_loss

        mask_pred = torch.sigmoid(mask_pred)

        output_dict = {
            # loss for backward
            "backward_loss": combined_loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": mask_pred,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": None,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "predict_loss": predict_loss,
                "edge_loss": edge_loss,
                "combined_loss": combined_loss
            },

            "visual_image": {
                "pred_mask": mask_pred,
                "edge_mask": edge_mask
            }
            # -----------------------------------------
        }

        return output_dict
