import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
import numpy as np
import os
import json
from . import PixelAttention as pa
from . import mantranet
from IMDLBenCo.registry import MODELS

@MODELS.register_module()
class SPAN(nn.Module):
    def __init__(self, weight_path, layers_steps=[1, 3, 9, 27, 81], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(SPAN, self).__init__()
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.featex = mantranet.IMTFE(device=device)
        self.featex.load_state_dict(torch.load(weight_path), strict=True)
        self.last_layer = self.Last_Layer_0725
        self.pixel_attention = nn.ModuleList([pa.PixelAttention(shift=step, in_channels=32, use_bn=False, use_res=True) for step in layers_steps]).to(device)
        self.outlier_trans = nn.Conv2d(self.featex.middle_and_last_block[-1].out_channels, 32, kernel_size=1, padding=0, bias=True).to(device)
        self.device = device

    def Last_Layer_0725(self, x):
        t = nn.Conv2d(32, 16, kernel_size=5, padding=2).to(x.device)(x)
        t = nn.ReLU().to(x.device)(t)
        t = nn.Conv2d(16, 8, kernel_size=5, padding=2).to(x.device)(t)
        t = nn.ReLU().to(x.device)(t)
        t = nn.Conv2d(8, 4, kernel_size=5, padding=2).to(x.device)(t)
        t = nn.ReLU().to(x.device)(t)
        t = nn.Conv2d(4, 1, kernel_size=5, padding=2).to(x.device)(t)
        return t

    def forward(self, image: torch.Tensor, mask=None, edge_mask=None, shape=None, *args, **kwargs):
        """denormalize image with imagenet mean and std
        then re-norm it to the preporcessing same with ManTra-Net (-1, 1)
        """        
        imagenet_mean=torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2).unsqueeze(0).to(self.device)
        imagenet_std=torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2).unsqueeze(0).to(self.device)
        image = image * imagenet_std # [B, 3, H, W]
        image = image + imagenet_mean
        
        image = image * 2 - 1
        x = image.to(self.device)
        x = self.featex(x)
        x = self.outlier_trans(x)
        for attention_layer in self.pixel_attention:
            x = attention_layer(x)
        mask_pred = self.last_layer(x)
        predict_loss = self.BCE_loss(mask_pred, mask)
        mask_pred = torch.sigmoid(mask_pred)
        output_dict = {
            "backward_loss": predict_loss,
            "pred_mask": mask_pred,
            "pred_label": None,
            "visual_loss": {
                "predict_loss": predict_loss,
            },

            "visual_image": {
                "pred_mask": mask_pred
            }
        }
        return output_dict

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPAN().to(device)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    mask_tensor = torch.randn(1, 1, 224, 224).to(device) 
    output_dict = model(input_tensor, mask=mask_tensor)
    print("output_dict['pred_mask'].shape:", output_dict["pred_mask"].shape)
    print("output_dict['backward_loss']:", output_dict["backward_loss"])
