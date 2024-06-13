import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .seg_hrnet_config import get_hrnet_cfg
from .seg_hrnet import get_seg_model
from .NLCDetection import NLCDetection
from .detection_head import DetectionHead

from IMDLBenCo.registry import MODELS


@MODELS.register_module()
class PSCC_Net(nn.Module):
    def __init__(self,
                 input_size: int = 256,
                 pretrain_path: str = None
                 ):
        super(PSCC_Net, self).__init__()
        self.FENet = get_seg_model(get_hrnet_cfg(pretrain_path))
        self.args = {'crop_size': [input_size, input_size],
                     'train_ratio': [0.5947, 0.25, 0.25, 0.25]}
        self.SegNet = NLCDetection(self.args)
        self.ClsNet = DetectionHead(self.args)
        authentic_ratio = self.args['train_ratio'][0]
        fake_ratio = 1 - authentic_ratio
        self.weights = nn.Parameter(torch.tensor([1. / authentic_ratio, 1. / fake_ratio]), requires_grad=False)

    def generate_4mask(self, mask):
        mask2 = TF.resize(mask, (mask.shape[2] // 2, mask.shape[3] // 2), antialias=True)
        mask3 = TF.resize(mask, (mask.shape[2] // 4, mask.shape[3] // 4), antialias=True)
        mask4 = TF.resize(mask, (mask.shape[2] // 8, mask.shape[3] // 8), antialias=True)

        mask = (mask > 0.5).float()
        mask2 = (mask2 > 0.5).float()
        mask3 = (mask3 > 0.5).float()
        mask4 = (mask4 > 0.5).float()

        return mask, mask2, mask3, mask4

    def get_mask_weight(self, mask):
        mask1, mask2, mask3, mask4 = self.generate_4mask(mask)

        # median-frequency class weighting
        mask1_balance = torch.ones_like(mask1)
        if (mask1 == 1).sum():
            mask1_balance[mask1 == 1] = 0.5 / ((mask1 == 1).sum().to(torch.float) / mask1.numel())
            mask1_balance[mask1 == 0] = 0.5 / ((mask1 == 0).sum().to(torch.float) / mask1.numel())
        else:
            print('Mask1 balance is not working!')

        mask2_balance = torch.ones_like(mask2)
        if (mask2 == 1).sum():
            mask2_balance[mask2 == 1] = 0.5 / ((mask2 == 1).sum().to(torch.float) / mask2.numel())
            mask2_balance[mask2 == 0] = 0.5 / ((mask2 == 0).sum().to(torch.float) / mask2.numel())
        else:
            print('Mask2 balance is not working!')

        mask3_balance = torch.ones_like(mask3)
        if (mask3 == 1).sum():
            mask3_balance[mask3 == 1] = 0.5 / ((mask3 == 1).sum().to(torch.float) / mask3.numel())
            mask3_balance[mask3 == 0] = 0.5 / ((mask3 == 0).sum().to(torch.float) / mask3.numel())
        else:
            print('Mask3 balance is not working!')

        mask4_balance = torch.ones_like(mask4)
        if (mask4 == 1).sum():
            mask4_balance[mask4 == 1] = 0.5 / ((mask4 == 1).sum().to(torch.float) / mask4.numel())
            mask4_balance[mask4 == 0] = 0.5 / ((mask4 == 0).sum().to(torch.float) / mask4.numel())
        else:
            print('Mask4 balance is not working!')

        return mask1_balance, mask2_balance, mask3_balance, mask4_balance

    def forward(self, image, mask, label, *args, **kwargs):

        label = label.float()

        # cross entropy loss

        CE_loss = nn.CrossEntropyLoss(weight=self.weights)
        # BCE_loss_full = nn.BCELoss(reduction='none')
        BCE_loss_full = nn.BCEWithLogitsLoss(reduction='none')

        # weight
        mask1, mask2, mask3, mask4 = self.generate_4mask(mask)
        mask1_balance, mask2_balance, mask3_balance, mask4_balance = self.get_mask_weight(mask)

        # forward
        feat = self.FENet(image)
        [pred_mask1, pred_mask2, pred_mask3, pred_mask4] = self.SegNet(feat)
        pred_mask = pred_mask1
        pred_logit = self.ClsNet(feat)
        pred_logit = torch.softmax(pred_logit, dim=1)
        pred_label = pred_logit[:, -1, ...]

        # loss
        mask1_loss = torch.mean(BCE_loss_full(pred_mask1, mask1) * mask1_balance)
        mask2_loss = torch.mean(BCE_loss_full(pred_mask2, mask2) * mask2_balance)
        mask3_loss = torch.mean(BCE_loss_full(pred_mask3, mask3) * mask3_balance)
        mask4_loss = torch.mean(BCE_loss_full(pred_mask4, mask4) * mask4_balance)
        seg_loss = mask1_loss + mask2_loss + mask3_loss + mask4_loss

        cls_loss = F.binary_cross_entropy(pred_label, label)

        combined_loss = seg_loss + cls_loss

        output_dict = {
            # loss for backward
            "backward_loss": combined_loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": pred_mask,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "seg_loss": seg_loss,
                "cls_loss": cls_loss,
                "combined_loss": combined_loss
            },
            "visual_image": {
                "pred_mask": pred_mask,
            }
            # -----------------------------------------
        }

        return output_dict
