import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from .cmx.builder_np_conf import myEncoderDecoder as confcmx
from .config import _C as config
from .config import update_config

from IMDLBenCo.registry import MODELS


@MODELS.register_module()
class Trufor(nn.Module):
    def __init__(self,
                 phase: int = 2,
                 np_pretrain_weights: str = None,
                 mit_b2_pretrain_weights: str = None,
                 config_path: str = None,
                 det_resume_ckpt: str = None
                 ):
        super(Trufor, self).__init__()
        update_config(config, None, config_path)
        self.model = confcmx(cfg=config)
        self.phase = phase
        # training phase
        if phase == 2:
            self.model.dncnn.load_state_dict(
                torch.load(np_pretrain_weights))
            print("load noiseprint++ weight success.")
            state_dict = torch.load(mit_b2_pretrain_weights)
            self.model.backbone.load_state_dict(state_dict, strict=False)
            print("load mit_b2 weight success.")
            # freeze noiseprint and confidence encoder
            self.model.dncnn.requires_grad_(False)
            self.model.detection.requires_grad_(False)
        elif phase == 3:
            # freeze noiseprint and anomaly decoder
            self.model.requires_grad_(False)
            self.model.detection.requires_grad_(True)
            self.load_state_dict(torch.load(det_resume_ckpt))
            print("load pretrain weight success.")
        else:
            raise NotImplementedError('Trufor training phase not implement!')

    def weighted_cross_entropy_loss(self, prediction, target, gamma_0=0.5, gamma_1=2.5):
        loss = - (gamma_0 * (1 - target) * torch.log(1 - prediction) +
                  gamma_1 * target * torch.log(prediction))
        return loss.mean()

    def dice_loss(self, prediction, target, smooth=1.0):
        target = target.reshape(-1)
        prediction = prediction.reshape(-1)
        intersection = (target * prediction).sum()
        dice = (2.0 * intersection + smooth) / (torch.square(target).sum() + torch.square(prediction).sum() + smooth)

        return 1.0 - dice

    def loss_2(self, loss_ce, loss_dice, lamda=0.3):
        return lamda * loss_ce + (1 - lamda) * loss_dice

    def confidence_loss(self, conf, mask, pred_mask):
        tcp_map = (1 - mask) * (1 - pred_mask) + mask * pred_mask
        return ((conf - tcp_map) ** 2).mean()

    def detection_loss(self, pred_label, label):
        label = label.unsqueeze(1)
        return F.binary_cross_entropy(pred_label, label)

    def loss_3(self, confidence_loss, detection_loss, lamda_det=0.5):
        return confidence_loss + lamda_det * detection_loss

    def forward(self, image, mask, label, *args, **kwargs):
        label = label.float()
        pred_mask, conf, det, npp = self.model(image)
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = pred_mask[:, -1, ...].unsqueeze(1)
        pred_label = torch.sigmoid(det)

        if self.phase == 2:
            loss_ce = self.weighted_cross_entropy_loss(pred_mask, mask)
            loss_dice = self.dice_loss(pred_mask, mask)
            combined_loss = self.loss_2(loss_ce, loss_dice, 0.3)

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
                    "loss_ce": loss_ce,
                    "dice_loss": loss_dice,
                    "combined_loss": combined_loss
                },
                "visual_image": {
                    "pred_mask": pred_mask
                }
                # -----------------------------------------
            }
        elif self.phase == 3:
            loss_conf = self.confidence_loss(conf, mask, pred_mask)
            loss_det = self.detection_loss(pred_label, label)
            combined_loss = self.loss_3(loss_conf, loss_det)

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
                    "loss_conf": loss_conf,
                    "loss_det": loss_det,
                    "combined_loss": combined_loss
                },
                "visual_image": {
                    "pred_mask": pred_mask
                }
                # -----------------------------------------
            }

        return output_dict
