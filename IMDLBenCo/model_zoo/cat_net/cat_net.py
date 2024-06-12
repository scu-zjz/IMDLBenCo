import torch
import torch.nn as nn
from .network_CAT import get_seg_model, CrossEntropy

from yacs.config import CfgNode
from torch.nn import functional as F
import yaml
import random

from IMDLBenCo.registry import MODELS

@MODELS.register_module()
class Cat_Net(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """

  def __init__(self, cfg_file):
    super(Cat_Net, self).__init__()
    cfg = None
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.safe_load(f)
        cfg = CfgNode(yaml_cfg)
    self.model = get_seg_model(cfg)
    self.loss = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL, weight=torch.FloatTensor([0.5, 2.5])).cuda()

  def forward(self, image, mask, DCT_coef, qtables, label, name, if_predcit_label=None, edge_mask=None, shape=None, ):
    images, masks = self.__post_process_tensor(image, mask, DCT_coef)
    images, masks = images.detach(), masks.squeeze(1).detach()
    qtables = qtables.unsqueeze(1)
    outputs = self.model(images.float(), qtables.float())
    loss = self.loss(outputs, masks.long())
    pred = F.softmax(outputs, dim=1)[:, 1].unsqueeze(1)
    pred = F.interpolate(pred, size=(image.shape[2], image.shape[3]), mode='bicubic')
    output_dict = {
        # loss for backward
        "backward_loss": loss,
        # predicted mask, will calculate for metrics automatically
        "pred_mask": pred,
        "pred_label": None,
        "visual_loss": {
            "predict_loss": loss,
            },

        "visual_image": {
            "pred_mask": pred,
        }

    }
    return output_dict

  def __post_process_tensor(self, image_tensor, mask, DCT_coef):
    ignore_index = -1
    crop_size = (512, 512)
    img_RGB = torch.permute(image_tensor, (0, 2, 3, 1))  # batch_size * h * w * 3
    h, w = img_RGB.shape[1], img_RGB.shape[2]
    t_RGB = (torch.permute(img_RGB, (0, 3, 1, 2)) - 127.5) / 127.5

    T = 20
    t_DCT_vol = torch.zeros(size=(DCT_coef.shape[0], T+1, DCT_coef.shape[1], DCT_coef.shape[2])).cuda()
    t_DCT_vol[:, 0] += (DCT_coef.cuda() == 0).float()

    for i in range(1, T):
        t_DCT_vol[:, i] += (DCT_coef == i).float()
        t_DCT_vol[:, i] += (DCT_coef == -i).float()
    t_DCT_vol[:, T] += (DCT_coef >= T).float()
    t_DCT_vol[:, T] += (DCT_coef <= -T).float()
    tensor = torch.cat([t_RGB, t_DCT_vol], dim=1)
    return tensor, mask.long()



