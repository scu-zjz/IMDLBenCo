import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .abstract_class import AbstractEvaluator
import torch.distributed as dist
import os
from sklearn.metrics import jaccard_score

class PixelIOU(AbstractEvaluator):
    def __init__(self, threshold=0.5, mode="origin") -> None:
        self.name = "pixel-level IOU"
        self.desc = "pixel-level IOU"
        self.threshold = threshold
        self.mode = mode
    
    def Cal_IOU(self, predict, mask, shape_mask=None):
        if shape_mask is not None:
            predict = predict * shape_mask
            mask = mask * shape_mask
        
        predict = (predict > self.threshold).float().flatten(1)
        mask = mask.flatten(1)

        # Exclude parts masked by shape_mask
        # if shape_mask is not None:
        #     valid_mask = shape_mask.flatten(1) > 0
        #     predict = predict[valid_mask]
        #     mask = mask[valid_mask]

        # Compute intersection and union
        intersection = torch.sum(predict * mask, dim=1)
        union = torch.sum(predict,dim=1) + torch.sum(mask,dim=1) - intersection

        iou = intersection / (union + 1e-8)  # Add small value to avoid division by zero

        return iou

    def Cal_IOU_2(self, predict, mask, shape_mask=None):
        # 确保predict和mask是二进制的
        # 这个为黑占黑
        predict = (predict > self.threshold).float().to(torch.int8)
        mask = mask.to(torch.int8)
        predict = 1 - predict
        mask = 1 - mask
        if shape_mask is not None:
            predict = predict * shape_mask.to(torch.int8)
            mask = mask * shape_mask.to(torch.int8)

        # Flatten the tensors
        predict = predict.flatten(1)
        mask = mask.flatten(1)
        print(predict.shape)
        # Compute intersection and union
        intersection = torch.sum(predict * mask, dim=1)
        union = torch.sum(predict,dim=1) + torch.sum(mask,dim=1) - intersection

        iou = intersection / (union + 1e-8)  # Add small value to avoid division by zero

        return iou
    
    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        self._check_pixel_level_params(predict, mask)
        if self.mode == "origin":
            IOU = self.Cal_IOU(predict, mask, shape_mask)
            # IOU2 = self.Cal_IOU_2(predict, mask, shape_mask)
        elif self.mode == "reverse":
            IOU = self.Cal_IOU(1 - predict, mask, shape_mask)
            # IOU2 = self.Cal_IOU_2(1 - predict, mask, shape_mask)
        elif self.mode == "double":
            normal_iou = self.Cal_IOU(predict, mask, shape_mask)
            reverse_iou = self.Cal_IOU(1 - predict, mask, shape_mask)
            IOU = torch.max(normal_iou, reverse_iou)
            # normal_iou2 = self.Cal_IOU_2(predict, mask, shape_mask)
            # reverse_iou2 = self.Cal_IOU_2(1 - predict, mask, shape_mask)
            # IOU2 = torch.max(normal_iou2, reverse_iou2)
        else:
            raise RuntimeError(f"Cal_AUC no mode name {self.mode}")
        return IOU
    
    def remain_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        return self.batch_update(predict, mask, shape_mask, *args, **kwargs)

    def epoch_update(self):

        return None
    
    def recovery(self):
        return None



# # 示例用法和对比
if __name__ == "__main__":
    # 生成一些示例数据
    batch_size, channels, height, width = 1, 1, 10, 10
    predict = torch.rand(batch_size, channels, height, width)
    mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    # 生成一个 shape_mask
    # shape_mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    shape_mask = None
    iou = PixelIOU(mode="origin")
    reverse_iou = PixelIOU(mode="reverse")
    double_iou = PixelIOU(mode="double")
    # image_iou = Image_IOU()

    iou_value_pytorch = iou.batch_update(predict, mask, shape_mask)
    reverse_iou_value_pytorch = reverse_iou.batch_update(predict, mask, shape_mask)
    double_iou_value_pytorch = double_iou.batch_update(predict, mask, shape_mask)
    # image_iou_value_pytorch = image_iou(predict, mask)

    # Function to compute sklearn IOU
    def compute_sklearn_iou(y_true, y_pred, threshold=0.5):
        y_true = y_true.view(-1).numpy()
        y_pred = (y_pred.view(-1).numpy() > threshold).astype(int)
        return jaccard_score(y_true, y_pred, average=None)

    iou_value_sklearn = compute_sklearn_iou(mask, predict)
    reverse_iou_value_sklearn = compute_sklearn_iou(mask, 1 - predict)
    # double_iou_value_sklearn = max(iou_value_sklearn, reverse_iou_value_sklearn)
    print("--"*10)
    print(f"PyTorch IOU: {iou_value_pytorch}")
    print(f"Sklearn IOU: {iou_value_sklearn}")
    print("--"*10)
    print(f"PyTorch Reverse IOU: {reverse_iou_value_pytorch}")
    print(f"Sklearn Reverse IOU: {reverse_iou_value_sklearn}")
    print("--"*10)
    print(f"PyTorch Double IOU: {double_iou_value_pytorch}")
    # print(f"Sklearn Double IOU: {double_iou_value_sklearn}")
    print("--"*10)
    # print(f"PyTorch Image IOU: {image_iou_value_pytorch}")

    # 定义两个集合的列表表示
    A = [1, 2, 3, 4, 5]
    B = [1, 2, 6, 7, 8]

    # 使用sklearn中的jaccard_score函数计算Jaccard相似系数
    jaccard_coefficient = jaccard_score(A, B, average=None) # Jaccard相似系数: [1. 1. 0. 0. 0. 0. 0. 0.]

    print("Jaccard相似系数:", jaccard_coefficient)