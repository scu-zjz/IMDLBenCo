from IMDLBenCo.registry import MODELS
import torch.nn as nn
import torch
import numpy as np

"""
=== Image-level Metrics ===
F1 Score: 0.4211
AUC Score: 0.4300
Accuracy: 0.4500

=== Pixel-level Metrics ===
F1 Score: 0.1983
Accuracy: 0.5892
"""

@MODELS.register_module()
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 放一个虚假的参数用于欺骗DDP
        self.param = nn.Parameter(torch.randn(1))
    def forward(self, image : torch.Tensor, mask : torch.Tensor, name, **kwargs):
        """
        对于固定的name,测试数据集返回固定的label确保可以复现：
        name=["01.jpg", "02.jpg", "03.jpg", "04.jpg"... "20.jpg"],
        label是固定的0~1的数值
        """
        # 将整个[b,c,h,w]的tensor拍扁成1维
        pred_mask = torch.mean(image, dim=1, keepdim=True)
        # 利用图片的左上角的像素值来保存该图片的label预测结果。小trick

        # 转回numpy并保存这个mask
        pred_mask = pred_mask.clip(0, 1)
        # 尽可能保证向后计算指标的图，与保存的图像一致（不会被int卡掉太多精度）
        pred_mask = (pred_mask * 255).to(torch.uint8).to(torch.float32) / 255
        pred_tensor_mask = pred_mask.clone()
        pred_mask = pred_mask.cpu().numpy()
        # print(pred_mask.shape)  
        pred_mask = np.uint8(pred_mask * 255)

        # 计算一个mse损失，没啥用
        loss = (image - mask) ** 2

        pred_label_dict = {
            "01.jpg": 0.1,
            "02.jpg": 0.2,
            "03.jpg": 0.3,
            "04.jpg": 0.4,
            "05.jpg": 0.5,
            "06.jpg": 0.6,
            "07.jpg": 0.7,
            "08.jpg": 0.1,
            "09.jpg": 0.9,
            "10.jpg": 1.0,
            "11.jpg": 0.1,
            "12.jpg": 0.2,
            "13.jpg": 0.3,
            "14.jpg": 0.4,
            "15.jpg": 0.5,
            "16.jpg": 0.6,
            "17.jpg": 0.7,
            "18.jpg": 0.8,
            "19.jpg": 0.9,
            "20.jpg": 1.0,
        }
        # 用name做映射，然后返回0~1的 tensor作为label
        pred_label = []
        for i in range(len(name)):
            pred_label.append(pred_label_dict[name[i]])
        pred_label = torch.tensor(pred_label).to(image.device)
        output_dict = {
                "backward_loss": loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": pred_tensor_mask,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label,
            "visual_loss": {
            },
            "visual_image": {
            }
        }
        return output_dict
    
if __name__ == "__main__":
    print(MODELS)