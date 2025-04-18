import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm

from IMDLBenCo.datasets import JsonDataset
from mymodel import MyModel  # 确保你已注册了这个模型

# 1. 加载数据集
dataset = JsonDataset(
    path="/mnt/data0/xiaochen/workspace/IMDLBenCo_pure/test_metrics/dataset.json",
    is_resizing=True,
    output_size=(512, 512),
)

# 2. DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 3. 加载模型
model = MyModel()
model.eval()

# 4. 存储所有结果
all_image_preds = []
all_image_labels = []

pixel_f1_list = []
pixel_auc_list = []
pixel_acc_list = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        image = batch['image']            # [B, C, H, W]
        mask = batch['mask']              # [B, 1, H, W]
        label = batch['label']            # scalar
        name = batch['name']      # list[str] of length B=1

        output = model(image, mask, name)
        
        # -------- image-level 预测 --------
        pred_label = output['pred_label'].cpu().numpy()  # shape: (B,)
        true_label = np.array([label])                   # shape: (B,)

        all_image_preds.extend(pred_label.tolist())
        all_image_labels.extend(true_label.tolist())

        # -------- pixel-level 预测 --------
        pred_mask = output['pred_mask'].cpu().numpy()    # [B, 1, H, W]
        true_mask = mask.cpu().numpy()                   # [B, 1, H, W]

        # flatten 到 [B, H*W]
        pred_flat = pred_mask.reshape(-1)
        true_flat = true_mask.reshape(-1)

        pixel_f1 = f1_score(true_flat, np.round(pred_flat), zero_division=1)
        # pixel_auc = roc_auc_score(true_flat, pred_flat)
        pixel_acc = accuracy_score(true_flat, np.round(pred_flat))
        pixel_f1_list.append(pixel_f1)
        # pixel_auc_list.append(pixel_auc)
        pixel_acc_list.append(pixel_acc)

# ---------- 计算 image-level metrics ----------
print("\n=== Image-level Metrics ===")
img_f1 = f1_score(all_image_labels, np.round(all_image_preds), zero_division=1)
img_auc = roc_auc_score(all_image_labels, all_image_preds)
img_acc = accuracy_score(all_image_labels, np.round(all_image_preds))

print(f"F1 Score: {img_f1:.4f}")
print(f"AUC Score: {img_auc:.4f}")
print(f"Accuracy: {img_acc:.4f}")

# ---------- 计算 pixel-level metrics ----------
print("\n=== Pixel-level Metrics ===")
# pix_f1 = f1_score(all_pixel_labels, np.round(all_pixel_preds), zero_division=1)
# pix_auc = roc_auc_score(all_pixel_labels, all_pixel_preds)
# pix_acc = accuracy_score(all_pixel_labels, np.round(all_pixel_preds))
pix_f1 = np.mean(pixel_f1_list)
# pix_auc = np.mean(pixel_auc_list)
pix_acc = np.mean(pixel_acc_list)


print(f"F1 Score: {pix_f1:.4f}")
# print(f"AUC Score: {pix_auc:.4f}")
print(f"Accuracy: {pix_acc:.4f}")
