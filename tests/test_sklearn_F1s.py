import numpy as np
import cv2
from sklearn.metrics import f1_score
import os

# 目录路径 (Directory path)
output_dir = "output_masks_shape"

import numpy as np
import cv2
from PIL import Image
import os

# 创建目录以保存图像 (Create directory to save images)
os.makedirs(output_dir, exist_ok=True)

# 设置图像的宽度和高度 (Set the width and height of the image)
width, height = 256, 256

def draw_random_polygons(image, num_polygons=5, min_vertices=3, max_vertices=8):
    for _ in range(num_polygons):
        num_vertices = np.random.randint(min_vertices, max_vertices + 1)
        points = np.random.randint(0, min(width, height), (num_vertices, 2))
        points = points.reshape((-1, 1, 2))
        color = (np.random.randint(256),)  # 单通道颜色 (Single channel color)
        cv2.fillPoly(image, [points], color)
    return image

for i in range(1, 9):
    # 创建空的灰度图像 (Create an empty grayscale image)
    prediction_mask = np.zeros((height, width), dtype=np.uint8)
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    # 在图像上绘制随机多边形 (Draw random polygons on the image)
    prediction_mask = draw_random_polygons(prediction_mask)
    gt_mask = draw_random_polygons(gt_mask) > 0.5

    # 将图像转换为Pillow图像并保存 (Convert image to Pillow image and save)
    prediction_image = Image.fromarray(prediction_mask)
    gt_image = Image.fromarray(gt_mask)

    # 保存预测mask和gt mask (Save prediction mask and ground truth mask)
    prediction_image.save(os.path.join(output_dir, f"image{i}.jpg"))
    gt_image.save(os.path.join(output_dir, f"mask{i}.jpg"))

print("图像和mask已成功生成并保存。 (Images and masks have been successfully generated and saved.)")


def calculate_binary_f1_score(pred, gt, return_cf_matrix=False):
    # 将图像展开成一维数组 (Flatten the images into 1D arrays)
    pred = pred.flatten()
    gt = gt.flatten()

    # 计算TP, FP, FN (Calculate TP, FP, FN)
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    # 计算Precision和Recall (Calculate Precision and Recall)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # 计算F1 Score (Calculate F1 Score)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    if return_cf_matrix == True:
        return f1, tp, fp, fn
    else:
        return f1


def calculate_macro_f1_score(pred, gt):
    f1 = calculate_binary_f1_score(pred, gt)
    f1_hat = calculate_binary_f1_score(1-pred, 1-gt)
    return (f1 + f1_hat) / 2


def calculate_micro_f1_score(pred, gt):
    f1, tp, fp, fn = calculate_binary_f1_score(pred, gt, return_cf_matrix=True)
    f1_, tp_, fp_, fn_ = calculate_binary_f1_score(1-pred, 1-gt, return_cf_matrix=True)
    
    tp += tp_
    fp += fp_
    fn += fn_
    
    # 计算Precision和Recall (Calculate Precision and Recall)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # 计算F1 Score (Calculate F1 Score)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return f1

# 初始化列表以存储F1 Score (Initialize list to store F1 Scores)
def check_f1(name, manual_f1):
    f1_scores = []
    manual_f1_scores = []
    for i in range(1, 9):
        # 读取预测mask和gt mask (Read prediction mask and ground truth mask)
        prediction_path = os.path.join(output_dir, f"image{i}.jpg")
        gt_path = os.path.join(output_dir, f"mask{i}.jpg")

        prediction_mask = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # 二值化图像 (Binarize the images - convert them to 0 and 1)
        _, prediction_mask_bin = cv2.threshold(prediction_mask, 127, 1, cv2.THRESH_BINARY)
        _, gt_mask_bin = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

        # 计算F1 Score (Calculate F1 Score)
        f1 = f1_score(gt_mask_bin.flatten(), prediction_mask_bin.flatten(), average=name)
        binary_f1 = f1_score(gt_mask_bin.flatten(), prediction_mask_bin.flatten(), average="binary")
        f1_manual = manual_f1(gt_mask_bin, prediction_mask_bin)
        f1_scores.append(f1)
        manual_f1_scores.append(f1_manual)

        print(f"Image {i} {name} F1 Score: {f1}")
        print(f"Image {i} binary F1 Score: {binary_f1}")
        print(f"Image {i} {name} manual F1 score {f1_manual}")
        print(f"\033[33m{name} F1 - Binary F1 == {f1 - binary_f1}\033[0m")
        if f1 == f1_manual:
            print(f"\033[92mImage {name} F1 - {i} - same with manual implementation!\033[0m")
        else:
            print(f"\033[91mImage {name} F1 - {i} - Incorrect with manual implementation!\033[0m")
        
    # 计算平均F1 Score (Calculate average F1 Score)
    mean_f1_score = np.mean(f1_scores)
    mean_manual_f1_score = np.mean(manual_f1_scores)
    print(f"Average F1 Score: {mean_f1_score}")
    print(f"Average manual F1 Score: {mean_manual_f1_score}")
    print(40*"*", '\n')
    
    
check_f1("binary", calculate_binary_f1_score)
check_f1("macro", calculate_macro_f1_score)
check_f1("micro", calculate_micro_f1_score)
