import os
import json
import random
import numpy as np
from PIL import Image

# 配置参数
IMAGE_SIZE = (512, 512)  # (height, width)
NUM_IMAGES = 20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建目录
os.makedirs(os.path.join(BASE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "masks"), exist_ok=True)

def generate_random_rectangle(img_size):
    """生成随机位置和大小的矩形区域（在图像中心附近）"""
    h, w = img_size
    # 随机中心点偏移范围
    max_offset = min(h, w) // 6
    cx = w//2 + random.randint(-max_offset, max_offset)
    cy = h//2 + random.randint(-max_offset, max_offset)
    
    # 随机尺寸
    rect_w = random.randint(min(h, w)//4, min(h, w)//2)
    rect_h = random.randint(min(h, w)//4, min(h, w)//2)
    
    # 计算边界
    x1 = max(0, cx - rect_w//2)
    y1 = max(0, cy - rect_h//2)
    x2 = min(w, cx + rect_w//2)
    y2 = min(h, cy + rect_h//2)
    return (x1, y1, x2, y2)

def generate_mask_with_blocks(img_size):
    """生成带随机黑块的噪声mask"""
    # 生成基础噪声
    mask = np.random.choice([0, 255], img_size, p=[0.5, 0.5]).astype(np.uint8)
    
    # 添加随机黑块
    for _ in range(random.randint(1, 3)):  # 1-3个黑块
        block_w = random.randint(20, 80)
        block_h = random.randint(20, 80)
        x = random.randint(0, img_size[1] - block_w)
        y = random.randint(0, img_size[0] - block_h)
        mask[y:y+block_h, x:x+block_w] = 0
    return mask

# 生成数据集
dataset = []

for idx in range(NUM_IMAGES):
    # 生成基础噪声图像
    img = np.random.randint(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8)
    
    # 添加随机矩形区域
    rect = generate_random_rectangle(IMAGE_SIZE)
    img[rect[1]:rect[3], rect[0]:rect[2], :] = 0
    
    # 保存图像
    img_name = f"{idx+1:02d}.jpg"
    img_path = os.path.join("images", img_name)
    Image.fromarray(img).save(os.path.join(BASE_DIR, img_path))
    
    # 处理前10张图的mask
    if idx < 10:
        mask = generate_mask_with_blocks(IMAGE_SIZE)
        mask_name = f"{idx+1:02d}_mask.jpg"
        mask_path = os.path.join("masks", mask_name)
        Image.fromarray(mask).save(os.path.join(BASE_DIR, mask_path))
        dataset.append([img_path, mask_path])
    else:
        dataset.append([img_path, "Negative"])

# 保存JSON文件
with open(os.path.join(BASE_DIR, "dataset.json"), "w") as f:
    json.dump(dataset, f, indent=2)

print("数据集生成完成！")