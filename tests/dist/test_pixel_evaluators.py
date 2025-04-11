import os
import json
import torch
import pytest
import datetime
import builtins
from argparse import Namespace
import torch.distributed as dist
from torch import nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
import albumentations as albu

from IMDLBenCo.datasets import ManiDataset, JsonDataset
from IMDLBenCo.training_scripts.tester import test_one_epoch as inference_one_epoch


from IMDLBenCo.evaluation import (
    ImageF1, PixelF1,
    ImageAUC, PixelAUC,
    ImageAccuracy, PixelAccuracy,
    PixelIOU
)
from IMDLBenCo.datasets import denormalize

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

DATA_DIR = Path("tmp_data/only_mani")
NUM_IMAGES = 20
JSON_PATH = DATA_DIR / "dataset.json"
PRED_JSON_PATH = DATA_DIR / "pred.json"
TEST_BATCH_SIZE = 3

FREEZE = False

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

EVALUATOR_LIST = [
    # ImageF1(),
    PixelF1(),
    # ImageAUC(),
    PixelAUC(),
    # ImageAccuracy(),
    PixelAccuracy(),
    # PixelIOU()
]

def generate_noise_image(size=(256, 256)):
    """返回一张 RGB 随机噪声图像（uint8）"""
    arr = np.random.randint(0, 256, size + (3,), dtype=np.uint8)
    return Image.fromarray(arr)


def generate_noise_mask(size=(256, 256)):
    """返回一张黑白随机掩码图像"""
    arr = np.random.choice([0, 255], size=size, p=[0.5, 0.5]).astype(np.uint8)
    return Image.fromarray(arr)

def generate_dataset():
    """只在 rank 0 执行，生成图像 + mask"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # generate manipulated dataset
    print(NUM_IMAGES//2)
    for i in range(NUM_IMAGES // 2):
        img = generate_noise_image()
        mask = generate_noise_mask()
        img.save(DATA_DIR / f"img_{i:02d}.png")
        mask.save(DATA_DIR / f"mask_{i:02d}.png")
    # # generate authentic dataset
    # for i in range(NUM_IMAGES // 2, NUM_IMAGES):
    #     img = generate_noise_image()
    #     img.save(DATA_DIR / f"img_{i:02d}.png")
    print(f"[Rank 0] Dataset generated at {DATA_DIR}")
    # 生成 JSON 文件
    json_data = [
        [str(DATA_DIR / f"img_{i:02d}.png"), str(DATA_DIR / f"mask_{i:02d}.png")]
        for i in range(NUM_IMAGES//2)
    ]
    # json_data_back = [
    #     [str(DATA_DIR / f"img_{i:02d}.png"), "Negative"]
    #     for i in range(NUM_IMAGES//2, NUM_IMAGES)
    # ]
    # json_data.extend(json_data_back)

    # 用于load pred后结果的小trick json
    json_pred_data = [
        [str(DATA_DIR / f"pre_{i:02d}.png"), str(DATA_DIR / f"mask_{i:02d}.png")]
        for i in range(NUM_IMAGES//2)
    ]
    # json_pred_data_back = [
    #     [str(DATA_DIR / f"pre_{i:02d}.png"), "Negative"]
    #     for i in range(NUM_IMAGES//2, NUM_IMAGES)
    # ]
    # json_pred_data.extend(json_pred_data_back)

    with open(JSON_PATH, 'w') as f:
        json.dump(json_data, f)
    with open(PRED_JSON_PATH, 'w') as f:
        json.dump(json_pred_data, f)
    print(f"[Rank 0] JSON file generated at {JSON_PATH}")


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        B, C, H, W = TEST_BATCH_SIZE, 3, 256, 256
        # self.x1 = np.random.randint(0, H // 2)
        # self.y1 = np.random.randint(0, W // 2)
        # self.x2 = np.random.randint(H // 2 + 1, H)
        # self.y2 = np.random.randint(W // 2 + 1, W)
        self.x1 = 50
        self.y1 = 50
        self.x2 = 150
        self.y2 = 150
        """
        如果这里不固定掩膜区域的话，会导致在remain dataset处理的时候，重新生成一组掩膜位置不同的mask，导致前后数据集不一致。出错。
        """
    def forward(self, image : torch.Tensor, mask : torch.Tensor, name, **kwargs):
        # 随机在images shape内部生成一个黑色矩形块覆盖上去
        # 随机生成左上角和右下角坐标
        # 这里假设输入的图像是 [b,c,h,w] 的格式
        B, C, H, W = image.shape
        for i in range(B):
            # 生成一个黑色矩形块
            if not FREEZE:
                image[i, :, self.x1:self.x2, self.y1:self.y2] = 0
            # 随机生成一个预测结果，通过系数使其更容易为negative

        pred_label = torch.rand(B) * 0.8 # shape [B]
        # 将整个[b,c,h,w]的tensor拍扁成1维
        pred_mask = torch.mean(image, dim=1, keepdim=True)
        # 利用图片的左上角的像素值来保存该图片的label预测结果。小trick
        pred_mask[:, 0, 0, 0] = pred_label
        # print("mask_shape:", pred_mask.shape)

        for i in range(B):
            assert pred_mask[i][0][0][0] == pred_label[i], f"mask {i} label error"  

        # 转回numpy并保存这个mask
        pred_mask = pred_mask.clip(0, 1)
        # 尽可能保证向后计算指标的图，与保存的图像一致（不会被int卡掉太多精度）
        pred_mask = (pred_mask * 255).to(torch.uint8).to(torch.float32) / 255
        pred_tensor_mask = pred_mask.clone()
        pred_mask = pred_mask.cpu().numpy()
        # print(pred_mask.shape)  
        pred_mask = np.uint8(pred_mask * 255)
        for i in range(B):
            # print(name[i])
            pred_msk = Image.fromarray(pred_mask[i][0], mode='L')
            if not FREEZE:
                pred_msk.save(DATA_DIR / f"pre{name[i][3:]}")

        # 计算一个mse损失，没啥用
        loss = (image - mask) ** 2

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

def setup_distributed():
    if not torch.distributed.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')
    builtin_print = builtins.print
    is_master = dist.get_rank() == 0
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (dist.get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

    return dist.get_rank(), dist.get_world_size()

def test_main_evaluators():
    rank, world_size = setup_distributed()

    # Dummy argparser
    args = Namespace(
        distributed=True,
        test_batch_size=TEST_BATCH_SIZE,
        no_model_eval=False,
    )

    # Rank 0 生成一次图像
    if rank == 0 and not FREEZE:
        print(f"[Rank {rank}] Generating image...")
        generate_dataset()
    
    # 所有 rank 等待 rank 0 完成生成
    dist.barrier()
    train_transform = albu.Compose([
        # Nothing here haha
    ])
    mydataset = JsonDataset(
        path=JSON_PATH,
        is_resizing=True,
        output_size=(256, 256),
        common_transforms=train_transform,
    )
    # 创建分布式采样器
    sampler = torch.utils.data.distributed.DistributedSampler(
        mydataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True
    )
    # 分布式数据加载器
    mydataloader = torch.utils.data.DataLoader(
        mydataset,
        sampler=sampler,
        batch_size=args.test_batch_size,
        num_workers=0,
        drop_last=True
    )
    # 分布式模型：
    model = DummyModel()

    test_status = inference_one_epoch(
        model=model,
        data_loader=mydataloader,
        evaluator_list=EVALUATOR_LIST,
        epoch=0,
        device=DEVICE,
        name="pytest",
        print_freq=1,
        args=args # Dummy args
    )
    for key in test_status.keys():
        print(key, test_status[key])

    dist.barrier()

    if rank == 0:
        # 用Sklearn计算pixel level 的F1 score进行验算
        # 读取JSON文件
        pred_dataset = JsonDataset(
            path=PRED_JSON_PATH,
            is_resizing=True,
            output_size=(256, 256),
            common_transforms=train_transform,
        )
        pixel_f1_list = []
        for data in pred_dataset:
            pred_img = data['image']
            # print(pred_img.shape)
            pred_img = denormalize(pred_img)
            # print(pred_img)
            pred_img = torch.mean(pred_img, dim=0, keepdim=True)
            pred_img = (pred_img > 0.5) * 1
            # print(pred_img)
            pred_mask = data['mask']
            sklearn_f1 = f1_score(
                pred_mask.flatten(),
                pred_img.flatten(),
            ) 
            pixel_f1_list.append(sklearn_f1)
        print("Pixel F1 Score:", pixel_f1_list)
        # 计算平均值    
        avg_pixel_f1 = sum(pixel_f1_list) / len(pixel_f1_list)
        print("Average Pixel F1 Score:", avg_pixel_f1)
        print("Pixel F1 Score:", test_status["pixel-level F1"])
        print("F1 gap" , abs(test_status["pixel-level F1"] - avg_pixel_f1))
        assert( abs(test_status["pixel-level F1"] - avg_pixel_f1) < 1e-5), "Pixel F1 Score mismatch"

        """
        同上，用Sklearn计算pixel-level的ACC
        """
        pixel_acc_list = []
        for data in pred_dataset:
            pred_img = data['image']
            # print(pred_img.shape)
            pred_img = denormalize(pred_img)
            # print(pred_img)
            pred_img = torch.mean(pred_img, dim=0, keepdim=True)
            pred_img = (pred_img >= 0.5) * 1
            # print(pred_img)
            pred_mask = data['mask']
            sklearn_acc = accuracy_score(
                pred_mask.flatten(),
                pred_img.flatten(),
            ) 
            pixel_acc_list.append(sklearn_acc)
        print("Pixel ACC Score:", pixel_acc_list)
        # 计算平均值
        avg_pixel_acc = sum(pixel_acc_list) / len(pixel_acc_list)
        print("Average Pixel ACC Score:", avg_pixel_acc)
        print("Pixel ACC Score:", test_status['pixel-level Accuracy'])
        print("gap" , abs(test_status['pixel-level Accuracy'] - avg_pixel_acc))
        assert( abs(test_status['pixel-level Accuracy'] - avg_pixel_acc) < 1e-5), "Pixel ACC Score mismatch"


        """
        同上，用Sklearn计算pixel-level的AUC
        """
        pixel_auc_list = []
        for data in pred_dataset:
            pred_img = data['image']
            # print(pred_img.shape)
            pred_img = denormalize(pred_img)
            # print(pred_img)
            pred_img = torch.mean(pred_img, dim=0, keepdim=True)
            # print(pred_img)
            pred_mask = data['mask']
            sklearn_auc = roc_auc_score(
                pred_mask.flatten(),
                pred_img.flatten(),
            ) 
            # 如果是NAN则填0
            if np.isnan(sklearn_auc):
                sklearn_auc = 0 
            pixel_auc_list.append(sklearn_auc)
        print("Pixel AUC Score:", pixel_auc_list)
        # 计算平均值
        avg_pixel_auc = sum(pixel_auc_list) / len(pixel_auc_list)
        print("Average Pixel AUC Score:", avg_pixel_auc)
        print("Pixel AUC Score:", test_status['pixel-level AUC'])
        print("AUC gap" , abs(test_status['pixel-level AUC'] - avg_pixel_auc))
        assert abs(test_status['pixel-level AUC'] - avg_pixel_auc) < 0.001, "Pixel AUC Score mismatch"
test_main_evaluators()