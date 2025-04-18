import sys
from pprint import pprint
sys.path.append(".")
from torch.utils.data import Dataset, DataLoader
import IMDLBenCo
import IMDLBenCo.datasets
from IMDLBenCo.datasets import ManiDataset
from IMDLBenCo.datasets.jpeg_dataset_deprecated import MetaCatnetDataset
from IMDLBenCo.registry import DATASETS
import torch

from IMDLBenCo.evaluation import generate_region_mask, cal_confusion_matrix, cal_F1
from IMDLBenCo.datasets import denormalize


IS_PADDING = True
IS_RESIZING = False
batch_size = 3

# 主要测试这个样例数据集的指标：
data = IMDLBenCo.datasets.ManiDataset("/mnt/data0/public_datasets/IML/basic_eval_dataset", 
                                      is_padding=IS_PADDING,
                                      is_resizing=IS_RESIZING,
                                      edge_width=None)


batch_size = 3
dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

F1_list = []

for i in dataloader:
    i['image'] = torch.mean(i['image'], dim=1, keepdim=True) # [3, 3, 1024, 1024] -> [3, 1, 1024, 1024]
    mask = i['mask']
    pred = i['image']
    shape_mask = i['shape_mask']
    shape = i['shape']
    print(shape)
    mask_crop = mask[:, :, 0:shape[0], 0:shape[1]]
    
        
    import matplotlib.pyplot as plt
    plt.subplot(3, 3, 1)
    plt.imshow(i['mask'][0][0])
    plt.subplot(3, 3, 2)
    plt.imshow(i['mask'][1][0])
    
    # test for evaluation split
    # regin_mask = genertate_region_mask(i['masks'], i['shapes'])
    plt.subplot(3, 2, 3)
    plt.title('shape_mask')
    plt.imshow(i['shape_mask'][0][0])
    plt.subplot(3, 2, 4)
    plt.imshow(mask_crop[0])
    
    plt.subplot(3, 2, 5)
    plt.imshow(i['image'][0][0])
    plt.subplot(3, 2, 6)
    plt.imshow(i['image'][1][0])
    
        
    plt.savefig("test.png")
    exit()


