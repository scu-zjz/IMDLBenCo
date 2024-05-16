import sys
from pprint import pprint
sys.path.append(".")
from torch.utils.data import Dataset, DataLoader
import IMDLBench
import IMDLBench.datasets
from IMDLBench.datasets import ManiDataset
from IMDLBench.datasets.jpeg_dataset import MetaCatnetDataset
from IMDLBench.registry import DATASETS
import torch

from IMDLBench.evaluation import genertate_region_mask, cal_confusion_matrix, cal_F1
from IMDLBench.datasets import denormalize


data = IMDLBench.datasets.ManiDataset("/home/psdz/Datasets/CASIA2.0_corrected", is_padding=True, edge_width= 7)
            # ['/mnt/data0/public_datasets/IML/IMD_20_1024', MetaCatnetDataset],
            # ['/mnt/data0/public_datasets/IML/tampCOCO/sp_COCO_list.json', MetaCatnetDataset],
            # ['/mnt/data0/public_datasets/IML/tampCOCO/cm_COCO_list.json', MetaCatnetDataset],
            # ['/mnt/data0/public_datasets/IML/tampCOCO/bcm_COCO_list.json', MetaCatnetDataset],
            # ['/mnt/data0/public_datasets/IML/tampCOCO/bcmc_COCO_list.json', MetaCatnetDataset]
# data = MetaCatnetDataset("/mnt/data0/public_datasets/IML/IMD_20_1024", is_padding=True, edge_width= 7)


batch_size = 3
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

for i in dataloader:
    for key in i.keys():
        if isinstance(i[key], torch.Tensor) and key != 'shapes':
            print(key, i[key].shape)
        else:
            print(key,  i[key])
            
            
    import matplotlib.pyplot as plt
    plt.subplot(3, 2, 1)
    plt.imshow(i['masks'][0][0])
    plt.subplot(3, 2, 2)
    plt.imshow(i['masks'][1][0])
    
    # test for evaluation split
    regin_mask = genertate_region_mask(i['masks'], i['shapes'])
    plt.subplot(3, 2, 3)
    plt.imshow(regin_mask[0][0])
    plt.subplot(3, 2, 4)
    plt.imshow(regin_mask[1][0])
    
    plt.subplot(3, 2, 5)
    plt.imshow(i['images'][0][0])
    plt.subplot(3, 2, 6)
    plt.imshow(i['images'][1][0])
    
        
        
        
    plt.savefig("test.png")
    exit()



