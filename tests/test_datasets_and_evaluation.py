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


data = IMDLBenCo.datasets.ManiDataset("/mnt/data0/public_datasets/IML/basic_eval_dataset", is_padding=True, edge_width= 7)
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
    plt.imshow(i['mask'][0][0])
    plt.subplot(3, 2, 2)
    plt.imshow(i['mask'][1][0])
    
    # test for evaluation split
    # regin_mask = genertate_region_mask(i['masks'], i['shapes'])
    plt.subplot(3, 2, 3)
    plt.title('shape_mask')
    plt.imshow(i['shape_mask'][0][0])
    plt.subplot(3, 2, 4)
    plt.imshow(i['shape_mask'][1][0])
    
    plt.subplot(3, 2, 5)
    plt.imshow(i['image'][0][0])
    plt.subplot(3, 2, 6)
    plt.imshow(i['image'][1][0])
    
        
        
        
    plt.savefig("test.png")
    exit()



