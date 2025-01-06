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




# data = IMDLBench.datasets.mani_dataset("/mnt/data0/public_datasets/IML/CASIA2.0", is_padding=True, edge_width= 7)
            # ['/mnt/data0/public_datasets/IML/IMD_20_1024', MetaCatnetDataset],
            # ['/mnt/data0/public_datasets/IML/tampCOCO/sp_COCO_list.json', MetaCatnetDataset],
            # ['/mnt/data0/public_datasets/IML/tampCOCO/cm_COCO_list.json', MetaCatnetDataset],
            # ['/mnt/data0/public_datasets/IML/tampCOCO/bcm_COCO_list.json', MetaCatnetDataset],
            # ['/mnt/data0/public_datasets/IML/tampCOCO/bcmc_COCO_list.json', MetaCatnetDataset]
data = MetaCatnetDataset("/mnt/data0/public_datasets/IML/IMD_20_1024", is_padding=True, edge_width= 7)


batch_size = 1
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

for i in data:
    for key in i.keys():
        if isinstance(i[key], torch.Tensor):
            print(key, i[key].shape)
    exit()