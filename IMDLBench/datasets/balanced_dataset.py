import random
from torch.utils.data import Dataset, DataLoader

from .iml_datasets import json_dataset, mani_dataset

from ..transforms import get_albu_transforms

from IMDLBench.registry import DATASETS




train_transform = get_albu_transforms('train')

def get_dataset(path, dataset_type):
    return dataset_type(path, 1024, train_transform, edge_width=7, if_return_shape=True)

@DATASETS.register_module()
class balanced_dataset(Dataset):
    def __init__(self, sample_number = 1840) -> None:
        self.sample_number = sample_number
        self.settings_list = [
            ['/mnt/data0/public_datasets/IML/CASIA2.0', mani_dataset],
            # ['/mnt/data0/public_datasets/IML/Fantastic_Reality_1024', mani_dataset], # TODO
            ['/mnt/data0/public_datasets/IML/IMD_20_1024', mani_dataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/sp_COCO_list.json', json_dataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/cm_COCO_list.json', json_dataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/bcm_COCO_list.json', json_dataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/bcmc_COCO_list.json', json_dataset]
        ]
        self.dataset_list = []
        for path, dataset_type in self.settings_list:
            self.dataset_list.append(
                get_dataset(path, dataset_type)
            )
        for i in self.dataset_list:
            print(len(i))
        
    def __len__(self):
        return self.sample_number * len(self.settings_list)    
    
    def __getitem__(self, index):
        dataset_index = index // self.sample_number

        selected_dataset = self.dataset_list[dataset_index]
        length = len(selected_dataset)
        selected_item = random.randint(0, length - 1)
        return selected_dataset[selected_item]

