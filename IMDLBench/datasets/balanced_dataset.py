import random
from torch.utils.data import Dataset, DataLoader

from .iml_datasets import JsonDataset, ManiDataset

from ..transforms import get_albu_transforms

from IMDLBench.registry import DATASETS

train_transform = get_albu_transforms('train')

def get_dataset(path, dataset_type):
    return dataset_type(path, 1024, train_transform, edge_width=7, if_return_shape=True)

@DATASETS.register_module()
class BalancedDataset(Dataset):
    """The BalancedDataset manages multiple iml_datasets, so it does not inherit from AbstractDataset.

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, sample_number = 1840) -> None:
        self.sample_number = sample_number
        self.settings_list = [
            ['/mnt/data0/public_datasets/IML/CASIA2.0', ManiDataset],
            # ['/mnt/data0/public_datasets/IML/Fantastic_Reality_1024', mani_dataset], # TODO
            ['/mnt/data0/public_datasets/IML/IMD_20_1024', ManiDataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/sp_COCO_list.json', JsonDataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/cm_COCO_list.json', JsonDataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/bcm_COCO_list.json', JsonDataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/bcmc_COCO_list.json', JsonDataset]
        ]
        
        self.dataset_list = [get_dataset(path, dataset_type) for path, dataset_type in self.settings_list]
        
    def __len__(self):
        return self.sample_number * len(self.settings_list)    
    
    def __getitem__(self, index):
        dataset_index = index // self.sample_number

        selected_dataset = self.dataset_list[dataset_index]
        length = len(selected_dataset)
        selected_item = random.randint(0, length - 1)
        return selected_dataset[selected_item]

