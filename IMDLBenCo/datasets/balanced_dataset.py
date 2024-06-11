import random
from torch.utils.data import Dataset, DataLoader

from .iml_datasets import JsonDataset, ManiDataset

from ..transforms import get_albu_transforms
from .utils import pil_loader, denormalize

from IMDLBenCo.registry import DATASETS
@DATASETS.register_module()
class BalancedDataset(Dataset):
    """The BalancedDataset manages multiple iml_datasets, so it does not inherit from AbstractDataset.

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, 
                 path = None, 
                 sample_number = 1840,
                 *args, 
                 **kwargs
                ) -> None:
        self.sample_number = sample_number
        if path == None:
            # Defalut
            self.settings_list = [
                ['/mnt/data0/public_datasets/IML/CASIA2.0', ManiDataset],
                ['/mnt/data0/public_datasets/IML/FantasticReality_v1/FantasticReality.json', JsonDataset],
                ['/mnt/data0/public_datasets/IML/IMD_20_1024', ManiDataset],
                ['/mnt/data0/public_datasets/IML/tampCOCO/sp_COCO_list.json', JsonDataset],
                ['/mnt/data0/public_datasets/IML/tampCOCO/cm_COCO_list.json', JsonDataset],
                ['/mnt/data0/public_datasets/IML/tampCOCO/bcm_COCO_list.json', JsonDataset],
                ['/mnt/data0/public_datasets/IML/tampCOCO/bcmc_COCO_list.json', JsonDataset]
            ]
        else:
            import json
            with open(path, "r") as f:
                setting_json = json.load(f)
            # self.settings_list = path_list
            self.settings_list = []
            for dataset_str,  dataset_path in setting_json:
                self.settings_list.append(
                    [
                        dataset_path,
                        DATASETS.get(dataset_str)
                    ]
                )
            

        self.dataset_list = [self._get_dataset(path, dataset_type, *args, **kwargs) for path, dataset_type in self.settings_list]
        
        
    def _get_dataset(self, path, dataset_type, *args, **kwargs):
        return dataset_type(path, *args, **kwargs)
        
        
    def __len__(self):
        return self.sample_number * len(self.settings_list)    
    
    def __getitem__(self, index):
        dataset_index = index // self.sample_number

        selected_dataset = self.dataset_list[dataset_index]
        length = len(selected_dataset)
        selected_item = random.randint(0, length - 1)
        return selected_dataset[selected_item]

