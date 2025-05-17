

import numpy as np
from ..registry import DATASETS
from .abstract_dataset import AbstractDataset
    

@DATASETS.register_module()
class DummyDataset(AbstractDataset):
    def _init_dataset_path(self, path):
        """
        Dummy dataset for testing purposes.
        Useless, return None.
        """
        self.entry_path = path
        tp_list = ["dummy_tp_path"]
        gt_list = ["dummy_gt_path"]
        return tp_list, gt_list
    def _get_image(self, index):
        """
        Dummy dataset for testing purposes.
        return random genrated images and masks.
        shape is (H, W, C), value range from 0 to 255.
        """
        tp_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        gt_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        label = np.random.randint(0, 2)
        tp_shape = (512, 512)
        gt_shape = (512, 512)
        tp_path = "dummy_tp_path"
        gt_path = "dummy_gt_path"
        return tp_image, gt_image, label, tp_shape, gt_shape, tp_path, gt_path
