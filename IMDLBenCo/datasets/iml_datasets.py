import json
import os


from ..registry import DATASETS
from .abstract_dataset import AbstractDataset
    
@DATASETS.register_module()
class ManiDataset(AbstractDataset):
    def _init_dataset_path(self, path):
        self.entry_path = path
        tp_dir = os.path.join(path, 'Tp')
        gt_dir = os.path.join(path, 'Gt')
        tp_list = os.listdir(tp_dir)
        gt_list = os.listdir(gt_dir)
        # Use sort mathod to keep order, to make sure the order is the same as the order in the tp_list and gt_list
        tp_list.sort()
        gt_list.sort()
        t_tp_list = [os.path.join(path, 'Tp', tp_list[index]) for index in range(len(tp_list))]
        t_gt_list = [os.path.join(path, 'Gt', gt_list[index]) for index in range(len(gt_list))]
        return t_tp_list, t_gt_list
    
@DATASETS.register_module()
class JsonDataset(AbstractDataset):
    """ init from a json file, which contains all the images path
        file is organized as:
            [
                ["./Tp/6.jpg", "./Gt/6.jpg"],
                ["./Tp/7.jpg", "./Gt/7.jpg"],
                ["./Tp/8.jpg", "Negative"],
                ......
            ]
        if path is "Neagative" then the image is negative sample, which means ground truths is a totally black image, and its label == 0.
        
    Args:
        path (_type_): _description_
        transform_albu (_type_, optional): _description_. Defaults to None.
        mask_edge_generator (_type_, optional): _description_. Defaults to None.
        if_return_shape
    """
    def _init_dataset_path(self, path):
        self.entry_path = path
        images = json.load(open(path, 'r'))
        tp_list = []
        gt_list = []
        for record in images:
            if os.path.isfile(record[0]):
                tp_list.append(record[0])
                gt_list.append(record[1])
            else: 
                raise TypeError("You have to pass a Json File to JsonDataset. Or try ManiDataset with a path. For more information please see the protocol here: https://scu-zjz.github.io/IMDLBenCo-doc/guide/quickstart/0_dataprepare.html#specific-format-definitions")
        return tp_list, gt_list

