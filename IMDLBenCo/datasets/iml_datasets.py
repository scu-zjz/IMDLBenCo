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
        
        assert os.path.isdir(path), NotADirectoryError(f"Get Error when loading from {self.entry_path}, the path is not a directory. Please check the path.")
        assert os.path.isdir(tp_dir), NotADirectoryError(f"Get Error when loading from {tp_dir}, the Tp directory is not exist. Please check the path.")
        assert os.path.isdir(gt_dir), NotADirectoryError(f"Get Error when loading from {gt_dir}, the Gt directory is not exist. Please check the path.")
        
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
        try:
            images = json.load(open(path, 'r'))
        except:
            raise TypeError(f"Get Error when loading from {self.entry_path}, please check the file format, it should be a json file, and the content should be like: [['./Tp/6.jpg', './Gt/6.jpg'], ['./Tp/7.jpg', './Gt/7.jpg'], ['./Tp/8.jpg', 'Negative'], ......]")
        tp_list = []
        gt_list = []
        for record in images:
            if os.path.isfile(record[0]):
                tp_list.append(record[0])
                gt_list.append(record[1])
            else: 
                raise TypeError(f"Get Error when loading from {self.entry_path}, the error record is: {record[0]}, which is not a file. Please check this file or try to use absolute path instead. Otherwise if you want to use ManiDataset with a path instead of JsonDataset, please pass a path into the 'train_*.sh'. For more information please see the protocol here: https://scu-zjz.github.io/IMDLBenCo-doc/guide/quickstart/0_dataprepare.html#specific-format-definitions")
        return tp_list, gt_list

