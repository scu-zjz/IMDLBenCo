# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import json
import os
from PIL import Image
import cv2
# Augmentation library

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from .utils import pil_loader, denormalize

from ..transforms import get_albu_transforms, EdgeMaskGenerator
from ..registry import DATASETS

@DATASETS.register_module()
class abstract_dataset(Dataset):
    def _init_dataset_path(self, path):
        tp_path = None # Tampered image
        gt_path = None # Ground truth
        
        raise NotImplementedError # abstract dataset!
    
        return tp_path, gt_path, labels
        
    def __init__(self, path, 
                is_padding = False,
                is_resizing = False,
                output_size = (1024, 1024),
                common_transforms = None, 
                edge_width = None, 
                ) -> None:
        super().__init__()
        self.tp_path, self.gt_path = self._init_dataset_path(path)
        
        if is_padding == True and is_resizing == True:
            raise AttributeError("is_padding and is_resizing can not be True at the same time")
        if is_padding == False and is_resizing == False:
            raise AttributeError("is_padding and is_resizing can not be False at the same time")

        # Padding or Resizing
        self.post_transform = None
        if is_padding == True:
            self.post_transform = get_albu_transforms(type_ = "pad", output_size = output_size)
        if is_resizing == True:
            self.post_transform = get_albu_transforms(type_ = "resize", output_size = output_size)
        
        # Common augmentations for augumentation
        self.common_transforms = common_transforms
        # Edge mask generator        
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)

        
    def __getitem__(self, index):

        data_dict = dict()
        
        tp_path = self.tp_path[index]
        gt_path = self.gt_path[index]
        
        tp_img = pil_loader(tp_path)
        tp_shape = tp_img.size
        
        # if "negative" then gt is a image with all 0
        if gt_path != "Negative":
            gt_img = pil_loader(gt_path)
            gt_shape = gt_img.size
            label = 1
        else:
            temp = np.array(tp_img)
            gt_img = np.zeros((temp.shape[0], temp.shape[1], 3))
            gt_shape = (temp.shape[1], temp.shape[0])
            label = 0
            
        assert tp_shape == gt_shape, "tp and gt image shape must be the same, but got {} and {}".format(tp_shape, gt_shape)
        
        tp_img = np.array(tp_img) # H W C
        gt_img = np.array(gt_img) # H W C
        
        # Do augmentations
        if self.common_transforms != None:
            res_dict = self.common_transforms(image = tp_img, mask = gt_img)
            tp_img = res_dict['image']
            gt_img = res_dict['mask']
        
        gt_img =  (np.mean(gt_img, axis = 2, keepdims = True)  > 127.5 ) * 1.0 # fuse the 3 channels to 1 channel, and make it binary(0 or 1)
        gt_img =  gt_img.transpose(2,0,1)[0] # H W C -> C H W -> H W
        masks_list = [gt_img]
        
        # if need to generate broaden edge mask
        if self.edge_mask_generator != None: 
            gt_img_edge = self.edge_mask_generator(gt_img)[0][0] # B C H W -> H W
            masks_list.append(gt_img_edge) # albumentation interface
        else:
            pass
            
        # Do post-transform (paddings or resizing)    
        res_dict = self.post_transform(image = tp_img, masks = masks_list)
        
        tp_img = res_dict['image']
        gt_img = res_dict['masks'][0].unsqueeze(0) # H W -> 1 H W \
            
        if self.edge_mask_generator != None:
            gt_img_edge = res_dict['masks'][1].unsqueeze(0) # H W -> 1 H W  
            data_dict['edge_mask'] = gt_img_edge

        # name of the image (mainly for testing)
        basename = os.path.basename(tp_path)
        
        data_dict['image'] = tp_img
        data_dict['mask'] = gt_img
        data_dict['label'] = label
        data_dict['shape'] = tp_shape
        data_dict['name'] = basename
        
        return data_dict
        
    def __len__(self):
        return len(self.tp_path)
    
@DATASETS.register_module()
class mani_dataset(abstract_dataset):
    def _init_dataset_path(self, path):
        path = path
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
class json_dataset(abstract_dataset):
    """ init from a json file, which contains all the images path
        file is organized as:
            [["./Tp/6.jpg", "./Gt/6.jpg"],
                ["./Tp/7.jpg", "./Gt/7.jpg"],
                ["./Tp/8.jpg", "Negative"],
                ......
            ]
        if path is "Neagative" then the image is negative sample, which means ground truths is a totally black image.
        
    Args:
        path (_type_): _description_
        transform_albu (_type_, optional): _description_. Defaults to None.
        mask_edge_generator (_type_, optional): _description_. Defaults to None.
        if_return_shape
    """
    def _init_dataset_path(self, path):
        images = json.load(open(path, 'r'))
        tp_list = []
        gt_list = []
        for record in images:
            tp_list.append(record[0])
            gt_list.append(record[1])
        return tp_list, gt_list

