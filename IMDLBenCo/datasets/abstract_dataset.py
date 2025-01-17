import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from .utils import pil_loader, denormalize

from ..transforms import get_albu_transforms, EdgeMaskGenerator

from IMDLBenCo.registry import DATASETS

@DATASETS.register_module()
class AbstractDataset(Dataset):
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
                img_loader = pil_loader,
                post_funcs = None
                ) -> None:
        super().__init__()
        self.entry_path = "Abstract"
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
        self.is_padding = is_padding
        self.is_resizing = is_resizing
        
        self.output_size = output_size
        
        # Common augmentations for augumentation
        self.common_transforms = common_transforms
        # Edge mask generator        
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)

        self.img_loader = img_loader
        self.post_funcs = post_funcs

        
    def __getitem__(self, index):

        data_dict = dict()
        
        tp_path = self.tp_path[index]
        gt_path = self.gt_path[index]
        
        # pil_loader or jpeg_loader
        tp_img = self.img_loader(tp_path)
        # shape, here is PIL Image
        tp_shape = tp_img.size
        
        # if "negative" then gt is a image with all 0
        if gt_path != "Negative":
            gt_img = self.img_loader(gt_path)
            gt_shape = gt_img.size
            label = 1
        else:
            temp = np.array(tp_img)
            gt_img = np.zeros((temp.shape[0], temp.shape[1], 3))
            gt_shape = (temp.shape[1], temp.shape[0])
            label = 0
            
        assert tp_shape == gt_shape, "tp and gt image shape must be the same, but got shape {} and {} for image '{}' and '{}'. Please check it!".format(tp_shape, gt_shape, tp_path, gt_path)
        
        tp_img = np.array(tp_img) # H W C
        gt_img = np.array(gt_img) # H W C
        
        # Do augmentations
        if self.common_transforms != None:
            res_dict = self.common_transforms(image = tp_img, mask = gt_img)
            tp_img = res_dict['image']
            gt_img = res_dict['mask']
            # copy_move may cause the label change, so we need to update the label
            if np.all(gt_img == 0):
                label = 0
            else:
                label = 1
            
        # redefine the shape, here is np.array
        tp_shape = tp_img.shape[0:2]  # H, W, 3 去掉最后一个3
        
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

            # =========output=====================
            data_dict['edge_mask'] = gt_img_edge
            # ====================================
        # name of the image (mainly for testing)
        basename = os.path.basename(tp_path)

        # =========output=====================
        data_dict['image'] = tp_img
        data_dict['mask'] = gt_img
        data_dict['label'] = label
       
        # 如果经过resize
        if self.is_resizing:
            tp_shape = self.output_size
            
        # 这里如果是（256， 384） 那么对应的图像是一个横着的 长的方块
        data_dict['shape'] = torch.tensor(tp_shape) # (H, W) 经过data loader后会变成三维矩阵，第0维是batch_index
        data_dict['name'] = basename
        
        # 如果padding则需要单独return一个shape_mask
        if self.is_padding:
            shape_mask = torch.zeros_like(gt_img)
            shape_mask[:, :tp_shape[0], :tp_shape[1]] = 1
            data_dict['shape_mask'] = shape_mask
        # ====================================
        # Post processing with callback functions on data_dict
        if self.post_funcs == None:
            pass    # Do nothing
        elif isinstance(self.post_funcs, list):
            # 如果是列表，循环调用列表中的每个回调函数
            for func in self.post_funcs:
                if callable(func):
                    func(data_dict)
                else:
                    raise NotImplementedError(f"Element {func} in list is not callable")
        elif callable(self.post_funcs):
            # 如果是单个回调函数，直接调用
            self.post_funcs(data_dict)
        else:
            # 其他类型抛出 NotImplementedError
            raise NotImplementedError(f"Unsupported type: {type(self.post_funcs)}")
        # ====================================

        return data_dict
        
    def __len__(self):
        return len(self.tp_path)
    
    def __str__(self):
        cls_name = self.__class__.__name__
        cls_path = self.entry_path
        cls_len = len(self.tp_path)
        info = f"[{cls_name}] at {cls_path}, with length of {cls_len:,}"
        return info
