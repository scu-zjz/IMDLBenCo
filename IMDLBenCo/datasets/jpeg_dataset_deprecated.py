import json
import os

from ..registry import DATASETS
from .abstract_dataset import AbstractDataset
from torch.utils.data import Dataset, DataLoader

from ..transforms import get_albu_transforms
import random
import numpy as np
from PIL import Image
import torch
from IMDLBenCo.datasets.utils import read_jpeg_from_memory

train_transform = get_albu_transforms('train')

def get_dataset(path, dataset_type):
    return dataset_type(path, 1024, train_transform, edge_width=7, if_return_shape=True)

@DATASETS.register_module()
class CatnetDataset(Dataset):
    """The BalancedDataset manages multiple iml_datasets, so it does not inherit from AbstractDataset.

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, sample_number = 1840) -> None:
        self.sample_number = sample_number
        self.settings_list = [
            ['/mnt/data0/public_datasets/IML/CASIA2.0', MetaCatnetDataset],
            # ['/mnt/data0/public_datasets/IML/Fantastic_Reality_1024', MetaCatnetDataset], # TODO
            ['/mnt/data0/public_datasets/IML/IMD_20_1024', MetaCatnetDataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/sp_COCO_list.json', MetaCatnetDataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/cm_COCO_list.json', MetaCatnetDataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/bcm_COCO_list.json', MetaCatnetDataset],
            ['/mnt/data0/public_datasets/IML/tampCOCO/bcmc_COCO_list.json', MetaCatnetDataset]
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

    
@DATASETS.register_module()
class MetaCatnetDataset(AbstractDataset):
    def _init_dataset_path(self, path):
        if path.endswith('.json'):
            tp_list, gt_list = self.__distract_dataset_json_path(path=path)
        elif os.path.isdir(path):
            tp_list, gt_list = self.__distract_dataset_folder_path(path=path)
        return tp_list, gt_list
    
    def __getitem__(self, index):
        data_dict = dict()
        
        tp_path = self.tp_path[index]
        gt_path = self.gt_path[index]
        
        # pil_loader or jpeg_loader
        tp_img = self.img_loader(tp_path)
        tp_shape = tp_img.size
        
        # if "negative" then gt is a image with all 0
        if gt_path != "Negative":
            gt_img = self.img_loader(gt_path)
            gt_shape = gt_img.size
            label = 1
        else:
            temp = np.array(tp_img)
            gt_img = np.zeros((temp.shape[0], temp.shape[1], 3))
            gt_shape = (temp.shape[1], temp.shape[0]) #FIXME: check the shape
            label = 0
            
        assert tp_shape == gt_shape, "tp and gt image shape must be the same, but got {} and {}".format(tp_shape, gt_shape)
        
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

        images, label, qtables = self.__post_process_tensor(tp_img, gt_img)
        
        data_dict['images'] = images
        data_dict['label'] = label
        data_dict['name'] = basename
        data_dict['qtables'] = qtables

        return data_dict
    
    def __distract_dataset_json_path(self, path):
        images = json.load(open(path, 'r'))
        tp_list = []
        gt_list = []
        for record in images:
            tp_list.append(record[0])
            gt_list.append(record[1])
        return tp_list, gt_list
    
    def __distract_dataset_folder_path(self, path):
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

    def __post_process_tensor(self, image_tensor, mask):
        # just shit code
        self._blocks = ('RGB', 'DCTvol', 'qtable')
        self._crop_size = (512, 512)
        self.DCT_channels = 1
        self._grid_crop = True
        ignore_index = -1
        img_RGB = torch.permute(image_tensor, (1, 2, 0)).numpy()
        mask = torch.squeeze(mask, 0)

        h, w = img_RGB.shape[0], img_RGB.shape[1]

        if 'DCTcoef' in self._blocks or 'DCTvol' in self._blocks or 'rawRGB' in self._blocks or 'qtable' in self._blocks:
            DCT_coef, qtables = self.__get_jpeg_info(image_tensor)

        if mask is None:
            mask = np.zeros((h, w))

        if self._crop_size is None and self._grid_crop:
            crop_size = (-(-h//8) * 8, -(-w//8) * 8)  # smallest 8x8 grid crop that contains image
        elif self._crop_size is None and not self._grid_crop:
            crop_size = None  # use entire image! no crop, no pad, no DCTcoef or rawRGB
        else:
            crop_size = self._crop_size

        if crop_size is not None:
            # Pad if crop_size is larger than image size
            if h < crop_size[0] or w < crop_size[1]:
                # pad img_RGB
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 127.5)
                temp[:img_RGB.shape[0], :img_RGB.shape[1], :] = img_RGB
                img_RGB = temp

                # pad mask
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1])), ignore_index)  # pad with ignore_index(-1)
                temp[:mask.shape[0], :mask.shape[1]] = mask
                mask = temp

                # pad DCT_coef
                if 'DCTcoef' in self._blocks or 'DCTvol' in self._blocks or 'rawRGB' in self._blocks:
                    max_h = max(crop_size[0], max([DCT_coef[c].shape[0] for c in range(self.DCT_channels)]))
                    max_w = max(crop_size[1], max([DCT_coef[c].shape[1] for c in range(self.DCT_channels)]))
                    for i in range(self.DCT_channels):
                        temp = np.full((max_h, max_w), 0.0)  # pad with 0
                        temp[:DCT_coef[i].shape[0], :DCT_coef[i].shape[1]] = DCT_coef[i][:, :]
                        DCT_coef[i] = temp

            # Determine where to crop
            if self._grid_crop:
                s_r = (random.randint(0, max(h - crop_size[0], 0)) // 8) * 8
                s_c = (random.randint(0, max(w - crop_size[1], 0)) // 8) * 8
            else:
                s_r = random.randint(0, max(h - crop_size[0], 0))
                s_c = random.randint(0, max(w - crop_size[1], 0))

            # crop img_RGB
            img_RGB = img_RGB[s_r:s_r+crop_size[0], s_c:s_c+crop_size[1], :]

            # crop mask
            mask = mask[s_r:s_r + crop_size[0], s_c:s_c + crop_size[1]]

            # crop DCT_coef
            if 'DCTcoef' in self._blocks or 'DCTvol' in self._blocks or 'rawRGB' in self._blocks:
                for i in range(self.DCT_channels):
                    DCT_coef[i] = DCT_coef[i][s_r:s_r+crop_size[0], s_c:s_c+crop_size[1]]
                t_DCT_coef = torch.tensor(DCT_coef, dtype=torch.float)  # final (but used below)

        # handle 'RGB'
        if 'RGB' in self._blocks:
            t_RGB = (torch.tensor(img_RGB.transpose(2,0,1), dtype=torch.float)-127.5)/127.5  # final

        # handle 'DCTvol'
        if 'DCTvol' in self._blocks:
            T = 20
            t_DCT_vol = torch.zeros(size=(T+1, t_DCT_coef.shape[1], t_DCT_coef.shape[2]))
            t_DCT_vol[0] += (t_DCT_coef == 0).float().squeeze()
            for i in range(1, T):
                t_DCT_vol[i] += (t_DCT_coef == i).float().squeeze()
                t_DCT_vol[i] += (t_DCT_coef == -i).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef >= T).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef <= -T).float().squeeze()

        # create tensor
        img_block = []
        for i in range(len(self._blocks)):
            if self._blocks[i] == 'RGB':
                img_block.append(t_RGB)
            elif self._blocks[i] == 'DCTcoef':
                img_block.append(t_DCT_coef)
            elif self._blocks[i] == 'DCTvol':
                img_block.append(t_DCT_vol)
            elif self._blocks[i] == 'qtable':
                continue
            else:
                raise KeyError("We cannot reach here. Something is wrong.")

        # final tensor
        tensor = torch.cat(img_block)

        if 'qtable' not in self._blocks:
            return tensor, torch.tensor(mask, dtype=torch.long), 0
        else:
            # print(tensor.shape, torch.tensor(mask, dtype=torch.long).shape, torch.tensor(qtables[:self.DCT_channels], dtype=torch.float).shape)
            return tensor, torch.tensor(mask, dtype=torch.long), torch.tensor(qtables[:self.DCT_channels], dtype=torch.float)

    def __get_jpeg_info(self, image_tensor):
        """
        :param im_path: JPEG image path
        :return: DCT_coef (Y,Cb,Cr), qtables (Y,Cb,Cr)
        """
        num_channels = 1
        jpeg = read_jpeg_from_memory(image_tensor)

        # determine which axes to up-sample
        ci = jpeg.comp_info
        need_scale = [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
        if num_channels == 3:
            if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
                need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
            if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
                need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
        else:
            need_scale[0][0] = 2
            need_scale[0][1] = 2

        # up-sample DCT coefficients to match image size
        DCT_coef = []
        for i in range(num_channels):
            r, c = jpeg.coef_arrays[i].shape
            coef_view = jpeg.coef_arrays[i].reshape(r//8, 8, c//8, 8).transpose(0, 2, 1, 3)
            # case 1: row scale (O) and col scale (O)
            if need_scale[i][0]==1 and need_scale[i][1]==1:
                out_arr = np.zeros((r * 2, c * 2))
                out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

            # case 2: row scale (O) and col scale (X)
            elif need_scale[i][0]==1 and need_scale[i][1]==2:
                out_arr = np.zeros((r * 2, c))
                DCT_coef.append(out_arr)
                out_view = out_arr.reshape(r*2//8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, :, :, :] = coef_view[:, :, :, :]
                out_view[1::2, :, :, :] = coef_view[:, :, :, :]

            # case 3: row scale (X) and col scale (O)
            elif need_scale[i][0]==2 and need_scale[i][1]==1:
                out_arr = np.zeros((r, c * 2))
                out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, ::2, :, :] = coef_view[:, :, :, :]
                out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

            # case 4: row scale (X) and col scale (X)
            elif need_scale[i][0]==2 and need_scale[i][1]==2:
                out_arr = np.zeros((r, c))
                out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, :, :, :] = coef_view[:, :, :, :]

            else:
                raise KeyError("Something wrong here.")

            DCT_coef.append(out_arr)

        # quantization tables
        qtables = [jpeg.quant_tables[ci[i].quant_tbl_no].astype(np.float64) for i in range(num_channels)]

        return DCT_coef, qtables

if __name__ == '__main__':
    pass

    exit(0)
    
    """ backup codes
    class JPEGDataset(JsonDataset):
        def __init__(self, path, 
                    is_padding=False,
                    is_resizing=False,
                    output_size=(1024, 1024),
                    common_transforms=None, 
                    edge_width=None) -> None:
            super().__init__(path, 
                            is_padding=is_padding,
                            is_resizing=is_resizing,
                            output_size=output_size,
                            common_transforms=common_transforms, 
                            edge_width=edge_width,
                            img_loader=jpeg_loader)
    """