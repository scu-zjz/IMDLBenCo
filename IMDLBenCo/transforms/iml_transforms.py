import cv2
import random
import numpy as np
# Augmentation library
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2

class RandomCopyMove(DualTransform):
    def __init__(self,
        max_h = 0.8,
        max_w = 0.8,
        min_h = 0.05,
        min_w = 0.05,
        mask_value = 255,
        always_apply = False,
        p = 0.5,  
    ):
        """Apply cope-move manipulation to the image, and change the respective region on the mask to <mask_value>

        Args:
            max_h (float, optional): (0~1), max window height rate to the full height of image . Defaults to 0.5.
            max_w (float, optional): (0~1), max window width rate to the full width of image . Defaults to 0.5.
            min_h (float, optional): (0~1), min window height rate to the full height of image . Defaults to 0.05.
            min_w (float, optional): (0~1), min window width rate to the full width of image . Defaults to 0.05.
            mask_value (int, optional): the value apply the tampered region on the mask. Defaults to 255.
            always_apply (bool, optional): _description_. Defaults to False.
            p (float, optional): _description_. Defaults to 0.5.
        """
        super(RandomCopyMove, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
        
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
        window_height = None, 
        window_width = None
    ):
        assert self.max_h < 1 and self.max_h > 0 
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0
        
        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)
        
        if window_width == None or window_height == None:
            window_h = np.random.randint(l_min_h, l_max_h)
            window_w = np.random.randint(l_min_w, l_max_w)
        else:
            window_h = window_height
            window_w = window_width

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
        
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        image = img.copy()
        H, W, _ = image.shape
        # copy region:
        c_pos_h, c_pos_w, c_window_h, c_window_w = self._get_random_window(H, W)
        
        # past region, window size is defined by copy region:
        self.p_pos_h, self.p_pos_w, self.p_window_h, self.p_window_w = self._get_random_window(H, W, c_window_h, c_window_w)
        
        copy_region = image[
            c_pos_h: c_pos_h + c_window_h, 
            c_pos_w: c_pos_w + c_window_w, 
            : 
        ]
        image[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
            :
        ] = copy_region
        return image
        

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
    
        manipulated_region =  np.full((self.p_window_h, self.p_window_w), 1)
        img = img.copy()
        img[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
        ] = self.mask_value
        return img
    # must be implemented for string output when print(RandomCopyMove())
    def get_transform_init_args_names(self):
        return ("max_h", "max_w", "min_h", "min_w", "mask_value", "always_apply", "p")
        
class RandomInpainting(DualTransform):
    def __init__(self,
        max_h = 0.8,
        max_w = 0.8,
        min_h = 0.05,
        min_w = 0.05,
        mask_value = 255,
        always_apply = False,
        p = 0.5,  
    ):
        super(RandomInpainting, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
    ):
        assert self.max_h < 1 and self.max_h > 0 
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0
        
        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)
    
        window_h = np.random.randint(l_min_h, l_max_h)
        window_w = np.random.randint(l_min_w, l_max_w)

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        img = np.uint8(img)
        H, W, C = img.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        # inpainting region
        self.pos_h, self.pos_w , self.window_h, self.window_w = self._get_random_window(H, W)
        mask[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = 1
        inpaint_flag = cv2.INPAINT_TELEA if random.random() > 0.5 else cv2.INPAINT_NS
        img = cv2.inpaint(img, mask, 3,inpaint_flag)
        return img
    
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
        img = img.copy()
        img[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = self.mask_value
        return img
    
    # must be implemented for string output when print(RandomInpainting())
    def get_transform_init_args_names(self):
        return ("max_h", "max_w", "min_h", "min_w", "mask_value", "always_apply", "p")

def get_albu_transforms(type_ = 'train', output_size = (1024, 1024)):
    """get albumentations transforms
        
        type_ (str): 
            if 'train', then return train transforms with
                random scale, flip, rotate, brightness, contrast, and GaussianBlur augmentation.
            if 'test' then return test transforms 
            if 'pad' then return zero-padding transforms
    """
    
    assert type_ in ['train', 'test', 'pad', 'resize'] , "type_ must be 'train' or 'test' of 'pad' "
    trans = None
    if type_ == 'train':
        trans = albu.Compose([
            # Rescale the input image by a random factor between 0.8 and 1.2
            albu.RandomScale(scale_limit=0.2, p=1), 
            RandomCopyMove(p = 0.1),
            RandomInpainting(p = 0.1),
            # Flips
            # albu.Resize(512, 512),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            # Brightness and contrast fluctuation
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=0.1,
                p=1
            ),
            albu.ImageCompression(
                quality_lower = 70,
                quality_upper = 100,
                p = 0.2
            ),
            # Rotate
            albu.RandomRotate90(p=0.5),
            # Blur
            albu.GaussianBlur(
                blur_limit = (3, 7),
                p = 0.2
            ),
        ])
    
    if type_ == 'test':
        trans = None
        trans = albu.Compose([
        # ---Blow for robustness evalution---
        # albu.Resize(512, 512),
        #   albu.JpegCompression(
        #         quality_lower = 89,
        #         quality_upper = 90,
        #         p = 1
        #   ),
        #  albu.GaussianBlur(
        #         blur_limit = (5, 5),
        #         p = 1
        #     ),
        
        # albu.GaussNoise(
        #     var_limit=(15, 15),
        #     p = 1
        # )
        ])
        
    if type_ == 'pad':
        trans = albu.Compose([
            albu.PadIfNeeded(          
                min_height=output_size[0],
                min_width=output_size[1], 
                border_mode=0, 
                value=0, 
                position= 'top_left',
                mask_value=0),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            albu.Crop(0, 0, output_size[0], output_size[1]),
            ToTensorV2(transpose_mask=True)
        ])
    if type_ == 'resize':
        trans = albu.Compose([
            albu.Resize(output_size[0], output_size[1]),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            albu.Crop(0, 0, output_size[0], output_size[1]),
            ToTensorV2(transpose_mask=True)
        ])

    return trans