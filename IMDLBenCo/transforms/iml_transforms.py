import cv2
import random
import numpy as np
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2


class RandomCopyMove(DualTransform):
    def __init__(
        self,
        max_h=0.8,
        max_w=0.8,
        min_h=0.05,
        min_w=0.05,
        mask_value=255,
        always_apply=False,
        p=0.5,
    ):
        super(RandomCopyMove, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value

    @property
    def targets_as_params(self):
        # For albumentations 1.x compatibility.
        return ["image"]

    def _randint(self, low, high_exclusive):
        if high_exclusive <= low:
            return low

        # Albumentations 2.x-style RNG
        if hasattr(self, "py_random"):
            return self.py_random.randrange(low, high_exclusive)

        # Legacy fallback
        return random.randrange(low, high_exclusive)

    def _sample_window(self, img_height, img_width, window_height=None, window_width=None):
        assert 0 < self.max_h < 1
        assert 0 < self.max_w < 1
        assert 0 < self.min_h < 1
        assert 0 < self.min_w < 1

        l_min_h = max(1, int(img_height * self.min_h))
        l_min_w = max(1, int(img_width * self.min_w))
        l_max_h = max(l_min_h + 1, int(img_height * self.max_h))
        l_max_w = max(l_min_w + 1, int(img_width * self.max_w))

        if window_height is None or window_width is None:
            window_h = self._randint(l_min_h, l_max_h)
            window_w = self._randint(l_min_w, l_max_w)
        else:
            window_h = window_height
            window_w = window_width

        window_h = min(window_h, img_height)
        window_w = min(window_w, img_width)

        pos_h = self._randint(0, img_height - window_h + 1)
        pos_w = self._randint(0, img_width - window_w + 1)

        return pos_h, pos_w, window_h, window_w

    def _sample_params(self, image):
        H, W = image.shape[:2]

        c_pos_h, c_pos_w, window_h, window_w = self._sample_window(H, W)
        p_pos_h, p_pos_w, _, _ = self._sample_window(H, W, window_h, window_w)

        return {
            "c_pos_h": c_pos_h,
            "c_pos_w": c_pos_w,
            "p_pos_h": p_pos_h,
            "p_pos_w": p_pos_w,
            "window_h": window_h,
            "window_w": window_w,
        }

    # Albumentations 2.x
    def get_params_dependent_on_data(self, params, data):
        return self._sample_params(data["image"])

    # Albumentations 1.x
    def get_params_dependent_on_targets(self, params):
        return self._sample_params(params["image"])

    def apply(
        self,
        img,
        c_pos_h=0,
        c_pos_w=0,
        p_pos_h=0,
        p_pos_w=0,
        window_h=1,
        window_w=1,
        **params,
    ):
        image = img.copy()

        copy_region = image[
            c_pos_h : c_pos_h + window_h,
            c_pos_w : c_pos_w + window_w,
            ...
        ].copy()

        image[
            p_pos_h : p_pos_h + window_h,
            p_pos_w : p_pos_w + window_w,
            ...
        ] = copy_region

        return image

    def apply_to_mask(
        self,
        mask,
        p_pos_h=0,
        p_pos_w=0,
        window_h=1,
        window_w=1,
        **params,
    ):
        mask = mask.copy()
        mask[
            p_pos_h : p_pos_h + window_h,
            p_pos_w : p_pos_w + window_w,
        ] = self.mask_value
        return mask

    def get_transform_init_args_names(self):
        return ("max_h", "max_w", "min_h", "min_w", "mask_value", "always_apply", "p")


class RandomInpainting(DualTransform):
    def __init__(
        self,
        max_h=0.8,
        max_w=0.8,
        min_h=0.05,
        min_w=0.05,
        mask_value=255,
        always_apply=False,
        p=0.5,
    ):
        super(RandomInpainting, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value

    @property
    def targets_as_params(self):
        return ["image"]

    def _randint(self, low, high_exclusive):
        if high_exclusive <= low:
            return low

        if hasattr(self, "py_random"):
            return self.py_random.randrange(low, high_exclusive)

        return random.randrange(low, high_exclusive)

    def _sample_window(self, img_height, img_width):
        assert 0 < self.max_h < 1
        assert 0 < self.max_w < 1
        assert 0 < self.min_h < 1
        assert 0 < self.min_w < 1

        l_min_h = max(1, int(img_height * self.min_h))
        l_min_w = max(1, int(img_width * self.min_w))
        l_max_h = max(l_min_h + 1, int(img_height * self.max_h))
        l_max_w = max(l_min_w + 1, int(img_width * self.max_w))

        window_h = self._randint(l_min_h, l_max_h)
        window_w = self._randint(l_min_w, l_max_w)

        window_h = min(window_h, img_height)
        window_w = min(window_w, img_width)

        pos_h = self._randint(0, img_height - window_h + 1)
        pos_w = self._randint(0, img_width - window_w + 1)

        return pos_h, pos_w, window_h, window_w

    def _sample_params(self, image):
        H, W = image.shape[:2]
        pos_h, pos_w, window_h, window_w = self._sample_window(H, W)

        if hasattr(self, "py_random"):
            flag = cv2.INPAINT_TELEA if self.py_random.random() > 0.5 else cv2.INPAINT_NS
        else:
            flag = cv2.INPAINT_TELEA if random.random() > 0.5 else cv2.INPAINT_NS

        return {
            "pos_h": pos_h,
            "pos_w": pos_w,
            "window_h": window_h,
            "window_w": window_w,
            "inpaint_flag": flag,
        }

    # Albumentations 2.x
    def get_params_dependent_on_data(self, params, data):
        return self._sample_params(data["image"])

    # Albumentations 1.x
    def get_params_dependent_on_targets(self, params):
        return self._sample_params(params["image"])

    def apply(
        self,
        img,
        pos_h=0,
        pos_w=0,
        window_h=1,
        window_w=1,
        inpaint_flag=cv2.INPAINT_TELEA,
        **params,
    ):
        image = img.copy()
        image = np.uint8(image)

        H, W = image.shape[:2]
        inpaint_mask = np.zeros((H, W), dtype=np.uint8)
        inpaint_mask[
            pos_h : pos_h + window_h,
            pos_w : pos_w + window_w,
        ] = 1

        image = cv2.inpaint(image, inpaint_mask, 3, inpaint_flag)
        return image

    def apply_to_mask(
        self,
        mask,
        pos_h=0,
        pos_w=0,
        window_h=1,
        window_w=1,
        **params,
    ):
        mask = mask.copy()
        mask[
            pos_h : pos_h + window_h,
            pos_w : pos_w + window_w,
        ] = self.mask_value
        return mask

    def get_transform_init_args_names(self):
        return ("max_h", "max_w", "min_h", "min_w", "mask_value", "always_apply", "p")


def get_albu_transforms(type_='train', output_size=(1024, 1024)):
    """get albumentations transforms

        type_ (str):
            if 'train', then return train transforms with
                random scale, flip, rotate, brightness, contrast, and GaussianBlur augmentation.
            if 'test' then return test transforms
            if 'pad' then return zero-padding transforms
    """

    assert type_ in ['train', 'test', 'pad', 'resize'], "type_ must be 'train' or 'test' of 'pad' "
    trans = None
    if type_ == 'train':
        trans = albu.Compose([
            albu.RandomScale(scale_limit=0.2, p=1),
            RandomCopyMove(p=0.1, mask_value=255),
            RandomInpainting(p=0.1, mask_value=255),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=0.1,
                p=1
            ),
            albu.ImageCompression(
                quality_lower=70,
                quality_upper=100,
                p=0.2
            ),
            albu.RandomRotate90(p=0.5),
            albu.GaussianBlur(
                blur_limit=(3, 7),
                p=0.2
            ),
        ])

    if type_ == 'test':
        trans = albu.Compose([
        ])

    if type_ == 'pad':
        trans = albu.Compose([
            albu.PadIfNeeded(
                min_height=output_size[0],
                min_width=output_size[1],
                border_mode=0,
                value=0,
                position='top_left',
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