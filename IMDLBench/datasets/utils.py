import cv2
import torch
from PIL import Image


def pil_loader(path: str) -> Image.Image:
    """PIL image loader

    Args:
        path (str): image path

    Returns:
        Image.Image: PIL image (after np.array(x) becomes [0,255] int8)
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
    
def jpeg_loader(path : str):
    pass # TODO TBK

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """denormalize image with mean and std
    """
    image = image.clone().detach().cpu()
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    return image
