import cv2
import torch
from PIL import Image
import numpy as np
from PIL import Image
import tempfile, os, io


def import_jpegio() -> None:
    try:
        import jpegio as jio
        return jio
    except ImportError:
        raise ImportError(
            'Please run "pip install jpegio" to install jpegio. This only support Linux system')

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

def convert_to_temp_jpeg(tensor):
    tensor = tensor.permute(1, 2, 0)
    img = Image.fromarray(tensor.numpy().astype('uint8'))
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        img.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name
    
    return temp_file_path

def read_jpeg_from_memory(tensor):
    jio = import_jpegio()
    temp_file_path = convert_to_temp_jpeg(tensor)
    jpeg = jio.read(temp_file_path)
    os.remove(temp_file_path)
    
    return jpeg


if __name__ == "__main__":
    # pass
    path = '/mnt/data0/public_datasets/IML/CASIA2.0/Tp/Tp_D_NRN_S_N_arc00090_art00010_00013.tif'
    print(read_jpeg_from_memory(path))