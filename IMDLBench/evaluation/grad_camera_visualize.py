import torch
import torch.nn as nn
from typing import Union, Tuple
from torch.utils.data import Dataset
from pytorch_grad_cam import GradCAM
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from ..datasets.utils import denormalize
import os
from PIL import Image
from tqdm import tqdm


class OutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(OutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)["mask_pred"]


class SemanticSegmentationTarget:
    def __init__(self, mask):
        self.mask = torch.squeeze(mask, 0)
        
    def __call__(self, model_output):
        return (model_output * self.mask).sum()


def grad_camera_visualize(model: nn.Module,
                        image: Union[Tuple[str, str], Dataset],
                        target_layers: list,
                        output_path: str) -> None:
    """
    visualize
    """
    if not isinstance(image, Dataset):
        # TODO
        pass 
    sampler = torch.utils.data.SequentialSampler(image)
    data_loader = torch.utils.data.DataLoader(
        image, 
        sampler=sampler,
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        drop_last=False,
    )
    model = OutputWrapper(model=model)
    for data in tqdm(data_loader):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cuda()
        input_tensor = data['image']
        cam = GradCAM(model=model, target_layers=target_layers)
        rgb_img = np.float32(torch.squeeze(denormalize(input_tensor), 0).permute(1, 2, 0).cpu())
        targets = [SemanticSegmentationTarget(data['mask'])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        Image.fromarray(cam_image).save(os.path.join(output_path, data['name'][0].split('.')[0] + '.jpg'))