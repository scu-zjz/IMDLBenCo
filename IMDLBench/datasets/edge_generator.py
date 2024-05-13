import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

class EdgeGenerator(torch.nn.Module):
    """generate the 'edge bar' for a 0-1 mask Groundtruth of a image
    Algorithm is based on 'Morphological Dilation and Difference Reduction'
    
    Which implemented with fixed-weight Convolution layer with weight matrix looks like a cross,
    for example, if kernel size is 3, the weight matrix is:
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]

    """
    def __init__(self, kernel_size = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    def _dilate(self, image, kernel_size=3):
        """Doings dilation on the image

        Args:
            image (_type_): 0-1 tensor in shape (B, C, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert image.shape[2] > kernel_size and image.shape[3] > kernel_size, "Image must be larger than kernel size"
        
        kernel = torch.zeros((1, 1, kernel_size, kernel_size))
        kernel[0, 0, kernel_size // 2: kernel_size//2+1, :] = 1
        kernel[0, 0, :,  kernel_size // 2: kernel_size//2+1] = 1
        kernel = kernel.float()
        # print(kernel)
        res = F.conv2d(image, kernel.view([1,1,kernel_size, kernel_size]),stride=1, padding = kernel_size // 2)
        return (res > 0) * 1.0


    def _find_edge(self, image, kernel_size=3, return_all=False):
        """Find 0-1 edges of the image

        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        image = torch.tensor(image).float()
        shape = image.shape
        
        if len(shape) == 2:
            image = image.reshape([1, 1, shape[0], shape[1]])
        if len(shape) == 3:
            image = image.reshape([1, shape[0], shape[1], shape[2]])   
        assert image.shape[1] == 1, "Image must be single channel"
        
        img = self._dilate(image, kernel_size=kernel_size)
        
        erosion = self._dilate(1-image, kernel_size=kernel_size)

        diff = -torch.abs(erosion - img) + 1
        diff = (diff > 0) * 1.0
        # res = dilate(diff)
        diff = diff.numpy()
        if return_all :
            return diff, img, erosion
        else:
            return diff
    
    def forward(self, x, return_all=False):
        """
        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        return self._find_edge(x, self.kernel_size, return_all=return_all)
    
    
"""Codes below are for testing"""
if __name__ == '__main__':
    lists = ['NC2016_1504.jpg', '519_mask.jpg', '526_mask.jpg', '528_mask.jpg', '534_mask.jpg']

    for i in lists:
        img = plt.imread(f'./components/Edge_generator/{i}')
        img = torch.tensor(img)
        print(img)
        img = (img > 127).float()
        plt.subplot(1, 4, 1)
        plt.imshow(img, cmap='gray')
        print(img)
        Edge = EdgeGenerator(kernel_size=11)
        
        raw_img = img.view(1, 1, img.shape[0], img.shape[1])


        # print(img)
        # plt.subplot(1,4, 2)
        # plt.imshow(diff.detach().numpy()[0, 0, :, :], cmap='gray')

        diff, img,erosion, = Edge(raw_img, return_all=True)
        
        plt.subplot(1,4, 2)
        plt.imshow(diff[0, 0, :, :], cmap='gray')
        
        # plt.subplot(1,4, 3)
        # plt.imshow(img.detach().numpy()[0, 0, :, :], cmap='gray')
        
        # plt.subplot(1, 4, 4)
        # plt.imshow(erosion.detach().numpy()[0, 0, :, :], cmap='gray')
        
        
        print(diff.shape)
        print(img.shape)
        print(erosion.shape)
        plt.show()