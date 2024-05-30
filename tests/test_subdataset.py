import torch
from torch.utils.data import DataLoader, Dataset

# 创建一个简单的自定义Dataset
class MyDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 实例化Dataset
dataset = MyDataset()

# 用DataLoader加载Dataset
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(len(dataloader.dataset))
# 获取DataLoader的Dataset引用
dataset_from_dataloader = dataloader.dataset

# 验证引用是否相同
print(f"Original Dataset ID: {id(dataset)}")
print(f"Dataset from DataLoader ID: {id(dataset_from_dataloader)}")
print(f"Is same object: {dataset is dataset_from_dataloader}")