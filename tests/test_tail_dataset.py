import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# 初始化分布式环境
dist.init_process_group(backend="nccl", init_method="env://")

# 获取当前进程的 rank 和总进程数
rank = dist.get_rank()
world_size = dist.get_world_size()

# 定义一个简单的 Dataset
class SmallDataset(Dataset):
    def __init__(self):
        # 数据集只有 3 个样本
        self.data = [1, 2, 3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 创建 Dataset 和 DistributedSampler
dataset = SmallDataset()
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

# 创建 DataLoader
batch_size = 5  # batch size 大于数据集长度
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
"""
重要！
当dataloader的drop_last=True时，且等效batchsize（num_tasks * batch_size）大于数据集长度时。
dataloader将无法迭代，返回不了任何东西。len(dataloader) = 0
"""

# 测试 DataLoader 的迭代
print(f"Rank {rank} iterating through DataLoader:")
for batch in dataloader:
    print(len(dataloader))
    print(f"Rank {rank} batch: {batch}")

# 清理分布式环境
dist.destroy_process_group()