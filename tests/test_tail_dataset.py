import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def distributed_gather_tensor(tensor: torch.Tensor, total_size: int = None):
    """
    Gather tensors from all processes. Optionally trim to `total_size` to remove padding from the last batch.
    Works only for 1D or 2D tensors (along dim=0).

    Args:
        tensor (torch.Tensor): Local tensor on each process
        total_size (int, optional): Total valid size after gathering. Used to remove padding.

    Returns:
        gathered_tensor (torch.Tensor): Full gathered tensor on all processes.
    """
    world_size = dist.get_world_size()
    tensor= tensor.to(device)
    # Create empty list to hold gathered tensors
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    gathered = torch.cat(gather_list, dim=0)

    if total_size is not None:
        gathered = gathered[:total_size]

    return gathered
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)  # 这一步是关键！！
device = torch.device("cuda", local_rank)
# 初始化分布式环境
dist.init_process_group(backend="nccl", init_method="env://")

# 获取当前进程的 rank 和总进程数
rank = dist.get_rank()
world_size = dist.get_world_size()

# 定义一个简单的 Dataset
class SmallDataset(Dataset):
    def __init__(self):
        # 数据集只有 3 个样本
        self.data = [torch.tensor(i).to(torch.float32) for i in range(9)]
        print(self.data[0])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 创建 Dataset 和 DistributedSampler
dataset = SmallDataset()
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

# 创建 DataLoader
batch_size = 5  # batch size 大于数据集长度
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False, shuffle=False)
"""
重要！
当dataloader的drop_last=True时，且等效batchsize（num_tasks * batch_size）大于数据集长度时。
dataloader将无法迭代，返回不了任何东西。len(dataloader) = 0
"""
res = []
# 测试 DataLoader 的迭代
print(f"Rank {rank} iterating through DataLoader:")
for batch in dataloader:
    batch.to(device)
    print(len(dataloader))
    print(f"Rank {rank} batch: {batch}")
    res.append(batch)
print(f"Rank {rank} result: {res}")

local_pred = torch.cat(res, dim=0)

gatherdata = distributed_gather_tensor(local_pred)
print(f"Rank {rank} gatherdata: {gatherdata}")

# check duplicate
if rank == 0:
    print(f"Rank {rank} gatherdata: {gatherdata}")
    print(f"Rank {rank} gatherdata size: {gatherdata.size()}")
    print(f"Rank {rank} gatherdata shape: {gatherdata.shape}")
    print(f"Rank {rank} gatherdata dtype: {gatherdata.dtype}")


# 清理分布式环境
dist.destroy_process_group()