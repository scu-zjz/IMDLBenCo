import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class DummyImageMaskDataset(Dataset):
    def __init__(self):
        # 假设每张图像是 3x32x32，mask 是 1x32x32
        self.images = [torch.rand(3, 32, 32) for _ in range(12)]
        self.masks = [torch.randint(0, 2, (1, 32, 32)) for _ in range(12)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

@pytest.fixture(scope="session")
def dummy_dataset():
    return DummyImageMaskDataset()

class DummyDataset(Dataset):
    def __init__(self):
        self.data = list(range(12))  # 100 个样本

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor([self.data[idx]], dtype=torch.float32)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo(rank, world_size):
    print(f"Rank {rank} starting.")
    setup(rank, world_size)

    # 构造 dataset 和 distributed sampler
    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    total = torch.tensor([0.0])

    # 迭代数据并求和
    for batch in dataloader:
        print(f"Rank {rank} batch: {batch}")
        total += batch.sum()

    print(f"Rank {rank} local sum before reduce: {total.item()}")

    # 分布式 reduce：将所有 rank 的 total 相加，放到 rank=0
    dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"Rank {rank} total sum after reduce: {total.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(demo, args=(world_size,), nprocs=world_size, join=True)
