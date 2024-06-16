import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from collections import Counter

""" Expected output:
------------------------------
Sampler drop_last=True, DataLoader drop_last=True,
 Sampler shuffle=True, DataLoader shuffle=default
Missing data: Counter({44: 1, 52: 1, 87: 1, 100: 1})
Extra data: Counter()
------------------------------
------------------------------
Sampler drop_last=True, DataLoader drop_last=True,
 Sampler shuffle=False, DataLoader shuffle=default
Missing data: Counter({97: 1, 98: 1, 99: 1, 100: 1})
Extra data: Counter()
------------------------------
------------------------------
Sampler drop_last=True, DataLoader drop_last=False,
 Sampler shuffle=True, DataLoader shuffle=default
Missing data: Counter({52: 1})
Extra data: Counter()
------------------------------
------------------------------
Sampler drop_last=True, DataLoader drop_last=False,
 Sampler shuffle=False, DataLoader shuffle=default
Missing data: Counter({100: 1})
Extra data: Counter()
------------------------------
------------------------------
Sampler drop_last=False, DataLoader drop_last=True,
 Sampler shuffle=True, DataLoader shuffle=default
Missing data: Counter()
Extra data: Counter({45: 1, 20: 1})
------------------------------
------------------------------
Sampler drop_last=False, DataLoader drop_last=True,
 Sampler shuffle=False, DataLoader shuffle=default
Missing data: Counter()
Extra data: Counter({1: 1, 2: 1})
------------------------------
------------------------------
Sampler drop_last=False, DataLoader drop_last=False,
 Sampler shuffle=True, DataLoader shuffle=default
Missing data: Counter()
Extra data: Counter({45: 1, 20: 1})
------------------------------
------------------------------
Sampler drop_last=False, DataLoader drop_last=False,
 Sampler shuffle=False, DataLoader shuffle=default
Missing data: Counter()
Extra data: Counter({1: 1, 2: 1})
------------------------------
"""

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def setup(rank, world_size):
    # 初始化分布式环境 Initialize distributed environment
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    # 销毁分布式环境 Destroy the distributed environment
    dist.destroy_process_group()

def run_config(rank, world_size, sampler_drop_last, dataloader_drop_last, sampler_shuffle):
    # 设置当前进程的GPU Set the GPU for the current process
    torch.cuda.set_device(rank)
    setup(rank, world_size)

    # 创建一个简单的数据集 Create a simple dataset
    data = torch.arange(1, 101).view(-1, 1)  # 数据集包含 100 个样本 Dataset contains 100 samples
    dataset = CustomDataset(data)

    # 设置批次大小，每个GPU上为2，总批次大小为6 Set batch size, 2 per GPU, total batch size is 6
    batch_size = 2

    # 创建分布式采样器 Create distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=sampler_drop_last, shuffle=sampler_shuffle)

    # 创建 DataLoader Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=dataloader_drop_last)

    # 用于收集每个进程的数据 Collect data from each process
    collected_data = []

    # 模拟训练循环 Simulate training loop
    for epoch in range(1):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            collected_data.extend(batch)

    # 将每个进程的数据转换为tensor Convert data from each process to tensor
    collected_data_tensor = torch.tensor(collected_data, dtype=torch.int32, device=torch.device(rank))

    # 将所有进程的数据聚合到主进程 Gather data from all processes to the master process
    gathered_data = [torch.zeros_like(collected_data_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_data, collected_data_tensor)

    if rank == 0:
        # 在主进程中合并并检查数据 Merge and check data in the master process
        all_data = torch.cat(gathered_data).cpu().numpy().tolist()

        # 使用Counter来统计每个数据的出现次数 Use Counter to count the occurrence of each data item
        all_data_counter = Counter(all_data)
        expected_data_counter = Counter(range(1, 101))

        missing_data = expected_data_counter - all_data_counter
        extra_data = all_data_counter - expected_data_counter
        print("---"*10)
        print(f"Sampler drop_last={sampler_drop_last}, DataLoader drop_last={dataloader_drop_last},\n Sampler shuffle={sampler_shuffle}, DataLoader shuffle=default")
        print(f"Missing data: {missing_data}")
        print(f"Extra data: {extra_data}")
        print("---"*10)

    cleanup()

def spawn_for_config(sampler_drop_last, dataloader_drop_last, sampler_shuffle):
    world_size = 3  # 总的进程数，即GPU数量 Total number of processes (GPUs)
    mp.spawn(run_config, args=(world_size, sampler_drop_last, dataloader_drop_last, sampler_shuffle), nprocs=world_size, join=True)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'  # 指定要使用的GPU Specify which GPUs to use
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 设置主节点地址 Set the master node address
    os.environ['MASTER_PORT'] = '29501'  # 设置主节点端口 Set the master node port

    configs = [
        (sampler_drop_last, dataloader_drop_last, sampler_shuffle)
        for sampler_drop_last in [True, False]
        for dataloader_drop_last in [True, False]
        for sampler_shuffle in [True, False]
    ]

    for config in configs:
        # print(f"Running config: Sampler drop_last={config[0]}, DataLoader drop_last={config[1]}, Sampler shuffle={config[2]}, DataLoader shuffle=defaults")
        spawn_for_config(*config)