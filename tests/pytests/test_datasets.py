# test_datasets.py
import pytest
import json
import os
import numpy as np
from PIL import Image
import tempfile
import torch

# 测试基础配置
@pytest.fixture
def setup_mani_dataset(tmp_path):
    # 创建临时目录结构
    dataset_path = tmp_path / "ManiDatasetTest"
    dataset_path.mkdir()
    tp_dir = tmp_path / "ManiDatasetTest" / "Tp"
    gt_dir = tmp_path / "ManiDatasetTest" / "Gt"
    tp_dir.mkdir()
    gt_dir.mkdir()
    
    # 创建测试图片
    for i in range(3):
        img = Image.new('RGB', (256, 256), color=(i*40, i*40, i*40))
        img.save(tp_dir / f"test_{i}.jpg")
        mask = Image.new('L', (256, 256), color=i)
        mask.save(gt_dir / f"test_{i}.jpg")
    
    return str(tmp_path / "ManiDatasetTest")

@pytest.fixture
def setup_json_dataset(tmp_path):
    # 创建测试JSON文件
    data = [
        [str(tmp_path/"tp1.jpg"), str(tmp_path/"gt1.jpg")],
        [str(tmp_path/"tp2.jpg"), "Negative"],
        [str(tmp_path/"tp3.jpg"), str(tmp_path/"gt3.jpg")]
    ]
    
    # 创建真实文件
    for tp, gt in data:
        if gt != "Negative":
            Image.new('RGB', (256, 256)).save(tp)
            Image.new('L', (256, 256)).save(gt)
        else:
            Image.new('RGB', (256, 256)).save(tp)
    
    json_path = tmp_path / "JsonDatasetTest.json"
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    return str(json_path)
@pytest.fixture
def setup_balanced_dataset(tmp_path, setup_mani_dataset, setup_json_dataset):
    setting_list = [
        ["JsonDataset", setup_json_dataset],
        ["ManiDataset", setup_mani_dataset]
    ]
    json_path = tmp_path / "test.json"
    with open(json_path, 'w') as f:
        json.dump(setting_list, f)
    return str(json_path)

# 测试ManiDataset
def test_mani_dataset(setup_mani_dataset):
    from IMDLBenCo.datasets import ManiDataset
    
    dataset = ManiDataset(
        path=setup_mani_dataset,
        is_resizing=True,
        output_size=(256, 256)
    )
    
    # 测试基础属性
    assert len(dataset.tp_path) == 3
    assert len(dataset.gt_path) == 3
    
    # 测试路径排序
    assert "test_0.jpg" in dataset.tp_path[0]
    assert "test_1.jpg" in dataset.tp_path[1]
    
    # 测试数据加载
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert sample['image'].shape == (3, 256, 256)
    assert sample['mask'].shape == (1, 256, 256)
    assert sample['label'] == 1

# 测试JsonDataset
def test_json_dataset(setup_json_dataset):
    from IMDLBenCo.datasets import JsonDataset
    
    dataset = JsonDataset(
        path=setup_json_dataset,
        is_resizing=True,
        output_size=(256, 256)
    )
    
    # 测试路径加载
    assert len(dataset.tp_path) == 3
    assert len(dataset.gt_path) == 3
    
    # 测试Negative样本
    assert dataset.gt_path[1] == "Negative"
    
    # 测试数据加载
    sample = dataset[1]
    assert sample['label'] == 0
    assert torch.all(sample['mask'] == 0)

# 测试BalancedDataset
def test_balanced_dataset(setup_balanced_dataset):
    
    from IMDLBenCo.datasets import BalancedDataset

    
    # 测试默认配置
    dataset = BalancedDataset(
        path = setup_balanced_dataset,
        sample_number=10,
        is_resizing=True,
        output_size=(256, 256)
    )
    
    # 验证配置列表
    assert len(dataset.settings_list) == 2

# 测试异常情况
def test_abstract_dataset_validation():
    from IMDLBenCo.datasets import ManiDataset, JsonDataset
    
    with pytest.raises((TypeError, FileNotFoundError)):
        # 测试无效的JSON路径
        JsonDataset(path="non_existent.json")
    
    with pytest.raises((AssertionError, FileNotFoundError)):
        # 测试ManiDataset路径不匹配
        dataset = ManiDataset(
            path="invalid_path",
            is_resizing=True,
            output_size=(256, 256)
        )

# 测试数据增强管道
def test_augmentation_pipeline(setup_mani_dataset):
    from IMDLBenCo.datasets import ManiDataset
    import albumentations as albu
    from albumentations import HorizontalFlip
    
    # 创建带增强的数据集
    transforms = albu.Compose([
        HorizontalFlip(p=1.0)  # 强制水平翻转
    ])

    dataset = ManiDataset(
        path=setup_mani_dataset,
        common_transforms=transforms,
        is_resizing=True,
        output_size=(256, 256)
    )
    
    original_sample = dataset[0]['image']
    flipped_sample = dataset[0]['image']
    
    # 验证是否应用了数据增强
    assert torch.all(torch.eq(
        original_sample,
        torch.flip(flipped_sample, [-1])
    ))

# 测试边缘情况
def test_edge_cases():
    from IMDLBenCo.datasets import (
        ManiDataset,
        JsonDataset
    )
    
    # 测试空目录
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises((FileNotFoundError, AssertionError, NotADirectoryError)):
            ManiDataset(path=empty_dir)
    
    # 测试无效JSON格式
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write("invalid json")
        f.flush()
        with pytest.raises((json.JSONDecodeError, TypeError)):
            JsonDataset(path=f.name)
            
"""
TODO: 完成对于if_paddding和 if_resizing的测试，并针对返回值进行约束。
"""