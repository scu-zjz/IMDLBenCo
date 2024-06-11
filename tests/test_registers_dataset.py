import sys
sys.path.append("./")
# from IMDLBench.datasets import DATASETS
from IMDLBenCo.registry import DATASETS

print(DATASETS)
obj = DATASETS.build("ManiDataset", path="/mnt/data0/public_datasets/IML/basic_eval_dataset", is_padding=True)
print(obj)