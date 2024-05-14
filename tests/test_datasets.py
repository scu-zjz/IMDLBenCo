import sys
from pprint import pprint
sys.path.append(".")
from torch.utils.data import Dataset, DataLoader
import IMDLBench
import IMDLBench.datasets
from IMDLBench.registry import DATASETS




data = IMDLBench.datasets.mani_dataset(r"G:\Datasets\IML\IML_Datasets_revised\CASIA1.0", is_padding=True, edge_width= 7)

# print(data[0:2])


batch_size = 3
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

for i in dataloader:
    pprint(i)
    import pdb
    pdb.set_trace()
    exit(0)
# print(dataloader)
print(repr(DATASETS))