import sys
from pprint import pprint
sys.path.append(".")

import IMDLBenCo
from IMDLBenCo.evaluation import PixelF1, ImageF1

list_e = [
    PixelF1(),
    ImageF1()
]


print(list_e[0], list_e[1])



def print_fun(obj):
    for i in obj:
        print(i)
        
        
        
print_fun(list_e)