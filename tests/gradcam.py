import torch
import sys
sys.path.append(".")
from IMDLBenCo.datasets import ManiDataset
from IMDLBenCo.transforms import get_albu_transforms
from IMDLBenCo.model_zoo import cat_net
from IMDLBench.model_zoo.cat_net.cat_net_post_function import cat_net_post_func
import torch
from IMDLBenCo.evaluation import grad_camera_visualize

if __name__ == '__main__':
    model = cat_net('/home/bingkui/IMDLBenCo/IMDLBench/training/CAT_full.yaml') # TODO 这里加载模型
    ckpt = '/home/bingkui/IMDLBenCo/output_dir_balance/checkpoint-44.pth' # TODO 这里填已经训练好的模型
    ckpt = torch.load(ckpt, map_location='cuda')
    model.load_state_dict(ckpt['model'])
    model.cuda()

    dataset = ManiDataset(path='/mnt/data0/public_datasets/IML/CASIA1.0',
                is_padding=False,
                is_resizing=True,
                output_size=(512, 512),
                common_transforms=get_albu_transforms('test'),
                edge_width=7)[0:1]
                #post_funcs=cat_net_post_func -> this argument is only for Cat-Net
    
    target_layers = [model.model.last_layer[-1]]
    grad_camera_visualize(model=model,
                          image=dataset,
                          target_layers=target_layers, # TODO 这里放你的模型结构中最后一个计算单元，用list装起来 
                          output_path='/home/bingkui/IMDLBenCo/images/CASIA1.0') # TODO 这里放图片输出的文件夹地址
