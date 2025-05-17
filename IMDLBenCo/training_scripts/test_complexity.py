import os
import json
import time
import torch
import types
import inspect
import argparse
import datetime
import numpy as np
from pathlib import Path
import albumentations as albu
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import FlopCountAnalysis, flop_count_table

import IMDLBenCo.training_scripts.utils.misc as misc
from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.datasets import DummyDataset

# robustness wrappers
def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBench Robustness test Launch!', add_help=True)
    # Model name
    parser.add_argument('--model', default=None, type=str,
                        help='The name of applied model', required=True)
    
    # 可以接受label的模型是否接受label输入，并启用相关的loss。
    parser.add_argument('--if_predict_label', action='store_true',
                        help='Does the model that can accept labels actually take label input and enable the corresponding loss function?')
    # ----Dataset parameters 数据集相关的参数----
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', action='store_true', 
                        help='resize all images to same resolution.')
    # If edge mask activated
    parser.add_argument('--edge_mask_width', default=None, type=int,
                        help='Edge broaden size (in pixels) for edge maks generator.')

    # ------------------------------------
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")

    # -----------------------
    # Since augmentation includes randomize functions, here need to set the seeds
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    args, remaining_args = parser.parse_known_args()
    # 获取对应的模型类
    model_class = MODELS.get(args.model)

    # 根据模型类动态创建参数解析器
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args

def main(args, model_args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_transform = albu.Compose([

    ])
    # Since augmentation includes randomize functions, here need to set the seeds
    # fix the seed for reproducibility
    seed = args.seed 
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = MODELS.get(args.model)
    # Filt usefull args
    if isinstance(model,(types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    combined_args.update({k: v for k, v in vars(model_args).items() if k in model_init_params})
    model = model(**combined_args)
    # ============================================

    model.to(device)

    print("Model = %s" % str(model))
    
    start_time = time.time()
    # get post function (if have)
    post_function_name = f"{args.model.lower()}_post_func"
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    try:
        post_function = POSTFUNCS.get_lower(post_function_name)
        print(f"Post function loaded: {post_function}")
    except Exception as e:
        print(f"Post function {post_function_name} not found, using default post function.")
        print(e)
        post_function = None

        dataset_test = DummyDataset(
            path=None,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=test_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )
        dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        print(f"Dummy Dataset loaded: {dataset_test}")
        print(f"Dummy Dataset test length: {len(dataset_test)}")
        
        model_forward_func = model.forward
        # get the function signature
        func_signature = inspect.signature(model_forward_func)
        # get the ordered_dict for parameters
        forward_para_list = func_signature.parameters
        # 用绿色字体打印，开始使用fvcore测试Parameter和FLOPS
        # 具体信息请参考：https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md
        print("\033[92m" + "Start testing FLOPS and Parameters..." + "\033[0m")
        print("\033[92m" + "Model name: {}".format(args.model) + "\033[0m")
        print("\033[92m" + "For more information, please refer to:")
        print("\033[92m" + "https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md" + "\033[0m")
        for data_dict in dataloader_test:
                    # move to device
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(device)
            data_list = []
            for key in forward_para_list.keys():
                if key in data_dict:
                    data_list.append(data_dict[key])
                else:
                    data_list.append(None)
                
            data_list = tuple(data_list)

            flops = FlopCountAnalysis(model, data_list)
            print(flops.total())
            print(flop_count_table(flops))
            print("\n")
        print("\033[92m" + "----Finish testing FLOPS and Parameters... Details please check the table above...----" + "\033[0m")
        local_time = time.time() - start_time
        local_time_str = str(datetime.timedelta(seconds=int(local_time)))
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total testing time {}'.format(total_time_str))
    exit(0)    
        


if __name__ == '__main__':
    args, model_args = get_args_parser()
    main(args, model_args)