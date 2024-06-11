# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(".")
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import utils.misc as misc

from IMDLBenCo.registry import MODELS
from IMDLBenCo.datasets import ManiDataset, JsonDataset
from IMDLBenCo.transforms import get_albu_transforms
from IMDLBenCo.evaluation import PixelF1, ImageF1

from tester import test_one_epoch

from IMDLBenCo.model_zoo import IML_ViT

def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBench testing', add_help=True)
    # ++++++++++++TODO++++++++++++++++
    # 这里是每个模型定制化的input区域，包括load与训练模型，模型的magic number等等
    # 需要根据你们的模型定制化修改这里 
    # 目前这里的内容都是仅仅给IML-ViT用的
    parser.add_argument('--vit_pretrain_path', default = None, type=str, help='path to vit pretrain model by MAE')
    parser.add_argument('--edge_broaden', default=7, type=int,
                        help='Edge broaden size (in pixels) for edge_generator.')
    parser.add_argument('--edge_lambda', default=20, type=float,
                        help='hyper-parameter of the weight for proposed edge loss.')
    parser.add_argument('--predict_head_norm', default="BN", type=str,
                        help="norm for predict head, can be one of 'BN', 'LN' and 'IN' (batch norm, layer norm and instance norm). It may influnce the result  on different machine or datasets!")
    # -------------------------------
    
    # ----Dataset parameters 数据集相关的参数----
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', action='store_true', 
                        help='resize all images to same resolution.')
    parser.add_argument('--test_data_json', default='/root/Dataset/CASIA1.0', type=str,
                        help='test dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    # ------------------------------------
    # Testing 相关的参数
    parser.add_argument('--checkpoint_path', default = '/root/workspace/IML-ViT/output_dir', type=str, help='path to the dir where saving checkpoints')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")

    # ----输出的日志相关的参数-----------
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    # -----------------------
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

def main(args):
    # init parameters for distributed training
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    
    test_transform = get_albu_transforms('test')

    with open(args.test_data_json, "r") as f:
        test_dataset_json = json.load(f)
    
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        global_rank = 0
    
    # ------------------ 
    # define the model and Evaluators here
    model = IML_ViT(
        vit_pretrain_path = args.vit_pretrain_path,
        predict_head_norm= args.predict_head_norm,
        edge_lambda = args.edge_lambda
    )
    
    evaluator_list = [
        PixelF1(threshold=0.5, mode="origin"),
        # ImageF1(threshold=0.5)
    ]
    
    # ------------------ TODO   
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    start_time = time.time()
    
    # Start go through each datasets:
    for dataset_name, dataset_path in test_dataset_json.items():
        args.full_log_dir = os.path.join(args.log_dir, dataset_name)

        if global_rank == 0 and args.full_log_dir is not None:
            os.makedirs(args.full_log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.full_log_dir)
        else:
            log_writer = None
        
        
        # TODO -------TBK的代码需要修改这里，其他人不用-------
        # ---- dataset with crop augmentation ----
        if os.path.isdir(dataset_path):
            dataset_test = ManiDataset(
                dataset_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=test_transform,
                edge_width=args.edge_broaden
            )
        else:
            dataset_test = JsonDataset(
                dataset_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=test_transform,
                edge_width=args.edge_broaden
            )
        # ------------------------------------
        print(dataset_test)
        print("len(dataset_test)", len(dataset_test))
        
        # Sampler
        if args.distributed:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, 
                num_replicas=num_tasks, 
                rank=global_rank, 
                shuffle=False
            )
            print("Sampler_test = %s" % str(sampler_test))
        else:
            sampler_test = torch.utils.data.RandomSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            sampler=sampler_test,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        print(f"Start testing on {dataset_name}! ")

        chkpt_list = os.listdir(args.checkpoint_path)
        print(chkpt_list)
        chkpt_pair = [(int(chkpt.split('-')[1].split('.')[0]) , chkpt) for chkpt in chkpt_list if chkpt.endswith(".pth")]
        chkpt_pair.sort(key=lambda x: x[0])
        print( "sorted checkpoint pairs in the ckpt dir: ",chkpt_pair)
        for epoch , chkpt_dir in chkpt_pair:
            if chkpt_dir.endswith(".pth"):
                print("Loading checkpoint: %s" % chkpt_dir)
                ckpt = os.path.join(args.checkpoint_path, chkpt_dir)
                ckpt = torch.load(ckpt, map_location='cuda')
                model.module.load_state_dict(ckpt['model'])            
                test_stats = test_one_epoch(
                    model=model,
                    data_loader=data_loader_test,
                    evaluator_list=evaluator_list,
                    device=device,
                    epoch=epoch,
                    log_writer=log_writer,
                    args=args
                )
                log_stats = {
                    **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch}
            
                if args.full_log_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.full_log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
        local_time = time.time() - start_time
        local_time_str = str(datetime.timedelta(seconds=int(local_time)))
        print(f'Testing on dataset {dataset_name} takes {local_time_str}')
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total testing time {}'.format(total_time_str))
    exit(0)    
        


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)