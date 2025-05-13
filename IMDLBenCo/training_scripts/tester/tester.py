import os
import re
import json
from PIL import Image
from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import IMDLBenCo.training_scripts.utils.misc as misc

from IMDLBenCo.datasets import denormalize
from IMDLBenCo.evaluation import AbstractEvaluator

def test_one_loader(model: torch.nn.Module,
                    data_loader: Iterable, 
                    evaluator_list: List[AbstractEvaluator],
                    device: torch.device, 
                    metric_logger=None,
                    print_freq=20,
                    header="Test:",
                    if_remain=False,
                    args=None
                   ):
    data_dict, output_dict = None, None # See https://github.com/scu-zjz/IMDLBenCo/blob/main/tests/test_tail_dataset.py for why this is needed

    # 具体data_dict的格式参考IMDLBench.datasets.abstract_dataset的108行 113行~117行
    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # move to device
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)
        # Forwarding on model
        output_dict = model(**data_dict)

        #---- Training evaluation ----
        # batch update in a evaluator
        # results, only for calculate batchsize
        predict = None
        if output_dict.get('pred_label') is not None:
            label_pred = output_dict['pred_label']
            predict = label_pred
        if output_dict.get('pred_mask') is not None:
            mask_pred = output_dict['pred_mask']
            predict = mask_pred
        BATCHSIZE = predict.shape[0]

        if BATCHSIZE != args.test_batch_size:
            print("=" * 20)
            print(f"A batch that is not fully loaded was detected at the end of the dataset. The actual batch size for this batch is {BATCHSIZE}: The default batch size is {args.test_batch_size}" )
            print("=" * 20)            
        for evaluator in evaluator_list:
            if if_remain == True:
                results = evaluator.remain_update(
                    **({"predict": output_dict["pred_mask"]} if output_dict.get("pred_mask") is not None else {"predict": None}),
                    **({"predict_label": output_dict["pred_label"]} if output_dict.get(
                        "pred_label") is not None else {"predict_label": None}),
                    **data_dict
                )
            else:
                results = evaluator.batch_update(
                    **({"predict": output_dict["pred_mask"]} if output_dict.get("pred_mask") is not None else {"predict": None}),
                    **({"predict_label": output_dict["pred_label"]} if output_dict.get(
                        "pred_label") is not None else {"predict_label": None}),
                    **data_dict
                )


            if results == None: # Image-level results, do nothing
                continue
            else:               # pixel-level results, update to logger
                assert BATCHSIZE == len(results) , f"Length of output results in evaluator {evaluator.name} does not match with the bachsize."
                # print(BATCHSIZE, results.shape)
                # print(results)
                # Only apply to SUM able metrics
                results = torch.sum(results)
                # print(results)
                # print(data_dict['name'])

                assert evaluator.name != "_n", f"name in evaluator {evaluator.name} can't set to '_n' to avoid conflicts in metric logger."
                
                world_size = misc.get_world_size()
                if if_remain != True:   # remain_dataset on the taile
                    kwargs_evaluator = {evaluator.name : results}
                    metric_logger.update(
                        **kwargs_evaluator,
                        _n= BATCHSIZE 
                    )
                else:                   # Common batch update
                    print("Actual Batchsize/ world_size", {"_n":BATCHSIZE / world_size})
                    kwargs_evaluator = {evaluator.name : results / world_size}
                    print(kwargs_evaluator)
                    metric_logger.update(
                        **kwargs_evaluator,
                        _n= BATCHSIZE / world_size
                    )

        # print(evaluator.name, metric_logger.meters[evaluator.name].count)
        # print(evaluator.name, metric_logger.meters[evaluator.name].total)
    return data_dict, output_dict


def test_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    evaluator_list: List[AbstractEvaluator],
                    device: torch.device, 
                    epoch: int,
                    name='', 
                    log_writer=None,
                    print_freq = 20,
                    args=None,
                    is_test=True):
      
    # print(data_loader.dataset.tp_path)
    
    with torch.no_grad():
        model.zero_grad()
        if args.no_model_eval == True:
            print("model.eval() IS NOT APPLIED")
        else:
            model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        # F1 evaluation for an Epoch during training
        header = 'Test: [{}]'.format(epoch)
        
        # Full test on the vanilla dataloader with drop_last==True
        data_dict, output_dict = test_one_loader(
            model=model,
            data_loader=data_loader,
            evaluator_list=evaluator_list,
            device=device,
            metric_logger=metric_logger,
            print_freq=print_freq,
            header=header,
            args=args
        )
        
        """
        check if needs a tail_dataset (when drop_last==True in Dataloader, use device:0 to compute the final results)
        """     
        # check if needs to do:
        if args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            batch_size = args.test_batch_size
            dataset_from_loader = data_loader.dataset
            dataset_len = len(dataset_from_loader)
            effective_batchsize = num_tasks * batch_size
            # print("dataset_len", dataset_len, num_tasks)
            # print(num_tasks)
            if dataset_len % effective_batchsize != 0:
                tail_dataset_band = "****An extra tail dataset should exist for accracy metrics!****"
                print("*" * len(tail_dataset_band))
                print(tail_dataset_band)
                print("*" * len(tail_dataset_band))
                #the tail_dataset exists
                    # only compute on device:0
                length_tail = dataset_len % effective_batchsize
                print(f"**** Length of tail: {length_tail} ****")
                start_idx = dataset_len - length_tail
                indices = list(range(dataset_len))
                remaining_indices = indices[start_idx:]
                remaining_subset = Subset(dataset_from_loader, remaining_indices)
                remaining_loader = DataLoader(remaining_subset, batch_size=batch_size, collate_fn=data_loader.collate_fn)
                # test on remaining loader
                remaining_header = 'Test <remaining>: [{}]'.format(epoch)
                rem_data_dict, rem_output_dict = test_one_loader(
                    model=model,
                    data_loader=remaining_loader,
                    evaluator_list=evaluator_list,
                    device=device,
                    metric_logger=metric_logger,
                    print_freq=1,
                    header=remaining_header,
                    if_remain=True,
                    args=args
                )        
        # Epoch level update of evaluators
        for evaluator in evaluator_list:
            results = evaluator.epoch_update()
            # recover the evaluator
            evaluator.recovery()
            if results == None:
                continue
            else:
                # results = results.item()
                kwargs_evaluator = {evaluator.name : results}
                assert evaluator.name != "_n", f"name in evaluator {evaluator.name} can't set to '_n' to avoid conflicts in metric logger."
                metric_logger.update(
                    **kwargs_evaluator,
                    _n=1
                )
        if data_dict is None:
            data_dict = rem_data_dict
            output_dict = rem_output_dict
        metric_logger.synchronize_between_processes()    
        print("---syncronized---")
        for evaluator in evaluator_list:
            print(evaluator.name, "reduced_count", metric_logger.meters[evaluator.name].count)
            print(evaluator.name, "reduced_sum", metric_logger.meters[evaluator.name].total)
        print('---syncronized done ---')
        if log_writer is not None:
            if is_test:
                for evaluator in evaluator_list:
                    log_writer.add_scalar(f'{name}_test_evaluators/{evaluator.name}',
                                          metric_logger.meters[evaluator.name].global_avg, epoch)
            if data_dict.get('image') is not None:
                log_writer.add_images(f'{name}_test/image', denormalize(data_dict['image']), epoch)
            if output_dict.get('pred_mask') is not None:
                log_writer.add_images(f'{name}_test/predict', output_dict['pred_mask'] * 1.0, epoch)
                log_writer.add_images(f'{name}_test/predict_threshold_0.5', (output_dict['pred_mask'] > 0.5) * 1.0,
                                      epoch)
            if data_dict.get('mask') is not None:
                log_writer.add_images(f'{name}_test/mask', data_dict['mask'], epoch)
            # log_writer.add_images('test/edge_mask', edge_mask, epoch)
            
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    


def inference_and_save_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, 
                    name='', 
                    log_writer=None,
                    print_freq = 20,
                    args=None,
                    is_test=True):
      
    # print(data_loader.dataset.tp_path)
    
    with torch.no_grad():
        model.zero_grad()
        if args.no_model_eval == True:
            print("model.eval() IS NOT APPLIED")
        else:
            model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        # F1 evaluation for an Epoch during training
        header = 'Test: [save images]'

        label_dict = {}
        for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(device)
            output_dict = model(**data_dict)

            filename = data_dict['name']
            mask_pred = output_dict['pred_mask']
            original_shape = data_dict['origin_shape']
            shape = data_dict['shape']
            # save pred_mask
            if output_dict.get('pred_mask') is not None:
                B, C, H, W = mask_pred.shape
                # padding:
                for i in range(B):
                    mask_pred_i = mask_pred[i] # 1, H, W
                    if args.if_padding:
                        mask_pred_i = mask_pred_i[:, :original_shape[i][0], :original_shape[i][1]]
                    if args.if_resizing:
                        mask_pred_i = F.interpolate(mask_pred_i.unsqueeze(0), size=(original_shape[i][0], original_shape[i][1]), mode='bilinear', align_corners=False)
                        mask_pred_i = mask_pred_i.squeeze(0)
                    mask_pred_i = mask_pred_i.squeeze(0)
                    mask_pred_i = mask_pred_i.cpu().numpy()
                    mask_pred_i = mask_pred_i * 255
                    mask_pred_i = mask_pred_i.astype('uint8')
                    filename_i = filename[i]
                    # 用正则表达式匹配拓展名，并替换为png
                    filename_i = re.sub(r'\.[^.]*$', '.png', filename_i)
                    # 如果需要，创建输出目录

                    output_dir = os.path.join(args.output_dir, "pred")
                    os.makedirs(output_dir, exist_ok=True)
                    filename_i = os.path.join(output_dir, os.path.basename(filename_i))
                    
                    # 使用PIL保存为PNG文件
                    img = Image.fromarray(mask_pred_i)
                    img.save(filename_i)
        
                    print(f'Saved mask to {filename_i} on RANK {misc.get_rank()}')

            # save pred_label
            if output_dict.get('pred_label') is not None:
                B = output_dict['pred_label'].shape[0]
                for i in range(B):
                    label_pred_i = output_dict['pred_label'][i].item()
                    label_dict[filename[i]] = label_pred_i
        if len(label_dict) > 0:
            rank = misc.get_rank()
            label_json_name = os.path.join(args.output_dir, f"pred_label_rank{rank}.json")
            with open(label_json_name, 'w') as f:
                json.dump(label_dict, f)
            print(f'Saved label to {label_json_name} on RANK {misc.get_rank()}')
            # barrier to ensure all ranks have finished saving
            if args.distributed:
                torch.distributed.barrier()
        if len (label_dict) > 0 and misc.get_rank() == 0:
            # combine all json files and drop duplicates
            all_json_files = []
            for i in range(misc.get_world_size()):
                rank_json_name = os.path.join(args.output_dir, f"pred_label_rank{i}.json")
                all_json_files.append(rank_json_name)
            combined_dict = {}
            for json_file in all_json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    combined_dict.update(data)
            combined_json_name = os.path.join(args.output_dir, "pred_label_combined.json")
            with open(combined_json_name, 'w') as f:
                json.dump(combined_dict, f)
            print(f'Saved combined label to {combined_json_name} on RANK {misc.get_rank()}')


