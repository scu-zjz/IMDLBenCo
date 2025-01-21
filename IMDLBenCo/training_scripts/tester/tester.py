
import math
import sys
from typing import Iterable, List
import torch
from torch.utils.data import Subset, DataLoader

import IMDLBenCo.training_scripts.utils.misc as misc

from IMDLBenCo.evaluation import genertate_region_mask, cal_confusion_matrix, cal_F1 # TODO remove this line
from IMDLBenCo.evaluation import AbstractEvaluator
from IMDLBenCo.datasets import denormalize


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
    # 具体data_dict的格式参考IMDLBench.datasets.abstract_dataset的108行 113行~117行
    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # move to device
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)
        # Forwarding on model
        output_dict = model(**data_dict)
        # results
        mask_pred = output_dict['pred_mask']
        label_pred = output_dict['pred_label']

        #---- Training evaluation ----
        # batch update in a evaluator
        BATCHSIZE, _, _, _ = mask_pred.shape
        if BATCHSIZE != args.test_batch_size:
            print("=" * 20)
            print(f"A batch that is not fully loaded was detected at the end of the dataset. The actual batch size for this batch is {BATCHSIZE}: The default batch size is {args.test_batch_size}" )
            print("=" * 20)
        
        for evaluator in evaluator_list:
            results = evaluator.batch_update(
                predict=mask_pred, 
                predict_label=label_pred,
                **data_dict
            )
            if results == None: # Image-level results, do nothing
                continue
            else:               # pixel-level results, update to logger
                assert BATCHSIZE == len(results) , f"Length of output results in evaluator {evaluator.name} does not match with the bachsize."
                # print(BATCHSIZE, results.shape)
                # print(results)
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
        print_freq = 20
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
                remaining_loader = DataLoader(remaining_subset, batch_size=batch_size)
                # test on remaining loader
                remaining_header = 'Test <remaining>: [{}]'.format(epoch)
                test_one_loader(
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
        metric_logger.synchronize_between_processes()    
        print("---syncronized---")
        for evaluator in evaluator_list:
            print(evaluator.name, "reduced_count", metric_logger.meters[evaluator.name].count)
            print(evaluator.name, "reduced_sum", metric_logger.meters[evaluator.name].total)
        print('---syncronized done ---')
        if log_writer is not None:
            if is_test:
                for evaluator in evaluator_list:
                    log_writer.add_scalar(f'{name}/test_evaluators/{evaluator.name}', metric_logger.meters[evaluator.name].global_avg, epoch)
            log_writer.add_images(f'{name}/test/image',  denormalize(data_dict['image']), epoch)
            log_writer.add_images(f'{name}/test/predict', output_dict['pred_mask'] * 1.0, epoch)
            log_writer.add_images(f'{name}/test/predict_threshold_0.5', (output_dict['pred_mask'] > 0.5)* 1.0, epoch)
            log_writer.add_images(f'{name}/test/mask', data_dict['mask'], epoch)
            # log_writer.add_images('test/edge_mask', edge_mask, epoch)
            
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}