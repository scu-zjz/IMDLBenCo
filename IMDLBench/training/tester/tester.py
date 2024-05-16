
import math
import sys
from typing import Iterable

import torch

import utils.misc as misc

from IMDLBench.evaluation import genertate_region_mask, cal_confusion_matrix, cal_F1
from IMDLBench.datasets import denormalize

def test_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, 
                    epoch: int, 
                    log_writer=None,
                    args=None):
    
    with torch.no_grad():
        model.zero_grad()
        model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        # F1 evaluation for an Epoch during training
        print_freq = 20
        header = 'Test: [{}]'.format(epoch)
        
        # 具体data_dict的格式参考IMDLBench.datasets.abstract_dataset的108行 113行~117行
        for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            
            # move to device
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(device)
                    

            output_dict = model(**data_dict)
            loss = output_dict['backward_loss']
            mask_pred = output_dict['pred_masks']
            
            mask_pred = mask_pred.detach()
            #---- Training evaluation ----
            # region_mask is for cutting of the zero-padding area.
            region_mask = genertate_region_mask(mask_pred, data_dict['shapes']) 
            TP, TN, FP, FN = cal_confusion_matrix(mask_pred, data_dict['masks'], region_mask)
        
            local_f1 = cal_F1(TP, TN, FP, FN)
            # print(local_f1)
            
            for i in local_f1: # merge batch
                metric_logger.update(average_f1=i)
                print(metric_logger.meters['average_f1'].count)
                print(metric_logger.meters['average_f1'].total)

        metric_logger.synchronize_between_processes()    
        # print("---syncronized---")
        # print(metric_logger.meters['average_f1'].count)
        # print(metric_logger.meters['average_f1'].total)
        # print('---syncronized done ---')
        if log_writer is not None:
            log_writer.add_scalar('F1/test_average', metric_logger.meters['average_f1'].global_avg, epoch)
            log_writer.add_images('test/image',  denormalize(data_dict['images']), epoch)
            log_writer.add_images('test/predict', (mask_pred > 0.5)* 1.0, epoch)
            log_writer.add_images('test/masks', data_dict['masks'], epoch)
            # log_writer.add_images('test/edge_mask', edge_mask, epoch)
            
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}