
import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
from IMDLBench.training.schedular.cos_lr_schedular import adjust_learning_rate # TODO

from IMDLBench.datasets import denormalize



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
            
    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # move to device
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        torch.cuda.synchronize()
        
        with torch.cuda.amp.autocast():
            output_dict = model(**data_dict)
            loss = output_dict['backward_loss']
            mask_pred = output_dict['pred_mask']
            
            visual_loss = output_dict['visual_loss']
            visual_loss_item = {}
            for k, v in visual_loss.items():
                visual_loss_item[k] = v.item()
                
            visual_image = output_dict['visual_image']
        
        predict_loss = loss / accum_iter
        loss_scaler(predict_loss,optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
                
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        lr = optimizer.param_groups[0]["lr"]
        # save to log.txt
        metric_logger.update(lr=lr)
        

        metric_logger.update(**visual_loss_item)
        # metric_logger.update(predict_loss= predict_loss_value)
        # metric_logger.update(edge_loss= edge_loss_value)
        
        visual_loss_reduced = {}
        for k, v in visual_loss_item.items():
            visual_loss_reduced[k] = misc.all_reduce_mean(v)

        if log_writer is not None and (data_iter_step + 1) % 50 == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            # Tensorboard logging
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
            for k, v in visual_loss_reduced.items():
                log_writer.add_scalar(f"train_loss/{k}", v, epoch_1000x)
            # log_writer.add_scalar('train_loss/predict_loss', loss_predict_reduce, epoch_1000x)
            # log_writer.add_scalar('train_loss/edge_loss', edge_loss_reduce, epoch_1000x)

    samples = data_dict['image']
    mask = data_dict['mask']
    
    if log_writer is not None:
        log_writer.add_images('train/image',  denormalize(samples), epoch)
        log_writer.add_images('train/predict', mask_pred, epoch)
        log_writer.add_images('train/predict_thresh0.5', (mask_pred > 0.5) * 1.0, epoch)
        log_writer.add_images('train/gt_mask', mask, epoch)
        
        for k, v in visual_image.items():
            log_writer.add_images(f'train/{k}', v, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


