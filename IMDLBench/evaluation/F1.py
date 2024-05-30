import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .abstract_class import AbstractEvaluator
import torch.distributed as dist
import os
from sklearn.metrics import f1_score

class ImageF1(AbstractEvaluator):
    def __init__(self, threshold = 0.5) -> None:
        super().__init__()
        self.name = "image-level F1"
        self.desc = "image-level F1"
        self.threshold = threshold
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.cnt = 0
    def batch_update(self, predict_label, label, *args, **kwargs):
        predict = (predict_label > self.threshold).float()
        self.TP += torch.sum(predict * label).item()
        self.TN += torch.sum((1-predict) * (1-label)).item()
        self.FP += torch.sum(predict * (1-label)).item()
        self.FN += torch.sum((1-predict) * label).item()
        self.cnt += len(predict_label)
        return None
    
    def epoch_update(self):
        t = torch.tensor([self.TP, self.TN, self.FP, self.FN, self.cnt],  dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        TP = t[0].item()
        TN = t[1].item()
        FP = t[2].item()
        FN = t[3].item()
        cnt = t[4].item()
        precision = TP / ( TP +  FP + 1e-9)
        recall =  TP / ( TP +  FN + 1e-9)
        F1 = 2 * precision * recall / (precision + recall + 1e-9)
        # F1 = torch.mean(F1) # fuse the Batch dimension
        return F1
    def recovery(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.cnt = 0

            
class PixelF1(AbstractEvaluator):
    def __init__(self, threshold = 0.5, mode = "origin") -> None:
        super().__init__()
        self.name = "pixel-level F1"
        self.desc = "pixel-level F1"
        self.threshold = threshold
        self.image_num = 0
        #  mode : "origin, reverse, double"
        self.mode = mode

    def Cal_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        if(shape_mask != None):
            TP = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum(predict * mask, dim=(1, 2, 3))  
            TN = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3)) 
            FP = torch.sum(predict * (1-mask), dim=(1, 2, 3)) 
            FN = torch.sum((1-predict) * mask, dim=(1, 2, 3))         
        return TP, TN, FP, FN

    def Cal_Reverse_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        if(shape_mask != None):
            TP = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum((1-predict) * mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask), dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3))
            FN = torch.sum(predict * mask, dim=(1, 2, 3))
        return TP, TN, FP, FN

    def Cal_F1(self, TP, TN, FP, FN):
        """_summary_

        Args:
            TP (_type_): _description_
            TN (_type_): _description_
            FP (_type_): _description_
            FN (_type_): _description_

        Returns:
            _type_: _description_
        """
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        # F1 = torch.mean(F1) # fuse the Batch dimension
        return F1

    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs): # 注意这里只有pixel-level需要的信息
        if self.mode == "origin":
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            F1 = self.Cal_F1(TP, TN, FP, FN)
        elif self.mode == "reverse":
            TP, TN, FP, FN = self.Cal_Reverse_Confusion_Matrix(predict, mask, shape_mask)
            F1 = self.Cal_F1(TP, TN, FP, FN)
        elif self.mode == "double":
            # todo
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            F1 = torch.max(self.Cal_F1(TP, TN, FP, FN), self.Cal_F1(FN, FP, TN, TP))
        else:
            raise RuntimeError(f"Cal_F1 no mode name {self.mode}")
        
        return F1
    def epoch_update(self):

        return None
    def recovery(self):
        self.image_num = 0


def test_origin_image_f1():
    # test imageF1
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://')
    
    num_gpus = torch.cuda.device_count()
    if dist.get_rank() == 0:
        print("number of GPUS", num_gpus)
    
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    DATA_LEN = 200
    float_tensor = torch.rand( DATA_LEN * num_gpus).cuda(local_rank)  # 生成一个长度为 5 的浮点数 tensor

    # 生成一个包含 0 或 1 的整数 tensor
    int_tensor = torch.randint(0, 2, (DATA_LEN * num_gpus,)).cuda(local_rank)
    # print(float_tensor)
    # print(int_tensor)
    
    evaluator = ImageF1(threshold=0.5)
    dist.barrier()
    dist.broadcast(float_tensor, src=0)
    dist.broadcast(int_tensor, src=0)
    # 收集所有的预测标签和真实标签，用于之后的 sklearn 验证
    all_predicts = []
    all_labels = []
    
    
    if dist.get_rank() != num_gpus - 1:
        idx = dist.get_rank() * DATA_LEN
        predict_labels = float_tensor[idx: idx + DATA_LEN].cuda(local_rank)
        true_labels = int_tensor[idx: idx + DATA_LEN].cuda(local_rank)
    else:
        idx = dist.get_rank() * DATA_LEN
        predict_labels = float_tensor[idx: idx + DATA_LEN-50].cuda(local_rank)
        true_labels = int_tensor[idx: idx + DATA_LEN-50].cuda(local_rank)


    if dist.get_rank() == 0:  # 只在 rank 0 进程中收集数据
        all_predicts = float_tensor.cpu().numpy()
        all_labels= int_tensor.cpu().numpy()
        # print(all_labels)
            

    # 运行 batch_update 更新统计数据
    evaluator.batch_update(predict_labels, true_labels)
    
    # 模拟一个 epoch 结束，调用 epoch_update 来计算 F1 分数
    gpu_f1_score = evaluator.epoch_update()
    if(dist.get_rank() == 0):
        print(f"F1 Score: {gpu_f1_score}")
        # 使用 sklearn 计算 F1 分数
        # all_predicts = np.concatenate(all_predicts)
        # all_labels = np.concatenate(all_labels)
        sklearn_f1 = f1_score(all_labels[:-50], (all_predicts[:-50] > 0.5).astype(int), average='binary')
        print(f"F1 Score (sklearn): {sklearn_f1}", "cnt = ", len(all_labels[:-50]))
    

    # 清理分布式环境
    dist.destroy_process_group()


            

if __name__ == "__main__":
    test_origin_image_f1()