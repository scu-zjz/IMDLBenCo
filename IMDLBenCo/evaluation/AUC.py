import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from abstract_class import AbstractEvaluator
from .abstract_class import AbstractEvaluator
import torch.distributed as dist
import os
from sklearn.metrics import roc_auc_score
class ImageAUC(AbstractEvaluator):
    def __init__(self) -> None:
        self.name = "image-level AUC"
        self.desc = "image-level AUC"
        self.predict_label = torch.tensor([], device='cuda')
        self.label = torch.tensor([], device='cuda')
        self.cnt = torch.tensor(0, device='cuda')
    
    def compute_auc(self, y_true, y_scores):
        # 排序索引
        desc_score_indices = torch.argsort(y_scores, descending=True)
        y_true_sorted = y_true[desc_score_indices]

        # 计算正负样本的数量
        n_pos = torch.sum(y_true_sorted).item()
        n_neg = len(y_true_sorted) - n_pos

        # 累积正样本和负样本的数量
        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        # 计算 TPR 和 FPR
        tpr = tps / n_pos
        fpr = fps / n_neg

        # 计算 AUC
        auc = torch.trapz(tpr, fpr)

        return auc.item()
    
    
    def batch_update(self, predict_label, label, *args, **kwargs):

        predict = predict_label.float().cuda()
        self.predict_label = torch.cat([self.predict_label, predict], dim=0)
        self.label = torch.cat([self.label, label], dim=0)
        self.cnt += torch.tensor(len(label), device='cuda')
        return None

    def epoch_update(self):
        # cnt = torch.tensor(self.cnt, dtype=torch.int64).cuda()
        cnt = self.cnt.clone().detach().cuda()
        t_gather_cnt = [torch.zeros(1, dtype=torch.int64, device='cuda') for _ in range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather(t_gather_cnt, cnt)
        
        max_cnt = torch.max(torch.stack(t_gather_cnt, dim=0), dim=0)[0].cuda()
        max_idx = torch.max(torch.stack(t_gather_cnt, dim=0), dim=0)[1].cuda()
        if max_cnt > self.cnt:
            self.predict_label = torch.cat([self.predict_label, torch.zeros(max_cnt-self.cnt, device='cuda')], dim=0)
            self.label = torch.cat([self.label, torch.zeros(max_cnt-self.cnt, device='cuda')], dim=0)

        t_label = self.label.float().cuda()
        t_predict_label = self.predict_label.float().cuda()

        t_gather_predict_label = [torch.zeros(max_cnt, dtype=torch.float32, device='cuda') for _ in range(dist.get_world_size())]
        t_gather_label = [torch.zeros(max_cnt, dtype=torch.float32, device='cuda') for _ in range(dist.get_world_size())]
        dist.barrier()

        dist.all_gather(t_gather_label, t_label)
        
        dist.barrier()
        dist.all_gather(t_gather_predict_label, t_predict_label)

        final_predict_label = torch.cat([t_gather_predict_label[idx][:cnt.item()] for idx, cnt in enumerate(t_gather_cnt)], dim=0).cuda()
        final_label = torch.cat([t_gather_label[idx][:cnt.item()] for idx, cnt in enumerate(t_gather_cnt)], dim=0).cuda()

        final_predict_label = final_predict_label.view(-1)
        final_label = final_label.view(-1)
        print(len(final_label))
        AUC = self.compute_auc(final_label, final_predict_label)
        return AUC
    
    def recovery(self):
        self.predict_label = torch.tensor([], device='cuda')
        self.label = torch.tensor([], device='cuda')
        self.cnt = torch.tensor(0, device='cuda')
    
    

class PixelAUC(AbstractEvaluator):
    # TODO 每张都单独算，不要一起算
    def __init__(self, threshold=0.5, mode="origin") -> None:
        self.name = "pixel-level AUC"
        self.desc = "pixel-level AUC"
        self.threshold = threshold
        self.mode = mode

    def Cal_AUC(self, y_true, y_scores, shape_mask=None):
        if shape_mask is not None:
            y_true = y_true * shape_mask
            y_scores = y_scores * shape_mask
        
        y_true = y_true.flatten()
        y_scores = y_scores.flatten()

        # 排除被 shape_mask 掩盖的部分
        if shape_mask is not None:
            valid_mask = shape_mask.flatten() > 0
            y_true = y_true[valid_mask]
            y_scores = y_scores[valid_mask]

        # 排序索引
        desc_score_indices = torch.argsort(y_scores, descending=True)
        y_true_sorted = y_true[desc_score_indices]

        # 计算正负样本的数量
        n_pos = torch.sum(y_true_sorted).item()
        n_neg = len(y_true_sorted) - n_pos

        # 累积正样本和负样本的数量
        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        # 计算 TPR 和 FPR
        tpr = tps / n_pos
        fpr = fps / n_neg

        # 计算 AUC
        auc = torch.trapz(tpr, fpr)

        return auc.item()
        
    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        # TODO
        AUC_list = []
        if self.mode == "origin":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask == None else shape_mask[idx]
                AUC_list.append(self.Cal_AUC(mask[idx], predict[idx], single_shape_mask))
        elif self.mode == "reverse":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask == None else shape_mask[idx]
                AUC_list.append(self.Cal_AUC(mask[idx], 1 - predict[idx], single_shape_mask))
        elif self.mode == "double":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask == None else shape_mask[idx]
                AUC_list.append(max(self.Cal_AUC(mask[idx], predict[idx], single_shape_mask), self.Cal_AUC(mask[idx], 1 - predict[idx], single_shape_mask)))
        else:
            raise RuntimeError(f"Cal_AUC no mode name {self.mode}")
        
        return torch.tensor(AUC_list)

    def epoch_update(self):

        return None
    
    def recovery(self):
        return None


def test_origin_image_AUC():
    # test imageauc
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
    
    evaluator = ImageAUC()
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
        print(f"Ours AUC Score: {gpu_f1_score}")
        print(f"Sklearn AUC Score:{roc_auc_score(all_labels[:-50], all_predicts[:-50])}")


    # 清理分布式环境
    dist.destroy_process_group()


            



# # 示例用法和对比
if __name__ == "__main__":
    # 生成一些示例数据
    # batch_size, channels, height, width = 1, 1, 10, 10
    # predict = torch.rand(batch_size, channels, height, width)
    # mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    # # 生成一个 shape_mask
    # shape_mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()

    # auc = PixelAUC()
    # reverse_auc = PixelAUC(mode="reverse")
    # double_auc = PixelAUC(mode="double")
    # # image_auc = Image_AUC()

    # auc_value_pytorch = auc.batch_update(predict, mask, shape_mask)
    # reverse_auc_value_pytorch = reverse_auc.batch_update(predict, mask, shape_mask)
    # double_auc_value_pytorch = double_auc.batch_update(predict, mask, shape_mask)
    # # image_auc_value_pytorch = image_auc(torch.tensor([[0.1],[0.3]]), torch.tensor([[1.],[0.]]))

    # # 转换为 numpy 数组用于 scikit-learn 计算
    # predict_np = (predict * shape_mask).flatten().numpy()
    # mask_np = (mask * shape_mask).flatten().numpy()

    # # 排除被 shape_mask 掩盖的部分
    # valid_mask_np = shape_mask.flatten().numpy() > 0
    # predict_np = predict_np[valid_mask_np]
    # mask_np = mask_np[valid_mask_np]

    # auc_value_sklearn = roc_auc_score(mask_np, predict_np)
    # reverse_auc_value_sklearn = roc_auc_score(mask_np, 1 - predict_np)
    # double_auc_value_sklearn = max(auc_value_sklearn, reverse_auc_value_sklearn)
    # # image_auc_value_sklearn = roc_auc_score(torch.tensor([[1],[0]]), torch.tensor([[0.1],[0.3]]))

    # print(f"PyTorch AUC: {auc_value_pytorch}")
    # print(f"scikit-learn AUC: {auc_value_sklearn}\n")

    # print(f"PyTorch Reverse AUC: {reverse_auc_value_pytorch}")
    # print(f"scikit-learn Reverse AUC: {reverse_auc_value_sklearn}\n")

    # print(f"PyTorch Double AUC: {double_auc_value_pytorch}")
    # print(f"scikit-learn Double AUC: {double_auc_value_sklearn}\n")

    # # print(f"PyTorch Image AUC: {image_auc_value_pytorch}")
    # # print(f"scikit-learn Image AUC: {image_auc_value_sklearn}")
    test_origin_image_AUC()
