import torch.nn as nn
"""
下面这个类很重要，可以自动管理在多卡之间的一个具体数值变量的reduce（显卡之间归并数据）
可以用于实现算法的参考。
"""
# from training.utils.misc import MetricLogger

"""
改变这个接口的主要目的是image-level的指标和pixel-level的指标的计算方式不同
"""
class AbstractEvaluator(object): # 想了想没必要用nn.module 反而可能会引起一些其他的问题，比如梯度追踪，或者Parameter的追踪等等问题？？？（我有点困不是很确定）
    def __init__(self) -> None:
        self.name = None
        self.desc = None
        self.threshold = None
    def _check_pixel_level_params(self, predict, mask):
        if predict is None:
            raise ValueError(f"Detect none mask predict from the model, cannot calculate {self.name}. Please remove Pixel-level metrics from the script, or check the model output.")
        if mask is None:
            raise ValueError(f"Detect none mask label from the dataset, cannot calculate {self.name}. Please remove Pixel-level metrics from the script, or check the dataset output.")
    def _chekc_image_level_params(self, predict_label, label):
        if predict_label is None:
            raise ValueError(f"Detect none image-level predict label from the model, cannot calculate {self.name}. Please remove Image-level metrics from the script, or check the model output.")
        if label is None:
            raise ValueError(f"Detect none image-level binary label from the dataset, cannot calculate {self.name}. Please remove Image-level metrics from the script, or check the dataset output.")
    def batch_update(self, predict, pred_label, mask, shape_mask=None, *args, **kwargs):
        """
        本函数在每个batch结尾update。
        """
        raise NotImplementedError
    def remain_update(self, predict, pred_label, mask, shape_mask=None, *args, **kwargs):
        """
        本函数在每个batch结尾update。
        主要用于处理在最后一个batch之后的剩余数据。
        """
        raise NotImplementedError
    def epoch_update(self):
        """
        理论上这个时候没有新的数据了，所以没有输入参数。
        
        功能：在显卡之间收集所有在整个epoch内统计的指标，然后返回最终期望的信息。
        """
        raise NotImplementedError
    def recovery(self):
        raise NotImplementedError




    
