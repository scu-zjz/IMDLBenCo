from IMDLBenCo.registry import MODELS
import torch.nn as nn
import torch

@MODELS.register_module()
class MyModel(nn.Module):
    def __init__(self, MyModel_Customized_param:int, pre_trained_weights:str) -> None:
        """
        The parameters of the `__init__` function will be automatically converted into the parameters expected by the argparser in the training and testing scripts by the framework according to their annotated types and variable names. 
        
        In other words, you can directly pass in parameters with the same names and types from the `run.sh` script to initialize the model.
        """
        super().__init__()
        
        self.MyModel_Customized_param = MyModel_Customized_param
        self.pre_trained_weights = pre_trained_weights

        # does not matter
        self.demo_layer = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.loss_func = torch.mean
        
    def forward(self, image, mask, label, *args, **kwargs):
        
        loss = self.loss_func(image)
        pred_mask = torch.mean(image, dim=1, keepdim=True)
        pred_label = torch.mean(image)
        
        inverse_mask = 1 - mask
        
        # ----------Output interface--------------------------------------
        output_dict = {
            # loss for backward
            "backward_loss": loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": pred_mask,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                # keys here can be customized by yourself.
                "predict_loss": loss,
                'lable_loss' : loss
            },

            "visual_image": {
                # keys here can be customized by yourself.
                # Various intermediate masks, heatmaps, and feature maps can be appropriately converted into RGB or single-channel images for visualization here.
                "pred_mask": pred_mask,
                "reverse_mask" : inverse_mask, 
            }
            # -------------------------------------------------------------
        }
        
        return output_dict
    
    
if __name__ == "__main__":
    print(MODELS)