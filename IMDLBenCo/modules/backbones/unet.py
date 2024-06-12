import torch
import torch.nn as nn
import torch.nn.functional as F
from IMDLBenCo.registry import MODELS
from ..extractors.high_frequency_feature_extraction import FFTExtractor, DCTExtractor
from ..extractors.sobel import SobelFilter
from ..extractors.bayar_conv import BayerConv
from ..extractors.srm_filter import SRMConv2D

class UNet(nn.Module):
    def __init__(self, input_head=None, num_channels=3):
        super(UNet, self).__init__()
        net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=1)
        self.input_head = input_head
        if input_head != None:
            original_first_layer = net.inc.double_conv[0]
            new_first_layer = nn.Conv2d(num_channels + 3, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
            new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
            if num_channels > 0:
                new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
            net.inc.double_conv[0] = new_first_layer
        last_layer = net.outc.conv 
        new_last_layer = nn.Conv2d(last_layer.in_channels, 1,
                                        kernel_size=last_layer.kernel_size, stride=last_layer.stride,
                                        padding=last_layer.padding, bias=False)
        net.outc.conv  = new_last_layer
        self.net = net
        self.loss_fun = nn.BCEWithLogitsLoss()

    def forward(self, image, mask, *args, **kwargs):
        if self.input_head != None:
            feature = self.input_head(image)
            input = torch.cat([image,feature],dim=1)
        else:
            input = image
        mask_pred = self.net(input)
        loss = self.loss_fun(mask_pred,mask)
        mask_pred = torch.sigmoid(mask_pred)
        output_dict = {
            # loss for backward
            "backward_loss": loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": mask_pred,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": None,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "predict_loss": loss,
            },

            "visual_image": {
                "pred_mask": mask_pred,
            }
            # -----------------------------------------
        }
        return output_dict


@MODELS.register_module()
def unet():
    return UNet(None)

@MODELS.register_module()
def fft_unet():
    return UNet(FFTExtractor(), 3)

@MODELS.register_module()
def dct_unet():
    return UNet(DCTExtractor(), 3)

@MODELS.register_module()
def sobel_unet():
    return UNet(SobelFilter(), 1)

@MODELS.register_module()
def bayar_unet():
    return UNet(BayerConv(), 3)

@MODELS.register_module()
def srm_unet():
    return UNet(SRMConv2D(), 9)


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table,flop_count_table

    model = UNet(None,0, 3, 1, False)
    flops = FlopCountAnalysis(model,(torch.randn(1,3,512,512),torch.randn(1,1,512,512)))
    # # 打印模型的参数信息
    print(flop_count_table(flops))