import timm
import torch


from fvcore.nn import FlopCountAnalysis,flop_count_table
import torch.nn as nn
from IMDLBenCo.registry import MODELS
from ..extractors.high_frequency_feature_extraction import(
    FFTExtractor,
    DCTExtractor
)
from ..extractors.sobel import SobelFilter
from ..extractors.bayar_conv import BayerConv
from ..extractors.srm_filter import SRMConv2D



class ResNet(nn.Module):
    def __init__(self, input_head=None,num_channels=3):
        super(ResNet, self).__init__()
        model = timm.create_model('resnet152', pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:7])
        self.deconv_model = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),  # [1, 512, 32, 32]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [1, 128, 64, 64]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # [1, 64, 128, 128]
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)      # [1, 1, 256, 256]
        )
        original_first_layer = list(model.children())[0]
        if input_head != None:
            self.input_head = input_head
            new_first_layer = nn.Conv2d(num_channels + 3, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
            new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
            if num_channels > 0:
                new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
            self.backbone[0] = new_first_layer
        else:
            self.input_head = None
        self.loss_fun = nn.BCEWithLogitsLoss()
    def forward(self, image, mask, *args, **kwargs):
        if self.input_head != None:
            feature = self.input_head(image)
            input = torch.cat([image,feature],dim=1)
        else:
            input = image
        mask_pred = self.deconv_model(self.backbone(input))
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
def resnet():
    return ResNet(None,0)

@MODELS.register_module()
def fft_resnet():
    return ResNet(FFTExtractor(),3)

@MODELS.register_module()
def dct_resnet():
    return ResNet(DCTExtractor(),3)

@MODELS.register_module()
def sobel_resnet():
    return ResNet(SobelFilter(), 1)

@MODELS.register_module()
def bayar_resnet():
    return ResNet(BayerConv(), 3) 

@MODELS.register_module()
def srm_resnet():
    return ResNet(SRMConv2D(),9)


if __name__ == '__main__':

    #创建模型
    model = ResNet()

    flops =  FlopCountAnalysis(model,(torch.randn(1,3,512,512),torch.randn(1,1,512,512)))
    print(flop_count_table(flops))

