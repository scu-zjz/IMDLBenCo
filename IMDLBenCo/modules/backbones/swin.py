import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from IMDLBenCo.registry import MODELS

from ..extractors.high_frequency_feature_extraction import FFTExtractor, DCTExtractor
from ..extractors.sobel import SobelFilter
from ..extractors.bayar_conv import BayerConv
from ..extractors.srm_filter import SRMConv2D

class Swin(timm.models.swin_transformer.SwinTransformer):
    def __init__(self, input_head=None, num_channels=3,pretrained=True):
        super(Swin, self).__init__(img_size=512,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
        if pretrained:
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
            self.load_state_dict(model.state_dict())
        original_first_layer = self.patch_embed.proj
        if input_head != None:
            self.input_head = input_head
            new_first_layer = nn.Conv2d(num_channels + 3, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
            new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
            if num_channels > 0:
                new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
            self.patch_embed.proj = new_first_layer
        else:
            self.input_head = None

        self.deconv_model = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: 1, 256, 64, 64
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: 1, 128, 128, 128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 输出: 1, 64, 256, 256
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),      # 输出: 1, 1, 512, 512
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)      # 输出: 1, 1, 512, 512
        )
        self.loss_fun = nn.BCEWithLogitsLoss()


    def forward(self, image, mask, *args, **kwargs):
        if self.input_head != None:
            feature = self.input_head(image)
            input = torch.cat([image, feature],dim=1)
        else:
            input = image
        # import pdb
        # pdb.set_trace()
        feature = self.forward_features(input)
        feature = feature.permute(0,3,1,2)
        mask_pred = self.deconv_model(feature)
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
def swin():
    return Swin(None)

@MODELS.register_module()
def fft_swin():
    return Swin(FFTExtractor(), 3)

@MODELS.register_module()
def dct_swin():
    return Swin(DCTExtractor(), 3)

@MODELS.register_module()
def sobel_swin():
    return Swin(SobelFilter(), 1)

@MODELS.register_module()
def bayar_swin():
    return Swin(BayerConv(), 3)

@MODELS.register_module()
def srm_swin():
    return Swin(SRMConv2D(), 9)

if __name__ == '__main__':

    import timm
    from pprint import pprint
    from fvcore.nn import FlopCountAnalysis, parameter_count_table,flop_count_table
    import torch
    model = Swin()
    flops =  FlopCountAnalysis(model,(torch.randn(1,3,512,512),torch.randn(1,1,512,512)))
    print(flop_count_table(flops))