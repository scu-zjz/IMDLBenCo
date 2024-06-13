import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from IMDLBenCo.registry import MODELS

from ..extractors.high_frequency_feature_extraction import FFTExtractor, DCTExtractor
from ..extractors.sobel import SobelFilter
from ..extractors.bayar_conv import BayerConv
from ..extractors.srm_filter import SRMConv2D

class ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, input_head=None, num_channels=3,pretrained=True):
        super(ViT, self).__init__()
        if pretrained:
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.load_state_dict(model.state_dict())
        original_first_layer = self.patch_embed.proj
        self.patch_embed.strict_img_size=False
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
        new_length = 1024
        self.deconv_model = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: 1, 256, 64, 64
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: 1, 128, 128, 128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 输出: 1, 64, 256, 256
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)      # 输出: 1, 1, 512, 512
        )
        
        original_pos_embed = model.pos_embed[:, 1:, :]
        new_pos_embed = F.interpolate(original_pos_embed.permute(0, 2, 1), size = new_length, mode='nearest').permute(0, 2, 1)

        # Create new positional embeddings tensor with space for class token
        new_pos_embed_with_cls = torch.zeros(1,new_length+1, 768, device=original_pos_embed.device)
        new_pos_embed_with_cls[:, 1:, :] = new_pos_embed  # fill in the interpolated embeddings
        new_pos_embed_with_cls[:, 0, :] = model.pos_embed[:, 0, :]  # copy the class token embeddings
        # 
        # Replace the original positional embeddings with the new ones
        self.pos_embed = nn.Parameter(new_pos_embed_with_cls)
        self.loss_fun = nn.BCEWithLogitsLoss()


    def forward(self, image, mask, *args, **kwargs):
        if self.input_head != None:
            feature = self.input_head(image)
            input = torch.concat([image,feature],dim=1)
        else:
            input = image
        x = self.patch_embed(input)
        x = self._pos_embed(x)
        x = self.blocks(x)
        feature = self.norm(x)
        h = w = int((feature.shape[1])**0.5)
        feature = feature.permute(0,2,1)[:,:,1:1025].reshape(image.shape[0],768,h,w)
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
def vit16():
    return ViT(None)

@MODELS.register_module()
def fft_vit16():
    return ViT(FFTExtractor(), 3)

@MODELS.register_module()
def dct_vit16():
    return ViT(DCTExtractor(), 3)

@MODELS.register_module()
def sobel_vit16():
    return ViT(SobelFilter(), 1)

@MODELS.register_module()
def bayar_vit16():
    return ViT(BayerConv(), 3)

@MODELS.register_module()
def srm_vit16():
    return ViT(SRMConv2D(), 9)

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table,flop_count_table

    model = ViT()
    flops = FlopCountAnalysis(model,(torch.randn(1,3,512,512),torch.randn(1,1,512,512)))
    # # 打印模型的参数信息
    print(flop_count_table(flops))
