import torch

import torch.nn as nn
from torchvision import models
from IMDLBenCo.registry import MODELS
from .glca_segformer import *

from .NoiseExtractor import *
from .utils import *

class SegFormerExtractor_bit5(nn.Module):
    def __init__(self,pretrain_path,glca = 0):
        super(SegFormerExtractor_bit5, self).__init__()
        self.segformer = mit_b5(
            pretrain_path = pretrain_path, glca=glca
        )

    def forward(self, x):
        features = self.segformer(x)
        return features[0], features[1], features[2], features[3] 
    
class SegFormerExtractor_bit4(nn.Module):
    def __init__(self,pretrain_path ='', glca = 0):
        super(SegFormerExtractor_bit4, self).__init__()
        self.segformer = mit_b4(
            pretrain_path = pretrain_path,glca=glca)
       
    def forward(self, x):
        features = self.segformer(x)
        return features[0], features[1], features[2], features[3] 
    
class SegFormerExtractor_bit2(nn.Module):
    def __init__(self,pretrain_path ='', glca = 0):
        super(SegFormerExtractor_bit2, self).__init__()
        self.segformer = mit_b2(
            pretrain_path = pretrain_path,glca=glca)

    def forward(self, x):
        features = self.segformer(x)
        return features[0], features[1], features[2], features[3]       
       
class SegFormerExtractor_bit3(nn.Module):
    def __init__(self,pretrain_path,glca = 0):
        super(SegFormerExtractor_bit3, self).__init__()
        self.segformer = mit_b3(
            pretrain_path = pretrain_path ,glca=glca
        )

    def forward(self, x):
        features = self.segformer(x)
        return features[0], features[1], features[2], features[3]       
   
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet101(pretrained=pretrained)

        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  

    def forward(self, x):
        x = self.initial(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        return f1,f2     

class AdaptiveGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        return self.gate(x)
    

class SE_Fusion(nn.Module):
    def __init__(self, in_channels):
        super(SE_Fusion, self).__init__()      
        self.SEBock = SEBlock(in_channels)

    def forward(self,feat1,feat2):  

        combined_out = feat1 + feat2
        weight = self.SEBock(combined_out)
        adaptive_feat1 = weight * feat1
        adaptive_feat2 = (1 - weight) * feat2
        return adaptive_feat1+adaptive_feat2

class MSCDILoss(nn.Module):
    def __init__(self, edge_weight=20.0, chi_weight=0.05, corr_weight=0):
        super(MSCDILoss, self).__init__()
        self.edge_weight = edge_weight
        #self.edge_weight = nn.Parameter(torch.tensor(edge_weight))
        self.chi_weight = chi_weight
        self.corr_weight = 0#corr_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def spatial_decorrelation_loss(self, f_tamp, f_seg):

        B, C, H, W = f_tamp.shape

        x = f_tamp.view(B, C, -1)  
        y = f_seg.view(B, C, -1)   
        
     
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)
        

        cross_cov = torch.bmm(x_centered, y_centered.transpose(1, 2))
        cross_cov = cross_cov / (H * W - 1)  
        

        loss = torch.norm(cross_cov, p='fro') ** 2
        loss = loss / (B * C * C)  
        return loss
    
    def cross_hierarchical_interaction_loss(self, multi_scale_features, gt_mask):
        """
       
        """
        total_loss = 0.0
        num_scales = len(multi_scale_features)
        

        gt_distributions = []
        for i, feat in enumerate(multi_scale_features):
            _, _, H, W = feat.shape

            gt_resized = F.interpolate(gt_mask, size=(H, W), mode='bilinear')
            gt_distributions.append(gt_resized)

        for i in range(num_scales):

            current_feat = multi_scale_features[i]
            B, C, H, W = current_feat.shape
            

            feat_flat = current_feat.view(B, C, -1)
            feat_norm = F.normalize(feat_flat, p=2, dim=-1)  # L2
            

            gt_current = gt_distributions[i].view(B, 1, -1)
            gt_norm = F.normalize(gt_current, p=2, dim=-1)

            gt_expanded = gt_norm.expand(B, C, -1)

            cosine_sim = (feat_norm * gt_expanded).sum(dim=-1)  # [B, C]

            consistency_loss = 1.0 - cosine_sim.mean()
            
            scale_weight = 1.0 / (2 ** (num_scales - i - 1))
            total_loss += scale_weight * consistency_loss
        
        return total_loss / num_scales
        
    def edge_aware_loss(self, pred, target, edge_mask):

        edge_loss = F.binary_cross_entropy_with_logits(
            input=pred,
            target=target,
            weight=edge_mask
        )*self.edge_weight
        
        return edge_loss
    
    def forward(self, pred, target, multi_scale_features=None, 
                edge_mask=None, f_tamp=None, f_seg=None):
        """

        
        Args:
            pred:  [B, 1, H, W]
            target: GT[B, H, W]
            multi_scale_features: [feat1, feat2, ...]
            edge_mask:  [B, H, W] 
            f_tamp:  [B, C, H, W]
            f_seg:  [B, C, H, W]
        """
        total_loss = 0.0

        bce_loss = self.bce_loss(pred, target)
        total_loss += bce_loss
        

        if edge_mask is not None:
            edge_loss = self.edge_aware_loss(pred, target, edge_mask)
            total_loss += edge_loss
        
        # if multi_scale_features is not None and self.chi_weight > 0:
        #     chi_loss = self.cross_hierarchical_interaction_loss(multi_scale_features, target)
        #     total_loss += self.chi_weight * chi_loss
        

        # if f_tamp is not None and f_seg is not None and self.corr_weight > 0:
        #     corr_loss = self.spatial_decorrelation_loss(f_tamp, f_seg)
        #     total_loss += self.corr_weight * corr_loss
        
        return total_loss   
    
@MODELS.register_module()
class MSCDI_Net(nn.Module):
    def __init__(self, backbone='mit_bit4',backbone_weights_path ='' ,noise_model = 'None', ff_domain = 0, fusion_mode = 2, glca :int = 12):
        """
        multi-scale cross domain interaction network for IML
        Args:
            backbone : 'mit_bit3','mit_bit3','mit_bit4','mit_bit5'
                       default: 'mit_bit4'   
            noise_model : 'Bayar','SRM','None'
            ff_domain : frequency feature domain/modal
                0: original feature map(segformer)
                1: segformer+wavelet domain
                2: resnet+wavelet
            fusion_mode : feature fusion mode
                0: concatation
                1: SE
                2: our interaction
            glca: 
                0: without glca
                >1: with glca,the number is glca layer , default=12
        """
        super(MSCDI_Net, self).__init__()
        if backbone_weights_path =='':
            assert False, "Please set the backbone weights path for segformer backbone using --backbone_weights_path!"
        # 
        if backbone == 'mit_bit5':
            self.segformer = SegFormerExtractor_bit5(backbone_weights_path, glca=glca)
        elif backbone == 'mit_bit2':
            self.segformer = SegFormerExtractor_bit2(backbone_weights_path, glca=glca)
        elif backbone == 'mit_bit3':
            self.segformer = SegFormerExtractor_bit3(backbone_weights_path, glca=glca)
        else:
            self.segformer = SegFormerExtractor_bit4(backbone_weights_path, glca=glca)

        if noise_model== 'SRM':
            self.NoiseExtractor = SRMNoiseExtractor()
        elif noise_model== 'Bayar': 
             self.NoiseExtractor = BayarNoiseExtractor()
        else:
            self.NoiseExtractor = None
         
        self.ff_domain = ff_domain
        self.fusion_mode = fusion_mode
       
        # 
        seg_upsampling_configs = [
            (512, 320, 2, False),  
            (320*2, 128,  2, False),  #f4
            #(256,  128,   2, False),  
            (128,  64,  2, False), #
            (64,  16,   2, False), #128->256
            (16,   4,   2, False)  #256->512
        ]

        feat_dims = [
            (512, 320,  16),  
            (320, 128,  32)   
        ]
        
        if self.ff_domain == 0:
            #self.crossatten = CrossAttentionBlock(512,512*2)
            self.downf1 = ChannelShuffleDownsample( in_channels=64, out_channels=320, scale_factor=4)
            self.downf2 = ChannelShuffleDownsample( in_channels=128, out_channels=512, scale_factor=4)
            #self.downf12 = ChannelShuffleDownsample( in_channels=64, out_channels=512, scale_factor=8)
        elif self.ff_domain == 1:
            # wavelet will shrink h,w to h/2,w/2
            self.downf1 = ChannelShuffleDownsample( in_channels=64, out_channels=320, scale_factor=2)
            self.downf2 = ChannelShuffleDownsample( in_channels=128, out_channels=512, scale_factor=2)
            # self.upf3 = PixelShuffleUpsample(in_channels=320, scale_factor=2, group_size=4, use_scope=False)  
            # self.upf4 = PixelShuffleUpsample(in_channels=512, scale_factor=2, group_size=4, use_scope=False)  
        elif self.ff_domain == 2:
            self.resnet = ResNetFeatureExtractor()
            self.downf1 = ChannelShuffleDownsample( in_channels=256, out_channels=320, scale_factor=4)  #128
            self.downf2 = ChannelShuffleDownsample( in_channels=512, out_channels=512, scale_factor=4)  #64


        if self.fusion_mode == 0:
            self.fusion_conv42 = nn.Conv2d(512*2,512,1)
            self.fusion_conv31 = nn.Conv2d(320*2,320,1)
            if self.NoiseExtractor != None:
                self.fusion_conv_noise = nn.Conv2d(128*2,128,1)
        elif self.fusion_mode == 1:
             self.SE_fusion42=SE_Fusion(512)
             self.SE_fusion31=SE_Fusion(320)
             if self.NoiseExtractor != None:
                self.SE_fusion_noise=SE_Fusion(128)
        elif self.fusion_mode == 2:
            self.interfuse1 = InteractiveFusion_C(512)
            self.interfuse2 = InteractiveFusion_C(320)
            if self.NoiseExtractor != None:
                self.interfuse_noise = InteractiveFusion_C(128)
        

        # if self.ff_domain == 1 or self.ff_domain == 2:
        #     self.wcams = nn.ModuleList()
        #     for in_ch,out_ch,size in feat_dims:
        #         self.wcams.append(
        #            WaveletCrossAttentionModule(in_ch)  # mode=1   
        #         )
        
        self.upsamplingers = nn.ModuleList()
        for idx, (in_ch, out_ch, scale, _) in enumerate(seg_upsampling_configs, 1):
            self.upsamplingers.append(
                nn.Sequential(   
                nn.Conv2d(in_ch, out_ch, kernel_size=1),          
                PixelShuffleUpsample(in_channels=out_ch, scale_factor=scale, group_size=4, use_scope=False)  
                )
            )
        
        self.final_conv = nn.Conv2d(4, 1, kernel_size=1)

        #self.edgeweight  = nn.Parameter(torch.tensor(20.0))
        self.loss_fn = MSCDILoss(edge_weight=20.0, chi_weight=0.5, corr_weight=1.0)
        self.multi_scale_features = []

    def forward(self, image:torch.tensor, mask, edge_mask, label, return_features=True, *args,  **kwargs):
         
        x=torch.tensor(image).to(image.device)

        self.multi_scale_features = []
        feat_1,feat_2,feat_3,feat_4=self.segformer(x)
        self.multi_scale_features = [feat_1, feat_2, feat_3, feat_4]

        if self.NoiseExtractor != None:
            noise_f = self.NoiseExtractor(x)
        else:
            noise_f = None

        if self.ff_domain == 0:
            feat_1 = self.downf1(feat_1)
            feat_2 = self.downf2(feat_2)
            
        # elif self.ff_domain == 1:
        #     feat_1 = self.wcams[1](self.downf1(feat_1))
        #     feat_2 = self.wcams[0](self.downf2(feat_2))
        # elif self.ff_domain == 2:
        #   feat_1,feat_2 =self.resnet(x)
        #   feat_1 = self.wcams[1](self.downf1(feat_1))
        #   feat_2 = self.wcams[0](self.downf2(feat_2))

        if self.fusion_mode == 0:
            feat_42 = torch.cat([feat_4,feat_2],dim=1)
            feat_42 = self.fusion_conv42(feat_42)
            feat_31 = torch.cat([feat_3,feat_1],dim=1)
            feat_31 = self.fusion_conv31(feat_31)
        elif self.fusion_mode == 1:
            feat_42 = self.SE_fusion42(feat_4,feat_2)
            feat_31 = self.SE_fusion31(feat_3,feat_1)
        elif self.fusion_mode == 2:
            feat_42= self.interfuse1(feat_4,feat_2)
            feat_31= self.interfuse2(feat_3,feat_1)

        feat_42 = self.upsamplingers[0](feat_42)
        fused_feat = torch.cat([feat_42,feat_31],dim=1)
        fused_feat = self.upsamplingers[1](fused_feat)
        feat_seg = fused_feat

        if self.NoiseExtractor != None:
            if self.fusion_mode == 0:
                fused_feat = torch.cat([fused_feat,noise_f],dim=1)
                fused_feat = self.fusion_conv42(fused_feat)
            elif self.fusion_mode == 1:
                fused_feat = self.SE_fusion_noise(fused_feat,noise_f)
            elif self.fusion_mode == 2:   
                fused_feat = self.interfuse_noise(fused_feat,noise_f)
        fused_feat = self.upsamplingers[2](fused_feat)       
        fused_feat = self.upsamplingers[3](fused_feat)   
        fused_feat = self.upsamplingers[4](fused_feat)
      
        fused_out = self.final_conv(fused_feat)
        pred_mask = torch.sigmoid(fused_out)

        # bce_loss = self.BCE_loss(fused_out, mask) #+ self.beta1*loss1 + self.beta2*loss2
        # edge_loss = F.binary_cross_entropy_with_logits(
        #     input=fused_out,
        #     target=mask,
        #     weight=edge_mask
        # )*self.edgeweight
        # total_loss = bce_loss +  edge_loss 
        total_loss = self.loss_fn(
            pred=fused_out, 
            target=mask,
            multi_scale_features=self.multi_scale_features,
            edge_mask=edge_mask,
            f_tamp=noise_f,  
            f_seg=feat_seg     
        )

        output_dict = {
            # loss for backward
            "backward_loss": total_loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": pred_mask,
            "mask":mask,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": None,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "predict_loss": total_loss
            },

            "visual_image": {
                "pred_mask": pred_mask
            }
        }
        return output_dict









