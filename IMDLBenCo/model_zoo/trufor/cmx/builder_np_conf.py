"""
Edited in September 2022
@author: fabrizio.guillaro, davide.cozzolino
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .utils.init_func import init_weight
import sys
from ..DnCNN import make_net

import logging


def preprc_imagenet_torch(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
    std  = torch.Tensor([0.229, 0.224, 0.225]).to(x.device)
    x = (x-mean[None, :, None, None]) / std[None, :, None, None]
    return x


def create_backbone(typ, norm_layer):
    channels = [64, 128, 320, 512]
    if typ == 'mit_b2':
        print('Using backbone: Segformer-B2')
        from .encoders.dual_segformer import mit_b2 as backbone_
        backbone = backbone_(norm_fuse=norm_layer)
    else:
        raise NotImplementedError('backbone not implemented')
    return backbone, channels


class myEncoderDecoder(nn.Module):
    def __init__(self, cfg=None, norm_layer=nn.BatchNorm2d):
        super(myEncoderDecoder, self).__init__()
        
        self.norm_layer = norm_layer
        self.cfg  = cfg.MODEL.EXTRA
        self.mods = cfg.MODEL.MODS
        
        # import backbone and decoder
        self.backbone, self.channels = create_backbone(self.cfg.BACKBONE, norm_layer)
        
        if 'CONF_BACKBONE' in self.cfg:
            self.backbone_conf, self.channels_conf = create_backbone(self.cfg.CONF_BACKBONE, norm_layer)
        else:
            self.backbone_conf = None

        if self.cfg.DECODER == 'MLPDecoder':
            logging.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.DATASET.NUM_CLASSES, norm_layer=norm_layer, embed_dim=self.cfg.DECODER_EMBED_DIM)

            if self.cfg.CONF:
                self.decode_head_conf = DecoderHead(in_channels=self.channels, num_classes=1, norm_layer=norm_layer, embed_dim=self.cfg.DECODER_EMBED_DIM)
            else:
                self.decode_head_conf = None
            
            self.conf_detection = None
            if self.cfg.DETECTION is not None:
                if self.cfg.DETECTION == 'none':
                    pass
                elif self.cfg.DETECTION == 'confpool':
                    self.conf_detection = 'confpool'
                    assert self.cfg.CONF
                    self.detection  = nn.Sequential(
                            nn.Linear(in_features=8, out_features=128),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(in_features=128, out_features=1),
                            )
                else:
                    raise NotImplementedError('Detection mechanism not implemented')

        else:
            raise NotImplementedError('decoder not implemented')

        num_levels = 17
        out_channel = 1
        self.dncnn = make_net(3, kernels=[3, ] * num_levels,
                       features=[64, ] * (num_levels - 1) + [out_channel],
                       bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                       acts=['relu', ] * (num_levels - 1) + ['linear', ],
                       dilats=[1, ] * num_levels,
                       bn_momentum=0.1, padding=1)
        
        if self.cfg.PREPRC == 'imagenet': #RGB (mean and variance)
            self.prepro = preprc_imagenet_torch
        else:
            assert False

        self.init_weights(pretrained=cfg.MODEL.PRETRAINED)

        
    
    def init_weights(self, pretrained=None):
        if pretrained:
            logging.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
            if self.backbone_conf is not None:
                self.backbone_conf.init_weights(pretrained=pretrained)

            np_weights = self.cfg.NP_WEIGHTS
            assert os.path.isfile(np_weights)
            dat = torch.load(np_weights, map_location=torch.device('cpu'))
            logging.info(f'Noiseprint++ weights: {np_weights}')
            if 'network' in dat:
                dat = dat['network']
            self.dncnn.load_state_dict(dat)

        logging.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, self.cfg.BN_EPS, self.cfg.BN_MOMENTUM,
                    mode='fan_in', nonlinearity='relu')




    def encode_decode(self, rgb, modal_x):

        if rgb is not None:
            orisize = rgb.shape
        else:
            orisize = modal_x.shape
        
        # cmx
        x = self.backbone(rgb, modal_x)
        out, feats = self.decode_head(x, return_feats=True)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        
        # confidence
        if self.decode_head_conf is not None:
            if self.backbone_conf is not None:
                x_conf = self.backbone_conf(rgb, modal_x)
            else:
                x_conf = x # same encoder of Localization Network

            conf = self.decode_head_conf(x_conf)
            conf = F.interpolate(conf, size=orisize[2:], mode='bilinear', align_corners=False)
        else:
            conf = None

        
        # detection
        if self.conf_detection is not None:
            if self.conf_detection == 'confpool':
                from .layer_utils import weighted_statistics_pooling
                f1 = weighted_statistics_pooling(conf).view(out.shape[0],-1)
                f2 = weighted_statistics_pooling(out[:,1:2,:,:]-out[:,0:1,:,:], F.logsigmoid(conf)).view(out.shape[0],-1)
                det = self.detection(torch.cat((f1,f2),-1))
            else:
                assert False
        else:
            det = None
        
        return out, conf, det


    def forward(self, rgb):

        # Noiseprint++ extraction
        if 'NP++' in self.mods:
            modal_x = self.dncnn(rgb)
            modal_x = torch.tile(modal_x, (3, 1, 1))
        else:
            modal_x = None

        if self.prepro is not None:
            rgb = self.prepro(rgb)

        out, conf, det = self.encode_decode(rgb, modal_x)
        return out, conf, det, modal_x
            