# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: davide.cozzolino
"""

import torch
import torch.nn.functional as F

    
def weighted_statistics_pooling(x, log_w=None):
        b = x.shape[0]
        c = x.shape[1]
        x = x.view(b,c,-1)
        
        if log_w is None:
            log_w = torch.zeros((b,1,x.shape[-1]), device=x.device)
        else:
            assert log_w.shape[0]==b
            assert log_w.shape[1]==1
            log_w = log_w.view(b,1,-1)
        
            assert log_w.shape[-1]==x.shape[-1]
        
        log_w = F.log_softmax(log_w, dim=-1)
        x_min = -torch.logsumexp(log_w-x, dim=-1)
        x_max =  torch.logsumexp(log_w+x, dim=-1)
        
        w = torch.exp(log_w)
        x_avg = torch.sum(w*x  , dim=-1)
        x_msq = torch.sum(w*x*x, dim=-1)
        
        x = torch.cat((x_min, x_max, x_avg, x_msq), dim=1)
        
        return x