#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LossFunction(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(LossFunction, self).__init__()
        
        self.test_normalize = True
        self.margin = margin
        
        print(f'Initialised Triplet Loss with Hard Negative Mining (margin={margin})')
    
    def forward(self, x, label=None):
        assert x.size()[1] == 2
        
        # Normalize anchor and positive
        out_anchor = F.normalize(x[:, 0, :], p=2, dim=1)
        out_positive = F.normalize(x[:, 1, :], p=2, dim=1)
        
        # Choose hard negative indices
        negidx = self.choose_negative(out_anchor.detach(), out_positive.detach(), type='hard')
        
        # Get negative pairs
        out_negative = out_positive[negidx, :]
        
        # Calculate positive and negative pair distances
        pos_dist = F.pairwise_distance(out_anchor, out_positive)
        neg_dist = F.pairwise_distance(out_anchor, out_negative)
        
        # Triplet loss function
        nloss = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))
        
        return nloss
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Hard Negative Mining
    ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def choose_negative(self, embed_a, embed_p, type=None):
        batch_size = embed_a.size(0)
        negidx = []
        allidx = range(0, batch_size)
        
        for idx in allidx:
            excidx = list(allidx)
            excidx.pop(idx)
            
            if type == 'hard':
                # Hard negative mining: choose the negative that is closest to anchor
                # (but still different identity)
                anchor = embed_a[idx:idx+1, :]  # [1, dim]
                candidates = embed_p[excidx, :]  # [batch_size-1, dim]
                
                # Compute distances between anchor and all candidates
                distances = F.pairwise_distance(
                    anchor.expand_as(candidates),
                    candidates
                )
                
                # Choose the closest one (hardest negative)
                hard_idx = torch.argmin(distances).item()
                negidx.append(excidx[hard_idx])
            
            else:
                raise ValueError('Undefined type of mining.')
        
        return negidx

