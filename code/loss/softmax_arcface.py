#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, scale=64, margin=0.5, weight_softmax=0.7, weight_arcface=0.3, **kwargs):
        super(LossFunction, self).__init__()
        
        self.test_normalize = True
        
        self.nOut = nOut
        self.nClasses = nClasses
        self.scale = scale
        self.margin = margin
        self.weight_softmax = weight_softmax
        self.weight_arcface = weight_arcface
        
        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos(m*theta) values for different m values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        self.criterion = nn.CrossEntropyLoss()
        
        print(f'Initialised Softmax + ArcFace Combined Loss')
        print(f'  (scale={scale}, margin={margin}, weights: softmax={weight_softmax}, arcface={weight_arcface})')
    
    def forward(self, x, label=None):
        # Normalize input features and weight matrix
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(x_norm, w_norm)
        cosine = cosine.clamp(-1, 1)  # Numerical stability
        
        # ===== Softmax Loss =====
        softmax_logits = cosine * self.scale
        softmax_loss = self.criterion(softmax_logits, label)
        
        # ===== ArcFace Loss =====
        # Compute theta (angle)
        theta = torch.acos(cosine)
        
        # Compute target logit with angular margin
        target_theta = theta[range(len(label)), label].view(-1, 1)
        target_theta_m = target_theta + self.margin
        target_cosine = torch.cos(target_theta_m)
        
        # Create output logits
        arcface_logits = cosine * 1.0
        arcface_logits[range(len(label)), label] = target_cosine.view(-1)
        arcface_logits = arcface_logits * self.scale
        
        arcface_loss = self.criterion(arcface_logits, label)
        
        # ===== Combined Loss =====
        combined_loss = self.weight_softmax * softmax_loss + self.weight_arcface * arcface_loss
        
        return combined_loss

