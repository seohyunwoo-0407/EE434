#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, scale=64, margin=0.5, **kwargs):
        super(LossFunction, self).__init__()
        
        self.test_normalize = True
        
        self.nOut = nOut
        self.nClasses = nClasses
        self.scale = scale
        self.margin = margin
        
        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos(m*theta) values for different m values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        self.criterion = nn.CrossEntropyLoss()
        
        print(f'Initialised ArcFace Loss (scale={scale}, margin={margin})')
    
    def forward(self, x, label=None):
        # Normalize input features and weight matrix
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(x_norm, w_norm)
        cosine = cosine.clamp(-1, 1)  # Numerical stability
        
        # Compute theta (angle)
        theta = torch.acos(cosine)
        
        # Compute target logit with angular margin
        target_theta = theta[range(len(label)), label].view(-1, 1)
        target_theta_m = target_theta + self.margin
        target_cosine = torch.cos(target_theta_m)
        
        # Create output logits
        output = cosine * 1.0
        output[range(len(label)), label] = target_cosine.view(-1)
        
        # Scale the logits
        output = output * self.scale
        
        # Compute loss
        loss = self.criterion(output, label)
        
        return loss

