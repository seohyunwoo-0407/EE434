#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, weight_decay, **kwargs):
    """
    AdamW Optimizer
    
    AdamW는 Adam의 개선 버전으로, weight decay를 분리하여
    더 효과적인 regularization을 제공합니다.
    
    Args:
        parameters: 모델 파라미터
        lr: learning rate
        weight_decay: weight decay (L2 regularization)
        **kwargs: 추가 파라미터 (beta1, beta2, eps 등)
    """
    print('Initialised AdamW optimizer')
    
    # AdamW의 기본 하이퍼파라미터
    betas = kwargs.get('betas', (0.9, 0.999))
    eps = kwargs.get('eps', 1e-8)
    
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

