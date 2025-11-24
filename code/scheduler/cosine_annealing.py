#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, max_epoch=10, eta_min=0.0001, **kwargs):
    """
    CosineAnnealingLR scheduler
    
    Cosine 함수를 따라 learning rate를 부드럽게 감소시킵니다.
    SGDR과 달리 warm restart 없이 한 번만 cosine annealing을 수행합니다.
    
    Args:
        optimizer: Optimizer
        max_epoch: 전체 epoch 수 (기본값: 10)
        eta_min: 최소 learning rate (기본값: 0.0001)
        **kwargs: 추가 파라미터
    
    Returns:
        scheduler: CosineAnnealingLR scheduler
        lr_step: 'epoch' (epoch마다 step 호출)
    """
    sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=eta_min
    )
    
    lr_step = 'epoch'
    
    print(f'Initialised CosineAnnealingLR scheduler')
    print(f'  T_max={max_epoch}, eta_min={eta_min}')
    
    return sche_fn, lr_step

