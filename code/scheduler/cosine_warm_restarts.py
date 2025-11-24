#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, T_0=10, T_mult=2, eta_min=0.0001, **kwargs):
    """
    SGDR (Stochastic Gradient Descent with Warm Restarts)
    CosineAnnealingWarmRestarts scheduler
    
    Cosine annealing을 사용하되, 주기적으로 learning rate를 다시 높여서
    warm restart를 수행합니다. 이는 local minimum에서 벗어나 더 좋은 해를
    찾는 데 도움이 됩니다.
    
    Args:
        optimizer: Optimizer
        T_0: 첫 번째 restart까지의 epoch 수 (기본값: 10)
        T_mult: restart 주기 배수 (기본값: 2)
                T_0=10, T_mult=2 -> restart at epoch 10, 20, 40, 80, ...
        eta_min: 최소 learning rate (기본값: 0.0001)
        **kwargs: 추가 파라미터
    
    Returns:
        scheduler: CosineAnnealingWarmRestarts scheduler
        lr_step: 'epoch' (epoch마다 step 호출)
    """
    sche_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    
    lr_step = 'epoch'
    
    print(f'Initialised CosineAnnealingWarmRestarts (SGDR) scheduler')
    print(f'  T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}')
    
    return sche_fn, lr_step

