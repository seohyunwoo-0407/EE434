#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision

def MainModel(nOut=256, **kwargs):
    """
    MobileNetV2 모델
    
    MobileNetV2는 경량 모델로, 2.22M 파라미터 (FC 제외)
    """
    model = torchvision.models.mobilenet_v2(num_classes=nOut)
    return model

