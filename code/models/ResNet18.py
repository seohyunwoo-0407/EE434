#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision

def MainModel(nOut=256, **kwargs):
    """
    ResNet18 모델
    11.4M 파라미터
    """
    return torchvision.models.resnet18(num_classes=nOut)
