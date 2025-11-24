#! /usr/bin/python
# -*- encoding: utf-8 -*-

import timm

def MainModel(nOut=256, **kwargs):
    """
    MobileViT-S 모델
    
    MobileViT-S는 Hybrid CNN-Transformer 모델
    도메인 적응(out-of-domain generalization) 태스크에 특히 적합
    
    파라미터 수: 약 4.94M (FC layer 제외)
    """
    model = timm.create_model('mobilevit_s', num_classes=nOut, pretrained=False)
    return model

