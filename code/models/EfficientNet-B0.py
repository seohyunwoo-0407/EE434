import torchvision

def MainModel(nOut=256, **kwargs):
    """
    EfficientNet-B0 모델
    4.66M 파라미터
    """
    return torchvision.models.efficientnet_b0(num_classes=nOut)