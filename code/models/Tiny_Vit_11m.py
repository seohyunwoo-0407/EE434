
import timm

def MainModel(nOut=256, **kwargs):
    """
    TinyViT-11M 모델
    
    TinyViT-11M은 약 10.55M 파라미터(FC layer 제외)
    """
    model = timm.create_model('tiny_vit_11m_224', num_classes=nOut, pretrained=False)
    return model

