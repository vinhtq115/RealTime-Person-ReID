import torch.nn as nn

from .config import CONFIG
from .texture.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50


__factory = {
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}


class SEMI(nn.Module):
    def __init__(self, app_model: str):
        super(SEMI, self).__init__()

        if __factory.get(app_model) is None:
            raise KeyError(f"Invalid appearance model: '{app_model}'")
        else:
            config = CONFIG()
            self.app_model = __factory[app_model](config)

    def forward(self, clip):
        features = self.app_model(clip)
        return features
