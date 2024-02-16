import torch
import yaml
from torch import nn

from ..closenet import CloSeNet
from ..utils.types import EasierDict


def load_model(file_path: str, device: str = 'cuda') -> nn.Module:
    model_cfg = EasierDict(
        yaml.load(open(file_path.replace('.pth', '_cfg.yaml'), 'r'), Loader=yaml.FullLoader)
    )

    model = CloSeNet(model_cfg)
    model.load_state_dict(torch.load(file_path, map_location=device))
    return model.to(device)
