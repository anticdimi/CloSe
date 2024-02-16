from typing import Union

import numpy as np
import torch


def to_torch(array: Union[np.ndarray, list]) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        return torch.tensor(array)


# ! Adapted from EasyDict: https://github.com/makinacorpus/easydict
METHOD_KEYS = ['update', 'pop', 'to', 'to_dict', 'detach']


class EasierDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and k not in METHOD_KEYS:
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, EasierDict):
            value = EasierDict(value)
        super(EasierDict, self).__setattr__(name, value)
        super(EasierDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasierDict, self).pop(k, d)

    def detach(self) -> 'EasierDict':
        for key in self:
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].detach()
        return self

    def to(self, device: Union[str, torch.device]) -> 'EasierDict':
        for key in self:
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].to(device)
        return self

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.items():
            if isinstance(v, EasierDict):
                d[k] = v.to_dict()
            elif isinstance(v, (list, tuple)):
                d[k] = type(v)(EasierDict(x).to_dict() if isinstance(x, dict) else x for x in v)
            else:
                d[k] = v
        return d


def to_easydict(d: dict) -> EasierDict:
    if isinstance(d, EasierDict):
        return d
    elif isinstance(d, dict):
        return EasierDict(d)
    else:
        raise ValueError(f'Cannot convert {type(d)} to EasyDict')
