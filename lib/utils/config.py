import torch
from torch import nn

from ..closenet import CloSeNet
from ..utils.types import EasierDict

import torch.optim as optim
import yaml
from pathlib import Path

import logging
import sys
from typing import Union


def load_model(file_path: str, device: str = 'cuda') -> nn.Module:
    model_cfg = EasierDict(
        yaml.load(open(file_path.replace('.pth', '_cfg.yaml'), 'r'), Loader=yaml.FullLoader)
    )

    model = CloSeNet(model_cfg)
    model.load_state_dict(torch.load(file_path, map_location=device))
    return model.to(device)


class CheckpointIO:
    def __init__(
        self,
        checkpoint_dir: Path,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.StepLR,
        cfg: EasierDict,
        check_ckpt: bool = True,
    ) -> None:
        self.module_dict_params = {
            f'{cfg.model}_model': model,
            f'{cfg.optim.name}_optimizer': optimizer,
            f'{cfg.model}_config': cfg,
            f'{cfg.model}_lr_scheduler': lr_scheduler,
        }
        self.ckpt_root = checkpoint_dir
        self.cfg = cfg
        self.latest_ckp = self._check_ckpts() if check_ckpt else None

    def _check_ckpts(self) -> Path:
        ckpts = list(self.ckpt_root.glob('*.pt'))
        if len(ckpts) == 0:
            return None
        else:
            try:
                ckpt = sorted(ckpts, key=lambda x: int(x.stem.split('_')[2]))[-1]
            except:  # noqa: E722
                ckpt = ckpts[0]
            return ckpt

    @staticmethod
    def load_model_config(filename: Path, device: str = 'cuda') -> dict:
        state_dict = torch.load(
            filename,
            map_location=torch.device(device),
        )

        for k in state_dict.keys():
            if '_config' in k:
                return state_dict[k]
        raise ValueError(f'No config found in {filename}')

    def save(self, filename, save_dir=None, **kwargs) -> None:
        out_dict = kwargs
        for k, v in self.module_dict_params.items():
            out_dict[k] = v
            if hasattr(v, 'state_dict'):
                out_dict[k] = v.state_dict()

        if save_dir is not None:
            save_path = save_dir / filename
        else:
            save_path = self.ckpt_root / filename
        torch.save(out_dict, save_path)

    def load_last_checkpoint(self) -> None:
        assert self.latest_ckp is not None, 'Check parameters, cannot find last checkpoint!'
        return self.load(self.latest_ckp)

    def init_training_state(self) -> dict:
        if self.latest_ckp is not None:
            return self.load(self.latest_ckp)
        else:
            return {}

    def load(self, filename: str) -> None:
        state_dict = torch.load(
            filename if filename is not None else self.latest_ckp,
            map_location=torch.device(self.cfg.device),
        )

        self.module_dict_params[f'{self.cfg.model}_model'].load_state_dict(state_dict)


class Logger:
    def __init__(self, out_dir: Union[Path, str], log_stdout: bool = False) -> None:
        self._instance = _mklogger(out_dir, log_stdout)

    def debug(self, msg: str) -> None:
        self._instance.debug(msg)

    def info(self, msg: str) -> None:
        self._instance.info(msg)

    def create_epoch_msg(self, epoch: int, it: int, loss: float, lr: float) -> str:
        return f'[Epoch {epoch:02d}] it={it:03d}, loss={loss:.8f}, lr={lr}'


def _mklogger(out_dir: Union[Path, str], log_stdout: bool = False) -> logging.Logger:
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    formatter = logging.Formatter('%(asctime)-15s %(msg)s')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(out_dir / 'debug.log')
    logger.addHandler(fh)
    fh.setFormatter(formatter)
    if log_stdout:
        ch = logging.StreamHandler(sys.stdout)
        logger.addHandler(ch)
        ch.setFormatter(formatter)
    return logger
