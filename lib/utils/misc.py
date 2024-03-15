import os
import random
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch

import json
import pickle
import time


def save_image(image: np.ndarray, path: str) -> None:
    from PIL import Image

    Image.fromarray(image.astype(np.uint8)).save(path)


def exists(path: str) -> bool:
    return Path(path).exists()


def mkdir(dir_path: str) -> None:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def fix_seeds(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def labels_to_colors(
    labels: Union[np.ndarray, list],
    colors: Union[np.ndarray, list] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
) -> np.ndarray:
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors)
    return colors[labels]


def format_time(seconds) -> str:
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def get_timestamp() -> str:
    return f'{datetime.now().strftime("%y%m%d-%H%M%S")}'


def load_npz(path: Union[Path, str]) -> dict:
    return dict(np.load(open(path, 'rb'), allow_pickle=True))


def load_json(path: Union[Path, str], mode: str = 'r') -> dict:
    return dict(json.load(open(path, mode)))


def load_pkl(path: Union[Path, str], encoding: str = 'ASCII') -> object:
    return pickle.load(open(path, 'rb'), encoding=encoding)


def worker_init_fn(worker_id, seed=None) -> None:
    if seed is not None:
        np.random.seed(seed)
    else:
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder='big')
        np.random.seed(base_seed + worker_id)
