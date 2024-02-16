import os
import random
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch


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


def get_timestamp():
    return f'{datetime.now().strftime("%y%m%d-%H%M%S")}'
