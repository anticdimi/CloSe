from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lib.utils.types import EasierDict, to_torch
from lib.utils.misc import load_json, load_npz, worker_init_fn

from functools import partial
from torch.utils.data import Dataset as TorchDataset


def load_scan_npz(
    scan_id: Union[str, Path],
    scalar_inputs: list[str] = ['points', 'colors', 'normals'],
    filter_bg: bool = False,
) -> EasierDict:
    scan_data = dict(np.load(scan_id, allow_pickle=True))

    faces = scan_data['faces'].astype(np.int32)
    betas = scan_data['betas'][:10].astype(np.float32)
    pose = scan_data['pose'].astype(np.float32)
    trans = scan_data['trans'].astype(np.float32)

    scalar_features = np.hstack([scan_data[k].squeeze().astype(np.float32) for k in scalar_inputs])
    # * If RGB values are in [0, 255] range, normalize them to [0, 1]
    if np.max(scalar_features[:, 3:6]) > 1.0:
        scalar_features[:, 3:6] = scalar_features[:, 3:6] / 255.0

    labels = scan_data['labels'].astype(np.int64) if 'labels' in scan_data.keys() else None
    garments = scan_data['garments'].astype(np.int32)
    canon_pose_coords = scan_data['canon_pose'].astype(np.float32)

    # * Filter out background points; eg. when training the model
    if filter_bg:
        non_bg_points = labels != -1
        labels = labels[non_bg_points]
        canon_pose_coords = canon_pose_coords[non_bg_points]
        scalar_features = scalar_features[non_bg_points]

    if np.max(garments) > 1:
        raise ValueError('Garment labels should be in [0, 1] range')

    return EasierDict(
        points=scalar_features,
        y=labels,
        garments=garments,
        scan_id=str(scan_id),
        canon_pose_coords=canon_pose_coords,
        pose=pose,
        betas=betas,
        trans=trans,
        faces=faces,
    )


class SingleScanDataset(Dataset):
    def __init__(self, scan_path: Union[str, Path], **kwargs) -> None:
        super().__init__(**kwargs)
        self.scan_path = Path(scan_path)
        assert self.scan_path.exists(), f'{self.scan_path} does not exist'

        scan_data = load_scan_npz(self.scan_path)
        # * Shortened for convenience
        self.points = scan_data.points
        self.labels = scan_data.y
        self.garments = scan_data.garments
        self.scan_id = scan_data.scan_id
        self.canon_pose_coords = scan_data.canon_pose_coords
        self.pose = scan_data.pose
        self.betas = scan_data.betas
        self.trans = scan_data.trans
        self.faces = scan_data.faces

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index: int) -> EasierDict:
        return EasierDict(
            points=to_torch(self.points[index]).float(),
            y=to_torch(self.labels[index]).long() if self.labels is not None else None,
            garments=to_torch(self.garments).int(),
            scan_id=str(self.scan_id),
            canon_pose_coords=to_torch(self.canon_pose_coords[index]).float(),
            pose=to_torch(self.pose).float(),
            betas=to_torch(self.betas).float(),
            trans=to_torch(self.trans).float(),
            idx=to_torch(index).long(),
        )

    def get_loader(self, batch_size: int, num_workers: int, **kwargs) -> DataLoader:
        def dummy_collate(batch) -> EasierDict:
            tensors = {
                key: torch.stack([b[key] for b in batch]).unsqueeze(0)
                for key in ['points', 'canon_pose_coords', 'y', 'idx']
            }
            tensors.update(
                {key: batch[0][key].unsqueeze(0) for key in ['garments', 'pose', 'betas', 'trans']}
            )
            tensors.update({'faces': to_torch(self.faces)})
            return EasierDict(tensors)

        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=dummy_collate,
            **kwargs,
        )


# --------------------------------------------------------------------------------------------#
class TorchCloSeDataset(TorchDataset):
    def __init__(
        self,
        mode: str,
        pointcloud_samples: int = 3000,
        data_path: str = 'data/',
        split_file: str = 'data/split_closedi.npz',
        palette: str = 'assets/demo/color_palette.npy',
        batch_size: int = 64,
        num_workers: int = 12,
        input: list = ['points', 'colors'],
        n_classes: int = 18,
        filter_bg: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.data_root = Path(data_path)

        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.filter_bg = filter_bg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_classes = n_classes
        self.pointcloud_samples = pointcloud_samples
        self.split = load_npz(split_file) if '.npz' in split_file else load_json(split_file)
        self.filter_bg = filter_bg
        self.data_list = self.split[mode]
        self.label_colors = palette
        self.inputs = input
        self.scalar_inputs = [inp for inp in self.inputs if 'smpl' not in inp]
        self.data = self.data_list

    def get(self, idx: int) -> EasierDict:
        return self.__getitem__(idx)

    def __getitem__(self, idx: int) -> EasierDict:
        sample = load_scan_npz(
            scan_id=self.data_list[idx],
            scalar_inputs=self.scalar_inputs,
            filter_bg=self.filter_bg,
        )

        indices = np.arange(sample['points'].shape[0])
        if self.pointcloud_samples == -1:
            sample_idxs = np.array(range(sample['points'].shape[0]))
        else:
            sample_idxs = np.random.choice(indices, size=self.pointcloud_samples)

        return EasierDict(
            points=sample['points'][sample_idxs],
            y=sample['y'][sample_idxs],
            garments=sample['garments'],
            scan_id=str(self.data_list[idx]),
            canon_pose_coords=sample['canon_pose_coords'][sample_idxs],
            pose=sample['pose'],
            betas=sample['betas'],
            trans=sample['trans'],
        )

    def __len__(self) -> int:
        return len(self.data_list)

    def get_loader(self, shuffle: bool = True, seed: int = None) -> DataLoader:
        if seed is not None:
            worker_init = partial(worker_init_fn, seed=seed)
        else:
            worker_init = worker_init_fn
        return DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            worker_init_fn=worker_init,
            drop_last=True,
        )

    def get_scan_loader(scan: EasierDict, batch_size: int, drop_last: bool = False) -> DataLoader:
        return SingleScanDataset(scan['scan_id']).get_loader(
            batch_size=batch_size, num_workers=8, drop_last=drop_last
        )
