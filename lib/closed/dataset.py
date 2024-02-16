from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.types import EasierDict, to_torch


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

    labels = scan_data['labels'].astype(np.int64)
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
            y=to_torch(self.labels[index]).long(),
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
