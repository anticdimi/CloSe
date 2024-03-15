from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.utils.types import EasierDict, to_torch
from lib.utils.misc import load_npz, worker_init_fn

from functools import partial
from typing import Any
from torch.utils.data import Dataset as TorchDataset


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
        prep_suffix: str = '',
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
        self.split = load_npz(split_file)
        self.filter_bg = filter_bg
        self.data_list = self.split[mode]
        self.label_colors = palette
        self.inputs = input
        self.scalar_inputs = [inp for inp in self.inputs if 'smpl' not in inp]

        self.prep_suffix = prep_suffix
        self.data = self.data_list

    @staticmethod
    def extract_sample(
        scan_id,
        scalar_inputs=['points', 'colors', 'normals'],
        filter_bg=True,
        n_classes=18,
    ):
        scan_data = np.load(scan_id)

        betas = (
            scan_data['betas'][:10].astype(np.float32)
            if 'betas' in scan_data.keys()
            else np.zeros(10)
        )
        pose = scan_data['pose'].astype(np.float32) if 'pose' in scan_data.keys() else None
        trans = scan_data['trans'].astype(np.float32) if 'trans' in scan_data.keys() else None

        scalar_features = np.hstack(
            [scan_data[k].squeeze().astype(np.float32) for k in scalar_inputs]
        )
        if np.max(scalar_features[:, 3:6]) > 1.0:
            scalar_features[:, 3:6] = scalar_features[:, 3:6] / 255.0

        labels = scan_data['labels'].astype(np.int64) if 'labels' in scan_data.keys() else None
        garments = scan_data['garments'].astype(np.int32)
        canon_pose_coords = scan_data['canon_pose'].astype(np.float32)
        if filter_bg:
            bg_points = labels != -1
            labels = labels[bg_points]
            canon_pose_coords = canon_pose_coords[bg_points]
            scalar_features = scalar_features[bg_points]

        if np.max(garments) > 1:
            new_garments = torch.zeros(n_classes)
            unique_lbls = np.unique(labels)
            garments = garments[garments >= 0]
            # hack for public dataset
            if np.any(garments != unique_lbls) and len(unique_lbls) != 1:
                garments = np.unique(labels)
            new_garments[garments[garments >= 0]] = 1
            garments = new_garments

        sample = {
            'points': scalar_features,
            'y': labels,
            'garments': garments,
            'scan_id': str(scan_id),
            'canon_pose_coords': canon_pose_coords,
            'pose': pose,
            'betas': betas,
            'trans': trans,
        }
        return sample

    def get_by_scan_id(self, scan_id: str) -> EasierDict:
        assert scan_id in self.data_list, f'{scan_id} not found in the list!'
        d = {self.data_list[idx]: idx for idx in range(len(self.data_list))}
        return self.get(d[scan_id])

    def get(self, idx: int) -> EasierDict:
        return self.__getitem__(idx)

    def __getitem__(self, idx: int) -> EasierDict:
        sample = TorchCloSeDataset.extract_sample(
            scan_id=self.data_list[idx],
            scalar_inputs=self.scalar_inputs,
            bp_name=self.bp_name,
            filter_bg=self.filter_bg,
            n_classes=self.n_classes,
        )

        indices = np.arange(sample['points'].shape[0])
        if self.pointcloud_samples == -1:
            sample_idxs = np.array(range(sample['points'].shape[0]))
        else:
            sample_idxs = np.random.choice(indices, size=self.pointcloud_samples)

        ret_sample = {
            'points': sample['points'][sample_idxs],
            'y': sample['y'][sample_idxs],
            'garments': sample['garments'],
            'scan_id': str(self.data_list[idx]),
            'canon_pose_coords': sample['canon_pose_coords'][sample_idxs],
            'pose': sample['pose'],
            'betas': sample['betas'],
            'trans': sample['trans'],
        }
        return EasierDict(ret_sample)

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

    def get_scan_loader(scan: dict, batch_size: int, drop_last: bool = False) -> DataLoader:
        class DummyScanDataset(TorchDataset):
            def __init__(
                self,
                points,
                y,
                pose=None,
                trans=None,
                betas=None,
                garments=None,
                canon_pose_coords=None,
                tgts=None,
                **kwargs,
            ) -> None:
                super().__init__()
                self.points = points.squeeze()
                self.y = y.squeeze()
                self.garments = garments
                self.tgts = tgts

                self.pose = pose
                self.trans = trans
                self.betas = betas

                if pose is not None:
                    self.pose = pose.squeeze()
                if trans is not None:
                    self.trans = trans.squeeze()
                if betas is not None:
                    self.betas = betas.squeeze()
                if canon_pose_coords is not None:
                    self.canon_pose_coords = canon_pose_coords.squeeze()

            def __len__(self) -> int:
                return self.points.shape[0]

            def __getitem__(self, index: int) -> dict[str, Any]:
                if self.tgts is None:
                    return {
                        'points': self.points[index],
                        'y': self.y[index],
                        'trans': self.trans,
                        'pose': self.pose,
                        'betas': self.betas,
                        'idx': index,
                        'garments': self.garments,
                        'canon_pose_coords': self.canon_pose_coords[index],
                    }
                else:
                    return {
                        'points': self.points[index],
                        'y': self.y[index],
                        'trans': self.trans,
                        'pose': self.pose,
                        'betas': self.betas,
                        'idx': index,
                        'garments': self.garments,
                        'canon_pose_coords': self.canon_pose_coords[index],
                        'tgts': self.tgts[index],
                    }

        def dummy_collate(batch) -> EasierDict:
            points, y, idx, canon_pose_coords, tgts = [], [], [], [], []
            for b in batch:
                points.append(torch.from_numpy(b['points']).to(torch.float32))
                canon_pose_coords.append(torch.from_numpy(b['canon_pose_coords']).to(torch.float32))
                y.append(torch.tensor(b['y'], dtype=torch.int64))
                idx.append(torch.tensor(b['idx']).to(torch.float32))
                if 'tgts' in b:
                    tgts.append(torch.tensor(b['tgts']).to(torch.long))

            keys = list(batch[0].keys())
            if 'tgts' in keys:
                ret_dict = {
                    'points': torch.stack(points).unsqueeze(0),
                    'canon_pose_coords': torch.stack(canon_pose_coords).unsqueeze(0),
                    'y': torch.stack(y).unsqueeze(0),
                    'idx': torch.stack(idx),
                    'tgts': torch.stack(tgts).unsqueeze(0),
                    'garments': to_torch(batch[0]['garments']).unsqueeze(0),
                }
            else:
                ret_dict = {
                    'points': torch.stack(points).unsqueeze(0),
                    'canon_pose_coords': torch.stack(canon_pose_coords).unsqueeze(0),
                    'y': torch.stack(y).unsqueeze(0),
                    'idx': torch.stack(idx),
                    'garments': to_torch(batch[0]['garments']).unsqueeze(0),
                }
            for k in ['pose', 'betas', 'trans']:
                if k in keys:
                    ret_dict[k] = to_torch(batch[0][k]).unsqueeze(0)
            return EasierDict(ret_dict)

        return DataLoader(
            DummyScanDataset(**{k: scan[k] for k in scan.keys()}),
            batch_size=batch_size,
            num_workers=8,
            shuffle=True,
            worker_init_fn=partial(worker_init_fn, seed=42),
            collate_fn=dummy_collate,
            drop_last=drop_last,
        )
