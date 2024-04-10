import copy
import json
import warnings
from collections import defaultdict
from pathlib import Path
import datetime

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .dataset import TorchCloSeDataset
from lib.utils.config import CheckpointIO
from lib.closenet.model import CloSeNet
from lib.closenet.trainers.closenet_trainer import CloSeNetTrainer
from lib.utils.misc import load_npz, mkdir, fix_seeds
from lib.utils.types import to_easydict

from lib.utils.misc import labels_to_colors
from lib.utils.viz import save_views

from .utils import set_model_grad
import yaml

fix_seeds(42)
warnings.filterwarnings('ignore')


class InferenceWrapper:
    def __init__(self, pretrained_path: str, **kwargs) -> None:
        ckpt_path = Path(pretrained_path)
        self.ckpt_path = ckpt_path
        config_yaml = ckpt_path.parent / f'{ckpt_path.stem}_cfg.yaml'
        self.cfg = to_easydict(yaml.load(open(config_yaml, 'r'), Loader=yaml.FullLoader))
        self.model_name = 'closenet'
        self.cfg.exp_logs_path = str(ckpt_path.parent.parent)
        self.cfg.data.pointcloud_samples = -1
        self.cfg.data.batch_size = 1
        self.model = CloSeNet(self.cfg)
        self.trainer = CloSeNetTrainer(
            model=self.model,
            train_dataset=None,
            val_dataset=None,
            test_dataset=None,
            cfg=self.cfg,
            check_ckpt=False,
        )

        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.ckpt = CheckpointIO(
            checkpoint_dir=ckpt_path.parent,
            model=self.model,
            optimizer=self.optim,
            lr_scheduler=None,
            cfg=self.cfg,
            check_ckpt=False,
        )
        _ = self.ckpt.load(ckpt_path)
        # Fix seeds
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.check_grads(self.old_model)

    def save_model(self, save_dir, filename):
        self.ckpt.save(filename, save_dir=save_dir)

    def reset_model(self):
        self.model = copy.deepcopy(self.old_model)
        set_model_grad(model=self.model)
        self.model.eval()

    def check_grads(self, model):
        for param in model.parameters():
            param.requires_grad = False
        assert model.training is False, 'Model is not trainable'

    def refine_weights(
        self,
        data_path,
        indices,
        tgt=None,
        changed_indices=[],
        mode='segm_dec',
        batched_backprop=True,
        steps=3,
    ) -> None:
        print('Refining weights')
        print('No of Indices:', len(indices))

        set_model_grad(mode=mode, model=self.model)

        if batched_backprop:
            _ = self.infer_refine_sequential(
                data_path, tgt=tgt, changed_indices=changed_indices, steps=steps
            )
        else:
            _ = self.infer_refine_sequential(
                data_path,
                tgt=tgt,
                changed_indices=changed_indices,
                steps=steps,
                batch_size=len(indices),
            )

    def infer(self, scan_path: str, scan_dataloader=None, scan=None) -> dict:
        outp_dict = defaultdict()
        eval_outp_dict = defaultdict(list)
        if scan is None:
            scan = TorchCloSeDataset.extract_sample(scan_path, filter_bg=False)
        if scan_dataloader is None:
            scan_dataloader = TorchCloSeDataset.get_scan_loader(
                scan, batch_size=2048, drop_last=False
            )
        # Run regular validation inference
        self.model.eval()
        for val_batch in tqdm(scan_dataloader):
            _, outp = self.trainer.val_step(val_batch, skip_loss=True)
            for k, v in outp.items():
                eval_outp_dict[k].append(v.cpu().detach().numpy())

            for k, v in eval_outp_dict.items():
                if v[0] is not None and k in ['idx', 'pred_labels', 'pred_logits', 'tgts']:
                    if k in ['idx']:
                        outp_dict[k] = np.concatenate(v, axis=0).astype(np.int32).squeeze()
                    elif k in ['pred_logits']:
                        outp_dict[k] = np.concatenate(v, axis=2)
                    else:
                        outp_dict[k] = np.concatenate(v, axis=1).squeeze()
        self.outp_dict = outp_dict
        return outp_dict

    def infer_refine_sequential(
        self,
        scan_path: str,
        tgt,
        changed_indices=[],
        steps=3,
        lamda_changed=0.0001,
        lamda_weight=10.0,
        lamda_fixed=1,
        lamda_present=1.0,
        batch_size=2048,
    ) -> dict:
        """
        Refine the weights of the model using the scan at scan_path

        Args:
            scan_path: path to the scan
            tgt: target label
            batched_backprop: whether to use batched backpropagation
            indices: indices of the points to refine
            changed_indices: indices of the points that diverge from the target label
            steps: number of epochs to refine the weights
        """
        scan = TorchCloSeDataset.extract_sample(scan_path, filter_bg=False)
        scan['tgts'] = torch.tensor(tgt, dtype=torch.long)
        all_g_idx = np.arange(0, 18, dtype=np.int32)
        present_g_idx = list(np.unique(scan['tgts'].to('cpu').numpy()))
        absent_g_idx = [i for i in all_g_idx if i not in present_g_idx]
        w_c = torch.ones(18, dtype=torch.float32).to(self.cfg.device)
        w_c[present_g_idx] *= lamda_present
        w_c[absent_g_idx] = 1.0

        scan_dataloader = TorchCloSeDataset.get_scan_loader(
            scan, batch_size=batch_size, drop_last=False
        )

        lookup = torch.zeros(len(scan['points']), dtype=torch.bool)
        lookup[changed_indices] = True

        old_weights = [param.data.clone() for param in self.model.parameters()]

        for _ in tqdm(range(steps)):
            loss_step = 0
            step_w_loss = 0

            for val_batch in scan_dataloader:
                tgts = val_batch['tgts'].to(self.cfg.device)
                self.optim.zero_grad()
                outp_dict = defaultdict()
                eval_outp_dict = defaultdict(list)

                if self.cfg.model == 'closenet_deltanet':
                    val_batch.pose = scan.pose
                    val_batch.betas = scan.betas
                    val_batch.trans = scan.trans
                _, outp = self.trainer.grad_enabled_val_step(val_batch, skip_loss=True)

                for k, v in outp.items():
                    eval_outp_dict[k].append(v.to(self.cfg.device))

                for k, v in eval_outp_dict.items():
                    if v[0] is not None and k in ['idx', 'pred_labels', 'pred_logits', 'tgts']:
                        if k in ['idx']:
                            outp_dict[k] = torch.cat(v, axis=0).squeeze()
                        elif k in ['pred_logits']:
                            outp_dict[k] = torch.cat(v, axis=2)
                        else:
                            outp_dict[k] = torch.cat(v, axis=1).squeeze()
                preds = outp_dict['pred_logits']
                val_batch_indices = val_batch['idx'].squeeze().to(torch.long)
                changed_idx = torch.where(lookup[val_batch_indices] == True)[0]
                fixed_idx = torch.where(lookup[val_batch_indices] == False)[0]
                assert len(changed_idx) + len(fixed_idx) == len(val_batch_indices)
                if len(changed_indices) > 0:
                    loss_changed = F.cross_entropy(
                        preds[:, :, changed_idx], tgts[:, changed_idx], weight=w_c
                    )
                    loss_fixed = F.cross_entropy(
                        preds[:, :, fixed_idx], tgts[:, fixed_idx], weight=w_c
                    )
                    base_loss = lamda_changed * loss_changed + lamda_fixed * loss_fixed
                else:
                    base_loss = F.cross_entropy(
                        preds[:, :, changed_idx], tgts[:, changed_idx], weight=w_c
                    ) + F.cross_entropy(preds[:, :, fixed_idx], tgts[:, fixed_idx], weight=w_c)
                loss = base_loss
                # Penalize the weights that are far from the initial weights
                w_loss = 0
                for param, data2 in zip(self.model.parameters(), old_weights):
                    w_loss += torch.mean(torch.abs(param.data - data2.detach()))
                w_loss = w_loss / len(old_weights)
                loss = base_loss + lamda_weight * w_loss
                loss.backward()
                step_w_loss += w_loss.item()
                loss_step += loss.item()
                self.optim.step()
            print(loss_step / len(scan_dataloader))
            print(step_w_loss / len(scan_dataloader))

    def evaluate(self, render=False, n_points=2048, split='val', palette=None):
        """
        Evaluate the trained model on the split set
        The model should be provided with a split_file.npz which contains the evaluation split indices
        Inputs:
        render: whether to render during evaluation
        n_points: number of points to evaluate for each scan
        split: the split to evaluate on
        palette_path: path to the color palette
        """
        cfg = copy.deepcopy(self.cfg)
        cfg.exp_logs_path = str(self.ckpt_path.parent.parent)
        cfg.data.pointcloud_samples = n_points
        cfg.data.batch_size = 1
        cfg.data.split_file = (
            'data/split_partial.npz'  # the split file should be in the data folder
        )
        cfg.exp_name = split

        render_device = cfg.device

        eval_dataset = TorchCloSeDataset(
            mode=split,
            split_file=cfg.data.split_file,
            palette=palette,
            pointcloud_samples=cfg.data.pointcloud_samples,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            n_classes=cfg.data.n_classes,
            input=cfg.data.input,
        )

        if split == 'val':
            self.trainer.val_dataset = eval_dataset
        elif split == 'test':
            self.trainer.test_dataset = eval_dataset
        eval_dict, outp_dict = self.trainer.evaluate_model(split, debug=False, seed=42)

        points = outp_dict['points']
        pred_labels = outp_dict['pred_labels']
        tgts = outp_dict['tgts']
        scan_ids = [s[0] for s in outp_dict['scan_id']]

        print(f'{eval_dict["IoU"]=}')
        print(f'{eval_dict["mIoU"]=}')
        print(f'{eval_dict["freq_IoU"]=}')

        logits = (
            outp_dict['pred_logits'].permute(0, 2, 1).contiguous()
            if 'closenet' not in cfg.model
            else outp_dict['pred_logits']
        )

        save_dict = {
            'points': points.cpu().numpy(),
            'pred_labels': pred_labels.cpu().numpy(),
            'logits': logits.cpu().numpy(),
            'scan_ids': scan_ids,
            'tgts': tgts.cpu().numpy(),
            'iou': eval_dict['IoU'],
            'miou': eval_dict['mIoU'],
            'freq_iou': eval_dict['freq_IoU'],
            'dataset': split,
        }

        save_path = mkdir(self.ckpt_path.parent.parent / split)
        mkdir(save_path.parent / 'eval_metrics')
        json.dump(
            eval_dict,
            open(
                save_path.parent
                / 'eval_metrics'
                / f'{datetime.datetime.now()}_{split}_{n_points}.json',
                'w',
            ),
            indent=4,
        )
        # np.savez(save_path.parent / f'{split}_{n_points}.npz', **save_dict)
        if render:
            save_img = save_path / f'{n_points}'
            mkdir(save_img)
            for i in tqdm(range(points.shape[0]), desc='Rendering'):
                npz = load_npz(scan_ids[i])
                save_views(
                    points[i, :, :3],
                    npz['faces'],
                    points[i, :, 3:6],
                    render_device,
                    [0, 90, 180, 270],
                    'pointcloud',
                    labels_to_colors(save_dict['pred_labels'][i], palette),
                    labels_to_colors(save_dict['tgts'][i], palette),
                    save_path=save_img,
                    rank=i,
                    scan_id=scan_ids[i].split('/')[-1],
                )
        return eval_dict, outp_dict
