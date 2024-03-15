import json
import time
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import tensorboardX as tbx
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from lib.utils.config import CheckpointIO, Logger
from lib.utils.metrics import IoU, frequency_weighted_IoU
from lib.utils.misc import format_time, mkdir
from lib.utils.types import EasierDict


class BaseTrainer(object):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        cfg: EasierDict,
        loss_weights: dict,
        **kwargs: dict,
    ) -> None:
        device = torch.device(
            cfg.device if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        )

        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.loss_weights = loss_weights

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg.optim.milestones,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.exp_path = mkdir(cfg.exp_logs_path)

        self.checkpoint_path = mkdir(self.exp_path / 'checkpoints')

        self.tb_logger = tbx.SummaryWriter(self.exp_path / 'summary')
        self.logger = Logger(self.exp_path, cfg.debug)

        self.ckpt = CheckpointIO(
            self.checkpoint_path,
            self.model,
            self.optimizer,
            self.lr_scheduler,
            cfg,
            **kwargs,
        )
        load_dict = self.ckpt.init_training_state()
        self.init_state = EasierDict(
            it=load_dict.get('it', 0),
            epoch_it=load_dict.get('epoch_it', 0),
            metric_val_best=load_dict.get(
                'metric_val_best',
                (1 if self.cfg.training.selection_mode == 'maximize' else -1) * (-np.inf),
            ),
        )

    def train_step(self, batch: EasierDict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict, outp_dict = self.step(batch)

        loss = loss_dict.get('total_loss')
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return {'loss_dict': loss_dict, 'outp_dict': outp_dict}

    def save_checkpoint(
        self,
        model_name: str,
        **kwargs: dict,
    ) -> None:
        assert model_name.split('.')[-1] == 'pt'
        self.ckpt.save(
            model_name,
            **kwargs,
        )

    def train_model(self) -> None:
        loss = 0

        epoch_it = self.init_state.epoch_it
        it = self.init_state.it
        print_it = self.cfg.training.print_it
        val_it = self.cfg.training.val_it
        backup_it = self.cfg.training.backup_it

        selection_metric = self.cfg.training.selection_metric
        selection_sign = 1 if self.cfg.training.selection_mode == 'maximize' else -1
        max_iters = self.cfg.training.max_iters
        max_epochs = self.cfg.training.max_epochs
        best_metric = {k: -selection_sign * np.inf for k in selection_metric.split(',')}

        self.logger.debug(f'Config:\n{json.dumps(self.cfg, indent=2)}')
        self.logger.debug(f'Init state:\t{json.dumps(self.init_state)}')
        start_time = time.time()

        while True:
            epoch_it += 1
            sum_loss = 0
            train_data_loader = self.train_dataset.get_loader()

            epoch_losses = defaultdict(list)
            train_bar = tqdm(train_data_loader)
            for batch in train_bar:
                it += 1
                train_dict = self.train_step(batch)
                loss_dict = train_dict['loss_dict']

                loss = loss_dict['total_loss']
                sum_loss += loss

                for k, v in loss_dict.items():
                    epoch_losses[k].append(v.item())

                log_msg = self.logger.create_epoch_msg(
                    epoch_it, it, loss, self.lr_scheduler.get_lr()[0]
                )
                train_bar.set_description(log_msg)
                # Print output
                if print_it > 0 and (it % print_it) == 0:
                    self.logger.debug(log_msg)

                # Backup if needed
                if backup_it > 0 and (it % backup_it) == 0:
                    self.save_checkpoint(
                        f'model_{epoch_it:d}_{it:d}.pt',
                        epoch_it=epoch_it,
                        it=it,
                        **best_metric,
                        selection_metric=selection_metric,
                    )

                # Run validation
                if val_it > 0 and (it % val_it) == 0:
                    val_dict, _ = self.compute_val_loss(self.val_dataset.get_loader(shuffle=False))
                    ckpt_name = f'valmin_{epoch_it:d}_{it:d}_val={val_dict["total_loss"]:.3f}'
                    ckpt_name += ''.join(
                        [f'_{k}={val_dict[k]:.3f}' for k in selection_metric.split(',')]
                    )

                    for sel_metric in selection_metric.split(','):
                        metric_val = val_dict[sel_metric]

                        if selection_sign * (metric_val - best_metric[sel_metric]) > 0:
                            best_metric[sel_metric] = metric_val
                            self.logger.debug(
                                f'New Best validation metric ({sel_metric}): {metric_val:.5f};'
                            )

                            self.save_checkpoint(
                                f'{ckpt_name}.pt',
                                epoch_it=epoch_it,
                                it=it,
                                overwrite=True,
                                **best_metric,
                            )
                        else:
                            self.logger.debug(
                                f'Validation metric ({sel_metric}): {metric_val:.5f};'
                            )

                    for k, v in val_dict.items():
                        if isinstance(v, list):
                            for i in range(len(v)):
                                self.tb_logger.add_scalar(f'val/{k}/{i}', v[i], it)
                        else:
                            self.tb_logger.add_scalar(f'val/{k}', v, it)

                if (0 < max_iters <= it) or (0 < max_epochs <= epoch_it):
                    self.logger.debug(
                        f'Maximum iteration/epochs ({it=}/{epoch_it=}) reached. Exiting.'
                    )
                    self.save_checkpoint(
                        f'model_{epoch_it:d}_{it:d}.pt',
                        epoch_it=epoch_it,
                        it=it,
                        **best_metric,
                    )

                    elapsed_time = time.time() - start_time
                    self.logger.debug(f'Elapsed time {format_time(elapsed_time)}')
                    return

            for k, v in epoch_losses.items():
                self.tb_logger.add_scalar(f'train/{k}', np.mean(v), it)

    def evaluate_model(
        self, dataset: str = 'val', num_batches: Optional[int] = None, **kwargs
    ) -> tuple[dict, dict]:
        dataloader = getattr(self, f'{dataset}_dataset').get_loader(
            shuffle=False, seed=kwargs.get('seed')
        )

        num_batches = num_batches or len(dataloader)
        return self.compute_val_loss(dataloader, **kwargs)

    @torch.no_grad()
    def val_step(self, data: dict, **kwargs) -> tuple[dict, dict]:
        self.model.eval()

        eval_loss_dict, outp = self.step(data, **kwargs)
        return {k: v.item() for k, v in eval_loss_dict.items()}, outp

    def step(self, *args, **kwargs) -> tuple[dict, dict]:
        raise NotImplementedError('Method step not implemented!')

    def grad_enabled_val_step(self, data: Union[EasierDict, Batch], **kwargs) -> tuple[dict, dict]:
        eval_loss_dict, outp = self.step(data, **kwargs)

        return {k: v.item() for k, v in eval_loss_dict.items()}, outp

    def compute_val_loss(
        self, dataloader: torch.utils.data.DataLoader, **kwargs
    ) -> tuple[dict, dict]:
        val_list, outps = defaultdict(list), defaultdict(list)
        for val_batch in tqdm(dataloader, desc='Running validation'):
            val_step_dict, outp = self.val_step(val_batch, **kwargs)
            for k, v in outp.items():
                outps[k].append(v)

            for k, v in val_step_dict.items():
                val_list[k].append(v)

        val_dict = {k: np.mean(v) for k, v in val_list.items()}
        outp_dict = {
            k: torch.vstack(v).squeeze() if not isinstance(v[0], list) else v
            for k, v in outps.items()
        }

        labels = outp_dict['pred_labels'].squeeze()
        tgts = outp_dict['tgts'].squeeze()
        val_dict['IoU'], val_dict['mIoU'] = IoU(labels, tgts, num_classes=self.cfg.data.n_classes)
        val_dict['freq_IoU'] = frequency_weighted_IoU(
            labels, tgts, num_classes=self.cfg.data.n_classes
        )

        return val_dict, outp_dict
