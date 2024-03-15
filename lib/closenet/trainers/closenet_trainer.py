from collections import defaultdict

import torch

from lib.utils.metrics import cross_entropy, labels_from_logits
from lib.utils.types import EasierDict
from lib.closenet.trainers.base_trainer import BaseTrainer
from typing import Any


class CloSeNetTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        cfg: EasierDict,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            model,
            train_dataset,
            val_dataset,
            test_dataset,
            cfg,
            cfg.training.loss_weights,
            **kwargs,
        )

        self.training_cfg = cfg.training

        def _optim_params(x) -> dict[str, Any]:
            return {
                'lr': x.get('lr', cfg.optim.lr),
                'weight_decay': x.get('weight_decay', cfg.optim.weight_decay),
            }

        self.optimizer = torch.optim.Adam(
            [
                {
                    'name': 'pc_params',
                    'params': model.pc_enc.parameters() if model.pc_enc is not None else [],
                    **_optim_params(cfg.model_arch['pc_enc']),
                },
                {
                    'name': 'garm_params',
                    'params': model.garm_enc.parameters() if model.garm_enc is not None else [],
                    **_optim_params(cfg.model_arch['garm_enc']),
                },
                {
                    'name': 'part_params',
                    'params': model.part_enc.parameters() if model.part_enc is not None else [],
                    **_optim_params(cfg.model_arch['part_enc']),
                },
                {
                    'name': 'dec_params',
                    'params': model.segm_dec.parameters() if model.segm_dec is not None else [],
                    **_optim_params(cfg.model_arch['segm_dec']),
                },
            ]
        )

    def _pack_data(
        self,
        batch: EasierDict,
        outp: dict,
    ) -> dict:
        ret_dict = {
            'points': batch.points.cpu() if hasattr(batch, 'points') else batch.x.cpu(),
            'pred_labels': labels_from_logits(outp['logits']).cpu(),
            'tgts': batch.y.cpu(),
            'pred_logits': outp['logits'].cpu(),
        }
        if outp.get('attn_weights', None) is not None:
            ret_dict['attn_weights'] = outp['attn_weights'].cpu()
        if hasattr(batch, 'idx'):
            ret_dict['idx'] = batch.idx.cpu()
        if hasattr(batch, 'scan_id'):
            ret_dict['scan_id'] = batch.scan_id
        return ret_dict

    def step(self, batch: EasierDict, skip_loss: bool = False, **kwargs: dict) -> dict:
        batch = batch.to(self.device)

        outp_dict = self.model(batch)

        loss_dict = defaultdict()
        if not skip_loss:
            loss_dict['segm_loss'] = cross_entropy(outp_dict['logits'], batch.y, smoothing=False)

            loss_dict['total_loss'] = sum(
                [self.loss_weights[k] * loss_dict[k] for k in self.loss_weights.keys()]
            )

        return loss_dict, self._pack_data(batch, outp_dict)
