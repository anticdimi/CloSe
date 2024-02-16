import torch
from torch.nn import functional as F

from ..utils.types import EasierDict
from .nets import AttentionGarmentEncoder, CanonEncoder, DGCNNBase, MLPDecoder


class CloSeNet(torch.nn.Module):
    def __init__(self, cfg: EasierDict) -> None:
        super(CloSeNet, self).__init__()
        # * Shortened
        model_cfg = cfg.model_arch
        pc_enc_cfg = model_cfg.pc_enc
        garm_enc_cfg = model_cfg.garm_enc
        segm_dec_cfg = model_cfg.segm_dec
        n_classes = 18

        # * Setup encoders
        self.pc_enc = DGCNNBase(
            inp_dim=pc_enc_cfg.inp_dim,
            emb_dim=pc_enc_cfg.emb_dim,
            k=pc_enc_cfg.k,
            use_tnet=pc_enc_cfg.use_tnet,
        )
        self.part_enc = CanonEncoder()
        self.garm_enc = AttentionGarmentEncoder(garm_enc_cfg)

        # * Setup MLP decoder
        segm_input = (
            self.pc_enc.emb_dim
            + self.pc_enc.channels_sum
            + self.part_enc.emb_dim
            + self.garm_enc.emb_dim
        )
        self.segm_dec = MLPDecoder(
            channels=[segm_input] + segm_dec_cfg.channels + [n_classes],
            dropout=segm_dec_cfg.dropout,
            slope=segm_dec_cfg.slope,
        )

    def _encode(self, data: EasierDict, **kwargs) -> EasierDict:
        x_max, conv_out, data = self.pc_enc(data, **kwargs)
        # * Get per-point features
        pc_features = (
            torch.cat([x_max] + conv_out, dim=1).permute(0, 2, 1).contiguous()
        )  # B, n_samples, K

        # * Get part features
        part_features = self.part_enc(data, **kwargs)

        # * Get garment features
        garm_features, attn_weights = self.garm_enc(data, [x_max] + conv_out, **kwargs)

        return EasierDict(
            pc_features=pc_features,
            part_features=part_features,
            garm_features=garm_features,
            attn_weights=attn_weights,
        )

    def _decode(self, data: EasierDict, **kwargs) -> torch.Tensor:
        encodings = (
            torch.cat([data.pc_features, data.part_features, data.garm_features], dim=-1)
            .permute(0, 2, 1)
            .contiguous()
        )
        return self.segm_dec(encodings, **kwargs)

    def forward(self, data: EasierDict, **kwargs) -> EasierDict:
        encodings = self._encode(data, **kwargs)
        logits = self._decode(encodings, **kwargs)
        return EasierDict(
            **data,
            logits=logits,
            labels=F.softmax(logits, dim=1).argmax(dim=1),
            encodings=encodings,
        )
