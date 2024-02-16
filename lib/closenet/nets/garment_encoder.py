import torch
from torch import nn
from .mlp import MLPDecoder
from ...utils.types import EasierDict


class AddNorm(nn.Module):
    def __init__(self, normalized_shape: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.ln(self.dropout(Y) + X)


class AttentionGarmentEncoder(nn.Module):
    def __init__(
        self,
        cfg,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_emb = True
        self.emb_dim = cfg.gar_emb_dim
        self.key_mask = cfg.key_mask
        self.garm_embedding = nn.Embedding(
            num_embeddings=cfg.n_embeddings,
            embedding_dim=cfg.gar_emb_dim,
            max_norm=cfg.max_norm,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.emb_dim,
            num_heads=cfg.n_heads,
            batch_first=True,
            kdim=self.emb_dim,
            vdim=self.emb_dim,
            dropout=cfg.dropout,
        )
        self.ln = AddNorm(cfg.emb_dim, cfg.dropout)
        self.ff = MLPDecoder([cfg.emb_dim, cfg.emb_dim], cfg.dropout, 0.2)
        self.norm = nn.LayerNorm(cfg.emb_dim)

    def forward(self, data: EasierDict, pc_features: torch.Tensor, **kwargs) -> torch.Tensor:
        k = self.garm_embedding.weight.repeat(data.points.shape[0], 1, 1)
        v = k.clone()

        q = pc_features[-1].permute(0, 2, 1).contiguous()

        mask = ~data.garments.bool()

        attn_outp, attn_weights = self.attn(
            q,
            k,
            v,
            key_padding_mask=mask,
        )

        proj = self.ff(attn_outp.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        attn_outp = self.ln(q, proj)

        return self.norm(attn_outp), attn_weights
