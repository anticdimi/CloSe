import torch

from ...utils.types import EasierDict


class CanonEncoder(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.emb_dim = 3

    def forward(self, batch: EasierDict) -> torch.Tensor:
        return batch.canon_pose_coords
