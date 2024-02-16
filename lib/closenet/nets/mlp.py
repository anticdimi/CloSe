import torch
from torch import nn


class MLPDecoder(nn.Module):
    def __init__(self, channels: list[int], dropout: float, slope: float) -> None:
        super().__init__()
        self.channels = channels

        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels[i - 1],
                        channels[i],
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels[i]),
                    nn.LeakyReLU(slope),
                    nn.Dropout(dropout) if i != len(channels) - 1 else nn.Identity(),
                )
            )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
