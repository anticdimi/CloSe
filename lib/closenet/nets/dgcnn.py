import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ...utils.types import EasierDict


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    batch_size = x.size(0)
    num_points = x.size(2)
    # Fix for the case when there is less points than needed for knn graph
    k = min(k, num_points)
    x = x.view(batch_size, -1, num_points)
    idx = _knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


# --------------------------------------------------------------------------------------------#
class TransformNet(nn.Module):
    def __init__(self) -> None:
        super(TransformNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(
            self.bn3(self.linear1(x)), negative_slope=0.2
        )  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(
            self.bn4(self.linear2(x)), negative_slope=0.2
        )  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class EdgeConv(nn.Module):
    def __init__(self, channels: list[int], k: int) -> None:
        super(EdgeConv, self).__init__()
        self.k = k
        modules = list()
        for i in range(1, len(channels)):
            in_ch = channels[i - 1]
            out_ch = channels[i]
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
        self.s_mlp = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_graph = get_graph_feature(x, self.k)
        out = self.s_mlp(x_graph)
        out = out.max(dim=-1, keepdim=False)[0]

        return out


class DGCNNBase(nn.Module):
    def __init__(self, inp_dim: int, emb_dim: int, k: int, use_tnet: bool) -> None:
        super(DGCNNBase, self).__init__()
        self.emb_dim = emb_dim
        self.k = k
        self.inp_dim = inp_dim

        self.transform_net = TransformNet() if use_tnet else None

        # Hardcoded for now
        # inp_dim = self.inp_dim if not use_tnet else 2 * self.inp_dim
        self.channels = [[max(self.inp_dim * 2, 12), 64, 64], [64 * 2, 64, 64], [64 * 2, 64]]
        self.channels_sum = 64 * 3
        self.convs = torch.nn.ModuleList()
        for ch_config in self.channels:
            self.convs.append(EdgeConv(ch_config, k))

        self.lin_global = nn.Sequential(
            nn.Conv1d(192, self.emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, data: EasierDict) -> tuple[torch.Tensor, list]:
        x = data.points
        x = x.float()
        if x.size(2) > 3:
            features = x[:, :, 3:].permute(0, 2, 1).contiguous()
            x = x[:, :, :3]
        else:
            features = x.permute(0, 2, 1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        num_points = x.size(2)

        if self.transform_net:
            x0 = get_graph_feature(
                x, k=self.k
            )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            t = self.transform_net(x0)  # (batch_size, 3, 3)
            x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
            x = torch.bmm(
                x, t
            )  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
            x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = torch.concat((x, features), axis=1)
        out = []
        for conv in self.convs:
            x = conv(x)
            out.append(x)

        x = torch.cat(out[-len(self.channels) :], dim=1)
        global_enc = self.lin_global(x).max(dim=-1, keepdim=True)[0]
        x_max = global_enc.repeat(1, 1, num_points)

        return x_max, out, data
