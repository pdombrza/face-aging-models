import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    # sinusoidal position encoding
    # https://arxiv.org/pdf/1706.03762 [3.5]
    def __init__(self, emb_dim: int = 256):
        self.emb_dim = emb_dim
        if emb_dim % 2 != 0:
            raise ValueError("Positional embedding dimension must be divisible by 2.")

    def forward(self, timestep):
        half_dim = self.emb_dim / 2
        device = timestep.device
        factor = 2 * torch.arange(0, half_dim, device=device) / half_dim
        denominator = torch.pow(10000, factor)
        embedding = timestep[:, None]
        embedding = embedding / denominator
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        return embedding


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int = 256):
        super(ResnetBlock, self).__init__()
        self.feature_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        )

        self.time_block = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(t_emb_dim, out_channels),
        )

        self.time_feature_block = nn.Sequential( # from SD
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x, time):
        residue = x
        x = self.feature_block
        time = self.time_block(time)
        time_feature = x + time.unsqueeze(-1).unsqueeze(-1) # need to check dimensions cause unsure but likely time has shape B x out_ch, and x has B x C x H x W
        time_feature = self.time_feature_block(time_feature)
        time_feature += residue
        return time_feature


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, t_emb_dim: int = 320):
        super(UNet, self).__init__()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        ...
