import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    # sinusoidal position encoding
    # https://arxiv.org/pdf/1706.03762 [3.5]
    def __init__(self, emb_dim: int = 256):
        super(SinusoidalPositionalEmbedding, self).__init__()
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


class NormActConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(NormActConv, self).__init__()
        self.norm_act_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.norm_act_conv(x)


class NormActConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(NormActConvTranspose, self).__init__()
        self.norm_act_convtranspose = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.norm_act_convtranspose(x)


class UNet(nn.Module):
    # https://arxiv.org/pdf/2104.05358 [Figure 4.],
    def __init__(self, in_channels: int = 3, t_emb_dim: int = 256):
        super(UNet, self).__init__()
        self.time_emb_mlp_layer = nn.Sequential(
            SinusoidalPositionalEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.ReLU(inplace=True)
        )
        #TODO refactor this shit
        self.down_ch = nn.ModuleList([
            NormActConv(in_channels=in_channels, out_channels=64),
            ResnetBlock(in_channels=64, out_channels=64, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=64, out_channels=128, stride=2),
            ResnetBlock(in_channels=128, out_channels=128, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=128, out_channels=256, stride=2),
            ResnetBlock(in_channels=256, out_channels=256, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=256, out_channels=512, stride=2),
            ResnetBlock(in_channels=512, out_channels=512, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=512, out_channels=512),
            ResnetBlock(in_channels=512, out_channels=512, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=512, out_channels=512, stride=2),
        ])

        self.mid_ch = nn.ModuleList([
            ResnetBlock(in_channels=512, out_channels=512, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=512, out_channels=512),

        ])

        self.up_ch = nn.ModuleList([
            ResnetBlock(in_channels=512, out_channels=512, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=512, out_channels=512),
            ResnetBlock(in_channels=1024, out_channels=1024, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=1024, out_channels=1024),
            NormActConvTranspose(in_channels=1024, out_channels=512, stride=2),
            ResnetBlock(in_channels=1024, out_channels=1024, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=1024, out_channels=1024),
            NormActConvTranspose(in_channels=1024, out_channels=256, stride=2),
            ResnetBlock(in_channels=512, out_channels=512, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=512, out_channels=512),
            NormActConvTranspose(in_channels=512, out_channels=128, stride=2),
            ResnetBlock(in_channels=256, out_channels=256, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=256, out_channels=256),
            NormActConvTranspose(in_channels=256, out_channels=64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        ])


    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        ...
