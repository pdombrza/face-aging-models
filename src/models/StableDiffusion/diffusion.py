import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        self.t_emb = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(4 * emb_dim, 4 * emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t_emb(x)
        return x


class ResidualBlock(nn.Module):
    ...


class AttentionBlock(nn.Module):
    ...


class Upsample(nn.Module):
    def __init__(self, channels: int):
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Upsample(channels, scale_factor=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int = 4, down_ch: int = 320, num_heads: int = 8, emb_dim: int = 40):
        super(UNet, self).__init__()
        self.down_ch = down_ch
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.down_channels = nn.ModuleList([
            nn.Conv2d(in_channels, self.down_ch, kernel_size=3, padding=1),
            ResidualBlock(self.down_ch, self.down_ch),
            AttentionBlock(self.num_heads, self.emb_dim),
            ResidualBlock(self.down_ch, self.down_ch),
            AttentionBlock(self.num_heads, self.emb_dim),
        ])
        self.emb_dim *= 2
        for _ in range(2):
            self.down_channels += [
                nn.Conv2d(self.down_ch, self.down_ch, kernel_size=3, stride=2, padding=1),
                ResidualBlock(self.down_ch, 2 * self.down_ch),
                AttentionBlock(self.num_heads, self.emb_dim),
                ResidualBlock(2 * self.down_ch, 2 * self.down_ch),
                AttentionBlock(self.num_heads, self.emb_dim),
            ]
            self.down_ch *= 2
            self.emb_dim *= 2
        self.down_channels += [
            nn.Conv2d(in_channels, self.down_ch, kernel_size=3, stride=2, padding=1),
            ResidualBlock(self.down_ch, self.down_ch),
            ResidualBlock(self.down_ch, self.down_ch),
        ]
        # B, 1280, H / 64, W / 64, emb_dim = 160

        self.mid_ch = self.down_ch
        self.mid_channels = nn.ModuleList([
            ResidualBlock(self.mid_ch, self.mid_ch),
            AttentionBlock(self.num_heads, self.emb_dim),
            ResidualBlock(self.mid_ch, self.mid_ch),
        ])

        # skip connection from down channels + output from mid channels = 2560 channels
        self.up_ch = self.mid_ch + self.down_ch  # 2560
        self.up_out_ch = self.up_ch / 2  # 1280
        self.up_channels = nn.ModuleList([
            ResidualBlock(self.up_ch, self.up_out_ch),
            ResidualBlock(self.up_ch, self.up_out_ch),
            ResidualBlock(self.up_ch, self.up_out_ch),
            Upsample(self.up_out_ch),
        ])
        for _ in range(2):
            self.up_channels += [
                ResidualBlock(self.up_ch, self.up_out_ch),
                AttentionBlock(self.num_heads, self.emb_dim),
            ]
        for _ in range(2):
            self.up_ch = self.mid_ch + self.down_ch / 2
            self.up_channels += [
                ResidualBlock(self.up_ch, self.up_out_ch),
                AttentionBlock(self.num_heads, self.emb_dim),
                Upsample(self.mid_ch - self.down_ch / 2),
                ResidualBlock(self.up_ch, self.up_out_ch / 2),
                AttentionBlock(self.num_heads, self.emb_dim / 2),
                ResidualBlock(self.mid_ch - self.down_ch / 2, self.up_out_ch / 2),
                AttentionBlock(self.num_heads, self.emb_dim / 2),
            ]
            self.up_out_ch /= 2
            self.mid_ch /= 2
            self.down_ch /= 2
            self.emb_dim /= 2

        self.up_channels += ([
            ResidualBlock(self.mid_ch - self.down_ch / 2, self.up_out_ch / 2),
            AttentionBlock(self.num_heads, self.emb_dim / 2),
        ])



class UNetOutputLayer(nn.Module):
    ...


class Diffusion(nn.Module):
    def __init__(self, t_emb_dim: int):
        super(Diffusion, self).__init__()
        self.timestemp_embed = TimeEmbedding(t_emb_dim)
        self.unet = UNet()
        self.out = UNetOutputLayer()

    def forward(self, x: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # x : B, n chans(4 chans in encoder output and this takes encoder output), H / 8, W / 8
        # context - from CLIP: B, seq len, embedding dim
        # timestep - (1, t_emb_dim)

        timestep = self.timestamp_embed(timestep)
        output = self.unet(x, context, timestep)
        output = self.out(output)
        return output
