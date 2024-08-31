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


class UNet(nn.Module):
    ...


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
