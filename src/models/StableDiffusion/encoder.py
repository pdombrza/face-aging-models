import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)
        )
        self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.res_block(x)
        if self.in_channels == self.out_channels:
            res = self.residual_layer(res)
        return x + res


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super(AttentionBlock, self).__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.group_norm(x)
        batch_size, n_channels, h, w = x.shape
        x = x.view((batch_size, n_channels, h * w))  # flatten the height and width into one dimension
        x = x.transpose(-1, -2)  # b_size, n_chans, h * w -> b_size, h * w, n_chans
        x = self.attention(x)
        x = x.transpose(-1, -2)  # back to original size
        x += residue  # skip connection
        return x


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 128, latent_dim: int = 8, num_blocks: int = 2):
        super(VAEEncoder, self).__init__()
        # by the end we have img width / 8 and img height / 8
        self.out_channels = out_channels
        self.encoder_modules = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=0),
                ResidualBlock(out_channels, out_channels),
                ResidualBlock(out_channels, out_channels),
            ]
        )
        for _ in range(num_blocks):
            self.encoder_modules += [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=0),
                ResidualBlock(self.out_channels, self.out_channels * 2),
                ResidualBlock(self.out_channels * 2, self.out_channels * 2),
            ]
            self.out_channels *= 2

        self.encoder_modules.extend([
                ResidualBlock(self.out_channels, self.out_channels),
                AttentionBlock(self.out_channels),
                ResidualBlock(self.out_channels, self.out_channels),
                nn.GroupNorm(num_groups=32, num_channels=self.out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.out_channels, latent_dim, kernel_size=3, padding=1),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding=0),
            ]
        )

    def reparameterization(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        z = torch.randn_like(mean) * var + mean
        return z

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for module in self.encoder_modules:
            if getattr(module, 'stride', None) == (2, 2):
                # pad only right and bottom
                x = F.pad(x, (1, 0, 1, 0))
            x = module(x)

        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        variance = torch.exp(log_var)
        stdev = torch.sqrt(variance)
        x = self.reparameterization(mean, stdev)

        x *= 0.18125  # magic constant from original paper

        return x
