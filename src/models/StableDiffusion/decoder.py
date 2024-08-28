import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import ResidualBlock, AttentionBlock


class VAEDecoder(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 512, num_res_blocks: int = 4):
        super(VAEDecoder, self).__init__()
        self.out_channels = out_channels
        self.decoder_block = [
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1),
            ResidualBlock(self.out_channels, self.out_channels),
            AttentionBlock(self.out_channels, self.out_channels),
        ]
        for _ in range(num_res_blocks):
            self.decoder_block.append(ResidualBlock(self.out_channels, self.out_channels))

        self.decoder_block.extend([
            ResidualBlock(self.out_channels, self.out_channels),
            nn.Upsample(scale_factor=2),  # H / 8, W / 8 -> H / 4, W / 4
        ])

        for _ in range(num_res_blocks):
            self.decoder_block.append(ResidualBlock(self.out_channels, self.out_channels))

        self.decoder_block.extend([
            nn.Upsample(scale_factor=2),  # H / 4, W / 4 -> H / 2, W / 2
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            ResidualBlock(self.out_channels, self.out_channels / 2)
        ])
        self.out_channels /= 2
        for _ in range(num_res_blocks - 1):
            self.decoder_block.append(ResidualBlock(self.out_channels, self.out_channels))

        self.decoder_block.extend([
            nn.Upsample(scale_factor=2),  # H / 2, W / 2 -> H, W
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            ResidualBlock(self.out_channels, self.out_channels / 2)
        ])

        self.out_channels /= 2
        for _ in range(num_res_blocks - 1):
            self.decoder_block.append(ResidualBlock(self.out_channels, self.out_channels))

        self.decoder_block.extend([
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.out_channels, 3, kernel_size=3, padding=1)
        ])

        self.decoder = nn.Sequential(*self.decoder_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18125  # reverse scaling from encoder
        x = self.decoder(x)
        return x
