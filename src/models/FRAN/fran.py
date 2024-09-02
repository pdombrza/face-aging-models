from copy import copy
from typing import Iterable

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurUpsample(nn.Module):
    def __init__(self, blur_kernel_size: Iterable = (3, 3), blur_kernel_sigma: Iterable = (1.5, 1.5), upsample_scale: Iterable = (2, 2)) -> None:
        super(BlurUpsample, self).__init__()
        self.blur_upsample = nn.Sequential(
            kornia.filters.GaussianBlur2d(kernel_size=(3, 3), sigma=(1.5, 1.5)),
            nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur_upsample(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpBlock, self).__init__()
        self.blur_upsample = BlurUpsample(in_channels)
        self.up_block = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, x_skip_conn: torch.Tensor) -> torch.Tensor:
        x = self.blur_upsample(x)
        # possibly need to pad x to have the same size as x_skip_connection?
        delta_height = x_skip_conn.size(2) - x.size(2)  # size(2) since we have B x C x H x W
        delta_width = x_skip_conn.size(3) - x.size(3)  # size(3) since we have B x C x H x W
        x = F.pad(x, (delta_width // 2, delta_width - delta_width // 2, delta_height // 2, delta_height - delta_height // 2)) # tuple left, right, top, bottom
        x = torch.cat([x, x_skip_conn], dim=1)
        # print("fin x:", x.shape)
        return self.up_block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DownBlock, self).__init__()
        self.down_block = nn.Sequential(
            kornia.filters.MaxBlurPool2D(kernel_size=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_block(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 5, num_channels: int = 64, num_unet_blocks: int = 4):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # downsampling
        self.downsample = nn.ModuleList()
        for _ in range(num_unet_blocks):
            self.downsample.append(DownBlock(self.num_channels, 2 * self.num_channels))
            self.num_channels *= 2
        # upsampling
        self.upsample = nn.ModuleList()
        for _ in range(num_unet_blocks):
            self.upsample.append(UpBlock(self.num_channels, self.num_channels // 2))
            self.num_channels //= 2

        self.out_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=3, kernel_size=1, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_layers(x)
        skip_values = [copy(x)]
        # downsampling
        for dlayer in self.downsample:
            x = dlayer(x)
            skip_values.append(x)
        # upsampling
        skip_values.pop()
        for ulayer in self.upsample:
            x = ulayer(x, skip_values.pop())
        x = self.out_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 4, num_channels: int = 64, num_blocks: int = 3, normalization: int = True):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.disc = [
            nn.Conv2d(in_channels, self.num_channels, kernel_size=3, stride=1, padding=1),
        ]
        if normalization:
            self.disc.append(nn.InstanceNorm2d(self.num_channels))
        self.disc.append(nn.LeakyReLU(0.2, inplace=True))
        self.disc.append(kornia.filters.MaxBlurPool2D(kernel_size=3))

        for i in range(num_blocks-1):
            layers = [nn.Conv2d(self.num_channels, 2 * self.num_channels, kernel_size=3, stride=1, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(2 * self.num_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if i != num_blocks - 1:
                layers.append(kornia.filters.MaxBlurPool2D(kernel_size=3))
            self.num_channels *= 2
            self.disc += layers

        self.disc.append(nn.Conv2d(self.num_channels, 1, kernel_size=3, stride=1, padding=1))

        self.discriminator = nn.Sequential(*self.disc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)
