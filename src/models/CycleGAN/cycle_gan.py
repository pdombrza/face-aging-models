import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, padding_type: str = '', conv_bias: bool = True) -> None:
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1), # maybe set bias to false?
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_block(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc: int = 3, n_layers: int = 3) -> None:
        super(Discriminator, self).__init__()
        # as in the original implementation
        self.discriminator = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),  # same here about the bias
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self, num_residual_blocks: int = 9):
        super(Generator, self).__init__()
        self.gen = [
            nn.ReflectionPad2d(3), # 3 input channels since we're dealing with rgb images
            nn.Conv2d(3, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        # Downsampling
        num_channels = 64
        for _ in range(2):
            self.gen += [
                nn.Conv2d(num_channels, 2 * num_channels, kernel_size=3, stride=2, padding=1), # maybe set bias to false?
                nn.InstanceNorm2d(2 * num_channels),
                nn.ReLU(inplace=True)
            ]
            num_channels *= 2
        # 9 resnet blocks
        self.gen += [ResidualBlock(num_channels) for _ in range(num_residual_blocks)]
        # Upsampling
        for _ in range(2):
            self.gen += [
                nn.ConvTranspose2d(num_channels, num_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(num_channels // 2),
                nn.ReLU(inplace=True)
            ]
            num_channels //= 2
        self.gen += [
            # nn.ReflectionPad2d(3), # this causes shape mismatch (makes sense)
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]
        self.generator = nn.Sequential(*self.gen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)
