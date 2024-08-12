import kornia
import torch.nn as nn


class BlurUpsample(nn.Module):
    def __init__(self, blur_kernel_size=(3, 3), blur_kernel_sigma=(1.5, 1.5), upsample_scale=(2, 2)):
        self.blur_upsample = nn.Sequential(
            kornia.filters.GaussianBlur2d(blur_kernel_size, blur_kernel_sigma),
            nn.Upsample(scale_factor=upsample_scale, align_corners=False)
        )

    def forward(self, x):
        return self.blur_upsample(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up_block = nn.Sequential(
            BlurUpsample(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # TODO: implement skip connections
        return self.up_block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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

    def forward(self, x):
        return self.down_block(x)


class Generator(nn.Module):
    def __init__(self, in_channels, num_channels=64, num_unet_blocks=4):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.gen = [
            nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # downsampling
        for _ in range(num_unet_blocks):
            self.gen += DownBlock(self.num_channels, 2 * self.num_channels)
            self.num_channels *= 2
        # upsampling
        for _ in range(num_unet_blocks):
            self.gen += UpBlock(self.num_channels, self.num_channels // 2)
            self.num_channels //= 2

        self.gen += [
            nn.Conv2d(in_channels=self.num_channels, out_channels=in_channels, kernel_size=1, stride=1)
        ]
        self.generator = nn.Sequential(*self.gen)

    def forward(self, x):
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, num_channels=64, num_blocks=3, normalization=True):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.disc = [
            nn.Conv2d(in_channels, self.num_channels, kernel_size=3, stride=1, padding=1),
        ]
        if normalization:
            self.disc.append(nn.InstanceNorm2d(64))
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

        self.disc.append(nn.Conv2d(num_channels, 1, kernel_size=3, stride=1, padding=1))

        self.discriminator = nn.Sequential(*self.disc)

    def forward(self, x):
        return self.discriminator(x)
