import kornia
import torch.nn as nn


class BlurUpsample(nn.Module):
    def __init__(self, blur_kernel_size=(3, 3), blur_kernel_sigma=(1.5, 1.5), upsample_scale=(2, 2)):
        self.blur_upsample = nn.Sequential(
            kornia.filters.GaussianBlur2d(blur_kernel_size),
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
        # TODO: take care of upsample size shenanigans
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
