import torch
import torch.nn as nn
import torch.nn.functional as F


def align_tensor_image(x: torch.Tensor, y: torch.Tensor):
    delta_height = y.size(2) - x.size(2)
    delta_width = y.size(3) - x.size(3)
    x = F.pad(x, (delta_width // 2, delta_width - delta_width // 2, delta_height // 2, delta_height - delta_height // 2))
    return x


class SinusoidalPositionalEmbedding(nn.Module):
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
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.time_block = nn.Sequential(
            nn.Linear(t_emb_dim, out_channels),
            nn.SiLU(inplace=True),
        )

        self.time_feature_block = nn.Sequential(  # from SD
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, time):
        residue = x
        x = self.feature_block(x)
        time = self.time_block(time)
        time_feature = x + time.unsqueeze(-1).unsqueeze(-1)
        time_feature = self.time_feature_block(time_feature)
        time_feature += residue
        return time_feature


class NormActConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(NormActConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_act_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.norm_act_conv(x)


class NormActConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(NormActConvTranspose, self).__init__()
        self.norm_act_convtranspose = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.norm_act_convtranspose(x)


class UNet(nn.Module):
    # https://arxiv.org/pdf/2104.05358 [Figure 4.],
    def __init__(self, in_channels: int = 6, starting_down_channels: int = 64, t_emb_dim: int = 256):
        super(UNet, self).__init__()
        self.down_ch = starting_down_channels
        self.time_emb_mlp_layer = nn.Sequential(
            SinusoidalPositionalEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.ReLU(inplace=True)
        )
        # set up down layers in unet
        self.down_layers = nn.ModuleList([
            NormActConv(in_channels=in_channels, out_channels=self.down_ch),
        ])
        for _ in range(3):
            self.down_layers += [
                ResnetBlock(in_channels=self.down_ch, out_channels=self.down_ch, t_emb_dim=t_emb_dim),
                NormActConv(in_channels=self.down_ch, out_channels=2 * self.down_ch, stride=2)
            ]
            self.down_ch *= 2

        self.down_layers += [
                ResnetBlock(in_channels=self.down_ch, out_channels=self.down_ch, t_emb_dim=t_emb_dim),
                NormActConv(in_channels=self.down_ch, out_channels=self.down_ch, stride=2),
            ]

        # set up the bottleneck
        self.mid_ch = self.down_ch
        self.mid_layers = nn.ModuleList()
        for _ in range(3):
            self.mid_layers += [
                ResnetBlock(in_channels=self.mid_ch, out_channels=self.mid_ch, t_emb_dim=t_emb_dim),
                NormActConv(in_channels=self.mid_ch, out_channels=self.mid_ch)
            ]

        # set up the up layers in unet
        self.up_ch = self.mid_ch
        self.up_layers = nn.ModuleList([
            ResnetBlock(in_channels=2 * self.up_ch, out_channels=2 * self.up_ch, t_emb_dim=t_emb_dim),
            NormActConv(in_channels=2 * self.up_ch, out_channels=2 * self.up_ch),
            NormActConvTranspose(in_channels=2 * self.up_ch, out_channels=self.up_ch, stride=2, output_padding=1),
        ])

        for _ in range(3):
            self.up_layers += [
                ResnetBlock(in_channels=2 * self.up_ch, out_channels=2 * self.up_ch, t_emb_dim=t_emb_dim),
                # NormActConv(in_channels=2 * self.up_ch, out_channels=2 * self.up_ch),
                NormActConvTranspose(in_channels=2 * self.up_ch, out_channels=self.up_ch // 2, stride=2, output_padding=1),
            ]

            self.up_ch //= 2

        self.up_layers += [
            NormActConvTranspose(in_channels=2 * self.up_ch, out_channels=3)
        ]

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = self.time_emb_mlp_layer(timestep)
        skip_connections = []
        for layer in self.down_layers:
            if isinstance(layer, NormActConv):
                x = layer(x)
                skip_connections.append(x)
            elif isinstance(layer, ResnetBlock):
                x = layer(x, t)
            else:
                raise ValueError("Invalid block in unet downsampling")

        for layer in self.mid_layers:
            if isinstance(layer, NormActConv):
                x = layer(x)
            elif isinstance(layer, ResnetBlock):
                x = layer(x, t)
            else:
                raise ValueError("Invalid block in unet bottleneck")

        for layer in self.up_layers[:-1]:
            if isinstance(layer, NormActConv) or isinstance(layer, NormActConvTranspose):
                x = layer(x)
            elif isinstance(layer, ResnetBlock):
                x_skip_conn = skip_connections.pop()
                x = align_tensor_image(x, x_skip_conn)
                x = torch.cat([x, x_skip_conn], dim=1)
                x = layer(x, t)
            else:
                raise ValueError("Invalid block in unet upsampling")

        # apply first skip connection to final upsampling layer - normactconvtranspose
        x_skip_conn = skip_connections.pop()
        x = align_tensor_image(x, x_skip_conn)
        x = torch.cat([x, x_skip_conn], dim=1)
        x = self.up_layers[-1](x)

        return x
