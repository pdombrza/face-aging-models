import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super(TimeEmbedding, self).__init__()
        self.t_emb = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(4 * emb_dim, 4 * emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t_emb(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = 1280):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.time_block = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.time_feature_block = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        )

        self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, time):
        residue = x
        x = self.feature_block(x)
        time = self.time_block(time)
        time_feature = x + time.unsqueeze(-1).unsqueeze(-1)
        time_feature = self.time_feature_block(time_feature)
        if self.in_channels == self.out_channels:
            time_feature += residue
        else:
            time_feature += self.residual_layer(residue)

        return time_feature


class AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, emb_dim: int, context_dim: int = 7680):
        # transformer-like attention block
        channels = n_heads * emb_dim
        self.input_block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
        )
        self.first_atn_block = nn.Sequential(
            nn.LayerNorm(channels),
            SelfAttention(n_heads, channels, input_projection_bias=False),
        )

        # second atn block
        self.layernorm2 = nn.LayerNorm(channels)
        self.cross_attention = CrossAttention(n_heads, channels, context_dim, input_projection_bias=False)

        # feed forward
        self.layernorm3 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, 4 * channels * 2)
        self.linear2 = nn.Linear(4 * channels, channels)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: image -> B, N_CHANS, H, W
        # context: text embedding -> B, seq_len, emb_dim
        residue_old = x

        x = self.input_block(x)

        batch_size, n_channels, h, w = x.shape
        x = x.view((batch_size, n_channels, h * w))
        x = x.transpose(-1, -2)  # B, H*W, N_CHANS

        residue = x
        x = self.first_atn_block(x)
        x += residue

        # cross attention
        residue = x
        x = self.layernorm2
        x = self.cross_attention(x, context)
        x += residue

        residue = x
        x = self.layernorm3(x)
        x, gate = self.linear1(x).chunk(2, dim=-1)
        x *= F.gelu(gate)
        x = self.linear2(x)
        x += residue

        x = x.transpose(-1, -2)  # restore back to B, N_CHANS, H*W
        x = x.view((batch_size, n_channels, h, w))
        x = self.conv_out(x)
        return x + residue_old


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super(Upsample, self).__init__()
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

    def forward(self, x):
        ...


class UNetOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetOutputLayer, self).__init__()
        self.unet_out_block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unet_out_block(x)
        return x


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
