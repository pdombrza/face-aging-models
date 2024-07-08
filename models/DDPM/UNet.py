import torch
import torch.nn as nn


class NormActConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_norm_groups=8, norm=True, act=True):
        super(NormActConv, self).__init__()
        conv_padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.GroupNorm(num_norm_groups, in_channels) if norm is True else nn.Identity(),
            nn.SiLU() if act is True else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=conv_padding)
        )

    def forward(self, x):
        return self.block(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_out, t_emb_dim):
        super(TimeEmbedding, self).__init__()
        self.te_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_out)
        )

    def forward(self, x):
        return self.te_block(x)


class AttentionBlock(nn.Module):
    def __init__(self, num_channels, num_heads=4, num_groups=8, norm=True):
        super(AttentionBlock, self).__init__()
        self.g_norm = nn.GroupNorm(num_groups, num_channels) if norm else nn.Identity()
        self.attn = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h*w)
        x = self.g_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, k=2, use_conv=True, use_mpool=True):
        # potentially adapt to avgpool
        super(Downsample, self).__init__()
        conv_out_ch = out_channels // 2 if use_mpool is True else out_channels
        pool_out_ch = out_channels // 2 if use_conv is True else out_channels
        self.use_conv = use_conv
        self.use_mpool = use_mpool
        if use_conv:
            self.cv = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
                    nn.Conv2d(in_channels, conv_out_ch, kernel_size=4, stride=k, padding=1),
                )
        else:
            self.cv = nn.Identity()

        if use_mpool:
            self.mpool = nn.Sequential(
                nn.MaxPool2d(k, k),
                nn.Conv2d(in_channels, out_channels, pool_out_ch, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.mpool = nn.Identity()

    def forward(self, x):
        if not self.use_conv:
            return self.mpool(x)
    
        if not self.use_mpool:
            return self.cv(x)
        
        return torch.cat([self.cv(x), self.mpool(x)], dim=1)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, k=2, use_conv=True, use_upsample=True):
        super(Upsample, self).__init__()
        self.use_conv = use_conv
        self.use_upsample = use_upsample
        conv_out_ch = out_channels // 2 if use_upsample is True else out_channels
        upsample_out_ch = out_channels // 2 if use_conv is True else out_channels
        if use_conv:
            self.cv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, conv_out_ch, kernel_size=4, stride=k, padding=1),
                nn.Conv2d(conv_out_ch, conv_out_ch, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.cv = nn.Identity()
        if use_upsample:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=k, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, upsample_out_ch, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.up = nn.Identity()

    def forward(self, x):
        if not self.use_conv:
            return self.up(x)
    
        if not self.use_upsample:
            return self.cv(x)
        
        return torch.cat([self.cv(x), self.up(x)], dim=1)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=128, num_layers=2, down_sample=True):
        super(DownConv, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([NormActConv(in_channels if i==0 else out_channels, out_channels) for i in range(num_layers)])
        self.conv2 = nn.ModuleList([NormActConv(out_channels, out_channels) for _ in range(num_layers)])
        self.te_block = nn.ModuleList([TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)])
        self.attn_block = nn.ModuleList([AttentionBlock(out_channels) for _ in range(num_layers)])
        self.down_block = Downsample(out_channels, out_channels) if down_sample else nn.Identity()
        self.res_block = nn.ModuleList()
        for i in range(num_layers):
            self.res_block.append(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            res_out = self.conv1[i](out)
            res_out += self.te_block[i](t_emb)[:, :, None, None]
            res_out = self.conv2[i](res_out)
            res_out += self.res_block[i](resnet_input)

            out_attn = self.attn_block[i](res_out)
            out = out_attn + res_out

        out = self.down_block(out)
        return out
    

class MidConv(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=128, num_layers=2):
        super(MidConv, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i == 1 else out_channels, out_channels) for i in range(num_layers + 1),
        ])
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, out_channels) for i in range(num_layers + 1),
        ])
        self.te_block = nn.ModuleList([TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers + 1)])
        self.attn_block = nn.ModuleList([AttentionBlock(out_channels) for _ in range(num_layers)])
        self.res_block = nn.ModuleList()
        for i in range(num_layers):
            self.res_block.append(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, t_emb):
        out = x
        resnet_input = out
        out = self.conv1[0](out)
        out += self.te_block[0](t_emb)[:, :, None, None]
        out = self.conv2[0](out)
        out += self.res_block[0](resnet_input)

        for i in range(self.num_layers):
            out_attn = self.attn_block[i](out)
            out += out_attn

            resnet_input = out
            out = self.conv1[i+1](out)
            out += self.te_block[i+1](t_emb)[:, :, None, None]
            out = self.conv2[i+1](out)
            out += self.res_block[i+1](resnet_input)

        return out
    

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=128, num_layers=2, up_sample=True):
        super(UpConv, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList([NormActConv(in_channels if i==0 else out_channels, out_channels) for i in range(num_layers)])
        self.conv2 = nn.ModuleList([NormActConv(out_channels, out_channels) for _ in range(num_layers)])
        self.te_block = nn.ModuleList([TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)])
        self.attn_block = nn.ModuleList([AttentionBlock(out_channels) for _ in range(num_layers)])
        self.up_block = Upsample(in_channels, in_channels // 2) if up_sample else nn.Identity()
        self.res_block = nn.ModuleList()
        for i in range(num_layers):
            self.res_block.append(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, down_out, t_emb):
        x = self.up_block(x)
        x = torch.cat([x, down_out], dim=1)
        out = x

        for i in range(self.num_layers):
            resnet_input = out
            out = self.conv1[i](out)
            out += self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)

            out_attn = self.attn_block[i](out)
            out += out_attn
        
        return out