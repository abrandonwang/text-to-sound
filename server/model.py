"""Model classes, copied from latent_diffusion.ipynb (U-Net v3) and mel_vae.ipynb."""

import math

import torch
import torch.nn as nn

T = 1000
SCALE = 0.6863


def time_embedding(t, dim = 32):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device = t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim = 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim = 128):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, heads = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first = True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + h


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_proj = nn.Linear(512, 128)
        self.inp = nn.Conv2d(4, 64, 3, padding = 1)
        self.block1 = ResBlock(64, 64)
        self.down1 = nn.Conv2d(64, 64, 3, stride = 2, padding = 1)
        self.block2 = ResBlock(64, 128)
        self.down2 = nn.Conv2d(128, 128, 3, stride = 2, padding = 1)
        self.block3 = ResBlock(128, 256)
        self.mid = ResBlock(256, 512)
        self.attn = Attention(512)
        self.up = nn.Upsample(scale_factor = 2)
        self.block4 = ResBlock(768, 256)
        self.block5 = ResBlock(384, 128)
        self.block6 = ResBlock(192, 64)
        self.out_norm = nn.GroupNorm(32, 64)
        self.out = nn.Conv2d(64, 4, 1)

    def forward(self, x, t, c):
        t_emb = time_embedding(t, dim = 128) + self.cond_proj(c)
        x = self.inp(x)
        d1 = self.block1(x, t_emb)
        h = self.down1(d1)
        d2 = self.block2(h, t_emb)
        h = self.down2(d2)
        d3 = self.block3(h, t_emb)
        h = self.mid(d3, t_emb)
        h = self.attn(h)
        h = torch.cat([h, d3], dim = 1)
        h = self.block4(h, t_emb)
        h = self.up(h)
        h = torch.cat([h, d2], dim = 1)
        h = self.block5(h, t_emb)
        h = self.up(h)
        h = torch.cat([h, d1], dim = 1)
        h = self.block6(h, t_emb)
        return self.out(nn.functional.silu(self.out_norm(h)))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(256, 8, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 256, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(256, 128, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(128, 64, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(64, 1, 3, 1, 1),
        )
