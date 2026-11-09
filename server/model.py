"""Model classes, copied from latent_diffusion.ipynb (U-Net v3) and mel_vae.ipynb."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# U-Net v4: bigger, built from AudioLDM's recipe (ADM-style U-Net).
#   - 128 base channels, x1/x2/x4 per level, 2 ResBlocks per level
#   - time and CLAP each go through a 2-layer MLP, then FiLM (scale + shift) every ResBlock
#   - self-attention (32 channels per head) at the two lower resolutions and the middle
#   - last conv of each block zero-initialised, so every block starts as the identity
# ---------------------------------------------------------------------------
def zero(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class ResBlock4(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero(nn.Conv2d(out_ch, out_ch, 3, padding = 1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim = 1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class Attention4(nn.Module):
    def __init__(self, ch, head_ch = 32):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.attn = nn.MultiheadAttention(ch, ch // head_ch, batch_first = True)
        zero(self.attn.out_proj)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights = False)
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding = 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor = 2, mode = "nearest"))


class UNet4(nn.Module):
    def __init__(self, base = 128, mult = (1, 2, 4), nres = 2, attn = (False, True, True), emb = 512, dropout = 0.0):
        super().__init__()
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cond_mlp = nn.Sequential(nn.Linear(512, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(4, base, 3, padding = 1)

        self.down = nn.ModuleList()
        skips, ch = [base], base
        for lvl, m in enumerate(mult):
            for _ in range(nres):
                blocks = [ResBlock4(ch, base * m, emb, dropout)]
                ch = base * m
                if attn[lvl]:
                    blocks.append(Attention4(ch))
                self.down.append(nn.ModuleList(blocks))
                skips.append(ch)
            if lvl != len(mult) - 1:                      # 20x64 -> 10x32 -> 5x16
                self.down.append(nn.ModuleList([Down(ch)]))
                skips.append(ch)

        self.mid = nn.ModuleList([ResBlock4(ch, ch, emb, dropout), Attention4(ch), ResBlock4(ch, ch, emb, dropout)])

        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mult))):
            for i in range(nres + 1):
                blocks = [ResBlock4(ch + skips.pop(), base * m, emb, dropout)]
                ch = base * m
                if attn[lvl]:
                    blocks.append(Attention4(ch))
                if lvl != 0 and i == nres:
                    blocks.append(Up(ch))
                self.up.append(nn.ModuleList(blocks))

        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), zero(nn.Conv2d(ch, 4, 3, padding = 1)))

    @staticmethod
    def run(blocks, h, emb):
        for b in blocks:
            h = b(h, emb) if isinstance(b, ResBlock4) else b(h)
        return h

    def forward(self, x, t, c):
        emb = self.time_mlp(time_embedding(t, self.base)) + self.cond_mlp(c)
        h = self.inp(x)
        hs = [h]
        for blocks in self.down:
            h = self.run(blocks, h, emb)
            hs.append(h)
        h = self.run(self.mid, h, emb)
        for blocks in self.up:
            h = self.run(blocks, torch.cat([h, hs.pop()], dim = 1), emb)
        return self.out(h)


def make_betas(schedule = "linear", beta_start = 1e-4, beta_end = 0.02, T = T):
    """'linear' = DDPM (what v3 used). 'scaled_linear' = LDM/AudioLDM: linear in sqrt(beta)."""
    if schedule == "scaled_linear":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2
    return torch.linspace(beta_start, beta_end, T)
