"""MuNet — the pseudo-CT network used for every model in Table 1 of the paper.

MuNet is the BIC-MAC organizers' residual 3D U-Net with three additions:

    1. Attention gates on the skip connections (Attention U-Net style).
    2. One multi-head self-attention block at each of the two deepest stages.
    3. Two deep-supervision heads predicting the CT at 1/2 and 1/4 resolution.

Together these are 1.4 M parameters, 5.8 % of the model (24.3 M vs 22.9 M without
them). Only two configurations appear in the paper, and they differ *only* in how
many input channels the stem reads:

    MuNet_2   in_channels=2   [NAC-PET, topogram]
    MuNet_4   in_channels=4   [NAC-PET, topogram, DIXON in-phase, DIXON out-phase]

All three additions are IDENTITY AT INITIALIZATION on purpose, so that a model
carrying them reproduces its plain-U-Net parent exactly at epoch 0 and a warm start
from a checkpoint without them is bit-exact. The mechanism differs per block and is
documented on each class below.

The network outputs CT in [0, 1]; callers convert to Hounsfield Units with
    HU = 1000 * (3 * x - 1)      i.e.  x * 3000 - 1000
which is the inverse of the [-1000, 2000] HU range normalization used in training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks (identical to the organizers' baseline)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Conv-Norm-Act x2 with an identity (or 1x1x1-projected) residual."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(identity)
        out = out + identity
        return self.relu(out)


class EncoderBlock(nn.Module):
    """One encoder stage: residual block, then 2x max-pool. Returns (skip, pooled)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.block(x)
        return x, self.pool(x)


# ---------------------------------------------------------------------------
# Addition 1 — attention gate on a skip connection
# ---------------------------------------------------------------------------

class AttentionGate3D(nn.Module):
    """Re-weight a skip connection per voxel, using the decoder signal as the query.

    IDENTITY AT INIT: `psi` is zero-initialized, so attn = tanh(0) = 0 and the block
    returns `x * (1 + 0) = x` exactly. After training each voxel of the skip is scaled
    by a factor in (0, 2), letting the decoder suppress the parts of the skip it does
    not need.
    """

    def __init__(self, ch: int, inter: int | None = None):
        super().__init__()
        inter = inter or max(1, ch // 2)
        self.w_x = nn.Conv3d(ch, inter, 1)
        self.w_g = nn.Conv3d(ch, inter, 1)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.psi = nn.Conv3d(inter, 1, 1)
        nn.init.zeros_(self.psi.weight)
        nn.init.zeros_(self.psi.bias)

    def forward(self, x, g):
        a = self.relu(self.w_x(x) + self.w_g(g))
        attn = torch.tanh(self.psi(a))          # (B, 1, ...) == 0 at init
        return x * (1.0 + attn)


class DecoderBlock(nn.Module):
    """Transposed-conv upsample, optional skip gating, concat, residual block."""

    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.attn = AttentionGate3D(out_ch) if use_attention else None
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if self.attn is not None:
            skip = self.attn(skip, x)
        return self.block(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Addition 2 — global self-attention at the deepest stages
# ---------------------------------------------------------------------------

class SelfAttention3D(nn.Module):
    """`out = x + gamma * MHA(LayerNorm(flatten(x)))`, with a per-channel LayerScale.

    IDENTITY AT INIT: `gamma` is zero-initialized, so the block returns `x` exactly.

    Applied only at the two deepest stages, where the token count is small enough for
    full attention to be affordable: a 208^3 patch is 13^3 = 2197 tokens there.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        b, c, d, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)                     # (B, N, C)
        tn = self.norm(t)
        a, _ = self.attn(tn, tn, tn, need_weights=False)
        a = (self.gamma * a).transpose(1, 2).reshape(b, c, d, h, w)
        return x + a


# ---------------------------------------------------------------------------
# MuNet
# ---------------------------------------------------------------------------

class MuNet(nn.Module):
    """Residual 3D U-Net, 4 encoder + 4 decoder stages, optional MuNet additions.

    Set `use_attention_skips=encoder_attention=deep_supervision=False` and this is
    byte-compatible with the organizers' baseline U-Net (same modules, same
    state-dict keys) — which is how the architecture ablation in the paper is run.

    Deep supervision is TRAINING-ONLY. Under `model.train()` forward returns a dict
    with the auxiliary logits; under `model.eval()` it returns the CT tensor alone, so
    inference code never has to know the flag was set. The aux heads still live in the
    state dict, which is why `deep_supervision` must be passed when rebuilding a
    checkpoint for inference — `load_meta()` in predict.py handles that automatically.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        bottleneck_dropout: float = 0.2,
        use_attention_skips: bool = False,
        encoder_attention: bool = False,
        attn_heads: int = 8,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.deep_supervision = bool(deep_supervision)
        self.encoder_attention = bool(encoder_attention)

        c1, c2, c3, c4, c5 = (base_channels * m for m in (1, 2, 4, 8, 16))

        # Encoder
        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)

        # Bottleneck. NOTE the state-dict keys differ between the two branches
        # (`bottleneck.0.conv1.weight` vs `bottleneck.conv1.weight`), so the dropout
        # value is recorded in the checkpoint meta sidecar. Every released checkpoint
        # uses 0.2.
        if bottleneck_dropout and bottleneck_dropout > 0:
            self.bottleneck = nn.Sequential(
                ResidualBlock(c4, c5), nn.Dropout3d(float(bottleneck_dropout))
            )
        else:
            self.bottleneck = ResidualBlock(c4, c5)

        if self.encoder_attention:
            self.attn_p4 = SelfAttention3D(c4, num_heads=attn_heads)
            self.attn_bott = SelfAttention3D(c5, num_heads=attn_heads)

        # Decoder
        self.dec4 = DecoderBlock(c5, c4, use_attention=use_attention_skips)
        self.dec3 = DecoderBlock(c4, c3, use_attention=use_attention_skips)
        self.dec2 = DecoderBlock(c3, c2, use_attention=use_attention_skips)
        self.dec1 = DecoderBlock(c2, c1, use_attention=use_attention_skips)

        # Output head. Zero weights and bias 0.333 make the untrained output the
        # constant 0.333, which decodes to HU = 0 — the body median. Starting there
        # rather than at random noise was a large early win.
        self.out_conv = nn.Conv3d(c1, out_channels, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.constant_(self.out_conv.bias, 0.333)

        # Addition 3 — deep supervision, same constant init as the main head so the
        # aux heads do not disturb a warm-started main path.
        if self.deep_supervision:
            self.ds2 = nn.Conv3d(c2, out_channels, 1)      # 1/2 resolution
            self.ds3 = nn.Conv3d(c3, out_channels, 1)      # 1/4 resolution
            for m in (self.ds2, self.ds3):
                nn.init.zeros_(m.weight)
                nn.init.constant_(m.bias, 0.333)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        if self.encoder_attention:
            p4 = self.attn_p4(p4)
        b = self.bottleneck(p4)
        if self.encoder_attention:
            b = self.attn_bott(b)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        out = self.out_conv(d1)

        if self.training and self.deep_supervision:
            # Training only. The loss downsamples the ground truth to match.
            return {"ct": out, "ds_aux": [self.ds2(d2), self.ds3(d3)]}
        return out


def build_model(cfg: dict) -> MuNet:
    """Build MuNet from a config dict (or from a checkpoint's meta sidecar).

    Both `configs/*.yaml` and `<checkpoint>.pth.meta.json` carry the same keys, so the
    same function serves training and inference. Defaults reproduce the organizers'
    plain baseline, i.e. all three MuNet additions off.
    """
    return MuNet(
        in_channels=int(cfg.get("in_channels", len(cfg.get("input_keys", ["nacpet"])))),
        out_channels=1,
        base_channels=int(cfg.get("base_channels", 32)),
        bottleneck_dropout=float(cfg.get("bottleneck_dropout", 0.2)),
        use_attention_skips=bool(cfg.get("use_attention_skips", False)),
        encoder_attention=bool(cfg.get("encoder_attention", False)),
        attn_heads=int(cfg.get("attn_heads", 8)),
        deep_supervision=bool(cfg.get("deep_supervision", False)),
    )
