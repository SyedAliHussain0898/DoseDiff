"""
UNet backbone for DoseDiff ‑ multi‑stage version
------------------------------------------------

* Shared encoder, stage‑specific decoders
* Supports CT + distance‑map fusion (ViT_fusion)
* Compatible with MultiStageSpacedDiffusion  (stage_indices passed
  through model_kwargs)

This is a *drop‑in replacement* for the old `unet.py` – just copy it to
`guided_diffusion/unet.py`.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# standard libs
# ----------------------------------------------------------------------
from abc import abstractmethod
import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# project‑local helpers (UNCHANGED)
# ----------------------------------------------------------------------
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from .vit import ViT_fusion

# ======================================================================
# basic building blocks (mostly identical to original)
# ======================================================================


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A Sequential that passes timestep embeddings to the children that
    support it (TimestepBlock); others ignore it.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# -------------------- (up‑/down‑sampling) -----------------------------

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels      = channels
        self.out_channels  = out_channels or channels
        self.use_conv      = use_conv
        self.dims          = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels,
                                3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x,
                (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode="nearest",
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels     = channels
        self.out_channels = out_channels or channels
        stride            = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, channels, self.out_channels, 3,
                stride=stride, padding=1
            )
        else:
            assert channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# -------------------- residual block ----------------------------------

class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels            = channels
        self.emb_channels        = emb_channels
        self.dropout             = dropout
        self.out_channels        = out_channels or channels
        self.use_conv            = use_conv
        self.use_checkpoint      = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(inplace=True),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(inplace=True),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm
                else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels,
                                           self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels,
                                           self.out_channels, 1)

    # ------------ forward -------------------------------------------------

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb),
                          self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


# -------------------- attention ---------------------------------------

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels       = channels
        self.use_checkpoint = use_checkpoint

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm   = normalization(channels)
        self.qkv    = conv_nd(1, channels, channels * 3, 1)
        self.attention = (
            QKVAttention(self.num_heads)
            if use_new_attention_order
            else QKVAttentionLegacy(self.num_heads)
        )
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,),
                          self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h   = self.attention(qkv)
        h   = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """Channel‑first implementation (legacy)"""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale   = 1 / math.sqrt(math.sqrt(ch))
        weight  = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight  = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a       = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """Separated‑heads implementation (new)"""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch           = width // (3 * self.n_heads)
        q, k, v      = qkv.chunk(3, dim=1)
        scale        = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct",
                      weight,
                      v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

# ======================================================================
# NEW  :  Multi‑stage former‑UNet
# ======================================================================


class IntermediateLoss(nn.Module):
    """
    Tiny head for optional deep supervision (not mandatory).
    """

    def __init__(self, in_ch, out_ch, dims):
        super().__init__()
        self.loss_head = nn.Sequential(
            normalization(in_ch),
            nn.SiLU(inplace=True),
            conv_nd(dims, in_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.loss_head(x)


class UNetModel_MS_Former_MultiStage(nn.Module):
    """
    Multi‑stage UNet.

    • shared encoder (CT, DIS, Dose fused)
    • stage‑specific decoders
    • ViT fusion in the bottleneck
    """

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        image_size,
        in_channels,
        ct_channels,
        dis_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        *,
        dropout            = 0.0,
        channel_mult       = (1, 2, 4, 8),
        conv_resample      = True,
        dims               = 2,
        num_classes        = None,
        use_checkpoint     = False,
        use_fp16           = False,
        num_heads          = 1,
        num_head_channels  = -1,
        num_heads_upsample = -1,
        use_scale_shift_norm = False,
        resblock_updown      = False,
        use_new_attention_order = False,
        num_stages         = 3,
        stage_channels     = None,   # list[int] or None
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size      = image_size
        self.model_channels  = model_channels
        self.out_channels    = out_channels
        self.num_classes     = num_classes
        self.use_checkpoint  = use_checkpoint
        self.dtype           = th.float16 if use_fp16 else th.float32
        self.num_stages      = num_stages

        # --------------------------------------------------------------
        # stage‑specific base widths
        # --------------------------------------------------------------
        if stage_channels is None:
            r = 0.7
            self.stage_channels = [
                int(model_channels * (1.5 * (r ** i)))
                for i in range(num_stages)
            ]
        else:
            assert len(stage_channels) == num_stages
            self.stage_channels = list(stage_channels)

        # ========== time / label embedding ==========
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(inplace=True),
            linear(time_embed_dim, time_embed_dim),
        )
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # =========================================================
        # encoder (Dose / CT / DIS processed in parallel)
        # =========================================================
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks     = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self.input_blocks_CT  = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, ct_channels, ch, 3, padding=1))]
        )
        self.input_blocks_DIS = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, dis_channels, ch, 3, padding=1))]
        )

        input_block_chans = [ch]
        ds = 1  # spatial down‑scale factor

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # ---------------- dose --------------
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout,
                            out_channels=int(mult * model_channels),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        ),
                        *(  # optional attention
                            [AttentionBlock(
                                int(mult * model_channels),
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )] if ds in attention_resolutions else []
                        ),
                    )
                )
                # --------------- ct ------------------
                self.input_blocks_CT.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout,
                            out_channels=int(mult * model_channels),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        ),
                        *(  # attention
                            [AttentionBlock(
                                int(mult * model_channels),
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )] if ds in attention_resolutions else []
                        ),
                    )
                )
                # --------------- dis -----------------
                self.input_blocks_DIS.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout,
                            out_channels=int(mult * model_channels),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        ),
                        *(  # attention
                            [AttentionBlock(
                                int(mult * model_channels),
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )] if ds in attention_resolutions else []
                        ),
                    )
                )

                ch = int(mult * model_channels)
                input_block_chans.append(ch)

            # downsample (except last level)
            if level != len(channel_mult) - 1:
                down = (
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=ch, dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                    if resblock_updown
                    else Downsample(ch, conv_resample, dims=dims)
                )
                self.input_blocks.append(TimestepEmbedSequential(down))
                self.input_blocks_CT.append(TimestepEmbedSequential(down))
                self.input_blocks_DIS.append(TimestepEmbedSequential(down))
                ds *= 2
                input_block_chans.append(ch)

        # ------------------ middle -------------------
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch, time_embed_dim, dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # ViT fusion (global)
        self.fusion = ViT_fusion(
            image_size=(
                image_size[0] // (2 ** (len(channel_mult) - 1)),
                image_size[1] // (2 ** (len(channel_mult) - 1)),
            ),
            patch_size=(
                image_size[0] // (2 ** len(channel_mult)),
                image_size[1] // (2 ** len(channel_mult)),
            ),
            dim=1024,
            heads=1,
            mlp_dim=2048,
            channels=model_channels * channel_mult[-1],
            dim_head=64,
        )

        # =========================================================
        # stage‑specific decoders + heads
        # =========================================================
        self.output_blocks            = nn.ModuleList()
        self.stage_heads              = nn.ModuleList()
        self.stage_intermediate_heads = nn.ModuleList()

        for s in range(num_stages):
            stage_ch      = self.stage_channels[s]
            stage_blocks  = nn.ModuleList()
            stage_ds      = 2 ** (len(channel_mult) - 1)
            stage_in_chs  = input_block_chans.copy()
            stage_ch_cur  = ch

            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(num_res_blocks + 1):
                    ich = stage_in_chs.pop()
                    layers = [
                        ResBlock(
                            stage_ch_cur + ich, time_embed_dim, dropout,
                            out_channels=int(stage_ch * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    stage_ch_cur = int(stage_ch * mult)

                    if stage_ds in attention_resolutions:
                        layers.append(
                            AttentionBlock(
                                stage_ch_cur,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )

                    # upsample (except very last)
                    if level and i == num_res_blocks:
                        layers.append(
                            ResBlock(
                                stage_ch_cur, time_embed_dim, dropout,
                                out_channels=stage_ch_cur, dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(stage_ch_cur, conv_resample, dims=dims)
                        )
                        stage_ds //= 2

                    stage_blocks.append(TimestepEmbedSequential(*layers))

            self.output_blocks.append(stage_blocks)

            # -------- head (stage_ch_cur → model_channels → out) --------
            self.stage_heads.append(
                nn.Sequential(
                    normalization(stage_ch_cur),
                    nn.SiLU(inplace=True),
                    conv_nd(dims, stage_ch_cur, self.model_channels, 1),
                    nn.SiLU(inplace=True),
                    zero_module(
                        conv_nd(dims, self.model_channels,
                                out_channels, 3, padding=1)
                    ),
                )
            )
            self.stage_intermediate_heads.append(
                IntermediateLoss(stage_ch_cur, out_channels, dims)
            )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    @th.no_grad()
    def get_stage_from_timestep(self, timesteps, num_timesteps=1000):
        frac  = timesteps.float() / (num_timesteps - 1)
        stage = (frac * (self.num_stages - 1)).round().long()
        return stage.clamp_(0, self.num_stages - 1)

    # -----------------------------------------

    def forward(
        self,
        x, timesteps,          # dose + time
        ct, dis,               # CT & distance maps
        y=None,                # optional label
        stage_indices=None,    # provided by MultiStageSpacedDiffusion
    ):
        assert (y is not None) == (self.num_classes is not None)

        if stage_indices is None:
            stage_indices = self.get_stage_from_timestep(timesteps)
        B = x.size(0)
        assert stage_indices.shape[0] == B

        # ---- embed time / label ----
        emb = self.time_embed(timestep_embedding(
            timesteps, self.model_channels, repeat_only=False
        ))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        # ---- encoder (three streams) ----
        hs   = []
        h    = x.to(dtype=self.dtype)
        ct_h = ct.to(dtype=self.dtype)
        dis_h= dis.to(dtype=self.dtype)

        for b, block in enumerate(self.input_blocks):
            ct_h  = self.input_blocks_CT[b](ct_h,  emb)
            dis_h = self.input_blocks_DIS[b](dis)_
