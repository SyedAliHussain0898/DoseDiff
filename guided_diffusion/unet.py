"""
UNet implementation used by DoseDiff
====================================

Key differences vs. the original repository
-------------------------------------------
1.  Works with *multi‑stage* training (`stage_indices` kwarg).
2.  Head projection is now robust to per‑stage channel counts.
3.  Skip‑connections are batched correctly even when samples in the
    same batch belong to different stages.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import List, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# helpers from the original code base
# ---------------------------------------------------------------------
from .nn import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from .vit import ViT_fusion

# ---------------------------------------------------------------------
# generic building blocks
# ---------------------------------------------------------------------


class TimestepBlock(nn.Module):
    """Any module that takes (x, emb) as input."""

    @abstractmethod
    def forward(self, x, emb):
        ...


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential block that passes timestep embeddings to its children."""

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, *, dims: int = 2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, *, dims: int = 2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A standard ResNet block that can change channels and optionally up/down‑sample.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        *,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(inplace=True),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # optional up/down sampling
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims=dims)
            self.x_upd = Upsample(channels, False, dims=dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims=dims)
            self.x_upd = Downsample(channels, False, dims=dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # embedding to hidden
        self.emb_layers = nn.Sequential(
            nn.SiLU(inplace=True),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # out path
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        # skip
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    # ------------------------------------------------------------------

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

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


# ---------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        *,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)

        Att = QKVAttention if use_new_attention_order else QKVAttentionLegacy
        self.attention = Att(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    # ------------------------------------------------------------------

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x_ = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_ + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.view(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


# ---------------------------------------------------------------------
# optional auxiliary head (deep supervision)
# ---------------------------------------------------------------------


class IntermediateLoss(nn.Module):
    def __init__(self, in_ch, out_ch, dims):
        super().__init__()
        self.loss_head = nn.Sequential(
            normalization(in_ch),
            nn.SiLU(inplace=True),
            conv_nd(dims, in_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.loss_head(x)


# ---------------------------------------------------------------------
# full UNet (multi‑stage variant)
# ---------------------------------------------------------------------


class UNetModel_MS_Former_MultiStage(nn.Module):
    """
    • Shared encoder
    • Stage‑specific decoders
    • Multi‑modal fusion (CT + distance maps)
    """

    # --------------------------------------------------

    def __init__(
        self,
        image_size: Sequence[int],
        *,
        in_channels: int,
        ct_channels: int,
        dis_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        dropout: float = 0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims: int = 2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        num_stages: int = 3,
        stage_channels: Sequence[int] | None = None,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # basic configs
        # ------------------------------------------------------------------
        self.image_size = image_size
        self.in_channels = in_channels
        self.ct_channels = ct_channels
        self.dis_channels = dis_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads if num_heads_upsample == -1 else num_heads_upsample
        self.num_stages = num_stages

        # ------------------------------------------------------------------
        # decide per‑stage channel width
        # ------------------------------------------------------------------
        if stage_channels is None:
            r = 0.7
            self.stage_channels = [
                int(model_channels * (1.5 * (r ** i))) for i in range(num_stages)
            ]
        else:
            assert len(stage_channels) == num_stages
            self.stage_channels = list(stage_channels)

        # ------------------------------------------------------------------
        # time / label embedding
        # ------------------------------------------------------------------
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(inplace=True),
            linear(time_embed_dim, time_embed_dim),
        )
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ------------------------------------------------------------------
        # ------------------------ SHARED ENCODER --------------------------
        # ------------------------------------------------------------------
        ch = in_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self.input_blocks_CT = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, ct_channels, ch, 3, padding=1))]
        )
        self.input_blocks_DIS = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, dis_channels, ch, 3, padding=1))]
        )

        input_block_chans: List[int] = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # three parallel ResBlocks for the three modalities ------------
                shared_kwargs = dict(
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=int(mult * model_channels),
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                layers = [ResBlock(ch, **shared_kwargs)]
                layers_CT = [ResBlock(ch, **shared_kwargs)]
                layers_DIS = [ResBlock(ch, **shared_kwargs)]

                ch = int(mult * model_channels)

                if ds in self.attention_resolutions:
                    AttKws = dict(
                        channels=ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order,
                    )
                    layers.append(AttentionBlock(**AttKws))
                    layers_CT.append(AttentionBlock(**AttKws))
                    layers_DIS.append(AttentionBlock(**AttKws))

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_CT.append(TimestepEmbedSequential(*layers_CT))
                self.input_blocks_DIS.append(TimestepEmbedSequential(*layers_DIS))
                input_block_chans.append(ch)

            # downsample (except for final level) ---------------------------
            if level != len(channel_mult) - 1:
                downsample = (
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                    if resblock_updown
                    else Downsample(ch, conv_resample, dims=dims, out_channels=ch)
                )
                self.input_blocks.append(TimestepEmbedSequential(downsample))
                self.input_blocks_CT.append(TimestepEmbedSequential(downsample))
                self.input_blocks_DIS.append(TimestepEmbedSequential(downsample))
                input_block_chans.append(ch)
                ds *= 2

        # ---------------- middle -----------------------------------------
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
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
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # transformer fusion at the bottleneck ----------------------------
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

        # ------------------------------------------------------------------
        # ---------------------- STAGE‑SPECIFIC DECODERS -------------------
        # ------------------------------------------------------------------
        self.output_blocks = nn.ModuleList()
        self.stage_outputs = nn.ModuleList()
        self.stage_intermediate_heads = nn.ModuleList()

        for stage in range(num_stages):
            stage_width = self.stage_channels[stage]
            stage_out_blocks = nn.ModuleList()
            stage_skip = input_block_chans.copy()
            stage_ds = 2 ** (len(channel_mult) - 1)
            stage_ch = ch

            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(num_res_blocks + 1):
                    skip_ch = stage_skip.pop()
                    layers = [
                        ResBlock(
                            stage_ch + skip_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(stage_width * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    stage_ch = int(stage_width * mult)

                    if stage_ds in self.attention_resolutions:
                        layers.append(
                            AttentionBlock(
                                stage_ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=self.num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )

                    if level and i == num_res_blocks:
                        up = (
                            ResBlock(
                                stage_ch,
                                time_embed_dim,
                                dropout,
                                out_channels=stage_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(stage_ch, conv_resample, dims=dims, out_channels=stage_ch)
                        )
                        layers.append(up)
                        stage_ds //= 2

                    stage_out_blocks.append(TimestepEmbedSequential(*layers))

            self.output_blocks.append(stage_out_blocks)

            # robust head: stage_ch → model_channels → out_channels
            self.stage_outputs.append(
                nn.Sequential(
                    normalization(stage_ch),
                    nn.SiLU(inplace=True),
                    conv_nd(dims, stage_ch, model_channels, 1),
                    nn.SiLU(inplace=True),
                    zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
                )
            )
            self.stage_intermediate_heads.append(IntermediateLoss(stage_ch, out_channels, dims))

    # ======================================================================
    # forward
    # ======================================================================

    def forward(
        self,
        x,
        timesteps: th.Tensor,
        ct,
        dis,
        y=None,
        *,
        stage_indices: th.Tensor | None = None,
    ):
        assert (y is not None) == (self.num_classes is not None)

        if stage_indices is None:
            stage_indices = self.get_stage_from_timestep(timesteps)

        # ---- embed time / label -----------------------------------------
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        # ---- encoder -----------------------------------------------------
        hs = []
        h = x.to(dtype=self.dtype)
        ct_h = ct.to(dtype=self.dtype)
        dis_h = dis.to(dtype=self.dtype)

        for block, block_CT, block_DIS in zip(
            self.input_blocks, self.input_blocks_CT, self.input_blocks_DIS
        ):
            ct_h = block_CT(ct_h, emb)
            dis_h = block_DIS(dis_h, emb)
            h = block(h, emb) + ct_h + dis_h
            hs.append(h)

        h = self.middle_block(h, emb)
        h = self.fusion(ct_h, dis_h, h) + h

        # ---- prepare per‑item skip stacks -------------------------------
        per_item_hs = [[layer[b : b + 1] for layer in hs] for b in range(h.shape[0])]

        # ---- decode per sample ------------------------------------------
        outs, inters = [], []
        for b, stage_idx in enumerate(stage_indices):
            dec = self.output_blocks[stage_idx]
            head = self.stage_outputs[stage_idx]
            aux = self.stage_intermediate_heads[stage_idx]

            h_b = h[b : b + 1]
            skip = per_item_hs[b]

            for mod in dec:
                h_b = th.cat([h_b, skip.pop()], dim=1)
                h_b = mod(h_b, emb[b : b + 1])

            outs.append(head(h_b))
            inters.append(aux(h_b))

        return th.cat(outs, 0), th.cat(inters, 0)

    # ------------------------------------------------------------------

    def get_stage_from_timestep(self, timesteps: th.Tensor, num_timesteps: int = 1000):
        """Simple linear mapping  t ∈ [0,T) → stage id."""
        frac = timesteps.float() / (num_timesteps - 1)
        idx = (frac * (self.num_stages - 1)).round().long()
        return idx.clamp_(0, self.num_stages - 1)
