from abc import abstractmethod
import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------
# The imports below are part of the existing code base ("nn.py", "vit.py", etc.).
# They remain unchanged.
# ------------------------------------------------------------------------------------
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


########################################
## Original Classes from Unet Old Full ##
## (Unchanged, except for additional comments)
########################################

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
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
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

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
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
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
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

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
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        # use_new_attention_order decides between QKVAttention and QKVAttentionLegacy
        self.attention = QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

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
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


#####################################################
## NEW/EDITED SECTION: Multi-stage diffusion "fix" ##
## UNetModel_MS_Former_MultiStage class            ##
#####################################################

# -----------------------------
# Additional: IntermediateLoss
# for optional intermediate supervision
# -----------------------------
class IntermediateLoss(nn.Module):
    def __init__(self, in_ch, out_ch, dims):
        super().__init__()
        self.loss_head = nn.Sequential(
            normalization(in_ch),
            nn.SiLU(),
            conv_nd(dims, in_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.loss_head(x)


class UNetModel_MS_Former_MultiStage(nn.Module):
    """
    Multi-stage version of UNetModel_MS_Former with:
    - Shared encoder across all stages
    - Stage-specific decoders
    - Multi-modal fusion (CT + distance maps)
    - Consistent channel dimensions

    NOTE: This class is newly added/edited in the codebase.
    """
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
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        num_stages=3,  # Number of diffusion stages
        stage_channels=None,  # Custom channels per stage
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.ct_channels = ct_channels
        self.dis_channels = dis_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.num_stages = num_stages

        # If no stage-specific channels given, create default distribution
        if stage_channels is None:
            base=model_channels
            self.stage_channels = [int(base*1.5), base, int(base*0.5)]
        else:
            self.stage_channels = stage_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        self.input_blocks_CT = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, ct_channels, ch, 3, padding=1))
        ])
        self.input_blocks_DIS = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, dis_channels, ch, 3, padding=1))
        ])

        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                layers_CT = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                layers_DIS = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)

                # If attention at this resolution
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    layers_CT.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    layers_DIS.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_CT.append(TimestepEmbedSequential(*layers_CT))
                self.input_blocks_DIS.append(TimestepEmbedSequential(*layers_DIS))
                input_block_chans.append(ch)

            # Downsample (except last)
            if level != len(channel_mult) - 1:
                out_ch = ch
                downsample_layer = (
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                    if resblock_updown
                    else Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch
                    )
                )
                self.input_blocks.append(TimestepEmbedSequential(downsample_layer))
                self.input_blocks_CT.append(TimestepEmbedSequential(downsample_layer))
                self.input_blocks_DIS.append(TimestepEmbedSequential(downsample_layer))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

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

        self.fusion = ViT_fusion(
            image_size=(
                image_size[0] // (2 ** (len(channel_mult) - 1)),
                image_size[1] // (2 ** (len(channel_mult) - 1))
            ),
            patch_size=(
                image_size[0] // (2 ** len(channel_mult)),
                image_size[1] // (2 ** len(channel_mult))
            ),
            dim=1024,
            heads=1,
            mlp_dim=2048,
            channels=model_channels * channel_mult[-1],
            dim_head=64
        )

        self.output_blocks = nn.ModuleList()
        self.stage_outputs = nn.ModuleList()
        self.stage_intermediate_heads = nn.ModuleList()  # new

        for stage in range(num_stages):
            stage_ch = self.stage_channels[stage]
            stage_output_blocks = nn.ModuleList()
            stage_input_block_chans = input_block_chans.copy()
            stage_ds = 2 ** (len(channel_mult) - 1)
            stage_ch_tracker = ch

            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(num_res_blocks + 1):
                    ich = stage_input_block_chans.pop()
                    layers = [
                        ResBlock(
                            stage_ch_tracker + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=int(stage_ch * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    stage_ch_tracker = int(stage_ch * mult)

                    # Optional attention in up block
                    if stage_ds in attention_resolutions:
                        layers.append(
                            AttentionBlock(
                                stage_ch_tracker,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )

                    # Upsample after finishing this level
                    if level and i == num_res_blocks:
                        out_ch = stage_ch_tracker
                        layers.append(
                            ResBlock(
                                stage_ch_tracker,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(
                                stage_ch_tracker, conv_resample, dims=dims, out_channels=out_ch
                            )
                        )
                        stage_ds //= 2

                    stage_output_blocks.append(TimestepEmbedSequential(*layers))

            self.output_blocks.append(stage_output_blocks)
            self.stage_outputs.append(
                nn.Sequential(
                    normalization(stage_ch_tracker),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
                )
            )

            # For optional intermediate supervision
            self.stage_intermediate_heads.append(IntermediateLoss(stage_ch_tracker, out_channels, dims))

    def forward(self, x, timesteps, ct, dis, y=None, stage_indices=None):
        """
        Multi-modal input forward, returns both final outputs and optional intermediate outputs
        """
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"

        if stage_indices is None:
            stage_indices = self.get_stage_from_timestep(timesteps)

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        hs = []
        h = x.type(self.dtype)
        ct_h = ct.type(self.dtype)
        dis_h = dis.type(self.dtype)

        for i, module in enumerate(self.input_blocks):
            ct_h = self.input_blocks_CT[i](ct_h, emb)
            dis_h = self.input_blocks_DIS[i](dis_h, emb)
            h = module(h, emb) + ct_h + dis_h
            hs.append(h)

        h = self.middle_block(h, emb)
        h = self.fusion(ct_h, dis_h, h) + h

        outputs = []
        intermediate_outputs = []

        for i, stage_idx in enumerate(stage_indices):
            stage_decoder = self.output_blocks[stage_idx]
            stage_out = self.stage_outputs[stage_idx]
            stage_aux = self.stage_intermediate_heads[stage_idx]

            sample_h = h[i : i + 1]
            sample_hs = [h_i[i : i + 1] for h_i in hs]

            for module in stage_decoder:
                sample_h = th.cat([sample_h, sample_hs.pop()], dim=1)
                sample_h = module(sample_h, emb[i : i + 1])

            # Final stage output
            outputs.append(stage_out(sample_h))
            # Intermediate output (useful if you want deeper supervision)
            intermediate_outputs.append(stage_aux(sample_h))

        # Now we return both final outputs and the intermediates
        return torch.cat(outputs, dim=0), torch.cat(intermediate_outputs, dim=0)

    def get_stage_from_timestep(self, timesteps, num_timesteps=1000):
         """
        Map DDPM timestep (0 … num_timesteps‑1) → stage index (0 … self.num_stages‑1).
        """
        # 1. [0,1] range
        normalized_t = timesteps.float() / (num_timesteps - 1)

        # 2. scale to number‑of‑stages and round
        stage_indices = (normalized_t * (self.num_stages - 1)).round().long()

        # 3. clamp for numerical safety
        stage_indices = stage_indices.clamp(0, self.num_stages - 1)
        return stage_indices
