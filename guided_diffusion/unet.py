from abc import abstractmethod
import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

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

class UNetModel_MS_Former_MultiStage(nn.Module):
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
        num_stages=3,
        stage_channels=None,
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

        if stage_channels is None:
            self.stage_channels = [max(model_channels, model_channels // (i+1)) for i in range(num_stages)]
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
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self.input_blocks_CT = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, ct_channels, ch, 3, padding=1))])
        self.input_blocks_DIS = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, dis_channels, ch, 3, padding=1))])

        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential())
                self.input_blocks_CT.append(TimestepEmbedSequential())
                self.input_blocks_DIS.append(TimestepEmbedSequential())
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential())
                self.input_blocks_CT.append(TimestepEmbedSequential())
                self.input_blocks_DIS.append(TimestepEmbedSequential())
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential()
        self.fusion = ViT_fusion(
            image_size=(image_size[0]//(2**(len(channel_mult)-1)), image_size[1]//(2**(len(channel_mult)-1))),
            patch_size=(image_size[0]//(2**(len(channel_mult))), image_size[1]//(2**(len(channel_mult)))),
            dim=1024,
            heads=1,
            mlp_dim=2048,
            channels=model_channels * channel_mult[-1],
            dim_head=64
        )

        self.output_blocks = nn.ModuleList()
        self.stage_outputs = nn.ModuleList()

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
                            else Upsample(stage_ch_tracker, conv_resample, dims=dims, out_channels=out_ch)
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
