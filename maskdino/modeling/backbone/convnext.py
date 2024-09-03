# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/ConvNeXt/blob/main/object_detection/mmdet/models/backbones/convnext.py
# ------------------------------------------------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
# from detrex.layers import LayerNorm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.backbone import Backbone


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r"""Implement paper `A ConvNet for the 2020s <https://arxiv.org/pdf/2201.03545.pdf>`_.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (Sequence[int]): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (List[int]): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
    """

    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=(0,1,2,3),
        frozen_stages=-1,
    ):
        super().__init__()

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        assert (
            self.frozen_stages <= 4
        ), f"only 4 stages in ConvNeXt model, but got frozen_stages={self.frozen_stages}."

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, channel_last=False),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, channel_last=False),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        norm_layer = partial(LayerNorm, eps=1e-6, channel_last=False)
        for i_layer in out_indices:
            layer = norm_layer(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()
        self._out_features = ["res{}".format(i+2) for i in self.out_indices]
        self._out_feature_channels = {"res{}".format(i+2): dims[i] for i in self.out_indices}
        self._out_feature_strides = {"res{}".format(i+2): 2 ** (i + 2) for i in self.out_indices}
        self._size_devisibility = 32

        self.apply(self._init_weights)

    def _freeze_stages(self):
        if self.frozen_stages >= 1:
            for i in range(0, self.frozen_stages):
                # freeze downsample_layer's parameters
                downsampler_layer = self.downsample_layers[i]
                downsampler_layer.eval()
                for param in downsampler_layer.parameters():
                    param.requires_grad = False

                # freeze stage layer's parameters
                stage = self.stages[i]
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        outs = {}
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x)
                outs["res{}".format(i)] = x_out

        return outs
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()

    def forward(self, x):
        """Forward function of `ConvNeXt`.

        Args:
            x (torch.Tensor): the input tensor for feature extraction.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "p1") to tensor
        """
        x = self.forward_features(x)
        return x



class LayerNorm(nn.Module):
    r"""LayerNorm which supports both channel_last (default) and channel_first data format.
    The inputs data format should be as follows:
        - channel_last: (bs, h, w, channels)
        - channel_first: (bs, channels, h, w)

    Args:
        normalized_shape (tuple): The size of the input feature dim.
        eps (float): A value added to the denominator for
            numerical stability. Default: True.
        channel_last (bool): Set True for `channel_last` input data
            format. Default: True.
    """

    def __init__(self, normalized_shape, eps=1e-6, channel_last=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.channel_last = channel_last
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """Forward function for `LayerNorm`"""
        if self.channel_last:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


    

@BACKBONE_REGISTRY.register()
class DConvNext(ConvNeXt , Backbone):
    def __init__(self,cfg,input_shape):
        in_chans_convnext = 3
        depths_convnext = cfg.MODEL.CONV.DEPTHS
        dims_convnext = cfg.MODEL.CONV.DIM 
        drop_path_rate_convnext = cfg.MODEL.CONV.DROP_RATE
        layer_scale_init_value_convnext = cfg.MODEL.CONV.LAYER_SCALE
        out_indices_convnext = (0,1,2,3)
        frozen_stages_convnext = -1
        super().__init__(
            in_chans=in_chans_convnext,
            depths=depths_convnext,
            dims=dims_convnext,
            drop_path_rate=drop_path_rate_convnext,
            layer_scale_init_value=layer_scale_init_value_convnext,
            out_indices=out_indices_convnext,
            frozen_stages=frozen_stages_convnext)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        y = super().forward(x)
        
        outputs = {}
        for i in range(len(y)):
            outputs[f"res{i+2}"] = y[f"res{i}"]
            # print(f"res{i}",outputs[f'res{i}'].shape)
        return outputs
        # return y

    # def output_shape(self):
    #     return {
    #         name: ShapeSpec(
    #             channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
    #         )
    #         for name in self._out_features
    #     }

    # @property
    # def size_divisibility(self):
    #     return 32