import logging
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from einops import rearrange

from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous

from .backbone import Backbone
from .utils import (
    PatchEmbed4D,
    LayerNorm2D,
    get_abs_pos,
    window_partition,
    window_unpartition,
)

import stnls
from .res_block import ResBlockList
from .proj import InputProjSeq,OutputProjSeq

logger = logging.getLogger(__name__)


__all__ = ["ViTStnls", "SimpleFeaturePyramidStnls", "get_vit_lr_decay_rate_stnls"]


class ResBottleneckBlockStnls(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = LayerNorm2D(bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = LayerNorm2D(bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, vid):
        B = vid.shape[0]
        vid = rearrange(vid,'b t c h w -> (b t) c h w')
        out = vid
        for layer in self.children():
            out = layer(out)
        out = vid + out
        vid = rearrange(vid,'(b t) c h w -> b t c h w',b=B)
        return out


class BlockStnls(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        attn_cfg, search_cfg, normz_cfg, agg_cfg,
        drop_path=0.0,
        act_layer=nn.GELU,
        use_residual_block=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_residual_block (bool): If True, use a residual block after the MLP block.
                parameter size.
        """
        super().__init__()
        from timm.models.layers import DropPath, Mlp

        edim = dim*num_heads
        self.norm0 = LayerNorm2D(edim)
        self.norm1 = LayerNorm2D(edim)
        self.attn = stnls.nn.NonLocalAttention(attn_cfg, search_cfg, normz_cfg, agg_cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.res0 = ResBlockList(attn_cfg.nres, edim, attn_cfg.res_ksize)
        self.use_residual_block = use_residual_block
        if use_residual_block:
            self.res1 = ResBottleneckBlockStnls(
                in_channels=edim,
                out_channels=edim,
                bottleneck_channels=edim // 2,
                act_layer=act_layer,
            )

    def forward(self, vid, flows=None):
        # -- replace flows --
        from dev_basics import flow
        if flows is None:
            flows = flow.run_zeros(vid)

        shortcut = vid
        vid = self.norm0(vid)
        vid = self.attn(vid,flows=flows)
        vid = shortcut + self.drop_path(vid)
        vid = vid + self.drop_path(self.res0(self.norm1(vid)))
        if self.use_residual_block:
            vid = self.res1(vid.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return vid


class ViTStnls(Backbone):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        embed_dim=9,
        in_feature=3,
        depth=10,
        num_heads=2,
        drop_path_rate=0.0,
        # norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        # out_feature="last_feat",
        # nres=3,
        # res_ksize=5,
        arch_cfg=None,
        attn_cfg=None,
        search_cfg=None,
        normz_cfg=None,
        agg_cfg=None,
        out_feature="last_feat",
    ):
        """
        Args:
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()

        # -- setup --
        self.in_feature = in_feature
        self.out_feature = out_feature

        # -- input/output --
        self.input_proj = InputProjSeq(depth=arch_cfg.input_proj_depth,
                                       in_channel=in_feature,
                                       out_channel=arch_cfg.embed_dim*num_heads,
                                       kernel_size=3, stride=1,
                                       act_layer=nn.LeakyReLU)
        self.output_proj = OutputProjSeq(depth=arch_cfg.output_proj_depth,
                                         in_channel=arch_cfg.embed_dim*num_heads,
                                         out_channel=arch_cfg.embed_dim,
                                         kernel_size=3, stride=1,
                                         act_layer=nn.LeakyReLU)


        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(arch_cfg.depth):
            block = BlockStnls(
                dim=arch_cfg.embed_dim,
                num_heads=arch_cfg.num_heads,
                attn_cfg=attn_cfg,
                search_cfg=search_cfg,
                normz_cfg=normz_cfg,
                agg_cfg=agg_cfg,
                drop_path=dpr[i],
                act_layer=act_layer,
            )
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: 1}
        self._out_features = [out_feature]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, vid, flows=None):

        # -- unpack --
        b,t,c,h,w = vid.shape

        # -- Input Projection --
        vid = self.input_proj(vid)
        for blk in self.blocks:
            vid = blk(vid,flows=flows)
        vid = self.output_proj(vid)
        outputs = {self._out_features[0]: vid}
        return outputs


class SimpleFeaturePyramidStnls(Backbone):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super().__init__()
        assert isinstance(net, Backbone)

        self.in_feature = in_feature
        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        # print("input_shapes: ",input_shapes)
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        # print("strides: ",strides)
        # _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif scale == 0.25:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                          nn.MaxPool2d(kernel_size=2, stride=2)]
            elif scale == 0.125:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            # stage = int(math.log2(strides[idx]))
            stage = idx
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        # self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        self._out_feature_strides = {"p{}".format(s): s for s,_ in enumerate(strides)}
        # print("_out_feature_strides: ",list(self._out_feature_strides.keys()))
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x, flows=None):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """


        bottom_up_features = self.net(x,flows=flows)
        features = bottom_up_features[self.in_feature]
        results = []
        # print("features.shape: ",features.shape)

        B = features.shape[0]
        features = rearrange(features,'b t c h w -> (b t) c h w')
        for stage in self.stages:
            sfeats = stage(features)
            # print("sfeats.shape: ",sfeats.shape)
            # sfeats = rearrange(sfeats,'(b t) c h w -> b t c h w',b=B)
            results.append(sfeats)

        # print(len(results))
        # print(self.top_block)
        # print(self._out_features)
        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            # print(top_"block_in_feature.shape
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}


def get_vit_lr_decay_rate_stnls(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)
