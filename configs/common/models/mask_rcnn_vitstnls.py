
import torch.nn as nn
from functools import partial
from detectron2.config import LazyCall as L
from detectron2.modeling import ViTStnls, SimpleFeaturePyramidStnls
from detectron2.modeling.backbone.fpn import LastLevelMaxPool,LastLevelMaxPoolStnls

from .mask_rcnn_fpn_stnls import model
from ..data.constants import constants

model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"

model.backbone = L(SimpleFeaturePyramidStnls)(
    net=L(ViTStnls)(
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    # scale_factors=(2.0, 2.0, 1.0, 0.5),
    scale_factors=(4., 2., 1., 0.5),
    top_block=L(LastLevelMaxPoolStnls)(),
    norm="LN",
    square_pad=128,
)

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "LN"

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
# model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.conv_dims = [64, 64, 64, 64]
model.roi_heads.box_head.fc_dims = [1024]
