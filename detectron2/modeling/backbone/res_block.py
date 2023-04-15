import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class ResBlockList(nn.Module):

    def __init__(self, nres, n_feats, kernel_size):
        super().__init__()
        if nres > 0:
            res = []
            for _ in range(nres):
                res.append(ResBlock(conv, n_feats, kernel_size))
            self.res = nn.Sequential(*res)
        else:
            self.res = nn.Identity()

    def forward(self,vid):
        B,T = vid.shape[:2]
        vid = rearrange(vid,'b t c h w -> (b t) c h w')
        vid = self.res(vid)
        vid = rearrange(vid,'(b t) c h w -> b t c h w',b=B)
        return vid

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res
