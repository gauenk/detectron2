# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- extra deps --
import math


# Input Projection
class InputProjSeq(nn.Module):

    def __init__(self, depth=1, in_channel=3, out_channel=64, kernel_size=3, stride=1,
                 norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.depth = depth
        self.in_channel = in_channel
        self.out_channel = out_channel
        layers = []
        for d in range(depth):
            if d > 0: in_channel = out_channel
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                    stride=stride, padding=kernel_size//2))
            layers.append(act_layer(inplace=True))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T,C,H,W)
        x = self.proj(x)
        x = x.view(B,T,-1,H,W)
        return x

    def flops(self, H, W):
        flops = H*W*self.in_channel*self.out_channel*3*3*self.depth
        return flops


class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1,
                 norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T,C,H,W)
        x = self.proj(x)
        # x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        # if self.norm is not None:
        #     x = self.norm(x)
        x = x.view(B,T,-1,H,W)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        # print("Input_proj:{%.2f}"%(flops/1e9))
        return flops

# Output Projection Sequence
class OutputProjSeq(nn.Module):
    def __init__(self, depth=1, in_channel=64, out_channel=3, kernel_size=3,
                 stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.depth = depth
        self.in_channel = in_channel
        self.out_channel = out_channel
        layers,out_chn = [],in_channel
        for d in range(depth):
            if d == (depth-1): out_chn = out_channel
            layers.append(nn.Conv2d(in_channel, out_chn, kernel_size=3,
                                    stride=stride, padding=kernel_size//2))
            layers.append(act_layer(inplace=True))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T,C,H,W)
        x = self.proj(x)
        x = x.view(B,T,-1,H,W)
        return x

    def flops(self, H, W):
        flops = H*W*self.in_channel*self.out_channel*3*3*self.depth
        return flops

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3,
                 stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T,C,H,W)
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))
        # x = x.transpose(1, 2).view(T*B, C, H, W)
        x = self.proj(x)
        BT,C,H,W = x.shape
        if self.norm is not None:
            x = self.norm(x)
        x = x.view(B,T,-1,H,W)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        # print("Output_proj:{%.2f}"%(flops/1e9))
        return flops

class SepConv2d(th.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = th.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = th.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, HW): 
        flops = 0
        flops += HW*self.in_channels*self.kernel_size**2/self.stride**2
        flops += HW*self.in_channels*self.out_channels
        # print("SeqConv2d:{%.2f}"%(flops/1e9))
        return flops

class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3,
                 q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q,k,v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = q_L*self.dim*self.inner_dim+kv_L*self.dim*self.inner_dim*2
        return flops

class ConvProjectionNoReshape(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, qk_frac=1.,
                 kernel_size=1,q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        inner_dim_qk = int(qk_frac*dim_head) * heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = nn.Conv2d(dim, inner_dim_qk, kernel_size=kernel_size,
                              stride=q_stride, padding=pad, bias=bias,
                              groups=1,padding_mode="reflect")
        self.to_k = nn.Conv2d(dim, inner_dim_qk, kernel_size=kernel_size,
                              stride=k_stride, padding=pad, bias=bias,
                              groups=1,padding_mode="reflect")
        self.to_v = nn.Conv2d(dim, inner_dim, kernel_size=kernel_size,
                              stride=v_stride, padding=pad, bias=bias,
                              groups=1,padding_mode="reflect")

    def forward(self, x, attn_kv=None):

        # -- unpack --
        b, c, h, w = x.shape
        nheads = self.heads
        attn_kv = x if attn_kv is None else attn_kv

        # -- forward --
        q = self.to_q(x)
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)

        return q,k,v

    def flops(self, H, W):
        flops = 0
        flops += conv2d_flops(self.to_q,H,W)
        flops += conv2d_flops(self.to_k,H,W)
        flops += conv2d_flops(self.to_v,H,W)
        return flops

class ConvQKV(nn.Module):
    def __init__(self, input_dim, heads = 8, dim_head = 64, qk_frac=1.,
                 kernel_size=1,q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        inner_dim_qk = int(qk_frac*dim_head) * heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = nn.Conv2d(input_dim, inner_dim_qk, kernel_size=kernel_size,
                              stride=q_stride, padding=pad, bias=bias,
                              groups=1,padding_mode="reflect")
        self.to_k = nn.Conv2d(input_dim, inner_dim_qk, kernel_size=kernel_size,
                              stride=k_stride, padding=pad, bias=bias,
                              groups=1,padding_mode="reflect")
        self.to_v = nn.Conv2d(input_dim, inner_dim, kernel_size=kernel_size,
                              stride=v_stride, padding=pad, bias=bias,
                              groups=1,padding_mode="reflect")

    def forward(self, x, attn_kv=None):

        # -- unpack --
        b, c, h, w = x.shape
        nheads = self.heads
        attn_kv = x if attn_kv is None else attn_kv

        # -- forward --
        q = self.to_q(x)
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)

        return q,k,v

    def flops(self, H, W):
        flops = 0
        flops += conv2d_flops(self.to_q,H,W)
        flops += conv2d_flops(self.to_k,H,W)
        flops += conv2d_flops(self.to_v,H,W)
        return flops

def conv2d_flops(conv,H,W):

    # -- unpack --
    ksize = conv.kernel_size
    stride = conv.stride
    groups = conv.groups
    # W = conv.weights
    # b = conv.bias
    in_C = conv.in_channels
    out_C = conv.out_channels

    # -- flop --
    flop = (H // stride[0]) * (W // stride[1]) * (ksize[0] * ksize[1])
    flop *= ((in_C//groups) * (out_C//groups) * groups)
    return flop
