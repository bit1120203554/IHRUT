import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_
import numbers

from . import MAUN_intf

# 简化版


#### Layer Norm
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type=None):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class NLSA(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super(NLSA, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StripeAttention(nn.Module):

    def __init__(self, num_feat, out_feat=0, squeeze_factor=4):
        super(StripeAttention, self).__init__()
        if out_feat == 0:
            out_feat = num_feat

        self.C1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat),
            nn.GELU(),
            nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.C2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )
        self.outnet = nn.Conv2d(num_feat, out_feat, 1, 1, 0, bias=False)

    def forward(self, x):
        # input zk - zk_1
        B, C, H, W = x.shape
        x = self.C1(x)
        CA = rearrange(x, " b c h w -> b (c w) h", c=C, w=W)
        CA = self.pool(CA)
        CA = rearrange(CA, " b (c w) h -> b c h w", c=C, w=W)
        CA = self.C2(CA)
        CA = torch.mul(x, CA)
        CA = self.outnet(CA)
        return CA


class FFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                groups=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            act_layer(),
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        # self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        B, C, H, W = x.size()
        x = rearrange(x, " b c h w -> b (h w) c", h=H, w=W)
        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, " b (h w) c -> b c h w", h=H, w=W)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)
        # flatten
        x = rearrange(x, " b c h w -> b (h w) c", h=H, w=W)
        x = self.linear2(x)
        x = rearrange(x, " b (h w) c -> b c h w", h=H, w=W)
        return x


# Prior
class TransBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        shift_size=0,
        ffn_ratio=2,
    ):
        super(TransBlock, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.swin_attn = NLSA(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
        )
        self.sc_attn = StripeAttention(dim)
        self.ffn = FFN(dim, int(dim * ffn_ratio))
        pass

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)
        # window operation
        x = rearrange(x, "b c h w -> b h w c")
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.swin_attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = rearrange(x, "b h w c -> b c h w")

        x = shortcut + self.sc_attn(x)
        x = x + self.ffn(self.norm2(x))
        return x


class IHRTrans(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=64,
        # window_size=8,
    ):
        super().__init__()
        self.stage_interaction = False
        self.block_interaction = False
        self.embedding = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1, padding=0)

        self.BottleNeck = TransBlock(dim=dim, num_heads=1)

        self.Decoder = nn.ModuleList(
            [
                TransBlock(dim=dim, num_heads=1),
                TransBlock(dim=dim, num_heads=1),
            ]
        )

        self.Downs = nn.ModuleList(
            [
                nn.Conv2d(dim, dim, 2, 2, 0, bias=False),
                nn.Conv2d(dim, dim, 2, 2, 0, bias=False),
            ]
        )
        self.cross_stage = nn.ModuleList(
            [
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
            ]
        )
        self.Ups = nn.ModuleList(
            [
                nn.ConvTranspose2d(dim, dim, 2, 2, 0, bias=False),
                nn.ConvTranspose2d(dim, dim, 2, 2, 0, bias=False),
                nn.ConvTranspose2d(dim, dim, 2, 2, 0, bias=False),
            ]
        )

        self.fusions = nn.ModuleList(
            [
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
            ]
        )

        self.mapping = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1, padding=0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, last_feats):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        if last_feats is not None:
            # lf1, lf2 = last_feats
            lf1, lf2, lf4 = last_feats

        this_feats = []
        x1 = self.embedding(x)

        if last_feats is None:
            lf1 = x1
        res1 = self.cross_stage[0](torch.cat([x1, lf1], dim=1))

        x2 = self.Downs[0](res1)

        if last_feats is None:
            lf2 = x2
        res2 = self.cross_stage[1](torch.cat([x2, lf2], dim=1))
        x4 = self.Downs[1](res2)

        if last_feats is None:
            lf4 = x4
        res4 = self.cross_stage[2](torch.cat([x4, lf4], dim=1))
        res4 = self.BottleNeck(res4)
        this_feats.insert(0, res4)

        dec_res2 = self.Ups[0](res4)
        dec_res2 = self.fusions[0](torch.cat([dec_res2, res2], dim=1))
        this_feats.insert(0, dec_res2)
        dec_res2 = self.Decoder[0](dec_res2)

        dec_res1 = self.Ups[1](dec_res2)
        dec_res1 = self.fusions[1](torch.cat([dec_res1, res1], dim=1))
        this_feats.insert(0, dec_res1)
        dec_res1 = self.Decoder[1](dec_res1)

        out = self.mapping(dec_res1) + x

        return out[:, :, :h_inp, :w_inp], this_feats


# Data
class DataBlock(nn.Module):
    def __init__(self, y_channels, out_channels, stage=0):
        super(DataBlock, self).__init__()
        self.stage = stage
        if stage != 0:
            self.sc_attn = StripeAttention(out_channels)

    # CA Module
    def forward(self, x_i, y, func_A, func_A_inv, appendix, alpha, k_last):
        # compute r_k
        yb = func_A(x_i, appendix)
        kai = alpha * (y - yb)
        kai = func_A_inv(kai, appendix)

        if self.stage == 0:
            mom = kai
        else:
            mom = self.sc_attn(kai + x_i - k_last)
        r_i = x_i + mom  # 1
        return r_i

    pass


class IHRUT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        func_0,
        func_A,
        func_A_inv,
        num_stages=5,
        n_feat=64,
        sharing=False,
    ):
        super(IHRUT, self).__init__()
        self.num_stages = num_stages  # >=2
        self.num_blocks = num_stages
        if sharing:
            self.num_blocks = 3
        self.func_0 = func_0
        self.func_A = func_A
        self.func_A_inv = func_A_inv

        self.para_estimator = StripeAttention(num_feat=in_channels, out_feat=num_stages)
        self.data_block = nn.ModuleList(
            [DataBlock(out_channels, out_channels, _) for _ in range(self.num_blocks)]
        )

        self.prior_block = nn.ModuleList(
            [
                IHRTrans(out_channels, out_channels, n_feat)
                for _ in range(self.num_blocks)
            ]
        )

    def forward(self, y, appendix):
        # utils.check_gpu_mem(y.device)
        _, _, H, W = y.shape
        j = 0
        x = self.func_A_inv(y, appendix)
        alphas = self.para_estimator(y)
        k_cur = x
        cross_last = None

        for i in range(self.num_stages):
            k_last = k_cur
            k_cur = x
            if self.num_blocks == 3:
                if i == self.num_stages - 1:
                    j = 2
                elif i == 0:
                    j = 0
                else:
                    j = 1

            alpha = alphas[:, i, :, :]
            x = self.data_block[j](
                x, y, self.func_A, self.func_A_inv, appendix, alpha, k_last
            )
            hb, wb = (32, 32)
            H_pad = (hb - H % hb) % hb
            W_pad = (wb - W % wb) % wb
            x = F.pad(x, (0, W_pad, 0, H_pad), mode="reflect")
            x, cross_last = self.prior_block[j](x, cross_last)  # [0]
            x = x[..., :H, :W]

            j += 1
            pass
        return x

    pass
