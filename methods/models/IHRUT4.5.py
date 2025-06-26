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

from . import duf_mixs2_intf

# 代码从头到尾大改，引入自己目前全部的设计


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


def nearest_2_power_int(x):
    return 2 ** (x - 1).bit_length()


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


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = (
        img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    )
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


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
        # t = [eval(f"t{i}") for i in range(6)]
        # print([t[i + 1] - t[i] for i in range(5)])
        return x


class StripeAttention(nn.Module):

    def __init__(self, num_feat, init_mu=0.5, squeeze_factor=8, memory_blocks=128):
        super(StripeAttention, self).__init__()
        self.mu = nn.Parameter(torch.full((1,), init_mu))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(
            nn.Linear(num_feat, num_feat // squeeze_factor),
            # nn.ReLU(inplace=True)
        )
        self.upnet = nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat),
            # nn.Linear(num_feat, num_feat),
            nn.Sigmoid(),
        )
        self.mb = torch.nn.Parameter(
            torch.randn(num_feat // squeeze_factor, memory_blocks)
        )
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):
        B, C, H, W = x.shape
        # B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * W, C, H)
        y = self.pool(x).reshape(B * W, C)

        low_rank_f = self.subnet(y).unsqueeze(2)  # B*W, C//4, 1
        mbg = self.mb.unsqueeze(0).repeat(B * W, 1, 1)  # B*W, C//4, 128
        f1 = (low_rank_f.transpose(1, 2)) @ mbg
        f_dic_c = F.softmax(
            f1 * (int(self.low_dim) ** (-0.5)), dim=-1
        )  # get the similarity information
        y1 = f_dic_c @ mbg.transpose(1, 2)
        y2 = self.upnet(y1).transpose(1, 2)
        x = x * y2 * self.mu
        x = x.reshape(B, W, C, H).permute(0, 2, 3, 1)
        return x


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
        self.sc_attn = StripeAttention(dim, init_mu=0.5, memory_blocks=dim // 2)
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

        x = shortcut + self.sc_attn(shortcut)
        x = x + self.ffn(self.norm2(x))

        return x


class Sample(nn.Module):
    def __init__(self, in_channels, out_channels, mode="down", type="W"):
        super(Sample, self).__init__()
        if mode == "up":
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, 2, 2, 0, bias=False
            )
        elif mode == "down":
            self.conv = nn.Conv2d(in_channels, out_channels, 2, 2, 0, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        pass

    def forward(self, x):
        return self.conv(x)


class IHRTrans(nn.Module):  # no drop path, u-shaped
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=64,
        window_size=8,
        depth=3,
    ):
        super(IHRTrans, self).__init__()
        self.depth = depth
        num_heads = [2**i for i in range(depth)]
        num_heads = num_heads + list(reversed(num_heads[:-1]))
        channels = [in_channels] + list(dim * np.array(num_heads)) + [out_channels]

        # channels = [dim * 2**i for i in range(depth)]
        # channels = (
        #     [in_channels] + channels + list(reversed(channels[:-1])) + [out_channels]
        # )

        self.body = nn.ModuleList()
        for i in range(2 * self.depth - 1):
            if i == 0:
                s_mode = ""
            elif i < depth:
                s_mode = "down"
            elif i < 2 * depth - 1:
                s_mode = "up"
            block = TransBlock(
                dim=channels[i + 1],
                num_heads=num_heads[i],
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
            )
            self.body.append(
                nn.Sequential(*[Sample(channels[i], channels[i + 1], s_mode), block])
            )

        self.body.append(
            nn.Sequential(
                *[nn.Conv2d(dim, dim, 3, 1, 1), nn.Conv2d(dim, out_channels, 3, 1, 1)]
            )
        )

    def forward(self, x):
        xs = []
        ra = torch.arange(2 * self.depth)
        for i in ra[: self.depth]:
            x = self.body[i](x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            xs.append(x)
        for i in ra[self.depth :]:
            x = self.body[i](xs[2 * self.depth - i - 1] + x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
        return x


# Data


class DataBlock(nn.Module):
    def __init__(self, y_channels, out_channels, init_mu):
        super(DataBlock, self).__init__()
        # self.norm = LayerNorm(out_channels, "BiasFree")
        self.sc_attn = StripeAttention(out_channels, init_mu=init_mu)

    # CA Module
    def forward(self, x_i, y, func_A, func_A_inv, appendix):
        # compute r_k
        yb = func_A(x_i, appendix)
        r_i = x_i - self.sc_attn(func_A_inv(y - yb, appendix))
        return r_i

    pass


class IHRUT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
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
            self.num_blocks = 2
        init_mu = 0.5
        self.func_A = func_A
        self.func_A_inv = func_A_inv

        self.data_block = nn.ModuleList(
            [
                DataBlock(in_channels, out_channels, init_mu)
                for _ in range(self.num_blocks)
            ]
        )

        # opt = duf_mixs2_intf.get_opt()
        # opt.in_dim = out_channels
        # opt.out_dim = out_channels
        # opt.dim = n_feat
        # self.prior_block = nn.ModuleList(
        #     [duf_mixs2_intf.MixS2_Transformer(opt) for _ in range(opt.stage - 2)]
        # )
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
        for i in range(self.num_stages):
            if self.num_blocks == 2:
                if i == self.num_stages - 1:
                    j = 1
                else:
                    j = 0

            x = self.data_block[j](x, y, self.func_A, self.func_A_inv, appendix)
            # padding to multiple of 32
            hb, wb = (32, 32)
            H_pad = (hb - H % hb) % hb
            W_pad = (wb - W % wb) % wb
            x = F.pad(x, (0, W_pad, 0, H_pad), mode="reflect")
            x = self.prior_block[j](x)
            x = x[..., :H, :W]
            j += 1
            pass
        return x

    pass
