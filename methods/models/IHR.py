import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_

from .network_dncnn import DnCNN
from .network_unet import UNetRes
from .restormer_arch import LayerNorm
from .restormer_arch import Restormer
from .network_scunet import SCUNet
from .qrnn import get_qrnn3d
from .SST import SST
from .sert import SERT
from .TDSAT import TDSAT


class IHR(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        func_0,
        prior_type,
        prior_nc,
        prior_nb,
        batch_size=1,
    ):
        super(IHR, self).__init__()
        self.func_0 = func_0

        self.prior_block = None
        if prior_type == "DnCNN":
            self.prior_block = DnCNN(out_channels, out_channels, prior_nc, prior_nb)
        elif prior_type == "ResUNet":
            self.prior_block = UNetRes(out_channels, out_channels, prior_nc, prior_nb)
        elif prior_type == "SST":
            self.prior_block = SST(
                inp_channels=out_channels,
                dim=90,
                window_size=8,
                depths=[6, 6, 6, 6, 6, 6],
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
            )
        elif prior_type == "SERT":
            self.prior_block = SERT(
                inp_channels=out_channels,
                dim=96,
                window_sizes=[16, 32, 32],
                depths=[4, 4, 4],
                num_heads=[6, 6, 6],
                split_sizes=[1, 2, 4],
                mlp_ratio=2,
                weight_factor=0.1,
                memory_blocks=128,
                down_rank=8,
            )
        elif prior_type == "Restormer":
            self.prior_block = Restormer(
                inp_channels=out_channels,
                out_channels=out_channels,
            )
        elif prior_type == "SCUNet":
            self.prior_block = SCUNet(
                in_nc=out_channels,
                config=[2, 2, 2, 2, 2, 2, 2],
                input_resolution=256,
            )
        elif prior_type == "TDSAT":
            self.prior_block = TDSAT(
                in_channels=batch_size,
                channels=16,
                num_half_layer=5,
                sample_idx=[1, 3],
            )

    def forward(self, y, appendix):
        # utils.check_gpu_mem(y.device)
        _, _, H, W = y.shape
        x = self.func_0(y, appendix)
        hb, wb = (32, 32)
        H_pad = (hb - H % hb) % hb
        W_pad = (wb - W % wb) % wb
        x = F.pad(x, (0, W_pad, 0, H_pad), mode="reflect")
        if self.prior_block is not None:
            x = self.prior_block(x)
        x = x[..., :H, :W]
        return x

    pass
