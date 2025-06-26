import torch.nn.functional as F
import torch
import torch.nn as nn


def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return inputs


def shift_back_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(
            inputs[:, i, :, :], shifts=(-1) * step * i, dims=2
        )
    return inputs


class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            # nn.Conv2d(64, 64, (1,2), padding=(0,1)),
            nn.ReLU(inplace=True),
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)

        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)

        x = self.upsample2(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs

        return out[:, :, :h_inp, :w_inp]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class GAP_net_intf(nn.Module):

    def __init__(self, in_channels, out_channels, func_0, func_A, func_A_inv):
        super(GAP_net_intf, self).__init__()
        self.func_0 = func_0
        self.func_A = func_A
        self.func_A_inv = func_A_inv

        self.unet1 = Unet(out_channels, out_channels)
        self.unet2 = Unet(out_channels, out_channels)
        self.unet3 = Unet(out_channels, out_channels)
        self.unet4 = Unet(out_channels, out_channels)
        self.unet5 = Unet(out_channels, out_channels)
        self.unet6 = Unet(out_channels, out_channels)
        self.unet7 = Unet(out_channels, out_channels)
        self.unet8 = Unet(out_channels, out_channels)
        self.unet9 = Unet(out_channels, out_channels)

    def forward(self, y, appendix):
        # if input_mask == None:
        #     Phi = torch.rand((1, 28, 256, 310)).cuda()
        #     Phi_s = torch.rand((1, 256, 310)).cuda()
        # else:
        #     Phi, Phi_s = input_mask, input_mask_s

        b, l, h_inp, w_inp = y.shape

        x_list = []
        x = self.func_A_inv(y, appendix)  # v0=H^T y
        ### 1-3
        yb = self.func_A(x, appendix)
        x = x + self.func_A_inv(y - yb, appendix)
        x = self.unet1(x)

        yb = self.func_A(x, appendix)
        x = x + self.func_A_inv(y - yb, appendix)
        x = self.unet2(x)

        yb = self.func_A(x, appendix)
        x = x + self.func_A_inv(y - yb, appendix)
        x = self.unet3(x)

        ### 4-6
        yb = self.func_A(x, appendix)
        x = x + self.func_A_inv(y - yb, appendix)
        x = self.unet4(x)

        yb = self.func_A(x, appendix)
        x = x + self.func_A_inv(y - yb, appendix)
        x = self.unet5(x)

        yb = self.func_A(x, appendix)
        x = x + self.func_A_inv(y - yb, appendix)
        x = self.unet6(x)

        # ### 7-9
        yb = self.func_A(x, appendix)
        x = x + self.func_A_inv(y - yb, appendix)
        x = self.unet7(x)

        x_list.append(x[:, :, :h_inp, :w_inp])
        yb = self.func_A(x, appendix)
        x = x + self.func_A_inv(y - yb, appendix)
        x = self.unet8(x)

        x_list.append(x[:, :, :h_inp, :w_inp])
        yb = self.func_A(x, appendix)
        x = x + self.func_A_inv(y - yb, appendix)
        x = self.unet9(x)

        return x[:, :, :h_inp, :w_inp]
