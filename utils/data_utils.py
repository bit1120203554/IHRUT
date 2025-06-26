import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
import torch
import scipy.stats as stats
import scipy.fft
import pywt

# import pytorch_msssim
from .metrix import ssim_torch
from . import base


# for numpy
def bits_to_level(bits):
    return (1 << bits) * 1.0 - 1


# 最接近的2的次幂
def nearest_2_power(x):
    return 2 ** int(np.ceil(np.log2(x)))


def odd_extension(x, mat, axis=None):
    mat_f = np.flip(mat, axis=axis)
    x_f = np.flip(2 * np.max(x) - x, axis=axis)
    mat_result = np.concatenate((mat, mat_f[1:-1]), axis=axis)
    x_result = np.concatenate((x, x_f[1:-1]), axis=axis)
    return x_result, mat_result


def occ_extension(x, mat, axis=None):
    mat_f = np.flip(mat, axis=axis)
    x_f = -np.flip(x, axis=axis)
    mat_result = np.concatenate((mat_f[:-1], mat), axis=axis)
    x_result = np.concatenate((x_f[:-1], x), axis=axis)
    return x_result, mat_result


# 矩阵傅里叶变换，变换最后一维，用torch计算
def FFT(x, y, axis=0):
    # 初始化
    if x.shape[0] != y.shape[axis]:
        print("FFT demension err")
        return

    # FFT
    Fs = 1 / (x[1] - x[0])
    L = x.shape[0]  # 信号长度
    x_fft = np.arange(L) * Fs / L
    y_fft = torch.fft.fft(torch.from_numpy(y), dim=axis).detach().numpy()

    return x_fft, y_fft


def iFFT(x, y, axis=0):
    x_ifft, y_ifft = FFT(x, y, axis)
    if x_ifft is None:
        return
    y_ifft /= x.shape[0]
    return x_ifft, y_ifft


def DCT(x, y, axis=0):
    # 初始化
    if x.shape[0] != y.shape[axis]:
        print("DCT demension err")
        return
    x_occ, y_occ = occ_extension(x, y, axis)

    # FFT
    L = x.shape[0]  # 信号长度
    x_dct_occ, y_dct_occ = FFT(x_occ, y_occ, axis)
    x_dct = np.take(x_dct_occ, np.arange(L), axis=axis)
    y_dct = np.take(y_dct_occ, np.arange(L), axis=axis)

    return x_dct, y_dct


def iDCT(x, y, axis=0):
    x_idct, y_idct = DCT(x, y, axis)
    if x_idct is None:
        return
    y_idct /= 2 * (x.shape[0] - 1)
    return x_idct, y_idct


# 批量计算x和y的线性回归R值和系数，在最后一维上计算
def count_linear(x, y, axis=0):
    x_avg_0 = x - np.mean(x, axis=axis, keepdims=True)
    y_avg_0 = y - np.mean(y, axis=axis, keepdims=True)
    xy_conv = np.sum(x_avg_0 * y_avg_0, axis=axis, keepdims=True)
    xx_conv = np.sum(x_avg_0 * x_avg_0, axis=axis, keepdims=True)
    yy_conv = np.sum(y_avg_0 * y_avg_0, axis=axis, keepdims=True)
    a = xy_conv / xx_conv
    b = np.mean(y, axis=axis, keepdims=True) - a * np.mean(x, axis=axis, keepdims=True)
    R = xy_conv / np.sqrt(xx_conv * yy_conv)
    return a, b, R


# 清理数据异常值，用中位值代替
def clean_abnormal_G(x, sigma=3.0, axis=None):
    std = np.std(x, axis=axis, keepdims=True)
    mean = np.mean(x, axis=axis, keepdims=True)
    # median = np.median(x, axis=axis, keepdims=True)
    bias = sigma * std

    return clean_abnormal(x, mean - bias, mean + bias)


def clean_abnormal_Q(x, rate=1.5, axis=None):
    x_Q1 = np.percentile(x, q=25, axis=axis, keepdims=True)
    # median = np.median(x, axis=axis, keepdims=True)
    x_Q3 = np.percentile(x, q=75, axis=axis, keepdims=True)
    IQR = x_Q3 - x_Q1

    return clean_abnormal(x, x_Q1 - rate * IQR, x_Q3 + rate * IQR)


def clean_abnormal(x, left, right):
    left = np.zeros_like(x) + left
    right = np.zeros_like(x) + right  # 去除维度的影响
    label = np.zeros_like(x)
    y = np.copy(x)
    r = x > right
    l = x < left
    label[r] = 1
    label[l] = -1
    change = np.zeros_like(x)
    change[r] = right[r]
    change[l] = left[l]
    y[change != 0] = 0
    y += change
    return y, label


# 小波变换滤波, axis是单个值, 硬阈值
def wavelet_filter(x, wavelet="db2", level=4, drop=0, method="hard", axis=-1):
    if drop > level:
        print("wavelet_filter error: drop>level")
        return
    x_extended = x
    coeffs = pywt.wavedec(x_extended, wavelet=wavelet, level=level, axis=axis)

    sigma = 1.4825
    abs_cd1 = np.abs(coeffs[-1])
    median_cd1 = np.median(abs_cd1, axis=axis, keepdims=True)
    sigma = 1.4875 * median_cd1
    lam = sigma * np.sqrt(2 * np.log(x.shape[axis]))

    for i in np.arange(level) + 1:
        if i + drop >= level + 1:
            coeffs[i] *= 0
            continue
        abs_coeff = np.abs(coeffs[i])
        threshold = lam / np.log2(level + 2 - i)
        coeffs[i][abs_coeff < threshold] = 0
        if method == "soft":
            coeffs[i][coeffs[i] < 0] += threshold
            coeffs[i][coeffs[i] > 0] -= threshold

    re = pywt.waverec(coeffs, wavelet=wavelet, axis=axis)
    result = np.take(re, np.arange(x.shape[axis]), axis=axis)
    return result


# 用于任意值通道（通道默认为第一个维度）上的线性插值
def interp(data, channels_old, channels, axis=0, fill_value="extrapolate"):
    interp_func = scipy.interpolate.interp1d(
        channels_old,
        data,
        axis=axis,
        kind="linear",
        bounds_error=False,
        fill_value=fill_value,
    )
    result = interp_func(channels)
    return result


# 谱域对数插值，在原参考指标为x的y要在两边插两个渐变的缩小值，再插值到target_x
# 要求所有channels同符号，且绝对值单调递增。
def side_log_interp(data, channels_old, channels, axis=0):
    # data_2d = data.reshape(data.shape[0], -1)

    channels_old_copy = channels_old.copy()
    channels_old_copy = np.insert(channels_old_copy, 0, [channels_old_copy[0] / 10])
    channels_old_copy = np.append(
        channels_old_copy, [2 * channels_old_copy[-1] - channels_old_copy[1]]
    )

    data_log = np.log(data + base.EPSILON)
    data_log = np.insert(data_log, 0, [data_log[0] - 10], axis=axis)
    data_log = np.append(data_log, [data_log[-1] - 10], axis=axis)

    result = data_log
    result = np.exp(interp(result, channels_old_copy, channels, axis))
    return result


def fit_tukeylambda_dist(x, vision=True, title=""):
    if vision:
        svals, ppcc = stats.ppcc_plot(x, -0.5, 0.2, N=50, plot=plt)
        plt.show()
        best_shape_val = svals[np.argmax(ppcc)]
        _, (scale, _, r) = stats.probplot(
            x, best_shape_val, dist="tukeylambda", plot=plt
        )
        plt.show()
    else:
        svals, ppcc = stats.ppcc_plot(x, -0.5, 0.2, N=50)
        best_shape_val = svals[np.argmax(ppcc)]
        _, (scale, _, r) = stats.probplot(x, best_shape_val, dist="tukeylambda")
    print(
        "%s Tukeylambda: mean:%f r2:%f  scale:%f shape:%f\n"
        % (title, np.mean(x), r**2, scale, best_shape_val)
    )
    return scale, best_shape_val, r**2


def fit_norm_dist(x, vision=True, title=""):
    if vision:
        _, (scale, _, r) = stats.probplot(x, plot=plt)
        plt.show()
    else:
        _, (scale, _, r) = stats.probplot(x)
    print("%s Gaussian: mean:%f r2:%f  scale:%f\n" % (title, np.mean(x), r**2, scale))
    return scale, r**2


def fit_uniform_dist(x, vision=True, title=""):
    if vision:
        _, (scale, _, r) = stats.probplot(x, dist=stats.uniform, plot=plt)
        plt.show()
    else:
        _, (scale, _, r) = stats.probplot(x, dist=stats.uniform)
    print("%s Uniform: mean:%f r2:%f  scale:%f\n" % (title, np.mean(x), r**2, scale))
    return scale, r**2


def KL_distance(src_array, tar_array, level):
    src_flat = np.floor(src_array.reshape(-1))
    tar_flat = np.floor(tar_array.reshape(-1))
    bins = np.arange(level + 2) - 0.1
    src_hist, edges = np.histogram(src_flat, bins, density=True)
    src_hist += base.EPSILON
    # show_line(np.arange(level + 1), src_hist)
    tar_hist, edges = np.histogram(tar_flat, bins, density=True)
    tar_hist += base.EPSILON
    # show_line(np.arange(level + 1), tar_hist)
    kl = tar_hist * np.log(tar_hist / src_hist)
    kl_d = np.sum(kl)
    return kl_d


# for tensor


# 将tensor的某一个dim和第一个维度交换转置
def tensor_dim_to_0(tensor, dim):
    if dim < 0:
        dim = tensor.dim() + dim
    return tensor.permute([dim] + list(range(dim)) + list(range(dim + 1, tensor.dim())))


def tensor_odd_extension(tensor, dim=0):
    tensor_f = torch.flip(tensor, dims=(dim,)).index_select(
        dim=dim,
        index=torch.tensor(np.arange(1, tensor.shape[dim] - 1), device=tensor.device),
    )
    if tensor_f.is_complex():
        tensor_f.imag = -tensor_f.imag
    tensor = torch.cat((tensor, tensor_f), dim=dim)
    return tensor


# 用于任意值通道（通道默认为第一个维度）上的线性插值
def tensor_interp(tensor, channels_old, channels, dim=0, fill_value="extrapolate"):
    # base.TimeCheck.init()
    if dim != 0:
        tensor = tensor_dim_to_0(tensor, dim)
    tensor_2d = tensor.reshape(tensor.shape[0], -1)
    if tensor.shape[0] != len(channels_old):
        print("interp error")
        return
    # channels_old = torch.from_numpy(channels_old)
    # channels = torch.from_numpy(channels)
    channels_old_diff = torch.diff(channels_old, dim=0)
    b = tensor_2d
    k = torch.diff(tensor_2d, dim=0).transpose(1, 0)
    k = (k / channels_old_diff).transpose(1, 0)
    mat_shape = list(tensor.shape[1:])
    new_shape = [len(channels)] + mat_shape
    result = torch.zeros(new_shape, device=tensor.device)
    if fill_value != "extrapolate":
        result += fill_value
    i = 0
    j = 0
    # base.TimeCheck.check(True)
    for c in channels:
        while i < len(channels_old_diff) and c >= channels_old[i]:
            i += 1
        mat = None
        if i == len(channels_old_diff) and c > channels_old[i]:
            if fill_value == "extrapolate":
                mat = b[i - 1] + k[i - 1] * (c - channels_old[i - 1])
            pass
        elif i > 0:
            mat = b[i - 1] + k[i - 1] * (c - channels_old[i - 1])
        else:
            if fill_value == "extrapolate":
                mat = b[i] + k[i] * (c - channels_old[i])
        if mat is not None:
            result[j] = mat.view(mat_shape)
        j += 1

    # base.TimeCheck.check(True)
    if dim != 0:
        result = tensor_dim_to_0(result, dim)
    # base.TimeCheck.check(True)
    return result


# 这个不能线性外插值，只能在外部补定值
def tensor_fast_interp(tensor, channels_old, channels, dim=0, fill_value=base.EPSILON):
    # base.TimeCheck.init()
    if dim != 0:
        tensor = tensor_dim_to_0(tensor, dim)
    if tensor.shape[0] != len(channels_old):
        print("interp error")
        return

    mat_shape = list(tensor.shape[1:])
    new_shape = [len(channels)] + mat_shape
    tensor_2d = tensor.reshape(tensor.shape[0], -1)
    # 左右补值
    fill = (
        torch.zeros_like(tensor_2d[0], device=tensor_2d.device) + fill_value
    ).unsqueeze(0)
    tensor_2d = torch.cat([tensor_2d, fill], dim=0)
    channels_old_diff = torch.diff(channels_old, dim=0)
    beta = (
        torch.zeros_like(channels, device=tensor_2d.device) + channels_old.shape[0] - 1
    )
    rate = torch.zeros_like(channels, device=tensor_2d.device) + 1

    i = 0
    j = 0
    # base.TimeCheck.check(True)
    for c in channels:
        while i < len(channels_old) and c >= channels_old[i]:
            i += 1
        if i > 0:
            beta[j] = i - 1
            if i < len(channels_old):
                rate[j] = (c - channels_old[i - 1]) / channels_old_diff[i - 1]
            elif c == channels_old[i - 1]:
                rate[j] = 0
        j += 1

    # base.TimeCheck.check(True)
    left_value = torch.index_select(tensor_2d, dim=0, index=beta)
    right_value = torch.index_select(tensor_2d, dim=0, index=(beta + 1))
    rate = rate.unsqueeze(-1)
    # beta0 = beta.cpu().detach().numpy()
    # rate0 = rate.cpu().detach().numpy()
    result = (1 - rate) * left_value + rate * right_value
    # result = result.cpu().detach().numpy()

    # base.TimeCheck.check(True)
    result = result.view(new_shape)
    if dim != 0:
        result = tensor_dim_to_0(result, dim)
    # base.TimeCheck.check(True)
    return result


def min_max_norm(tensor):
    t_min = tensor.min()
    t_max = tensor.max()
    tensor = (tensor - t_min) / (t_max - t_min)
    return tensor, t_min, t_max


def src_tar_norm(src_tensor, tar_tensor):
    tar_norm, tar_min, tar_max = min_max_norm(tar_tensor)
    src_norm = (src_tensor - tar_min) / (tar_max - tar_min)
    return src_norm, tar_norm


def level_norm(data, from_level, to_level=1):
    return data / from_level * to_level


def mse(src_tensor, tar_tensor):
    imdff = src_tensor - tar_tensor
    mse = (imdff**2).mean()
    return mse


def rmse(src_tensor, tar_tensor):
    return torch.sqrt(mse(src_tensor, tar_tensor))


def psnr(src_tensor, tar_tensor, level):
    src_tensor = torch.clamp(src_tensor, 0, level)
    tar_tensor = torch.clamp(tar_tensor, 0, level)
    mse0 = mse(src_tensor, tar_tensor)
    ps = 10 * torch.log10(level**2 / (mse0 + base.EPSILON))
    return ps


def ssim(src_tensor, tar_tensor, level):
    src_tensor = torch.clamp(src_tensor, 0, level)
    tar_tensor = torch.clamp(tar_tensor, 0, level)
    return ssim_torch.ssim(
        torch.unsqueeze(src_tensor, 0), torch.unsqueeze(tar_tensor, 0)
    )


def sam(src_tensor, tar_tensor):
    src1 = torch.reshape(src_tensor, (src_tensor.shape[0], -1))
    tar1 = torch.reshape(tar_tensor, (tar_tensor.shape[0], -1))
    mole = torch.sum(src1 * tar1, dim=0, keepdim=False)
    src1_nor2 = torch.sqrt(torch.sum(torch.square(src1), dim=0))
    tar1_nor2 = torch.sqrt(torch.sum(torch.square(tar1), dim=0))
    cosine = (mole + base.EPSILON) / (src1_nor2 * tar1_nor2 + base.EPSILON)
    torch.clamp(cosine, -1, 1)
    sam = torch.acos(cosine)
    return torch.mean(sam)
    # return torch.mean(torch.rad2deg(sam))


def egras(src_tensor, tar_tensor, level):
    src_tensor = torch.clamp(src_tensor, 0, level)
    tar_tensor = torch.clamp(tar_tensor, 0, level)
    imdff = src_tensor - tar_tensor
    n = tar_tensor.shape[0]
    mse = torch.reshape(imdff**2, (n, -1))
    mse = torch.mean(mse, dim=1)
    tar1 = torch.reshape(tar_tensor, (n, -1))
    y = torch.mean(tar1, dim=1)
    rmse = torch.sqrt(mse)
    eg = torch.square((100 / level) * torch.sqrt(1 / mse) * (rmse * y))
    egras = torch.sqrt(torch.mean(eg))
    return egras


def fwhm(src_tensor, tar_tensor):
    fwhm = 0
    return fwhm


# ndarray和tensor通用
def random_patch(src, label, patch_size):
    H_offset, W_offset = (0, 0)
    _, H, W = src.shape
    if patch_size is None:
        patch_size = (H, W)
    patch_H, patch_W = patch_size

    if H <= patch_H:
        H_offset = 0
        patch_H = H
    else:
        H_offset = np.random.randint(0, max(0, H - patch_H))
    if W <= patch_W:
        W_offset = 0
        patch_W = W
    else:
        W_offset = np.random.randint(0, max(0, W - patch_W))

    # 修正patch_size
    patch_size = [patch_H, patch_W]
    offset = [H_offset, W_offset]

    src_patch = src[..., H_offset : H_offset + patch_H, W_offset : W_offset + patch_W]
    label_patch = label[
        ..., H_offset : H_offset + patch_H, W_offset : W_offset + patch_W
    ]

    return src_patch, label_patch, patch_size, offset


# visualization


def show_line(x, y, title=None, vis=True, save_path=None):
    # plt.style.use("_mpl-gallery")
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    if isinstance(y, list):
        for y0 in y:
            ax.plot(x, y0, linewidth=2.0)
    else:
        ax.plot(x, y, linewidth=2.0)
    if vis:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def show_scatter(x, y, title=None, vis=True, save_path=None):
    # plt.style.use("_mpl-gallery")
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    if isinstance(y, list):
        for y0 in y:
            ax.scatter(x, y0, 1)
    else:
        ax.scatter(x, y, 1)
    if vis:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


# 插值的矩阵化表示


def show_info(x, label):
    print(
        "%s\nmean:%f std:%f min:%f median:%f max:%f\n"
        % (label, np.mean(x), np.std(x), np.min(x), np.median(x), np.max(x))
    )


if __name__ == "__main__":

    pass
