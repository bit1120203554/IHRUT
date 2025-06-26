import os
import copy
import numpy as np
import cv2
import skimage
import scipy
from enum import Enum
import torch
import hdf5storage  # 注意安装和导入hdf5plugin，否则h5文件读不了
import hdf5plugin

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

from . import base
from . import data_utils


class ImagePostfix(Enum):
    NUMPY = ".npy"
    RAW = ".raw"
    JPG = ".jpg"
    PNG = ".png"
    TIF = ".tif"
    TIFF = ".tiff"
    MAT = ".mat"
    FRED = ".fred"
    H5 = ".h5"


class ImageType(Enum):
    GRAYSCALE = 0
    NORMAL_RGB = 1
    RAW_RGB = 2
    HSI = 3
    LASIS_INTF = 4


class Image:
    # channels
    BGR = np.array([482.0, 539.0, 627.0])

    def __init__(
        self, filename=None, level=None, channels=None, data=None, mat_key=None
    ):
        self.filename = filename
        self.data = data
        self.channels = channels
        self.mat_key = mat_key  # 图像为mat格式时的存取键
        self.level = level
        self.dtype = None
        # 有文件名、并且数据为空的时候才读取
        if self.filename is not None and self.data is None:
            self.load()

    def postfix(self):
        postfix = "." + self.filename.split(".")[-1]
        return postfix

    def to_tensor(self):
        if self.data is None:
            print("err: data in Image is None")
        return data_utils.level_norm(torch.from_numpy(self.data.copy()), self.level)
        # ndarray_chw = np.ascontiguousarray(self.data)
        # return torch.from_numpy(ndarray_chw.astype("float32"))

    def from_tensor(self, tensor: torch.Tensor):
        self.data = np.round(tensor.detach().numpy() * self.level).astype(self.dtype)
        return self.data

    # 深拷贝，可以多态
    def copy(self):
        image_copy = copy.deepcopy(self)
        return image_copy

    # 用于做模板通过深拷贝衍生出真实数据对象，可以多态
    def template_load(self, filename, mat_key=None):
        image_copy = self.copy()
        image_copy.filename = filename
        if mat_key is not None:
            image_copy.mat_key = mat_key
        image_copy.load()
        return image_copy

    # 基类里的load、save放置了几种比较常见的。子类可以重写，也可以在重写时调用该函数。
    # 加载，这个基类里放置了几种比较常见的。子类可以重写，也可以在重写时调用该函数。
    # 谱域是第一维度，目的是和网络的输入格式一致。
    def load(self):
        postfix = self.postfix()
        if postfix == ImagePostfix.NUMPY.value:
            self.data = np.load(self.filename)
        elif postfix in [ImagePostfix.JPG.value, ImagePostfix.PNG.value]:
            self.data = cv2.imread(self.filename)
        elif postfix == ImagePostfix.MAT.value:
            data = hdf5storage.loadmat(self.filename)
            self.data = data[self.mat_key]
        elif postfix == ImagePostfix.H5.value:
            data = hdf5storage.h5py.File(self.filename, "r")[self.mat_key]
            self.data = data[:].transpose(1, 2, 0) * self.level
        elif postfix in [ImagePostfix.TIF.value, ImagePostfix.TIFF.value]:
            self.data = skimage.io.imread(self.filename).astype(np.uint16)

        else:
            self.data = np.fromfile(self.filename, dtype=np.uint16)
        if len(self.data.shape) == 3:
            if postfix == ImagePostfix.TIF.value:  # TIF转置的特殊处理
                # self.data = np.flip(self.data.transpose(0, 2, 1), axis=(0, 1))
                self.data = np.flip(self.data.transpose(0, 2, 1), axis=(1, 2))
            else:
                self.data = self.data.transpose(2, 0, 1)
        # self.data[self.data > self.level] = self.level  # 限制最大值
        self.dtype = self.data.dtype

    def save(self, data=None, dtype=None):
        if data is None:
            data = self.data
        postfix = self.postfix()
        if postfix == ImagePostfix.TIF.value:  # TIF转置的特殊处理
            # data = np.flip(data, axis=(0, 1)).transpose(0, 2, 1)
            data = np.flip(data, axis=(1, 2)).transpose(0, 2, 1)
        else:
            data = np.round(data).astype(dtype).transpose(1, 2, 0)
        # self.data[self.data > self.level] = self.level
        if postfix == ImagePostfix.NUMPY.value:
            np.save(self.filename, data)
        elif postfix in [ImagePostfix.JPG.value, ImagePostfix.PNG.value]:
            cv2.imwrite(self.filename, data)
        elif postfix == ImagePostfix.MAT.value:
            mat = {self.mat_key: data}
            hdf5storage.savemat(self.filename, mat)
        elif postfix in [ImagePostfix.TIF.value, ImagePostfix.TIFF.value]:
            skimage.io.imsave(self.filename, data)
        else:
            data.tofile(self.filename)

    def min_max_norm(self, clean=True, clean_rate=1.5):
        if clean:
            data, _ = data_utils.clean_abnormal_Q(self.data, clean_rate)
        data = (data - data.min()) / (data.max() - data.min()) * self.level
        return data

    def std_norm(self):
        data = ((self.data - self.data.mean()) / self.data.std()) / 2 + 0.5
        return data

    def resize_rate(self, rate):
        data = cv2.resize(
            self.data.transpose(1, 2, 0),
            dsize=(
                int(self.data.shape[2] * rate),
                int(self.data.shape[1] * rate),
            ),
            interpolation=cv2.INTER_LINEAR,
        ).transpose(2, 0, 1)
        return data

    def resize2(self):
        data = cv2.resize(
            self.data.transpose(1, 2, 0),
            dsize=(
                data_utils.nearest_2_power(self.data.shape[2]),
                data_utils.nearest_2_power(self.data.shape[1]),
            ),
            interpolation=cv2.INTER_LINEAR,
        ).transpose(2, 0, 1)
        return data

    def level_norm(self, level):
        return np.array(self.data / self.level * level)

    def dtype_norm(self, dtype):
        tar_bits = dtype.itemsize * 8
        level = (1 << tar_bits) * 1.0 - 1
        data = np.array(self.data / self.level * level).astype(dtype)
        data[data > level] = level  # 限制最大值
        return data

    # 最近邻插值
    def visualize_grayscale(self, channel, vis=True, save_path=None):
        x0 = self.channels
        if x0 is None:
            channel_no = int(channel) % (self.data.shape[0])
        else:
            y0 = range(len(x0))
            f = scipy.interpolate.interp1d(x0, y0, kind="nearest")
            channel_no = int(f(channel))
        channel_no = [channel_no, channel_no, channel_no]
        data = self.dtype_norm(np.dtype("uint8"))
        img_mat = data[channel_no]
        data = cv2.cvtColor(cv2.merge(img_mat), cv2.COLOR_BGR2RGB)
        if vis:
            cv2.imshow(self.filename, data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_path is not None:
            cv2.imwrite(save_path, data)
        pass


class GrayscaleImage(Image):
    def __init__(self, filename=None, level=255, channels=np.arange(1), data=None):
        super().__init__(filename, level, channels, data)


class NormalRGBImage(Image):
    def __init__(self, filename=None, level=255, channels=Image.BGR, data=None):
        super().__init__(filename, level, channels, data)

    def load(self):
        self.data = cv2.imread(self.filename).astype(self.dtype)

    def save(self):
        super().save(self.data, np.uint8)

    def visualize_RGB(self, vis=True, save_path=None):
        data = cv2.merge(self.dtype_norm(np.dtype("uint8")))
        if vis:
            cv2.imshow(self.filename, data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_path is not None:
            cv2.imwrite(save_path, data)


class RawRGBImage(Image):
    def __init__(self, filename=None, level=4095, channels=Image.BGR, data=None):
        super().__init__(filename, level, channels, data)

    def load(self):
        super().load()
        pass

    def visualize_RGB(self, vis=True, save_path=None):
        pass


class HSImage(Image):
    # mat_key表示在mat格式下的键值，对于非mat格式则为空
    def __init__(
        self,
        filename=None,
        level=4095,
        channels=None,
        data=None,
        mat_key=None,
    ):
        self.mat_key = mat_key
        super().__init__(filename, level, channels, data, mat_key)

    def load(self):
        super().load()

    def save(self):
        super().save(self.data)

    # 自动插值到相应波段
    def visualize_RGB(self, vis=True, save_path=None, rate=1):
        x0 = self.channels
        if x0 is None:
            print("No RGB showing for " + self.postfix.value)
            return
        y0 = range(len(x0))
        f = scipy.interpolate.interp1d(x0, y0, kind="nearest")
        BGR_channel_no = np.array(f(Image.BGR)).astype(int)
        data = np.clip(self.dtype_norm(np.dtype("uint8")) * rate, 0, 255)
        img_mat = data[BGR_channel_no]
        data = cv2.merge(img_mat)
        if vis:
            cv2.imshow(self.filename, data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_path is not None:
            cv2.imwrite(save_path, data)
        pass

    # 光谱到干涉，其中的OPDs最好等距（上采样、变换、建图）
    def spec2intf(self, OPDs):
        if len(self.channels) < 2:
            print("Unable to conduct FFT because len(channels) < 2.")
            return None

        # 采样，求出目标光程差需要的真实波数坐标，并通过谱域对数插值把原数据插值到相应的位置。
        x, center = LASISIntfImage.OPDs2bands(OPDs)
        # 谱域对数插值
        y = self.spec2bands_upsample(self.data, self.channels, x)
        y_intf, x_intf = LASISIntfImage.bands2intf_fft(self.data, x, center)
        result = LASISIntfImage(
            level=self.level,
            L=y_intf.shape[0],
            W=y_intf.shape[-1],
            center=center,
            dOPD=x_intf[1] - x_intf[0],
            data=y_intf,
        )

        # FFT
        shape = list(y.shape).copy()
        y_t = y.reshape((shape[0], -1)).transpose()
        x_inf, y_inf_t = data_utils.DCT(x, y_t)
        x_inf, y_inf = data_utils.odd_extension(x_inf, y_inf_t.transpose(), axis=0)
        shape[0] = x_inf.shape[0]
        y_inf = y_inf.reshape(shape)

        # 最后将变换好的图像再插值到目标光程差中
        rate = len(x)
        y_inf = -data_utils.interp(y_inf, x_inf, OPDs) / rate
        result = LASISIntfImage(channels=OPDs, data=y_inf, level=self.level)
        return result

    # 光谱反采样为波数曲线
    @staticmethod
    def spec2bands_upsample(data, channels_old, bands):
        # 最长波长之后补0，封住左边
        channels_old = np.append(
            channels_old,
            [
                6 * channels_old[-1] - 5 * channels_old[-2],
                11 * channels_old[-1] - 10 * channels_old[-2],
            ],
        )
        data = np.append(data, [data[-1] / 2, data[-1] / 64], axis=0)
        # 波长转波数
        wave_numbers = np.flip(1 / (channels_old + base.EPSILON))
        data = np.flip(data, axis=0)
        y = data_utils.interp(data, wave_numbers, bands, fill_value=base.EPSILON)
        return y

    @staticmethod  # 可适应batch
    def tensor_spec2bands_upsample(data, channels_old, bands):
        # 波长转波数
        channels_old = torch.cat(
            [
                channels_old,
                (6 * channels_old[-1] - 5 * channels_old[-2]).unsqueeze(0),
                (11 * channels_old[-1] - 10 * channels_old[-2]).unsqueeze(0),
            ],
            dim=0,
        )
        data = torch.cat(
            [
                data,
                (data[..., -1, :, :] / 2).unsqueeze(-3),
                (data[..., -1, :, :] / 64).unsqueeze(-3),
            ],
            dim=-3,
        )
        # 波长转波数
        wave_numbers = torch.flip(1 / (channels_old + base.EPSILON), dims=(0,))
        y = data_utils.tensor_interp(
            data, wave_numbers, bands, dim=-3, fill_value=base.EPSILON
        )
        return y

    # 波数曲线采样光谱
    @staticmethod
    def bands2spec_sample(data, bands, channels):
        # 波数转波长
        x_spec = np.flip(1 / (bands + base.EPSILON))
        y_spec = np.flip(data, axis=0)
        y_spec = data_utils.interp(y_spec, x_spec, channels, dim=-3)
        return y_spec

    @staticmethod  # 可适应batch
    def tensor_band2spec_sample(data, bands, channels):
        # 波数转波长
        x_spec = torch.flip(1 / (bands + base.EPSILON), dims=(0,))
        y_spec = torch.flip(data, dims=(-3,))
        y_spec = data_utils.tensor_interp(y_spec, x_spec, channels, dim=-3)
        return y_spec


class LASISIntfImage(Image):
    def __init__(
        self,
        filename=None,
        level=4095,
        L=256,
        W=2048,
        center=35,
        dOPD=150,
        data=None,
    ):
        self.L = L
        self.W = W
        self.head = None
        self.dOPD = dOPD
        self.center = center
        channels = (np.arange(self.L) - center) * dOPD
        super().__init__(filename, level, channels, data)

    def load(self):
        super().load()
        self.data = self.data.reshape(-1, self.L + 1, self.W).transpose(
            1, 0, 2
        )  # 这个加载的时候实际上没有经过转置运算
        self.head = self.data[0:1]
        self.data = self.data[1:]
        pass

    def save(self):
        data = np.concatenate((self.head, self.data), axis=0)
        super().save(data.transpose(2, 1, 0), np.uint16)  # 这个是为了配合super里的转置
        pass

    # 干涉到光谱（变换、下采样、建图）
    def intf2spec(self, channels):
        y_spec, x_spec = self.intf2bands_fft(self.data, self.channels)
        y_spec = HSImage.bands2spec_sample(self.data, x_spec, channels)
        result = HSImage(channels=channels, data=y_spec, level=self.level)
        return result

    @staticmethod
    def apodization(data, OPDs, anti=False):
        l = np.arange(len(OPDs))
        center = l[np.abs(OPDs - 0) <= base.EPSILON][0]  # 找光程差的中间0点坐标
        x0 = OPDs[center:]  # 切片取横坐标
        beta = 1.1
        x = len(OPDs) - len(x0)
        apd = np.hamming(2 * x + 1) * beta
        apd = data_utils.side_log_interp(apd, np.arange(apd.shape[0])+1, l+1)
        apd = apd.reshape(1, -1, 1, 1)
        # data_utils.show_line(OPDs, apd.squeeze(), "apd")
        if anti:
            result = data / apd
        else:
            result = data * apd
        return result

    @staticmethod
    # 求出目标光程差需要的波数坐标
    def OPDs2bands(OPDs):
        l = np.arange(len(OPDs))
        center = l[np.abs(OPDs - 0) <= base.EPSILON][0]  # 找光程差的中间0点坐标
        x0 = OPDs[center:]  # 切片取横坐标
        L = 2 * (len(x0) - 1)
        dOPD = OPDs[1] - OPDs[0]
        x = np.arange(L) / L / dOPD
        return x[: len(x0)], center

    @staticmethod
    # 求出目标光程差需要的波数坐标
    def bands2OPDs(bands, center):
        x = bands
        L = 2 * (len(x) - 1)
        dx = x[1] - x[0]
        l = center + len(x)
        OPDs = (np.arange(L) - center) / L / dx
        return OPDs[:l]

    @staticmethod
    def bands2intf_fft(data, bands, center):
        # 波数曲线偶延拓
        x_intf = LASISIntfImage.bands2OPDs(bands, center)
        y_intf = (
            LASISIntfImage.tensor_bands2intf_fft(torch.from_numpy(data), center)
            .detach()
            .numpy()
        )
        return y_intf, x_intf

    @staticmethod  # 可适应batch
    def tensor_bands2intf_fft(tensor, center):
        len1 = tensor.shape[-3]
        tensor = data_utils.tensor_odd_extension(tensor, dim=-3)
        len2 = tensor.shape[-3]
        tensor = torch.fft.fft(tensor, dim=-3) / len2
        tensor = torch.cat(
            (tensor[..., len2 - center :, :, :], tensor[..., :len1, :, :]), dim=-3
        )
        return tensor.real

    @staticmethod  # 返回波数图像，经过采样可得光谱图像
    def intf2bands_fft(data, OPDs):
        x_spec, center = LASISIntfImage.OPDs2bands(OPDs)
        if len(OPDs) < 2:
            print("Unable to conduct FFT because len(OPDs) < 2.")
            return None
        # pre-process
        y_spec = (
            LASISIntfImage.tensor_intf2bands_fft(torch.from_numpy(data), center)
            .detach()
            .numpy()
        )
        return y_spec, x_spec

    @staticmethod  # 可适应batch
    def tensor_intf2bands_fft(tensor, center):
        small = tensor[..., :center, :, :]  # 小双边
        big = tensor[..., center:, :, :]
        len1 = big.shape[-3]
        tensor = data_utils.tensor_odd_extension(big, dim=-3)
        len2 = tensor.shape[-3]
        tensor[..., len2 - center :, :, :] = small
        tensor = torch.fft.ifft(tensor, dim=-3) * len2
        return tensor[..., :len1, :, :]


def get_image_list(path, postfix, image_template: Image):
    paths = data_utils.base.get_file_paths(path, postfix)
    images = []
    for p in paths:
        i = image_template.template_load(p)
        images.append(i)
    return images


def visualize_HSIs(
    path, output_dir, template: HSImage, mat_key=None, norm=False, clean_rate=1.5
):
    # base.TimeCheck.init()
    paths = os.listdir(path)
    for p in paths:
        full_p = os.path.join(path, p)  # 获得完整路径
        _, p_text, p_postfix = base.cut_path(p)
        if p_postfix in [
            ImagePostfix.JPG.value,
            ImagePostfix.PNG.value,
        ]:
            continue
        template = template.template_load(full_p, mat_key)
        # template.data = template.resize_rate(0.5)
        if norm:
            template.data = data_utils.wavelet_filter(
                template.data, wavelet="sym4", level=4, axis=0
            )
            template.data = template.min_max_norm(clean_rate=clean_rate)

        template.visualize_RGB(
            False, os.path.join(output_dir, p_text + ImagePostfix.PNG.value)
        )
        print(full_p)
        # base.TimeCheck.check(True)
    pass


# if __name__ == "__main__":
#     pass
