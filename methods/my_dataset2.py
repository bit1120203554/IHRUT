import json
import os
from torch.utils.data import Dataset
import numpy as np
import shutil
from torch.utils.data import DataLoader

import utils
from .config import Config


# 数据集的对象
class MyDataset2:
    gain_code = {1: "G1", 2: "G2", 3: "G3", 4: "G4", 5: "G5"}
    DN_code = {
        0: "DARK",
        500: "DN500",
        1000: "DN1000",
        1500: "DN1500",
        2000: "DN2000",
        2500: "DN2500",
        3000: "DN3000",
        3500: "DN3500",
    }

    def __init__(self, config: Config):
        # super(MyDataset2, self).__init__()
        # normal
        self.root = config.dataset_path
        self.info = None
        with open(os.path.join(self.root, "info.json"), "r") as f:
            self.info = json.load(f)
        self.dOPD = self.info["dOPD"]
        self.center = self.info["center"]
        self.level = self.info["level"]
        self.L = self.info["L"]
        self.W = self.info["W"]
        self.max_W = self.info["max_W"]  # <=W
        self.specs = np.array(self.info["specs"])
        self.fwh = np.array(self.info["fwh"])
        self.paths = {}  # 路径字典，格式在load中制定
        self.load_paths(self.paths, self.root)
        self.intf_template = utils.LASISIntfImage(
            level=self.level,
            L=self.L,
            W=self.W,
            center=self.center,
            dOPD=self.dOPD,
        )
        self.spec_template = utils.HSImage(level=self.level, channels=self.specs)

    # 专门用于读取路径，建立字典，以便访问的类。路径固定2层（subset和type各占一层），同一路径下只能放一种后缀名的tif/tiff/raw或者fred文件，但是jpg和png等其他格式的可以随便放。
    def load_paths(self, dict: dict, path: str, subset="root", type=None):
        paths = os.listdir(path)
        for p in paths:
            full_p = os.path.relpath(os.path.join(path, p))  # 获得完整路径
            _, p_text, p_postfix = utils.cut_path(p)
            if not os.path.isdir(full_p):  # 解析文件名
                if (
                    subset == "calibration"
                    or type == "intf"
                    # or (subset == "test" and type == "HSI")
                ):
                    mode, gain, DN_or_name = self.cut_special_path(p_text, p_postfix)
                    if mode is None:
                        continue
                    if mode not in dict.keys():
                        dict[mode] = {}
                    if gain not in dict[mode].keys():
                        dict[mode][gain] = {}
                    dict[mode][gain][DN_or_name] = full_p
                else:
                    if p_postfix not in [
                        utils.ImagePostfix.JPG.value,
                        utils.ImagePostfix.PNG.value,
                    ]:
                        dict[p_text] = full_p
            else:  # 解析子路径
                dict[p] = {}
                # 如果是根路径则确定当前子集，如果不是则不改变当前子集
                if subset == "root":
                    self.load_paths(dict[p], full_p, p)
                elif type is None:
                    self.load_paths(dict[p], full_p, subset, p)
                else:
                    self.load_paths(dict[p], full_p, subset, type)
        pass

    # 查询路径
    def get_path(
        self,
        subset="calibration",
        type="intf",
        set_name="relative",
        mode="ZZ",
        gain=1,
        name=None,
        DN=None,
    ):
        path = None
        # 读取路径
        if subset == "calibration" or type == "intf":
            if DN is not None:
                name = self.DN_code[DN]
            gain = self.gain_code[gain]
            path = self.paths[subset][type][set_name][mode][gain][name]
        else:
            path = self.paths[subset][type][set_name][name]
        return path

    # 读取数据
    def fetch_data(self, type, path, mat_key=None):
        result = None
        p_cut = path.split(".")
        p_postfix = "." + p_cut[-1]
        # 读取数据
        if p_postfix in [utils.ImagePostfix.JPG.value, utils.ImagePostfix.PNG.value]:
            result = utils.NormalRGBImage(path)
        elif type == "intf":
            result = self.intf_template.template_load(path, mat_key)
        elif type == "HSI":
            result = self.spec_template.template_load(path, mat_key)
        return result

    def get_data(
        self,
        subset="calibration",
        type="intf",
        set_name="relative",
        mode="ZZ",
        gain=1,
        name="",
        DN=None,
        mat_key=None,
    ):
        return self.fetch_data(
            type,
            self.get_path(subset, type, set_name, mode, gain, name, DN),
            mat_key,
        )

    def cut_special_path(self, p_text, p_postfix):
        p_split = p_text.split("_")
        mode = None
        gain = None
        DN_or_name = None
        if p_postfix in [
            utils.ImagePostfix.FRED.value,
            utils.ImagePostfix.RAW.value,
        ]:
            mode = p_split[-3]
            gain = p_split[-2]
            DN_or_name = p_split[-1].split("+")[0]  # +号之后是偏移值
        # elif p_postfix == ".asd":
        #     mode = p_split[-4]
        #     gain = p_split[-3]
        #     DN_or_name = p_split[-2]
        elif p_postfix in [
            utils.ImagePostfix.TIF.value,
            utils.ImagePostfix.TIFF.value,
        ]:
            mode = p_split[-6]
            gain = p_split[-5]
            DN_or_name = p_split[-4].split("+")[0]  # +号之后是偏移值
        return mode, gain, DN_or_name

    def cut_metadata(
        self,
        subset,
        type,
        set_name,
        channels,
        mat_key=None,
        cut_size=(256, 2048),
        bias=0,
        norm=False,
        resize=1,
    ):
        metadata_dir = os.path.join(self.root, "metadata", type, set_name)
        output_dir = os.path.join(self.root, subset, type, set_name)
        shutil.rmtree(output_dir, "r")
        utils.mkdir(output_dir)
        print("cut_metadata")

        paths = os.listdir(metadata_dir)
        for p in paths:
            full_p = os.path.join(metadata_dir, p)  # 获得完整路径
            _, p_text, p_postfix = utils.cut_path(p)
            if p_postfix in [
                utils.ImagePostfix.JPG.value,
                utils.ImagePostfix.PNG.value,
            ]:
                continue

            self.cut_data(
                type,
                full_p,
                output_dir,
                channels,
                mat_key,
                cut_size,
                bias,
                norm,
                resize,
            )
        pass

    def cut_data(
        self,
        type,
        src_path,
        output_dir,
        channels,
        mat_key,
        cut_size,
        bias,
        norm=False,
        resize=1,
    ):
        src_dir, p_text, p_postfix = utils.cut_path(src_path)

        meta = self.fetch_data(type, src_path, mat_key=mat_key)
        postfix = utils.ImagePostfix.PNG.value

        # meta.data = utils.interp(meta.data, channels, meta.channels)  # 这个容易爆内存

        if norm:  # 归一化处理
            meta.data = utils.wavelet_filter(meta.data, wavelet="sym4", level=4, axis=0)
            meta.data = meta.min_max_norm()

        # 自动去边resize+切割
        meta.data = meta.resize_rate(resize)
        if meta.data.shape[1] < cut_size[0] + 2 * bias:
            resize = (cut_size[0] + 2 * bias) / meta.data.shape[1]
            meta.data = meta.resize_rate(resize)
        if meta.data.shape[2] < cut_size[1] + 2 * bias:
            resize = (cut_size[1] + 2 * bias) / meta.data.shape[2]
            meta.data = meta.resize_rate(resize)

        if type == "intf":
            cut = self.intf_template.copy()
            postfix = utils.ImagePostfix.RAW.value
        elif type == "HSI":
            cut = self.spec_template.copy()
            postfix = utils.ImagePostfix.TIFF.value
            meta.visualize_RGB(
                False,
                os.path.join(src_dir, p_text + utils.ImagePostfix.PNG.value),
            )

        H = meta.data.shape[1] - 2 * bias
        W = meta.data.shape[2] - 2 * bias
        H = H - H % cut_size[0]
        W = W - W % cut_size[1]
        meta.data = meta.data[:, bias : bias + H, bias : bias + W]

        H /= cut_size[0]
        W /= cut_size[1]
        cuts = np.split(meta.data, H, axis=1)
        cuts = [np.split(t, W, axis=2) for t in cuts]
        cuts = np.array(cuts)

        if type == "intf":
            cut.head = meta.head[:, : cut_size[0], : cut_size[1]]

        for i0 in np.arange(cuts.shape[0]):
            for i1 in np.arange(cuts.shape[1]):
                cut.data = cuts[i0][i1]
                cut.data = utils.interp(cut.data, channels, cut.channels)
                cut.filename = os.path.join(
                    output_dir, "%s-%d-%d" % (p_text, i0, i1) + postfix
                )
                cut.save()
                if type == "HSI":
                    cut.visualize_RGB(
                        False,
                        os.path.join(
                            output_dir,
                            "%s-%d-%d" % (p_text, i0, i1)
                            + utils.ImagePostfix.PNG.value,
                        ),
                    )
                print(cut.filename)
        pass
