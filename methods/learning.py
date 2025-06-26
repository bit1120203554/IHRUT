import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader

import utils
from .config import Config
from . import MyDataset2
from . import degradation_model
from . import models


class ISRSet(utils.ImgSet):
    def __init__(
        self,
        name,
        online_simulation: bool,  # 这个决定是读取src还是在线生成仿真的src用于训练/测试。如果在线仿真，切割之后会随机重置W_offset，可以适用于小图片切割的情况
        deg_model: degradation_model.DegregationModel,  # 在线仿真用的退化模型
        src_paths,
        label_paths,
        src_template: utils.LASISIntfImage,
        label_template: utils.HSImage,
        patch_size=None,
        max_W=None,
        gain=1,
    ):  # max_W: 在线仿真时，W_offset的最大值
        super(ISRSet, self).__init__(
            name, src_paths, label_paths, src_template, label_template, patch_size
        )
        self.online_simulation = online_simulation
        self.deg_model = deg_model
        self.gain = gain
        self.max_W = max_W
        if self.max_W is None:
            self.max_W = self.deg_model.W
        # params
        self.OPDs = deg_model.OPDs
        self.bands = torch.from_numpy(deg_model.bands).to(torch.float32)
        self.sample = torch.from_numpy(deg_model.sample)

        l = np.arange(self.OPDs.shape[0])
        self.center = int(l[np.abs(self.OPDs) <= utils.EPSILON][0])
        self.elc_gain = deg_model.elc_gain[gain - 1]
        self.A = torch.from_numpy(deg_model.A).to(torch.complex64)
        self.beta = torch.from_numpy(deg_model.beta).to(torch.float32)
        self.M = torch.from_numpy(deg_model.M[gain - 1]).to(torch.float32)
        self.K = torch.from_numpy(deg_model.K).to(torch.float32)
        self.D0 = (deg_model.D0 / deg_model.level).astype("float32")
        self.N_base = (
            torch.from_numpy(deg_model.N_base)[gain - 1].to(torch.float32)
            / deg_model.level
        )

    def __getitem__(self, index):
        specs = torch.from_numpy(self.label_template.channels).to(torch.float32)
        src_path = self.src_paths[index]
        label_path = self.label_paths[index]
        label_img = self.label_template.template_load(label_path)
        if self.online_simulation:
            src_img = self.src_template.copy()
            label_dir, label_name, label_postfix = utils.cut_path(label_path)
            src_path = "%s_%s_%s%s" % (
                self.deg_model.mode,
                self.gain,
                label_name,
                utils.ImagePostfix.RAW.value,
            )
            src_path = os.path.join(label_dir, src_path)
            src_img.data = label_img.data
        else:
            src_img = self.src_template.template_load(src_path)

        src_img.data, label_img.data, self.patch_size, self.offset = utils.random_patch(
            src_img.data[..., : self.max_W],
            label_img.data[..., : self.max_W],
            self.patch_size,
        )
        patch_W = self.patch_size[1]
        if self.online_simulation:
            W = min(self.max_W, src_img.data.shape[-1])
            if W > patch_W:
                self.offset[1] = np.random.randint(0, max(0, W - patch_W))
            else:
                self.offset[1] = 0
            src_img.data = self.deg_model.simulation(
                label_img.data,
                label_img.channels,
                self.gain,
                self.offset[1],
            )
        else:
            _, src_text, _ = utils.cut_path(src_path)
            src_name = src_text.split("+")
            if len(src_name) > 1:
                self.offset[1] = eval(src_name[1])

        W_offset = self.offset[1]

        return (
            src_img.to_tensor().to(torch.float32),
            label_img.to_tensor().to(torch.float32),
            src_path,
            label_path,
            self.center,
            self.bands,
            specs,
            self.sample,
            self.elc_gain,
            self.A[..., W_offset : W_offset + patch_W],
            self.beta[..., W_offset : W_offset + patch_W],
            self.M[..., W_offset : W_offset + patch_W],
            self.K[..., W_offset : W_offset + patch_W],
            self.D0 + self.elc_gain * self.N_base[..., W_offset : W_offset + patch_W],
        )

    def batch(
        self,
        *args,
    ):
        return utils.Batch(
            src=args[0],
            label=args[1],
            src_path=args[2],
            label_path=args[3],
            center=args[4],
            bands=args[5],
            specs=args[6],
            sample=args[7],
            elc_gain=args[8],
            A=args[9],
            beta=args[10],
            M=args[11],
            K=args[12],
            D=args[13],
        )


def generate_loader(
    config: Config,
    train: bool,
    cropped: bool,  # 数据集是不是已经切成了patch
    online_simulation: bool,  # 这个决定是读取src还是在线生成仿真的src用于训练/测试
    degradation_model: degradation_model.DegregationModel,  # 在线仿真用的退化模型
    dataset: MyDataset2,
    set_name,
    gains=None,
):
    if gains is None:
        gains = dataset.gain_code.keys()

    intf_template = dataset.intf_template.copy()
    if train:
        subset = "train"
    else:
        subset = "test"

    if cropped:
        intf_template.W = config.patch_size[1]

    hsi_set = dataset.paths[subset]["HSI"][set_name]
    train_list = [[], []]
    val_list = [[], []]
    for gain in gains:
        cut = np.floor(len(hsi_set.keys()) * config.trainset_rate)
        if not train:  # 测试模式训练集比率为0
            cut = 0
        gc = dataset.gain_code[gain]
        i = 0
        for origin in hsi_set.keys():
            label = hsi_set[origin]
            if online_simulation:
                src = None
            else:
                intf_set = dataset.paths[subset]["intf"][set_name]
                src = intf_set[degradation_model.mode][gc][origin]
            if i < cut:
                train_list[0].append(src)
                train_list[1].append(label)
            else:  # 测试或者验证的时候不切patch
                val_list[0].append(src)
                val_list[1].append(label)
            i += 1
        pass

    train_set = ISRSet(
        name=set_name,
        online_simulation=online_simulation,
        deg_model=degradation_model,
        src_paths=train_list[0],
        label_paths=train_list[1],
        src_template=intf_template,
        label_template=dataset.spec_template,
        patch_size=config.patch_size,
        max_W=dataset.max_W,
        gain=gain,
    )
    val_set = ISRSet(
        name=set_name,
        online_simulation=online_simulation,
        deg_model=degradation_model,
        src_paths=val_list[0],
        label_paths=val_list[1],
        src_template=intf_template,
        label_template=dataset.spec_template,
        patch_size=None,
        max_W=dataset.max_W,
        gain=gain,
    )
    if not train:
        return val_set
    return train_set, val_set


class IHRFramework(utils.Framework):
    def __init__(
        self,
        config: Config,
        intf_template: utils.LASISIntfImage,
        spec_template: utils.HSImage,
    ):

        if config.model_name is None:  # 没有模型名
            model_name = "ISR_%s_%s_%d" % (
                config.model_type,
                config.prior_type,
                config.num_stages,
            )
        else:  # 有模型名
            model_name = config.model_name

        self.intf_template = intf_template
        self.spec_template = spec_template
        super(IHRFramework, self).__init__(
            config.model_type, config.loss_type, model_name, config
        )

    def get_model(self):
        type = self.model_type
        func_0 = FFT_inv
        func_A = Phi
        func_A_inv = Phi_inv
        if type == "IHRUT":
            return models.IHRUT(
                len(self.intf_template.channels),
                # len(utils.LASISIntfImage.OPDs2bands(self.intf_template.channels)[0]),
                len(self.spec_template.channels),
                func_0,
                func_A,
                func_A_inv,
                num_stages=self.config.num_stages,
                sharing=self.config.sharing,
            )
        elif type == "mixs2":
            opt = models.get_opt()
            opt.stage = self.config.num_stages
            return models.DUF_MixS2_intf(
                opt,
                len(self.intf_template.channels),
                # len(utils.LASISIntfImage.OPDs2bands(self.intf_template.channels)[0]),
                len(self.spec_template.channels),
                func_0,
                func_A,
                func_A_inv,
            )
        elif type == "PADUT":
            return models.PADUT_intf(
                len(self.intf_template.channels),
                # len(utils.LASISIntfImage.OPDs2bands(self.intf_template.channels)[0]),
                len(self.spec_template.channels),
                func_0,
                func_A,
                func_A_inv,
                nums_stages=self.config.num_stages,
            )
        elif type == "DAUHST":
            return models.DAUHST(
                len(self.intf_template.channels),
                # len(utils.LASISIntfImage.OPDs2bands(self.intf_template.channels)[0]),
                len(self.spec_template.channels),
                func_0,
                func_A,
                func_A_inv,
                nums_stages=self.config.num_stages,
            )
        elif type == "MAUN":
            return models.MAUN(
                len(self.intf_template.channels),
                # len(utils.LASISIntfImage.OPDs2bands(self.intf_template.channels)[0]),
                len(self.spec_template.channels),
                func_0,
                func_A,
                func_A_inv,
                num_iterations=self.config.num_stages,
            )
        elif type == "GAPNet":
            return models.GAP_net_intf(
                len(self.intf_template.channels),
                # len(utils.LASISIntfImage.OPDs2bands(self.intf_template.channels)[0]),
                len(self.spec_template.channels),
                func_0,
                func_A,
                func_A_inv,
            )
        elif type == "INV":
            func_0 = Phi_inv
        # elif type == "FFT":
        else:  # "FFT"
            func_0 = FFT_inv
        return models.IHR(
            len(self.intf_template.channels),
            # len(utils.LASISIntfImage.OPDs2bands(self.intf_template.channels)[0]),
            len(self.spec_template.channels),
            func_0,
            self.config.prior_type,
            self.config.prior_nc,
            self.config.prior_nb,
        )

    def forward(self, data: utils.Batch):
        data.src -= data.D
        model_out = self.model(data.src, data.__dict__)  # 传字典能算FLOPS
        # model_out = utils.HSImage.tensor_band2spec_sample(
        #     model_out, data.bands[0], data.specs[0]
        # )  # 这个比较慢
        model_out = torch.clamp(model_out, 0, 1)
        return model_out

    @staticmethod
    def eval_single(src: torch.Tensor, label: torch.Tensor):
        result = {}
        result["psnr"] = utils.psnr(src, label, 1.0)
        result["rmse"] = utils.rmse(src, label)
        result["ssim"] = utils.ssim(src, label, 1.0)
        result["sam"] = utils.sam(src, label)
        result["egras"] = utils.egras(src, label, 1.0)
        return result

    @staticmethod
    def save_single(
        val_set: utils.ImgSet,
        model_out: torch.Tensor,
        src_path,
        label_path,
        out_dir,
    ):
        utils.Framework.save_single(val_set, model_out, src_path, label_path, out_dir)
        _, src_name, _ = utils.cut_path(src_path)
        val_set.label_template.visualize_RGB(
            False, os.path.join(out_dir, src_name + utils.ImagePostfix.PNG.value)
        )


def Phi(tensor, data: dict):
    tensor = utils.HSImage.tensor_spec2bands_upsample(
        tensor, data["specs"][0], data["bands"][0]
    )  # 这个比较慢
    background = torch.mean(tensor, dim=1, keepdim=True) * data["beta"]
    tensor = tensor * data["A"]
    tensor = utils.LASISIntfImage.tensor_bands2intf_fft(tensor, data["center"])
    tensor = torch.abs(tensor + background) * data["M"]
    return tensor


def Phi_inv(tensor, data: dict):
    tensor /= data["M"]
    tensor = utils.LASISIntfImage.tensor_intf2bands_fft(tensor, data["center"])
    # utils.check_gpu_mem(tensor.device)
    tensor[:, : data["sample"][0][0]] = 0  # remove background
    tensor[:, data["sample"][0][1] :] = 0

    tensor = tensor / data["A"]

    tensor = torch.abs(tensor)
    tensor = utils.HSImage.tensor_band2spec_sample(
        tensor, data["bands"][0], data["specs"][0]
    )  # 这个比较慢
    return tensor


def FFT(tensor, data: dict):
    tensor = utils.HSImage.tensor_spec2bands_upsample(
        tensor, data["specs"][0], data["bands"][0]
    )
    tensor = utils.LASISIntfImage.tensor_bands2intf_fft(tensor, data["center"])
    return tensor


def FFT_inv(tensor, data: dict):
    tensor = utils.LASISIntfImage.tensor_intf2bands_fft(tensor, data["center"])
    # tensor /= torch.mean(data['A'], dim=-1, keepdim=True)
    tensor = torch.abs(tensor)
    tensor = utils.HSImage.tensor_band2spec_sample(
        tensor, data["bands"][0], data["specs"][0]
    )
    return tensor


def Phi_inv0(tensor, data: utils.Batch):
    return tensor


def train(
    config: Config,
    deg_model: degradation_model.DegregationModel,
    dataset: MyDataset2,
    set_name,
    cropped=True,
    online_simulation=True,
    gains=None,
    save_model_out=False,
):
    train_set, val_set = generate_loader(
        config, True, cropped, online_simulation, deg_model, dataset, set_name, gains
    )
    resume = False
    if config.model_name is not None:
        resume = True
    model = IHRFramework(config, dataset.intf_template, dataset.spec_template)
    loss_list = model.train(config, train_set, val_set, resume, save_model_out)
    return loss_list


def test(
    config: Config,
    deg_model: degradation_model.DegregationModel,
    dataset: MyDataset2,
    set_name,
    gains=None,
    online_simulation=False,
    save_model_out=True,
):
    test_set = generate_loader(
        config, False, False, online_simulation, deg_model, dataset, set_name, gains
    )
    model = IHRFramework(config, dataset.intf_template, dataset.spec_template)

    metrix = model.test(config, test_set, save_model_out)
    return metrix


def complexity(
    config: Config,
    deg_model: degradation_model.DegregationModel,
    dataset: MyDataset2,
    set_name,
    cropped=True,
    online_simulation=True,
    gains=None,
):
    train_set, val_set = generate_loader(
        config, True, cropped, online_simulation, deg_model, dataset, set_name, gains
    )
    model = IHRFramework(config, dataset.intf_template, dataset.spec_template)
    test_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    inputs = next(iter(test_loader))
    inputs = train_set.batch(*inputs)
    inputs.to(model.device)
    flops, n_params = model.complexity(config, (inputs.src, inputs.__dict__))
    return flops, n_params
