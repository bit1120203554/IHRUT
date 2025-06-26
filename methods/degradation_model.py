import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy
from statsmodels.tsa.stattools import adfuller as ADF
import time
import os
import pickle

import utils
from .config import Config
from . import MyDataset2

# 特点：H是推扫方向，而所有的噪声在幅度W方向差别较大，在L方向差别较小。

"""退化模型"""


class DegregationModel:
    def __init__(self):
        self.mode = "ZZ"
        self.L = 256
        self.W = 2048
        self.level = 4095
        self.OPDs = []
        self.bands = []
        self.sample = []
        #### gain
        self.elc_gain = []
        self.elc_range = 0.1
        #### optical degradation
        self.A = []  #
        self.beta = []  # 谱域的低频信号
        self.M = []  # signal bias and baseline 相对校正系数
        #### electrical degradation
        ### sensing
        self.K = []  # sensor response
        ### signal-independent
        self.D0 = []  # dark current
        self.N_base = []
        self.N_sigma = []  #
        self.N_L_sigma = []  # stripe noise

        pass

    # 以mat文件形式存储
    def load(self, load_path=None):
        with open(load_path, "rb") as f:
            obj = pickle.load(f)
            for k, v in vars(obj).items():
                setattr(self, k, v)
        pass

    def save(self, save_path=None):
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        pass

    # calibration
    def calibration(self, config: Config, dataset: MyDataset2, mode="ZZ"):
        self.mode = mode
        self.L = dataset.L
        self.W = dataset.W
        self.level = dataset.level
        self.OPDs = dataset.intf_template.channels

        print("calibration started.\n path:%s, mode:%s" % (dataset.root, mode))
        self.calibrate_degradation(dataset, mode)  # 逐增益的参数估计
        self.calibrate_joint()  # 增益求解和制作矩阵
        save_path = os.path.join(
            config.result_path,
            "calibration",
            mode + "_" + str(round(time.time())) + ".pkl",
        )
        self.save(save_path)
        # self.validation( dataset, mode)

    def calibrate_degradation(self, dataset: MyDataset2, mode):
        L = dataset.L
        W = dataset.W
        l = np.arange(L)
        w = np.arange(W)
        specs = dataset.specs
        OPDs = dataset.intf_template.channels
        bands, center = utils.LASISIntfImage.OPDs2bands(OPDs)
        self.bands = bands
        # 波数曲线上的低中频和中高频信号的分界线，最近邻插值
        f = scipy.interpolate.interp1d(bands, np.arange(len(bands)), kind="nearest")
        medium = int(f(1 / (6 * specs[-1] - 5 * specs[-2])))
        high = int(f(1 / (specs[0])))
        self.sample = np.array([medium, high])

        g = 0
        # 逐个标定不同增益下的结果
        for gain in dataset.gain_code.keys():
            print("gain: %d" % gain)
            ### 数据读取和预处理
            absolutes = []
            relatives = []
            spec_refs = []
            for DN in dataset.DN_code.keys():
                absolute = dataset.get_data(
                    "calibration",
                    "intf",
                    "absolute",
                    mode,
                    gain=gain,
                    DN=DN,
                ).data
                absolutes.append(absolute)
                relative = dataset.get_data(
                    "calibration",
                    "intf",
                    "relative",
                    mode,
                    gain=gain,
                    DN=DN,
                ).data
                relatives.append(relative)
                if DN != 0:
                    spec_ref = dataset.get_data(
                        "calibration",
                        "HSI",
                        "absolute",
                        mode,
                        gain=gain,
                        DN=DN,
                    ).data
                    spec_refs.append(spec_ref)

            dark = absolutes[0]
            dark_h_mean = np.mean(dark, axis=1, keepdims=True)
            absolutes = np.array(absolutes)

            # absolutes_h_mean = np.mean(absolutes, axis=2, keepdims=True)
            # absolute_band, _ = utils.LASISIntfImage.intf2bands_fft(
            #     -absolutes_h_mean[1], OPDs
            # )
            # z = np.zeros_like(absolute_band)
            # z[medium:] = absolute_band[medium:]
            # utils.show_line(
            #     bands,
            #     [np.real(z[..., 63]), np.imag(z[..., 63])],
            #     "dark_band",
            # )
            # z1_back, _ = utils.LASISIntfImage.bands2intf_fft(z, bands, center)
            # utils.show_line(l, z1_back[..., 63], "z1_back")

            absolutes = (absolutes - absolutes[0])[1:]  # 去暗电流
            absolutes_h_mean = np.mean(absolutes, axis=2, keepdims=True)

            relatives = np.array(relatives)
            relatives_h_var = np.var(relatives, axis=2, keepdims=True)
            relatives_h_var = (relatives_h_var - relatives_h_var[0])[1:]
            relatives = (relatives - relatives[0])[1:]  # 去暗电流
            relatives_h_mean = np.mean(relatives, axis=2, keepdims=True)

            ### step1
            print("# electrical degradation")

            print("### D & noises")
            """D & N_D"""
            D0 = np.median(dark_h_mean)
            N_base = dark_h_mean - D0

            """N_L & N_read"""
            N = dark - dark_h_mean
            print("stripes")
            print(np.std(np.mean(N, axis=0)))
            print(np.std(np.mean(N, axis=1)))
            print(np.std(np.mean(N, axis=2)))

            N_L = np.mean(N, axis=0, keepdims=True)
            N_L_sigma = np.std(N_L, axis=1, keepdims=True)
            # utils.show_line(w, N_L_sigma.flatten(), "N_L_sigma")
            utils.fit_norm_dist((N_L / N_L_sigma).flatten(), vision=False, title="N_L")

            N -= N_L
            # utils.show_line(l, N_L.flatten(), "N_L_sigma")
            N_sigma = np.std(N, axis=1, keepdims=True)
            # utils.show_line(l, N_sigma[..., 63], "N_sigma")
            # utils.show_line(l, np.mean(N_sigma, axis=(1, 2)), "N_sigma_L")
            # utils.show_line(w, np.mean(N_sigma, axis=(0, 1)), "N_sigma_W")

            read = N / N_sigma
            read = read.transpose(0, 2, 1).reshape(-1, N.shape[1])
            np.random.seed(123)
            indices = np.random.choice(read.shape[0], size=1000, replace=False)
            x = read[indices]
            utils.fit_norm_dist(x.flatten(), vision=False, title="N")

            # utils.show_line(
            #     l, relatives_h_mean[0][..., 63].flatten(), title="relatives_63"
            # )
            # utils.show_line(l, relatives_h_var[0][..., 63].flatten(), title="var_63")
            # utils.show_line(
            #     l, np.mean(relatives_h_mean[0], axis=2).flatten(), title="relatives_l"
            # )
            # utils.show_line(
            #     l, np.mean(relatives_h_var[0], axis=2).flatten(), title="var_l"
            # )

            """K"""
            K0 = relatives_h_var / relatives_h_mean
            K = np.mean(K0, axis=0)
            # mi, ma = (np.min(K), np.max(K))
            # utils.show_line(w, np.mean(K, axis=0).squeeze(), "K")
            # utils.show_line(l, np.mean(K, axis=-1).squeeze(), "K")

            ### step2
            print("# optical degradation")

            """M_sensing"""
            M_sensing0 = relatives_h_mean[:, 0:1]
            M_sensing = relatives_h_mean / np.mean(
                M_sensing0, axis=(1, 2, 3), keepdims=True
            )
            M_sensing = np.mean(M_sensing, axis=0)
            # utils.show_line(l, np.mean(M_sensing, axis=(1, 2)), "M_L")
            # utils.show_line(l, M_sensing[..., 63], "M_L_eg.")
            # utils.show_line(l, np.mean(M_sensing, axis=0).flatten(), title="M_L_eg.")
            # utils.show_line(w, np.mean(M_sensing, axis=(0, 1)), "M_W_eg.")

            # ab_bands = []
            Ms = []
            betas = []
            As = []
            for DN in np.arange(absolutes.shape[0]):
                spec_ref = spec_refs[DN]
                spec_ref = np.median(spec_ref, axis=(1, 2), keepdims=True)
                # spec_ref = np.zeros_like(spec_ref) + np.mean(
                #     spec_ref, axis=(1, 2), keepdims=True
                # ) * np.mean(spec_ref, axis=(0, 1), keepdims=True) / np.mean(spec_ref)
                spec_ref_band = utils.HSImage.spec2bands_upsample(
                    spec_ref, specs, bands
                )
                # utils.show_line(bands, spec_ref_band.squeeze(), "spec_ref_band")

                """M_base"""
                ab = absolutes_h_mean[DN] / M_sensing
                absolute_band, _ = utils.LASISIntfImage.intf2bands_fft(ab, OPDs)
                z = np.zeros_like(absolute_band)
                z[: int(medium / 2)] = absolute_band[: int(medium / 2)]
                # utils.show_line(
                #     bands,
                #     [np.real(z[..., 63]), np.imag(z[..., 63])],
                #     "beta",
                # )
                z_back, _ = utils.LASISIntfImage.bands2intf_fft(z, bands, center)
                # utils.show_line(l, [ab[..., 63], z_back[..., 63]], "z_back")

                background = np.mean(z_back)
                M_base = z_back / background  # 三维
                # utils.show_line(l, M_base[..., 63], "M_base")
                M = M_base * M_sensing
                # utils.show_line(l, M[..., 63], "M")
                ab /= M_base

                # ADF check
                # utils.show_line(l, ab[..., 63], "db_intf")
                # adf_result = ADF(ab[..., 63].squeeze())
                # print("ADF Statistic: %f" % adf_result[0])
                # print("p-value: %f" % adf_result[1])
                # print("Lags Used: %f" % adf_result[2])
                # print("Number of Observations Used: %f" % adf_result[3])

                # utils.show_line(l, np.mean(M, axis=2).flatten(), title="M_L_eg.")
                # utils.show_line(w, np.mean(M, axis=(0, 1)), "M_W_eg.")

                """A"""
                absolute_band, _ = utils.LASISIntfImage.intf2bands_fft(ab, OPDs)
                # z = utils.HSImage.spec2bands_upsample(
                #     utils.HSImage.bands2spec_sample(absolute_band, bands, specs),
                #     specs,
                #     bands,
                # )
                z = np.zeros_like(absolute_band)
                z[medium:high] = absolute_band[medium:high]
                z += utils.EPSILON
                # utils.show_line(
                #     bands,
                #     [np.real(z[..., 63]), np.imag(z[..., 63])],
                #     "signal_band",
                # )
                A = spec_ref_band / z
                A.real = utils.clean_abnormal_Q(A.real, axis=0)[0]
                A = 1 / A
                A.real = utils.clean_abnormal_Q(A.real, axis=-1)[0]
                A[:medium] = 1 + 0j
                A[high:] = 1 + 0j  # 处理特殊值/0值/无穷大值
                # utils.show_line(
                #     bands,
                #     [np.real(A[..., 63]), np.imag(A[..., 63])],
                #     "A",
                # )
                # utils.show_line(
                #     bands[medium:high],
                #     [np.real(A[medium:high, :, 63]), np.imag(A[medium:high, :, 63])],
                #     "A",
                # )
                # z_back, _ = utils.LASISIntfImage.bands2intf_fft(z, bands, center)
                # utils.show_line(l, [z_back[..., 63], ab[..., 63]], "z_back")

                """beta"""
                beta_band = absolute_band - z
                # utils.show_line(
                #     bands,
                #     [np.real(beta_band[..., 63]), np.imag(beta_band[..., 63])],
                #     "background_band",
                # )
                beta, _ = utils.LASISIntfImage.bands2intf_fft(beta_band, bands, center)
                # utils.show_line(l, [beta[..., 63], ab[..., 63]], "beta")
                beta = beta / np.mean(spec_ref_band, axis=0, keepdims=True)
                # utils.show_line(l, [beta[..., 63]], "beta")

                Ms.append(M)
                betas.append(beta)
                As.append(A)
                pass

            print(
                "M %f %f"
                % (np.mean(np.std(np.array(Ms), axis=0)), np.mean(np.array(Ms)))
            )
            M = np.abs(np.mean(np.array(Ms), axis=0)) + utils.EPSILON
            beta = np.mean(np.array(betas), axis=0)
            A = np.mean(np.array(As), axis=0)

            self.D0.append(D0)
            self.N_base.append(N_base)
            self.N_L_sigma.append(N_L_sigma)
            self.N_sigma.append(N_sigma)
            self.K.append(K)
            self.M.append(M)
            self.beta.append(beta)
            self.A.append(A)
            g += 1

    # 联合标定
    def calibrate_joint(self):
        def count_gain(data):
            result = np.mean(np.array(data), axis=(1, 2, 3))
            result /= result[0]
            return np.array(result)

        def remove_gain(data, gain):
            result = data / gain.reshape(-1, 1, 1, 1)
            print(
                "## %f %f" % (np.mean(np.std(result, axis=0)), np.mean(result))
            )  # 误差估计
            result = np.mean(result, axis=0)
            return result

        L = self.OPDs.shape[0]
        W = self.W
        bands = self.bands
        l = np.arange(L)
        w = np.arange(W)
        # elc_gain的估计
        K_gain = count_gain(self.K)
        N_L_gain = 0.5 + 0.5 * count_gain(self.N_L_sigma)
        N_gain = count_gain(self.N_sigma)
        N_base_gain = []
        for g in np.arange(len(self.K)):
            # utils.show_line(w, np.mean(self.N_base[g], axis=(0, 1)), "N_base")
            # utils.show_line(w, self.N_L_sigma[g].flatten(), "N_L_sigma")
            # utils.show_line(w, np.mean(self.N_sigma[g], axis=(0, 1)), "N_sigma")
            # utils.show_line(w, np.mean(self.K[g], axis=(0, 1)), "K")
            # utils.show_line(l, np.mean(self.M[g], axis=-1).flatten(), "M")
            # utils.show_line(l, np.mean(self.beta[g], axis=-1).flatten(), "beta")
            # utils.show_line(
            #     bands,
            #     [
            #         np.real(np.mean(self.A[g], axis=-1)),
            #         np.imag(np.mean(self.A[g], axis=-1)),
            #     ],
            #     "A",
            # )

            a, b, c = utils.count_linear(
                self.N_base[0][0].flatten(),
                self.N_base[g][0].flatten(),
            )
            N_base_gain.append(a)
        N_base_gain = np.array(N_base_gain).squeeze()
        N_base_gain /= N_base_gain[0]

        self.elc_gain = np.mean(
            np.array([K_gain, N_L_gain, N_gain, N_base_gain]), axis=0
        )

        self.K = remove_gain(self.K, K_gain)
        self.N_L_sigma = remove_gain(self.N_L_sigma, N_L_gain)
        self.N_sigma = remove_gain(self.N_sigma, N_gain)
        self.N_base = np.array(self.N_base) / N_base_gain.reshape(
            -1, 1, 1, 1
        )  # 不取均值，因为每个增益下波形不一样

        # 误差估计
        N_base_cut = self.N_base[-2:]
        print(
            "D base %f %f" % (np.mean(np.std(N_base_cut, axis=0)), np.mean(N_base_cut))
        )
        self.A = np.array(self.A)
        print(
            "A %f %f"
            % (
                np.mean(
                    np.std(np.abs(self.A[:, self.sample[0] : self.sample[1]]), axis=0)
                ),
                np.mean(np.abs(self.A[:, self.sample[0] : self.sample[1]])),
            )
        )
        print("beta %f %f" % (np.mean(np.std(self.beta, axis=0)), np.mean(self.beta)))
        print("D0 %f %f" % (np.mean(np.std(self.D0, axis=0)), np.mean(self.D0)))

        # 直接估计
        self.D0 = np.mean(self.D0)
        self.M = np.array(self.M)
        self.beta = np.mean(np.array(self.beta), axis=0)
        self.A = np.mean(np.array(self.A), axis=0)
        pass

    # 从dataset的HSI生成intf的训练集和数据集
    def simulation_dataset(
        self,
        dataset: MyDataset2,
        subset,
        set_name,
        gains=None,
        rand_offset=True,
    ):
        if gains is None:
            gains = dataset.gain_code.keys()

        set_dir = os.path.join(dataset.root, subset, "HSI", set_name)
        output_dir = os.path.join(dataset.root, subset, "intf", set_name)
        utils.mkdir(output_dir)
        paths = os.listdir(set_dir)
        for p in paths:
            full_p = os.path.join(set_dir, p)
            _, p_text, p_postfix = utils.cut_path(p)
            if p_postfix in [
                utils.ImagePostfix.JPG.value,
                utils.ImagePostfix.PNG.value,
            ]:
                continue
            print("now: %s" % p)
            cut = dataset.spec_template.template_load(full_p)
            for gain in gains:
                if rand_offset and cut.data.shape[2] < dataset.max_W:
                    W_offset = np.random.randint(0, dataset.max_W - cut.data.shape[2])
                else:
                    W_offset = 0
                cut_intf = dataset.intf_template.copy()
                # utils.TimeCheck.init()
                # utils.show_line(cut.channels, cut.data[:, 100, 1000].squeeze())
                cut_intf.data = self.simulation_ablation4_standard(
                    cut.data, cut.channels, gain, W_offset
                )
                # cut_intf.visualize_grayscale(0, True, "C:/Users/25785/Desktop/a.png")
                cut_intf.head = np.zeros((1, cut.data.shape[1], cut.data.shape[2]))
                cut_intf.head[..., 63] = gain * 256
                cut_intf.filename = os.path.join(
                    output_dir,
                    "%s_%s_%s+%d%s"
                    % (
                        self.mode,
                        dataset.gain_code[gain],
                        p_text,
                        W_offset,
                        utils.ImagePostfix.RAW.value,
                    ),
                )
                cut_intf.save()
                pass
            pass
        pass

    # 从dataset的HSI生成intf的训练集和数据集
    # simulation
    def simulation(self, data, specs, gain=1, W_offset=0):
        W = data.shape[2]
        if W_offset + W > self.W:
            print("the'W' out of bound.")
            return
        # utils.TimeCheck.init()
        data = utils.HSImage.spec2bands_upsample(data, specs, self.bands)
        data = self.simulate_optical_degradation(data, gain, W_offset)
        # utils.show_line(np.arange(256), result1[:, 100, 2020].squeeze(), "w")
        # utils.TimeCheck.check(True)
        data = self.simulate_electrical_degradation(data, gain, W_offset)
        # utils.TimeCheck.check(True)
        return data

    def simulate_optical_degradation(self, data, gain=1, W_offset=0):
        l = np.arange(len(self.OPDs))
        W = data.shape[2]
        center = l[np.abs(self.OPDs - 0) <= utils.EPSILON][0]
        background = (
            np.mean(data, axis=0, keepdims=True)
            * self.beta[..., W_offset : W_offset + W]
        )
        data = data * self.A[..., W_offset : W_offset + W]
        data, _ = utils.LASISIntfImage.bands2intf_fft(data, self.bands, center)
        data = np.abs(data + background)
        data *= self.M[gain - 1, ..., W_offset : W_offset + W]
        return data

    def simulate_electrical_degradation(self, data, gain=1, W_offset=0, use_dark=True):
        def get_elec_gain(gain):  # 根据log线性的假设，随机取参数
            return np.exp(np.random.randn(1) * self.elc_range) * self.elc_gain[gain - 1]

        data = np.copy(data)
        L = data.shape[0]
        H = data.shape[1]
        W = data.shape[2]
        g = gain - 1

        # dark
        D0 = self.D0
        N_base = self.N_base[g, ..., W_offset : W_offset + W] * get_elec_gain(gain)
        N_L_sigma = self.N_L_sigma[..., W_offset : W_offset + W] * get_elec_gain(gain)
        N_sigma = self.N_sigma[..., W_offset : W_offset + W] * get_elec_gain(gain)
        D = D0 + N_base
        N_L = np.random.randn(1, H, W) * N_L_sigma
        N_read = np.random.randn(L, H, W) * N_sigma
        dark = D + N_L + N_read

        # sensing
        K = self.K[..., W_offset : W_offset + W] * get_elec_gain(gain)
        data = np.random.poisson(data / K) * K
        if use_dark:
            data += dark
            data = np.round(data)
        return data

    def simulation_ablation0(self, data, specs, gain=1, W_offset=0):
        W = data.shape[2]
        if W_offset + W > self.W:
            print("the'W' out of bound.")
            return
        # utils.TimeCheck.init()
        data = utils.HSImage.spec2bands_upsample(data, specs, self.bands)
        l = np.arange(len(self.OPDs))
        W = data.shape[2]
        center = l[np.abs(self.OPDs - 0) <= utils.EPSILON][0]
        data, _ = utils.LASISIntfImage.bands2intf_fft(data, self.bands, center)
        data = np.abs(data)
        return data

    def simulation_ablation1(self, data, specs, gain=1, W_offset=0):
        W = data.shape[2]
        if W_offset + W > self.W:
            print("the'W' out of bound.")
            return
        # utils.TimeCheck.init()
        data = utils.HSImage.spec2bands_upsample(data, specs, self.bands)
        l = np.arange(len(self.OPDs))
        W = data.shape[2]
        center = l[np.abs(self.OPDs - 0) <= utils.EPSILON][0]
        background = (
            np.mean(data, axis=0, keepdims=True)
            * self.beta[..., W_offset : W_offset + W]
        )
        data = data * self.A[..., W_offset : W_offset + W]
        data, _ = utils.LASISIntfImage.bands2intf_fft(data, self.bands, center)
        data = np.abs(data + background)
        # utils.show_line(np.arange(256), result1[:, 100, 2020].squeeze(), "w")
        # utils.TimeCheck.check(True)
        # utils.TimeCheck.check(True)
        return data

    def simulation_ablation2(self, data, specs, gain=1, W_offset=0):
        W = data.shape[2]
        if W_offset + W > self.W:
            print("the'W' out of bound.")
            return
        # utils.TimeCheck.init()
        data = utils.HSImage.spec2bands_upsample(data, specs, self.bands)
        data = self.simulate_optical_degradation(data, gain, W_offset)
        # utils.show_line(np.arange(256), result1[:, 100, 2020].squeeze(), "w")
        # utils.TimeCheck.check(True)
        # utils.TimeCheck.check(True)
        return data

    def simulation_ablation3(self, data, specs, gain=1, W_offset=0):
        W = data.shape[2]
        if W_offset + W > self.W:
            print("the'W' out of bound.")
            return
        # utils.TimeCheck.init()
        data = utils.HSImage.spec2bands_upsample(data, specs, self.bands)
        data = self.simulate_optical_degradation(data, gain, W_offset)

        L = data.shape[0]
        H = data.shape[1]
        W = data.shape[2]
        g = gain - 1
        D0 = self.D0
        N_base = self.N_base[g, ..., W_offset : W_offset + W]
        N_L_sigma = self.N_L_sigma[..., W_offset : W_offset + W]
        N_sigma = self.N_sigma[..., W_offset : W_offset + W]
        D = D0 + N_base
        N_L = np.random.randn(1, H, W) * N_L_sigma
        N_read = np.random.randn(L, H, W) * np.mean(N_sigma)
        dark = D + N_L + N_read
        # utils.show_line(np.arange(256), result1[:, 100, 2020].squeeze(), "w")
        # utils.TimeCheck.check(True)
        # utils.TimeCheck.check(True)
        data += dark
        return data

    def simulation_ablation4(self, data, specs, gain=1, W_offset=0):
        W = data.shape[2]
        if W_offset + W > self.W:
            print("the'W' out of bound.")
            return
        # utils.TimeCheck.init()
        data = utils.HSImage.spec2bands_upsample(data, specs, self.bands)
        data = self.simulate_optical_degradation(data, gain, W_offset)

        L = data.shape[0]
        H = data.shape[1]
        W = data.shape[2]
        g = gain - 1
        D0 = self.D0
        N_base = self.N_base[g, ..., W_offset : W_offset + W]
        N_L_sigma = self.N_L_sigma[..., W_offset : W_offset + W]
        N_sigma = self.N_sigma[..., W_offset : W_offset + W]
        D = D0 + N_base
        N_L = np.random.randn(1, H, W) * N_L_sigma
        N_read = np.random.randn(L, H, W) * np.mean(N_sigma)
        dark = D + N_L + N_read
        # utils.show_line(np.arange(256), result1[:, 100, 2020].squeeze(), "w")
        # utils.TimeCheck.check(True)
        # utils.TimeCheck.check(True)
        K = self.K[..., W_offset : W_offset + W]
        data = np.random.poisson(data / K) * K
        data += dark
        return data

    def simulation_ablation4_standard(self, data, specs, gain=1, W_offset=0):
        W = data.shape[2]
        if W_offset + W > self.W:
            print("the'W' out of bound.")
            return
        # utils.TimeCheck.init()
        data = utils.HSImage.spec2bands_upsample(data, specs, self.bands)
        data = self.simulate_optical_degradation(data, gain, W_offset)

        L = data.shape[0]
        H = data.shape[1]
        W = data.shape[2]
        g = gain - 1
        D0 = self.D0
        N_base = self.N_base[g, ..., W_offset : W_offset + W]
        N_L_sigma = self.N_L_sigma[..., W_offset : W_offset + W]
        N_sigma = self.N_sigma[..., W_offset : W_offset + W]
        D = D0 + N_base
        N_L = np.random.randn(1, H, W) * np.mean(N_L_sigma)
        N_read = np.random.randn(L, H, W) * np.mean(N_sigma)
        dark = D + N_L + N_read
        # utils.show_line(np.arange(256), result1[:, 100, 2020].squeeze(), "w")
        # utils.TimeCheck.check(True)
        # utils.TimeCheck.check(True)
        K = np.mean(self.K[..., W_offset : W_offset + W])
        data = np.random.poisson(data / K) * K
        data += dark
        return data

    def simulation_ablation5(self, data, specs, gain=1, W_offset=0):
        W = data.shape[2]
        if W_offset + W > self.W:
            print("the'W' out of bound.")
            return
        # utils.TimeCheck.init()
        data = utils.HSImage.spec2bands_upsample(data, specs, self.bands)
        data = self.simulate_optical_degradation(data, gain, W_offset)

        L = data.shape[0]
        H = data.shape[1]
        W = data.shape[2]
        g = gain - 1
        D0 = self.D0
        N_base = self.N_base[g, ..., W_offset : W_offset + W]
        N_L_sigma = self.N_L_sigma[..., W_offset : W_offset + W]
        N_sigma = self.N_sigma[..., W_offset : W_offset + W]
        D = D0 + N_base
        N_L = np.random.randn(1, H, W) * N_L_sigma
        N_read = np.random.randn(L, H, W) * N_sigma
        dark = D + N_L + N_read
        # utils.show_line(np.arange(256), result1[:, 100, 2020].squeeze(), "w")
        # utils.TimeCheck.check(True)
        # utils.TimeCheck.check(True)
        # sensing
        K = self.K[..., W_offset : W_offset + W]
        data = np.random.poisson(data / K) * K
        data += dark
        data = np.round(data)
        return data

    # validation
    def validation(self, dataset: MyDataset2, mode="ZZ"):
        L = dataset.L
        H = 120
        W = dataset.W
        l = np.arange(L)
        w = np.arange(W)
        g = 0
        e = np.copy(self.elc_range)
        self.elc_range = 0
        for gain in dataset.gain_code.keys():
            print("gain: %d" % gain)
            # ### 数据读取和预处理
            absolutes = []
            relatives = []
            spec_refs = []
            for DN in dataset.DN_code.keys():
                absolute = dataset.get_data(
                    "calibration",
                    "intf",
                    "absolute",
                    mode,
                    gain=gain,
                    DN=DN,
                ).data
                absolutes.append(absolute)
                relative = dataset.get_data(
                    "calibration",
                    "intf",
                    "relative",
                    mode,
                    gain=gain,
                    DN=DN,
                ).data
                relatives.append(relative)
                if DN != 0:
                    spec_ref = dataset.get_data(
                        "calibration",
                        "HSI",
                        "absolute",
                        mode,
                        gain=gain,
                        DN=DN,
                    ).data
                    spec_refs.append(spec_ref)

            dark = absolutes[0]
            dark_h_mean = np.mean(dark, axis=1, keepdims=True)

            absolutes = (absolutes)[1:]  # - absolutes[0]
            absolutes_h_mean = np.mean(absolutes, axis=2, keepdims=True)

            # utils.show_line(
            #     l, np.mean(absolutes_h_mean[1], axis=2).flatten(), title="abs_L"
            # )
            # utils.show_line(
            #     w, np.mean(absolutes_h_mean[1], axis=0).flatten(), title="abs_W"
            # )

            # test1: dark
            # pic0 = np.zeros((L, 120, W))
            # pic0 = self.simulate_electrical_degradation(pic0, gain=gain)
            # pic0_h_mean = np.mean(pic0, axis=1, keepdims=True)

            # # utils.show_scatter(l, pic0_h_mean[..., 63].flatten(), title="dark_L_63")
            # # utils.show_scatter(l, pic0_h_mean[..., 131].flatten(), title="dark_L_131")
            # # utils.show_line(w, pic0_h_mean[127][0], title="dark_example_127")

            # pic0_N = pic0 - pic0_h_mean
            # pic0_N_h_std = np.std(pic0, axis=1)
            # # utils.show_line(w, np.mean(pic0_N_h_std, axis=0), "N_h_std")  # 针对
            # print("stripes:")
            # print(np.std(np.mean(pic0_N, axis=0)))
            # print(np.std(np.mean(pic0_N, axis=2)))
            # print("KL: %f" % (utils.KL_distance(pic0, dark, self.level)))

            for DN in np.arange(len(spec_refs)):
                # utils.show_line(
                #     l,
                #     absolutes_h_mean[DN, ..., 63].flatten(),
                #     title="abs_L",
                # )
                # utils.show_line(
                #     w,
                #     np.mean(absolutes_h_mean[DN], axis=0).flatten(),
                #     title="abs_W",
                # )
                spec_ref = np.median(spec_refs[DN], axis=(1, 2), keepdims=True)
                pic2 = np.zeros((spec_ref.shape[0], 120, W)) + spec_ref
                pic2 = self.simulation(pic2, dataset.specs, gain=gain)
                pic2_mean1 = np.mean(pic2, axis=1, keepdims=True)
                # utils.show_line(l, pic2_mean1[..., 63], "pic2_L")
                # utils.show_line(w, np.mean(pic2_mean1, axis=0).flatten(), "pic2_W")
                print("KL: %f" % (utils.KL_distance(pic2, absolutes[DN], self.level)))
                g += 1
            pass
        self.elc_range = e
        pass
