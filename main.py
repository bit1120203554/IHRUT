import os
import numpy as np
import cv2

import utils
import methods
import methods.models

if __name__ == "__main__":
    # 运行准备
    config_path = "./config.json"
    config = methods.Config(config_path)
    gains = [4]
    my_dataset2 = methods.MyDataset2(config)
    # inv = np.diff(my_dataset2.spec_template.channels)
    # pass

    # d1 = my_dataset2.spec_template.template_load(
    #     "results/output/cali/ISR_INV0_DnCNN_0/20191027_120610_manual_120f_VNIR_ZZ_G4_DN3500.tiff"
    # )
    # d2 = my_dataset2.get_data(
    #     "test",
    #     "HSI",
    #     "Houston",
    #     "ZZ",
    #     gain=gains[0],
    #     name="Houston13-0-0",
    # )

    # utils.show_line(
    #     my_dataset2.spec_template.channels,
    #     [d2.data[:, 60, 63]],
    #     "ooo",
    #     True,
    #     "./xibar6.png",
    # )

    # d1 = my_dataset2.spec_template.template_load(
    #     "results/output/chikusei/ISR_INV_DnCNN_0/ZZ_G4_Chikusei-1-0.tiff"
    # )
    # d1.data = d1.data[:, :, 1000:1800]
    # d2 = my_dataset2.get_data(
    #     "test",
    #     "HSI",
    #     "chikusei",
    #     "ZZ",
    #     gain=gains[0],
    #     name="Chikusei-1-0",
    # )
    # d2.data = d2.data[:, :, 1000:1800]

    # d2 = my_dataset2.get_data(
    #     "train",
    #     "HSI",
    #     "real",
    #     "ZZ",
    #     gain=gains[0],
    #     name="10-1-0",
    # )
    # utils.show_line(
    #     my_dataset2.spec_template.channels,
    #     [d2.data[:, 60, 63]],
    #     "ooo",
    #     True,
    #     "./xibar5.png",
    # )

    # utils.show_line(
    #     my_dataset2.spec_template.channels,
    #     [d1.data[:, 100, 350], d2.data[:, 100, 350]],
    #     "ooo",
    #     False,
    #     "./xibar2.png",
    # )
    # t1 = utils.rmse(d1.to_tensor(), d2.to_tensor())
    pass
    # utils.show_line(
    #     my_dataset2.spec_template.channels,
    #     [d1.data[:, 60, 63], d2.data[:, 60, 63]],
    #     "ooo",
    #     False,
    #     "./xibar2.png",
    # )

    # y, x = utils.LASISIntfImage.intf2bands_fft(d1.data, d1.channels)
    # utils.show_line(x[21:], -y[21:, 60, 63], "111")

    # HSOD_channels = np.linspace(400, 1000, num=200)
    # chikusei_channels = np.linspace(363, 1018, num=128)
    # WDC_channels = np.linspace(400, 2400, num=191)
    # real_channels = np.linspace(450, 900, num=34)
    # Houston_channels = np.linspace(364, 1046, num=144)

    # 数据集预处理

    # utils.visualize_HSIs(
    #     "F:/dataset/357",
    #     "F:/dataset/vis1_5",
    #     utils.HSImage(
    #         filename=None,
    #         level=4095,
    #         channels=HSOD_channels,
    #         data=None,
    #     ),
    #     mat_key="dataset",
    #     norm=True,
    #     clean_rate=1.5,
    # )

    # my_dataset2.cut_metadata(
    #     "train",
    #     "HSI",
    #     "hsod-train",
    #     HSOD_channels,
    #     "dataset",
    #     config.patch_size,
    #     bias=0,
    #     norm=True,
    #     resize=0.5,
    # )

    # my_dataset2.cut_metadata(
    #     "train",
    #     "HSI",
    #     "real",
    #     real_channels,
    #     None,
    #     config.patch_size,
    #     bias=0,
    #     norm=False,
    # )

    # my_dataset2.cut_metadata(
    #     "test",
    #     "HSI",
    #     "hsod-test",
    #     HSOD_channels,
    #     "dataset",
    #     (config.patch_size[0], my_dataset2.W),
    #     bias=0,
    #     norm=True,
    # )

    # my_dataset2.cut_metadata(
    #     "train",
    #     "HSI",
    #     "chikusei",
    #     chikusei_channels,
    #     "chikusei",
    #     (config.patch_size[0], my_dataset2.W),
    #     bias=50,
    # )

    # my_dataset2.cut_metadata(
    #     "test",
    #     "HSI",
    #     "chikusei",
    #     chikusei_channels,
    #     "chikusei",
    #     (config.patch_size[0], my_dataset2.W),
    #     bias=50,
    # )

    # my_dataset2.cut_metadata(
    #     "test",
    #     "HSI",
    #     "WDC",
    #     WDC_channels,
    #     cut_size=(config.patch_size[0], my_dataset2.W),
    #     bias=0,
    #     norm=True,
    # )

    # my_dataset2.cut_metadata(
    #     "test",
    #     "HSI",
    #     "Houston",
    #     Houston_channels,
    #     cut_size=(config.patch_size[0], my_dataset2.W),
    #     bias=0,
    #     norm=True,
    # )

    # my_dataset2.cut_metadata(
    #     "test",
    #     "intf",
    #     "qiao",
    #     my_dataset2.intf_template.channels,
    #     cut_size=(256, my_dataset2.W),
    # )

    # 这段是用来给cali数据集生成GT和GT的可视化结果的。此外，传统方法复原的数据也可以用这段代码生成可视化结果。
    # spec_template0 = my_dataset2.spec_template.copy()
    # # 1
    # output_dir = os.path.join(my_dataset2.root, "test", "HSI", "cali")
    # # 2
    # # output_dir = "results/output/cali/traditional"
    # for gain in gains:
    #     for DN in my_dataset2.DN_code.keys():
    #         if DN == 0:
    #             spec_ref = np.zeros((70, 120, 2048))
    #         else:
    #             spec_ref = my_dataset2.get_data(
    #                 "calibration",
    #                 "HSI",
    #                 "absolute",
    #                 "ZZ",
    #                 gain=gain,
    #                 DN=DN,
    #             ).data
    #             # 1
    #             spec_ref = np.zeros_like(spec_ref) + np.median(
    #                 spec_ref, axis=(1, 2), keepdims=True
    #             )
    #             # spec_ref = np.zeros_like(spec_ref) + np.mean(
    #             #     spec_ref, axis=(1, 2), keepdims=True
    #             # ) * np.mean(spec_ref, axis=(0, 1), keepdims=True) / np.mean(spec_ref)
    #             # 2
    #             pass

    #         spec_template0.data = spec_ref
    #         spec_template0.filename = os.path.join(
    #             output_dir,
    #             my_dataset2.DN_code[DN] + utils.ImagePostfix.TIFF.value,
    #         )
    #         spec_template0.visualize_RGB(
    #             False,
    #             os.path.join(
    #                 output_dir,
    #                 my_dataset2.DN_code[DN] + utils.ImagePostfix.PNG.value,
    #             ),
    #         )
    #         spec_template0.save()

    # # 建模、标定、初步验证
    degregation_model = methods.DegregationModel()
    # degregation_model.calibration(config, my_dataset2, mode="ZZ")

    # # 读取标定参数
    degregation_model.load(
        os.path.join(config.result_path, "calibration", "ZZ_1733764060.pkl")
    )
    # degregation_model.validation(my_dataset2, mode="ZZ")

    # # 制作仿真测试数据、进一步验证
    # degregation_model.simulation_dataset(my_dataset2, "test", "chikusei", gains=gains)
    # degregation_model.simulation_dataset(my_dataset2, "test", "hsod-test", gains=gains)
    # degregation_model.simulation_dataset(
    #     my_dataset2, "train", "hsod-train-a4-standard", gains=gains, rand_offset=True
    # )
    # degregation_model.simulation_dataset(
    #     my_dataset2, "train", "real", gains=gains, rand_offset=True
    # )
    # degregation_model.simulation_dataset(my_dataset2, "test", "Houston", gains=gains)

    # 传统方法的可视化和指标计算
    # trad = "hsod-test1"
    # trad_path = "results/output/%s/traditional" % (trad)
    # utils.visualize_HSIs(trad_path, trad_path, my_dataset2.spec_template)
    # trad_paths = os.listdir(trad_path)
    # for p in trad_paths:
    #     full_p = os.path.join(trad_path, p)  # 获得完整路径
    #     _, p_text, p_postfix = utils.cut_path(p)
    #     _, _, p_text = my_dataset2.cut_special_path(p_text, p_postfix)
    #     if p_postfix in [
    #         utils.ImagePostfix.JPG.value,
    #         utils.ImagePostfix.PNG.value,
    #     ]:
    #         continue
    #     # mode, gain, DN_or_name = my_dataset2.cut_special_path(p_text, p_postfix)
    #     DN_or_name = p_text
    #     src = my_dataset2.spec_template.template_load(full_p)
    #     src = src.to_tensor()
    #     spec_ref = my_dataset2.get_data(
    #         "test",
    #         "HSI",
    #         trad,
    #         None,
    #         name=DN_or_name,
    #     )
    #     spec_ref = spec_ref.to_tensor()
    #     result = methods.IHRFramework.eval_single(src, spec_ref)
    #     print(full_p)
    #     print(result)

    # # 仿真数据训练 仿真数据测试 真实数据测试
    # 测试FFT和INV
    # config.set_model_params(model_type="FFT", num_stages=0)
    # model = methods.IHRFramework(
    #     config, my_dataset2.intf_template, my_dataset2.spec_template
    # )
    # test_set = methods.generate_loader(
    #     config, False, False, False, degregation_model, my_dataset2, "cali", gains
    # )
    # test_set = methods.generate_loader(
    #     config, True, False, True, degregation_model, my_dataset2, "hsod-train", gains
    # )[0]

    # metrix, details = model.evaluate(config, test_set, True, False)

    # print(metrix)
    # print(details)

    # 测试其他模型
    # config.set_model_params(model_type="FFT", num_stages=1)
    # config.set_model_params(
    #     model_type="FFT",
    #     prior_type="ResUNet",
    #     num_stages=1,
    #     prior_nc=[64, 128, 256, 512],
    #     prior_nb=4,
    # # )
    # config.set_model_params(
    #     model_type="GAPNet",
    #     prior_type="",
    # )
    # methods.train(
    #     config, degregation_model, my_dataset2, "hsod-train", True, False, gains, True
    # )

    # config.set_model_params(model_type="INV", num_stages=1)
    # config.set_model_params(
    #     model_type="INV",
    #     prior_type="ResUNet",
    #     num_stages=1,
    #     prior_nc=[64, 128, 256, 512],
    #     prior_nb=4,
    # )
    # methods.train(
    #     config, degregation_model, my_dataset2, "hsod-train", True, False, gains, True
    # )
    # methods.train(
    #     config, degregation_model, my_dataset2, "hsod-train", True, True, gains, True
    # )

    # config.set_model_params(
    #     model_type="FFT", prior_type="SERT", num_stages=1, sharing=True
    # )
    # config.set_model_params(
    #     model_type="MAUN", prior_type="4.17", num_stages=3, sharing=False
    # )

    # config.set_model_params(
    #     model_type="IHRUT",
    #     prior_type="rbt2-2",
    #     model_name="ISR_IHRUT_rbt2-2_5_hsod-train_1747124085_hsod-train_1747156779",
    #     num_stages=5,
    #     sharing=False,
    # )
    # #
    # methods.train(
    #     config, degregation_model, my_dataset2, "hsod-train", True, False, gains, False
    # )
    # config.set_model_params(
    #     model_type="IHRUT",
    #     prior_type="deg_ab4-standard_5",
    #     model_name="ISR_IHRUT_deg_ab4-standard_5_hsod-train-a4-standard_1747118974",
    #     num_stages=5,
    #     sharing=False,
    # )
    # #
    # methods.train(
    #     config, degregation_model, my_dataset2, "hsod-train-a4-standard", True, False, gains, False
    # )

    config.set_model_params(
        model_type="IHRUT",
        prior_type="rbtF2_in",
        model_name="ISR_IHRUT_4.24_5_hsod-train_1740034733",
        num_stages=5,
        sharing=False,
    )
    # methods.test(config, degregation_model, my_dataset2, "chikusei", gains, False, True)
    # methods.test(config, degregation_model, my_dataset2, "cali", gains, False, True)
    # pass

    # config.set_model_params(
    #     model_type="FFT",
    #     prior_type="SERT",
    #     model_name="ISR_FFT_SERT_1_hsod-train_1736777233",
    #     num_stages=1,
    #     sharing=False,
    # )
    # methods.test(
    #     config, degregation_model, my_dataset2, "hsod-test1", gains, False, False
    # )
    # methods.test(
    #     config, degregation_model, my_dataset2, "chikusei", gains, False, False
    # )
    # methods.test(config, degregation_model, my_dataset2, "cali", gains, False, True)
    methods.test(config, degregation_model, my_dataset2, "Houston", gains, False, True)
    # methods.test(config, degregation_model, my_dataset2, "qiao", gains, False, True)
    # methods.complexity(
    #     config, degregation_model, my_dataset2, "hsod-debug", True, False, gains
    # )
    # pass
