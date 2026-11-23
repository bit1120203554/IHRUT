# Physical Degradation Model-Guided Interferometric Hyperspectral Reconstruction with Unfolding Transformer

**Yuansheng Li, Yunhao Zou, Linwei Chen, and Ying Fu · ICCV 2025**

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Physical_Degradation_Model-Guided_Interferometric_Hyperspectral_Reconstruction_with_Unfolding_Transformer_ICCV_2025_paper.html) · [Supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Li_Physical_Degradation_Model-Guided_ICCV_2025_supplemental.pdf) · [arXiv](https://arxiv.org/abs/2506.21880)

IHRUT reconstructs hyperspectral images from interferograms captured by a large-aperture static imaging spectrometer (LASIS). A degradation model fitted to radiometric calibration measurements generates training interferograms from hyperspectral images. At each unfolding stage, the reconstruction network combines a data update guided by the optical operators with a spatial–spectral transformer prior. Stripe Pattern Enhancement models the column-dependent spectral distortions of the instrument. See Figure 3 and §§3.2–3.3 of the [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Physical_Degradation_Model-Guided_Interferometric_Hyperspectral_Reconstruction_with_Unfolding_Transformer_ICCV_2025_paper.pdf).

## Released files

This repository contains the IHRUT model, degradation simulator, fitted LASIS calibration parameters, and two pretrained checkpoints.

| Model | Bundled checkpoint | Training epoch |
| --- | --- | ---: |
| IHRUT-5 | [model_latest.pth](results/checkpoints/ISR_IHRUT_4.24_5_hsod-train_1740034733/model_latest.pth) | 300 |
| IHRUT-7 | [model_latest.pth](results/checkpoints/ISR_IHRUT_4.24_7_hsod-train_1740139801_hsod-train_1740363592/model_latest.pth) | 300 |

The bundled calibration parameters are in [ZZ_1733764060.pkl](results/calibration/ZZ_1733764060.pkl). Load checkpoint and calibration files from trusted sources because the legacy formats use Python pickle.

## Datasets and reproducibility

The checkout includes the LASIS data format in [`info.json`](datasets/my_dataset2/info.json). Reproducing the paper requires the prepared datasets and experiment split below. The archived [`main.py`](main.py) also relies on legacy model imports that need updating for a fresh checkout.

| Dataset | Experiment inputs needed | Source |
| --- | --- | --- |
| HSOD-BIT | 276 HSI training patches of 256 × 256 × 70 and six prepared test scenes with simulated interferograms | [HSOD-BIT source images](https://github.com/anonymous0519/HSOD-BIT) |
| Houston | One prepared 256 × 2048 × 70 test scene and its simulated interferogram | [2013 IEEE GRSS Data Fusion Contest](https://www.grss-ieee.org/community/technical-committees/2013-ieee-grss-data-fusion-contest/) |
| LASIS | Paired measured interferograms and reference HSIs at DN1000, DN2000, and DN3000 | LASIS calibration measurements |

The paper uses 256 interferogram samples per pixel and reconstructs 70 bands from 450 to 900 nm. It trains with Adam, L1 loss, batch size 1, and an initial learning rate of 1e-4 over 300 epochs, with linear warm-up and cosine annealing on one RTX 3090. See §4.1 of the paper and §3.1 of the supplement for dataset preparation. [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Physical_Degradation_Model-Guided_Interferometric_Hyperspectral_Reconstruction_with_Unfolding_Transformer_ICCV_2025_paper.pdf) · [Supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Li_Physical_Degradation_Model-Guided_ICCV_2025_supplemental.pdf)

## Results reported in the paper

The values below come from Table 2. Each quality score is PSNR in dB / SSIM. The HSOD-BIT and LASIS columns are averages over six synthetic scenes and three real calibration scenes, respectively. [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Physical_Degradation_Model-Guided_Interferometric_Hyperspectral_Reconstruction_with_Unfolding_Transformer_ICCV_2025_paper.pdf)

| Model | Parameters | GFLOPs | HSOD-BIT | Houston | LASIS calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| IHRUT-5 | 3.20 M | 106.36 | 44.25 / 0.988 | 39.73 / 0.994 | 39.46 / 0.992 |
| IHRUT-7 | 4.42 M | 145.80 | 44.30 / 0.988 | 39.86 / 0.995 | 39.61 / 0.992 |

## Code map

| Path | Contents |
| --- | --- |
| [`methods/models/`](methods/models/) | IHRUT and comparison model implementations |
| [`methods/degradation_model.py`](methods/degradation_model.py) | Calibration and optical/electronic degradation simulation |
| [`methods/learning.py`](methods/learning.py) | Dataset batches, optical operators, and training and evaluation |
| [`config.json`](config.json) | Data paths and training settings |
| [`results/checkpoints/`](results/checkpoints/) | Released checkpoints and archived experiment logs |

## Citation

```bibtex
@InProceedings{Li_2025_ICCV,
  author    = {Li, Yuansheng and Zou, Yunhao and Chen, Linwei and Fu, Ying},
  title     = {Physical Degradation Model-Guided Interferometric Hyperspectral Reconstruction with Unfolding Transformer},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2025},
  pages     = {13815--13825},
  doi       = {10.1109/ICCV51701.2025.01282}
}
```
