# LDCT: Low-Dose CT Denoising with Diffusion Models

> 基于扩散模型（DDPM）的低剂量 CT（LDCT）图像去噪与重建

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.10-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<!--
<img width="2115" height="736" alt="result_0605" src="https://github.com/user-attachments/assets/9b56e67b-89c2-4500-9191-511ffaedc5fa" />

-->
![cover](assets/cover.png)

---

## 📖 简介

低剂量 CT（Low-Dose CT, LDCT）扫描能显著降低患者的辐射暴露，但代价是图像中会引入明显的噪声和伪影，
影响放射科医生的诊断精度。本项目基于**去噪扩散概率模型（DDPM）**，将低剂量 CT 图像映射为接近标准剂量
（Normal-Dose CT, NDCT）质量的图像，在保留解剖结构和病灶细节的同时抑制噪声。

本方法相对常规 CNN/GAN 去噪的核心创新在于：采用 RRDB 编码器 + 双尺度 Transformer 瓶颈（局部窗口注意力 + 全局扩张卷积）+ CSAS 跨尺度注意力与 CSG 通道-空间门控解码器 的混合架构，
在捕获长程依赖的同时通过残差密集块与边缘/频域/小波感知损失精细保留解剖细节；引入 2.5D 相邻切片堆叠输入 与 PixelUnshuffle 无损下采样，并配合零初始化多尺度残差头，使网络从恒等映射起步，
避免早期细节损失。训练与推理端进一步通过渐进式 patch 调度、warmup-EMA、纯 MSE 精调 + SWA 以及 8 变换 TTA + 高斯加权拼接，显著提升低剂量 CT 去噪的 PSNR/SSIM 与视觉保真度，而无需对抗训练的不稳定性。

**核心特点**
- 🩻 基于 DDPM 的条件扩散去噪框架
- 📉 显著提升 PSNR / SSIM，同时抑制过平滑伪影
- ⚡ 采样加速，DDIM / 少步采样
- 🔧 可配置的噪声调度（noise schedule）与训练超参数

---

## 🖼️ 效果展示

<!--
<img width="1284" height="1236" alt="grid_3top2bot (1)" src="https://github.com/user-attachments/assets/10aa8b2d-2fbf-4d14-b8d9-d8557a64e1eb" />
<img width="4767" height="1229" alt="result_0004" src="https://github.com/user-attachments/assets/cb269047-dfa3-4069-8d1a-2ab70ef26c7f" />


-->

| 低剂量 CT（输入） | 本方法去噪结果 | 标准剂量 CT（Ground Truth） |
|:---:|:---:|:---:|
<img width="1773" height="602" alt="val_ep340_case_000" src="https://github.com/user-attachments/assets/1393a84a-5c72-4ee4-a91a-8cf38d7d3a93" />


<!-- 如有训练过程中的 loss 曲线或采样过程 GIF，也可以放在这里 -->
<img width="4800" height="600" alt="training_curves" src="https://github.com/user-attachments/assets/8eb11e9b-58f4-4aa5-b491-eda8dff88bbb" />


---

## 📊 实验结果

**数据集**：<!-- AAPM 2016 Low-Dose CT Grand Challenge / Mayo Clinic 数据集 -->

**评价指标**：PSNR（峰值信噪比）、SSIM（结构相似性）>

<img width="2328" height="870" alt="image" src="https://github.com/user-attachments/assets/8076bb16-dbf4-4fa8-8106-37aefd4c3064" />


<!-- 可选：加一张柱状图或折线图，直观比较各方法指标 -->

---

## 📁 项目结构



```
LDCT/
├── configs/                # 训练/采样配置文件（.yaml）
│   └── default.yaml
├── data/                   # 数据集加载与预处理脚本
│   └── dataset.py
├── models/                 # 网络结构定义（U-Net / 扩散模型核心）
│   ├── unet.py
│   └── diffusion.py
├── assets/                 # README 用到的图片、GIF
├── train.py                # 训练入口
├── sample.py                # 推理/采样入口（去噪生成）
├── eval.py                 # 计算 PSNR/SSIM 等指标
├── requirements.txt
└── README.md
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
git clone https://github.com/jiangboyu66/LDCT.git
cd LDCT
pip install -r requirements.txt
```

<!-- 
- Python >= 3.8
- torch >= 1.10, torchvision
- numpy, scipy, pydicom（如果读取 DICOM 格式的 CT 数据）
-->

### 2. 数据准备

.npy文件

```
data/
├── train/
│   ├── ldct/     # 低剂量CT切片
│   └── ndct/     # 标准剂量CT切片（作为监督标签）
└── test/
    ├── ldct/
    └── ndct/
```



### 3. 训练

```bash
python train.py --config configs/default.yaml
```

**关键训练参数**（可在 `configs/default.yaml` 中修改）：

| 参数 | 说明 | 默认值 |
|------|------|:------:|
| `timesteps` | 扩散总步数 T | [待补充，如 1000] |
| `beta_schedule` | 噪声调度方式 | [待补充，如 linear / cosine] |
| `batch_size` | 批大小 | [待补充] |
| `lr` | 学习率 | [待补充] |
| `epochs` | 训练轮数 | [待补充] |

### 4. 推理 / 去噪采样

```bash
python sample.py --weights checkpoints/best.pt --input path/to/ldct_image.png --output results/
```

### 5. 评估指标

```bash
python eval.py --pred results/ --gt data/test/ndct/
```

---

## 🧠 方法概述

<!--

“前向”过程：网络直接接收低剂量 CT（LDCT）作为输入，通过端到端回归预测对应的正常剂量 CT（NDCT）。训练时使用归一化后的 LDCT 切片（或 2.5D 相邻切片堆叠）与 NDCT 目标，优化复合损失（Charbonnier + SSIM + 感知 + 频域/小波 + 边缘感知等），不涉及对 NDCT 或残差逐步加噪。
条件机制：LDCT 图像本身即作为网络的输入条件。编码器（RRDB）提取多尺度特征，Transformer 瓶颈与解码器通过跨尺度注意力（CSAS）和通道-空间门控（CSG）融合这些特征，最终由 MultiScaleHead 输出去噪结果（残差连接中心切片）。推理时支持 TTA 与高斯加权拼接，无需反向采样。
网络结构：采用 RRDB 编码器（残差密集块）+ DualScale Transformer 瓶颈（局部窗口注意力 + 全局扩张卷积）+ 带 CSAS/CSG 的解码器 + 多尺度残差头。下采样使用 PixelUnshuffle（信息无损），无时间步嵌入或 U-Net 式扩散时间条件。整体为确定性前馈网络，配合 EMA、SWA 与 fine-tune 阶段纯 MSE 优化。

-->

<img width="1168" height="784" alt="wY14Z" src="https://github.com/user-attachments/assets/cc2b05f3-3949-4357-9259-e2f3bd923432" />


---

## 📌 TODO / 未来计划

将 2.5D 切片堆叠扩展为真正的 3D 体数据建模（引入 3D 卷积/注意力或体积 Transformer），进一步利用完整空间上下文
 在更多公开低剂量 CT 数据集（如 Mayo Clinic、AAPM Grand Challenge、多厂商多剂量数据）上验证跨中心、跨设备泛化能力
 探索模型轻量化与推理加速（知识蒸馏、剪枝、量化或高效注意力变体），以满足临床实时部署需求

---

## 📝 引用

如果本项目对你的研究有帮助，欢迎 star ⭐，也可以按如下方式引用：

```bibtex
@misc{jiang2026ldct,
  title  = {LDCT: Low-Dose CT Denoising with Diffusion Models},
  author = {Jiang, Boyu},
  year   = {2026},
  url    = {https://github.com/jiangboyu66/LDCT}
}
```

<!-- 如果这个项目是基于某篇论文或参考实现开发的，请在此致谢/引用原始论文 -->

---

## 📄 License

本项目基于 [MIT License](LICENSE) 开源。

---

## 🙋 联系方式

如有问题或合作意向，欢迎通过 GitHub Issue 联系，或邮箱：[待补充]
