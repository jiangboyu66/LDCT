"""
    Transformer-Based Attention Framework for
    Noise Reduction and Detail Preservation in Low-Dose CT Imaging — V4-Final (单卡版, 单次训练版)
    ==========================================================================
    修复清单:
      FIX-1  BatchNorm2d → GroupNorm
      FIX-2  patch_inference 中心裁剪拼接
      FIX-3  PerceptualLoss mean/std 动态对齐 device/dtype
      FIX-4  patch_size 切换时重建 DataLoader
      FIX-5  CSAS 显式传入 H, W
      FIX-6  EdgeAwareLoss Sobel kernel 动态对齐 device/dtype
      FIX-7  safe_heads() 确保 num_heads 能整除 dim，修复 bc=88 reshape 错误
      FIX-8  GradScaler / autocast 使用新 API

    新增优化:
      OPT-1  tta_inference() —— 8 种几何变换 TTA，推理时直接涨分，无需重训
      OPT-2  Fine-tune 阶段 (epoch 301–400)：lr=5e-6, CosineAnnealingWarmRestarts,
              更重视 SSIM+Edge 的 CompositeLoss 权重，让模型在已收敛的基础上
              继续精调高频细节

    本版改动:
      - 去掉 4-Fold 交叉验证，改为单次 train/val/test 患者级划分
      - evaluate() 同步输出 MSE + PSNR + SSIM
      - 单次 checkpoint 目录 + 单次最终汇总
      - per-slice CSV 增加 mse 字段，方便后续配对检验

    单卡改动说明 (相对于 DDP 版):
      - 移除 torch.distributed / DDP / DistributedSampler / dist.all_reduce
      - main() 不再接收 rank/world_size，直接使用 cuda:0（或 cpu）
      - DataLoader 改回普通 shuffle=True 模式
      - EMA.update() 不再限制 rank==0，每步都更新
      - 保留全部功能：EMA、TTA、fine-tune、checkpoint 恢复

    患者划分 (10个患者):
      见 TRAIN_PATIENTS / VAL_PATIENTS / TEST_PATIENTS

    ==========================================================================
    V5 补丁清单:
      BUGFIX-A  resume 到 fine-tune 分界点时 optimizer_ft 未初始化导致的 AttributeError
      BUGFIX-B  resume 时不恢复 scheduler state_dict，LR 被打回 warmup
      BUGFIX-C  EMA decay 过大 (0.9999) 导致有效窗口横跨多个 patch-size 阶段，
                改为 warmup 式 decay 且窗口收紧；每次验证同时评测 EMA 和裸模型，取较优者
      FEAT-1    2.5D 输入：in_ch 可配置为奇数(3/5/7)，Dataset 按窗口堆叠相邻切片，
                target 仍只用中心切片；MultiScaleHead 相应只取中心通道做残差连接
      FEAT-2    下采样 MaxPool2d(2) → PixelUnshuffle(2) + 1x1 conv（信息无损）
      FEAT-3    MultiScaleHead 最后一层 zero-init，训练起点即为恒等映射
      FEAT-4    Fine-tune 阶段损失换成几乎纯 MSE
      FEAT-5    推理拼接：中心裁剪硬拼接 → 二维高斯权重加权拼接，消除接缝台阶
      FEAT-6    SWA：fine-tune 阶段最后 N 个 checkpoint 权重做算术平均
      FEAT-7    去掉 drop/attn_drop（默认置 0），weight decay 按参数维度分组
      FEAT-8    SSIMLoss 方差项 clamp(min=0)，避免 fp16 下负方差
    ==========================================================================
"""

import os
import random
import math
import csv
import glob
import re

import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import models
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import mean_squared_error as mse_sk
from copy import deepcopy

# =============================================================================
# 患者划分（单次训练/验证/测试，替代原 4-Fold）
# =============================================================================
ALL_PATIENTS = ['L067', 'L096', 'L109', 'L143', 'L192',
                'L286', 'L291', 'L310', 'L333', 'L506']

# 按需自行调整
TEST_PATIENTS  = ['L333', 'L506']
VAL_PATIENTS   = ['L291', 'L310']
TRAIN_PATIENTS = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286']

# =============================================================================
# Fine-tune 阶段超参（OPT-2）
# =============================================================================
FINETUNE_START = 300
FINETUNE_EPOCHS = 100
FINETUNE_LR = 5e-6
FINETUNE_ETA_MIN = 1e-7
FINETUNE_T0 = 20
FINETUNE_T_MULT = 2

# =============================================================================
# FEAT-1: 2.5D 输入配置
# =============================================================================
IN_CH = 3

# =============================================================================
# FEAT-6: SWA 配置
# =============================================================================
SWA_LAST_N = 6


# =============================================================================
# 工具函数
# =============================================================================
def _infer_hw(L, hint_H=None, hint_W=None):
    if hint_H is not None and hint_W is not None:
        assert hint_H * hint_W == L, f"hint_H*hint_W={hint_H * hint_W} ≠ L={L}"
        return hint_H, hint_W
    H = int(math.isqrt(L))
    while H > 1 and L % H != 0:
        H -= 1
    return H, L // H


def GN(ch):
    groups = min(32, ch)
    while ch % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, ch)


def safe_heads(dim, target_div=64):
    target = max(1, dim // target_div)
    nh = target
    while nh > 1 and dim % nh != 0:
        nh -= 1
    return nh


def smart_imshow(ax, img_hu, title=''):
    v_mean = float(np.mean(img_hu))
    v_std = float(np.std(img_hu))
    if v_mean < -360 or v_mean > 440 or v_std < 1.0:
        vmin = v_mean - 2 * v_std - 1
        vmax = v_mean + 2 * v_std + 1
    else:
        vmin, vmax = -160, 240
    ax.imshow(img_hu, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')


# =============================================================================
# EMA
# =============================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.step = 0
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(d).add_(m.data, alpha=1 - d)

    def eval(self):
        self.shadow.eval()
        return self.shadow

    def state_dict_step(self):
        return {'decay': self.decay, 'step': self.step}

    def load_step(self, d):
        if d is None:
            return
        self.decay = d.get('decay', self.decay)
        self.step = d.get('step', self.step)


# =============================================================================
# FEAT-2: PixelUnshuffle 下采样
# =============================================================================
class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(ch * 4, ch, 1, bias=False),
        )

    def forward(self, x):
        return self.op(x)


# =============================================================================
# RRDB Encoder
# =============================================================================
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth, 3, padding=1, bias=False)
        self.norm = GN(growth)
        self.act = nn.GELU()

    def forward(self, x):
        return torch.cat([x, self.act(self.norm(self.conv(x)))], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth=32):
        super().__init__()
        self.layers = nn.Sequential(
            DenseLayer(in_ch, growth),
            DenseLayer(in_ch + growth, growth),
            DenseLayer(in_ch + growth * 2, growth),
            DenseLayer(in_ch + growth * 3, growth),
        )
        self.proj = nn.Conv2d(in_ch + growth * 4, in_ch, 1, bias=False)
        self.norm = GN(in_ch)

    def forward(self, x):
        return self.norm(self.proj(self.layers(x))) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, in_ch, growth=32):
        super().__init__()
        self.db1 = DenseBlock(in_ch, growth)
        self.db2 = DenseBlock(in_ch, growth)
        self.db3 = DenseBlock(in_ch, growth)

    def forward(self, x):
        return self.db3(self.db2(self.db1(x))) * 0.2 + x


class RRDBEncoder(nn.Module):
    def __init__(self, in_ch=1, bc=88, growth=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, bc, 3, padding=1, bias=False),
            GN(bc), nn.GELU())
        self.enc1 = nn.Sequential(RRDB(bc, growth), RRDB(bc, growth))
        self.down1 = Down(bc)
        self.enc2 = nn.Sequential(
            nn.Conv2d(bc, bc * 2, 1, bias=False), GN(bc * 2), nn.GELU(),
            RRDB(bc * 2, growth), RRDB(bc * 2, growth))
        self.down2 = Down(bc * 2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(bc * 2, bc * 4, 1, bias=False), GN(bc * 4), nn.GELU(),
            RRDB(bc * 4, growth), RRDB(bc * 4, growth))
        self.down3 = Down(bc * 4)
        self.enc4 = nn.Sequential(
            nn.Conv2d(bc * 4, bc * 8, 1, bias=False), GN(bc * 8), nn.GELU(),
            RRDB(bc * 8, growth), RRDB(bc * 8, growth))
        self.down4 = Down(bc * 8)

    def forward(self, x):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        bot = self.down4(e4)
        return [e1, e2, e3, e4], bot


# =============================================================================
# Window helpers
# =============================================================================
def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(wins, ws, H, W):
    nH, nW = H // ws, W // ws
    B = wins.shape[0] // (nH * nW)
    x = wins.view(B, nH, nW, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# =============================================================================
# WindowAttention
# =============================================================================
class WindowAttention(nn.Module):
    def __init__(self, dim, ws, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        while num_heads > 1 and dim % num_heads != 0:
            num_heads -= 1
        self.dim = dim
        self.ws = ws
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.rpb = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), indexing='ij'))
        cf = coords.flatten(1)
        rel = cf[:, :, None] - cf[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer('rpi', rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        rpb = self.rpb[self.rpi.view(-1)].view(N, N, self.num_heads)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# =============================================================================
# SwinBlock
# =============================================================================
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, ws=8, shift=False,
                 mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.ws = ws
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, dim), nn.Dropout(drop))
        self.attn = WindowAttention(dim, ws, num_heads, attn_drop, drop)

    def _build_mask(self, H, W, ws, shift, device):
        if not shift or min(H, W) <= ws:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for hs in h_slices:
            for ws_ in w_slices:
                img_mask[:, hs, ws_, :] = cnt
                cnt += 1
        mw = window_partition(img_mask, ws).view(-1, ws * ws)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.).masked_fill(mask == 0, 0.)
        return mask

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        ws = min(self.ws, H, W)
        shift = ws // 2 if self.shift and min(H, W) > ws else 0

        sc = x
        x = self.norm1(x).view(B, H, W, C)
        if shift > 0:
            x = torch.roll(x, (-shift, -shift), (1, 2))

        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x.permute(0, 3, 1, 2),
                      (0, pad_r, 0, pad_b)).permute(0, 2, 3, 1)
        _, pH, pW, _ = x.shape

        mask = self._build_mask(pH, pW, ws, shift > 0, x.device)
        xw = window_partition(x, ws)
        xw = self.attn(xw.view(-1, ws * ws, C), mask)
        xw = xw.view(-1, ws, ws, C)
        x = window_reverse(xw, ws, pH, pW)

        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()
        if shift > 0:
            x = torch.roll(x, (shift, shift), (1, 2))
        x = x.view(B, H * W, C) + sc
        return x + self.mlp(self.norm2(x))


# =============================================================================
# DualScaleBlock
# =============================================================================
class DualScaleBlock(nn.Module):
    def __init__(self, dim, num_heads, ws=8, shift=False, drop=0., attn_drop=0.):
        super().__init__()
        self.local_attn = SwinBlock(dim, num_heads, ws, shift,
                                    drop=drop, attn_drop=attn_drop)
        dil = 4
        self.glob_dw = nn.Conv2d(dim, dim, 3, padding=dil, dilation=dil,
                                 groups=dim, bias=False)
        self.glob_pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.glob_norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        xl = self.local_attn(x, H, W)
        x2d = x.transpose(1, 2).view(B, C, H, W)
        xg = self.glob_pw(self.glob_dw(x2d)).flatten(2).transpose(1, 2)
        xg = self.glob_norm(xg + x)
        gate = self.gate(torch.cat([xl, xg], dim=-1))
        return self.out_norm(gate * xl + (1 - gate) * xg)


# =============================================================================
# CSAS
# =============================================================================
class CSAS(nn.Module):
    def __init__(self, dim, pool_grid=8):
        super().__init__()
        self.pool_grid = pool_grid
        nh = safe_heads(dim, target_div=64)
        self.nh = nh
        self.scale = (dim // nh) ** -0.5
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, query, kv, H=None, W=None):
        B, Lq, C = query.shape
        H, W = _infer_hw(Lq, H, W)
        g = min(self.pool_grid, H, W)

        kv_2d = kv.transpose(1, 2).view(B, C, H, W)
        kv_pool = F.adaptive_avg_pool2d(kv_2d, (g, g))
        kv_flat = kv_pool.flatten(2).transpose(1, 2)

        nh, D = self.nh, C // self.nh
        q = self.q(self.nq(query)).view(B, Lq, nh, D).transpose(1, 2)
        k = self.k(self.nk(kv_flat)).view(B, g * g, nh, D).transpose(1, 2)
        v = self.v(kv_flat).view(B, g * g, nh, D).transpose(1, 2)

        attn = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Lq, C)
        return query + self.o(out)


# =============================================================================
# CSG
# =============================================================================
class CSG(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        rd = max(dim // r, 4)
        self.ch_fc = nn.Sequential(
            nn.Linear(dim, rd), nn.ReLU(True), nn.Linear(rd, dim), nn.Sigmoid())
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x, H, W):
        B, L, C = x.shape
        ch = self.ch_fc(x.mean(dim=1))
        x = x * ch.unsqueeze(1)
        x2d = x.transpose(1, 2).view(B, C, H, W)
        sp = self.sp_conv(torch.cat(
            [x2d.mean(1, keepdim=True), x2d.max(1, keepdim=True).values], 1))
        return (x2d * sp).flatten(2).transpose(1, 2)


# =============================================================================
# TransformerBottleneck
# =============================================================================
class TransformerBottleneck(nn.Module):
    def __init__(self, dim, train_grid=16, num_heads=8,
                 depth=4, ws=8, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.train_grid = train_grid
        self.pos = nn.Parameter(
            torch.zeros(1, train_grid * train_grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        num_heads = safe_heads(dim, target_div=dim // max(1, num_heads))
        self.blocks = nn.ModuleList([
            DualScaleBlock(dim, num_heads, ws,
                           shift=(i % 2 == 1),
                           drop=drop, attn_drop=attn_drop)
            for i in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def _get_pos(self, H, W):
        G = self.train_grid
        if H == G and W == G:
            return self.pos
        pe = self.pos.reshape(1, G, G, self.dim).permute(0, 3, 1, 2)
        pe = F.interpolate(pe.float(), (H, W), mode='bilinear', align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, H * W, self.dim)

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2) + self._get_pos(H, W)
        for blk in self.blocks:
            t = blk(t, H, W)
        return self.norm(t).transpose(1, 2).view(B, C, H, W)


# =============================================================================
# PatchExpand
# =============================================================================
class PatchExpand(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.expand = nn.Linear(in_dim, in_dim * 4, bias=False)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=False) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, H, W):
        x = self.expand(x)
        B, L, C4 = x.shape
        C = C4 // 4
        x = x.view(B, H, W, 2, 2, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = self.norm(x.view(B, 2 * H * 2 * W, C))
        return self.proj(x), 2 * H, 2 * W


# =============================================================================
# DecoderStage
# =============================================================================
class DecoderStage(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, num_heads,
                 ws=8, depth=2, drop=0., attn_drop=0.):
        super().__init__()
        self.expand = PatchExpand(in_dim, out_dim)
        self.skip_proj = nn.Linear(skip_dim, out_dim, bias=False) \
            if skip_dim != out_dim else nn.Identity()
        self.csas = CSAS(out_dim)
        num_heads = safe_heads(out_dim, target_div=out_dim // max(1, num_heads))
        self.blocks = nn.ModuleList([
            DualScaleBlock(out_dim, num_heads, ws,
                           shift=(i % 2 == 1),
                           drop=drop, attn_drop=attn_drop)
            for i in range(depth)])
        self.csg = CSG(out_dim)

    def forward(self, x, skip, H_in, W_in):
        x, H, W = self.expand(x, H_in, W_in)

        B, C_s, Hs, Ws = skip.shape
        if Hs != H or Ws != W:
            skip = F.interpolate(skip.float(), (H, W),
                                 mode='bilinear', align_corners=False)
        skip_t = self.skip_proj(skip.flatten(2).transpose(1, 2))

        x = self.csas(x, skip_t, H=H, W=W)
        for blk in self.blocks:
            x = blk(x, H, W)
        return self.csg(x, H, W), H, W


# =============================================================================
# MultiScaleHead
# =============================================================================
class MultiScaleHead(nn.Module):
    def __init__(self, dim, in_ch=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim // 2, 1, bias=False), nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1))
        self.detail = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, tokens, H, W, x_in):
        B, L, C = tokens.shape
        feat = self.norm(tokens).transpose(1, 2).view(B, C, H, W)
        main = self.main(feat)

        c = x_in.shape[1]
        x_c = x_in[:, c // 2: c // 2 + 1]

        lp = F.avg_pool2d(x_c, 3, stride=1, padding=1)
        detail = self.detail(x_c - lp)
        return (x_c + main + self.alpha.clamp(0, 1) * detail).clamp(-1., 1.)


# =============================================================================
# Full Model
# =============================================================================
class LDCTDenoiserV4(nn.Module):
    def __init__(self, in_ch=1, bc=88, growth=32,
                 bot_depth=4, bot_heads=8, ws=8,
                 dec_depths=(2, 2, 2, 2),
                 drop=0., attn_drop=0.):
        super().__init__()
        self.encoder = RRDBEncoder(in_ch, bc, growth)
        self.bottleneck = TransformerBottleneck(
            dim=bc * 8, train_grid=16, num_heads=bot_heads,
            depth=bot_depth, ws=ws, drop=drop, attn_drop=attn_drop)

        self.dec4 = DecoderStage(bc * 8, bc * 8, bc * 4, safe_heads(bc * 4),
                                 ws, dec_depths[0], drop, attn_drop)
        self.dec3 = DecoderStage(bc * 4, bc * 4, bc * 2, safe_heads(bc * 2),
                                 ws, dec_depths[1], drop, attn_drop)
        self.dec2 = DecoderStage(bc * 2, bc * 2, bc, safe_heads(bc),
                                 ws, dec_depths[2], drop, attn_drop)
        self.dec1 = DecoderStage(bc, bc, bc, safe_heads(bc),
                                 ws, dec_depths[3], drop, attn_drop)
        self.head = MultiScaleHead(bc, in_ch=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)

        nn.init.zeros_(self.head.main[-1].weight)
        nn.init.zeros_(self.head.main[-1].bias)
        nn.init.zeros_(self.head.detail[-1].weight)
        nn.init.zeros_(self.head.detail[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        skips, bot = self.encoder(x)
        bot = self.bottleneck(bot)
        bH, bW = H // 16, W // 16
        t = bot.flatten(2).transpose(1, 2)

        t, h, w = self.dec4(t, skips[3], bH, bW)
        t, h, w = self.dec3(t, skips[2], h, w)
        t, h, w = self.dec2(t, skips[1], h, w)
        t, h, w = self.dec1(t, skips[0], h, w)
        return self.head(t, h, w, x)


# =============================================================================
# Loss Functions
# =============================================================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, data_range=2.0, levels=3):
        super().__init__()
        self.dr = data_range
        self.levels = levels
        self.ws = window_size
        g = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(g ** 2) / (2 * 1.5 ** 2))
        g /= g.sum()
        self.register_buffer('win', g.outer(g).unsqueeze(0).unsqueeze(0))

    def _ssim(self, x, y):
        C1, C2 = (0.01 * self.dr) ** 2, (0.03 * self.dr) ** 2
        pad = self.ws // 2
        w = self.win.to(x.device, x.dtype)
        mx = F.conv2d(x, w, padding=pad)
        my = F.conv2d(y, w, padding=pad)
        mxx = (F.conv2d(x * x, w, padding=pad) - mx ** 2).clamp(min=0)
        myy = (F.conv2d(y * y, w, padding=pad) - my ** 2).clamp(min=0)
        mxy = F.conv2d(x * y, w, padding=pad) - mx * my
        return ((2 * mx * my + C1) * (2 * mxy + C2) /
                ((mx ** 2 + my ** 2 + C1) * (mxx + myy + C2))).mean()

    def forward(self, x, y):
        loss = 0.
        for i in range(self.levels):
            loss += 1. - self._ssim(x, y)
            if i < self.levels - 1:
                x = F.avg_pool2d(x, 2)
                y = F.avg_pool2d(y, 2)
        return loss / self.levels


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        fp = torch.fft.rfft2(pred.float(), norm='ortho')
        ft = torch.fft.rfft2(target.float(), norm='ortho')
        return F.l1_loss(fp.abs(), ft.abs())


class HaarWaveletLoss(nn.Module):
    @staticmethod
    def _dwt(x):
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 1::2, 0::2]
        c = x[:, :, 0::2, 1::2]
        d = x[:, :, 1::2, 1::2]
        return ((a + b + c + d) * 0.25, (a - b + c - d) * 0.25,
                (a + b - c - d) * 0.25, (a - b - c + d) * 0.25)

    def forward(self, pred, target):
        _, lhp, hlp, hhp = self._dwt(pred)
        _, lht, hlt, hht = self._dwt(target)
        return (F.l1_loss(lhp, lht) + F.l1_loss(hlp, hlt) +
                0.5 * F.l1_loss(hhp, hht)) / 2.5


class NoiseAwareLoss(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.k = k

    def forward(self, pred, target, ldct):
        k, p = self.k, self.k // 2
        mu = F.avg_pool2d(ldct, k, stride=1, padding=p)
        var = (F.avg_pool2d(ldct ** 2, k, stride=1, padding=p) - mu ** 2).clamp(0)
        w = (var / (var.mean() + 1e-6)).clamp(0.5, 3.0)
        return (F.l1_loss(pred, target, reduction='none') * w).mean()


class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def forward(self, p, t):
        sx = self.sx.to(p.device, p.dtype)
        sy = self.sy.to(p.device, p.dtype)
        ep = torch.sqrt(F.conv2d(p, sx, padding=1) ** 2 +
                        F.conv2d(p, sy, padding=1) ** 2 + 1e-6)
        et = torch.sqrt(F.conv2d(t, sx, padding=1) ** 2 +
                        F.conv2d(t, sy, padding=1) ** 2 + 1e-6)
        return F.l1_loss(ep, et)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
        self.feats, self.hooks = {}, []
        for name in ('layer1', 'layer2', 'layer3'):
            h = dict(resnet.named_modules())[name].register_forward_hook(
                lambda m, i, o, n=name: self.feats.update({n: o}))
            self.hooks.append(h)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = resnet.to(dev)
        for p in self.resnet.parameters():
            p.requires_grad_(False)
        self.layers = ('layer1', 'layer2', 'layer3')
        self.register_buffer('mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        def prep(t):
            t3 = t.float().repeat(1, 3, 1, 1)
            mean = self.mean.to(t3.device)
            std = self.std.to(t3.device)
            return ((t3 + 1) / 2 - mean) / std

        self.feats.clear()
        self.resnet(prep(x))
        xf = self.feats.copy()
        self.feats.clear()
        self.resnet(prep(y))
        yf = dict(self.feats)
        return sum(F.mse_loss(xf[l], yf[l]) for l in self.layers) / 3

    def __del__(self):
        for h in self.hooks:
            try:
                h.remove()
            except:
                pass


class CompositeLoss(nn.Module):
    def __init__(self, lc=1.0, ls=1.5, lp=0.05, lf=0.15, lw=0.4, ln=0.3, le=0.3,
                 lmse=0.0):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ssim = SSIMLoss(data_range=2.0, levels=3)
        self.perc = PerceptualLoss()
        self.freq = FrequencyLoss()
        self.wav = HaarWaveletLoss()
        self.noise = NoiseAwareLoss()
        self.edge = EdgeAwareLoss()
        self.lc, self.ls, self.lp = lc, ls, lp
        self.lf, self.lw, self.ln = lf, lw, ln
        self.le = le
        self.lmse = lmse

    def forward(self, pred, target, ldct=None):
        lc = self.charb(pred, target)
        ls = self.ssim(pred, target)
        lp = self.perc(pred, target) if self.lp > 0 else torch.zeros(1, device=pred.device)
        lf = self.freq(pred, target) if self.lf > 0 else torch.zeros(1, device=pred.device)
        lw = self.wav(pred, target) if self.lw > 0 else torch.zeros(1, device=pred.device)
        le = self.edge(pred, target) if self.le > 0 else torch.zeros(1, device=pred.device)
        ln = self.noise(pred, target, ldct) if (ldct is not None and self.ln > 0) \
            else torch.zeros(1, device=pred.device)
        lmse = F.mse_loss(pred, target) if self.lmse > 0 else torch.zeros(1, device=pred.device)

        total = (self.lc * lc + self.ls * ls + self.lp * lp +
                 self.lf * lf + self.lw * lw + self.ln * ln + self.le * le +
                 self.lmse * lmse)
        subs = dict(charb=lc.item(), ssim=ls.item(),
                    perc=lp.item() if torch.is_tensor(lp) else lp,
                    freq=lf.item() if torch.is_tensor(lf) else lf,
                    wav=lw.item() if torch.is_tensor(lw) else lw,
                    edge=le.item() if torch.is_tensor(le) else le,
                    noise=ln.item() if torch.is_tensor(ln) else ln,
                    mse=lmse.item() if torch.is_tensor(lmse) else lmse)
        return total, subs


# =============================================================================
# Dataset (FEAT-1: 2.5D)
# =============================================================================
class LDCTDataset(Dataset):
    def __init__(self, ldct_root, ndct_root, patients, mode='train',
                 patch_size=128, in_ch=1):
        assert in_ch % 2 == 1, "in_ch (2.5D 窗口) 必须是奇数"
        self.in_ch = in_ch
        self.half = in_ch // 2

        self.samples = []
        avail = []
        for p in patients:
            ld = os.path.join(ldct_root, p)
            nd = os.path.join(ndct_root, p)
            if not (os.path.isdir(ld) and os.path.isdir(nd)):
                print(f"  [警告] 患者 {p} 目录不存在，已跳过")
                continue
            avail.append(p)
            lf = sorted(f for f in os.listdir(ld) if f.endswith('.npy'))
            nf = sorted(f for f in os.listdir(nd) if f.endswith('.npy'))
            n = min(len(lf), len(nf))
            ld_paths = [os.path.join(ld, lf[i]) for i in range(n)]
            nd_paths = [os.path.join(nd, nf[i]) for i in range(n)]

            for i in range(n):
                idxs = [min(max(i + off, 0), n - 1)
                        for off in range(-self.half, self.half + 1)]
                window_ld_paths = [ld_paths[j] for j in idxs]
                self.samples.append((window_ld_paths, nd_paths[i]))

        self.patch_size = patch_size
        self.is_train = (mode == 'train')
        print(f"[{mode.upper()}] 患者={avail}  共 {len(self.samples)} 个切片对  "
              f"(2.5D in_ch={in_ch})")

    def set_patch_size(self, ps):
        self.patch_size = ps

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_norm(path):
        arr = np.clip(np.load(path).astype(np.float32), -1000, 1500)
        arr = (arr + 1000) / 2500 * 2 - 1
        return arr

    def __getitem__(self, idx):
        ld_paths, nd_path = self.samples[idx]

        ld_slices = [self._load_norm(p) for p in ld_paths]
        ld = np.stack(ld_slices, axis=0)          # (in_ch, H, W)
        nd = self._load_norm(nd_path)[None]        # (1, H, W)

        ld = torch.from_numpy(ld)
        nd = torch.from_numpy(nd)

        if self.is_train and self.patch_size > 0:
            _, h, w = ld.shape
            ps = (min(self.patch_size, h, w) // 16) * 16
            ps = max(ps, 64)
            i = random.randint(0, h - ps)
            j = random.randint(0, w - ps)
            ld = TF.crop(ld, i, j, ps, ps)
            nd = TF.crop(nd, i, j, ps, ps)
            if random.random() > 0.5: ld, nd = TF.hflip(ld), TF.hflip(nd)
            if random.random() > 0.5: ld, nd = TF.vflip(ld), TF.vflip(nd)
            k = random.randint(0, 3)
            if k: ld, nd = torch.rot90(ld, k, [1, 2]), torch.rot90(nd, k, [1, 2])
            if random.random() > 0.7:
                f = random.uniform(0.95, 1.05)
                ld = (ld * f).clamp(-1, 1)
                nd = (nd * f).clamp(-1, 1)
        return ld, nd


# =============================================================================
# DataLoader
# =============================================================================
def get_batch_size(patch_size):
    if patch_size <= 128: return 72
    if patch_size <= 192: return 32
    if patch_size <= 256: return 20
    return 12


def make_loader(dataset, batch_size, num_workers, shuffle=True):
    if shuffle and len(dataset) == 0:
        raise RuntimeError(
            "训练集为空！请检查 LDCT_ROOT / NDCT_ROOT 路径及 .npy 文件是否存在。")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


# =============================================================================
# OPT-1: TTA Inference
# =============================================================================
@torch.no_grad()
def tta_inference(model, img, tile=256, overlap=64, device='cuda'):
    def fwd0(x): return x
    def inv0(x): return x
    def fwd1(x): return torch.flip(x, [-1])
    def inv1(x): return torch.flip(x, [-1])
    def fwd2(x): return torch.flip(x, [-2])
    def inv2(x): return torch.flip(x, [-2])
    def fwd3(x): return torch.rot90(x, 1, [-2, -1])
    def inv3(x): return torch.rot90(x, -1, [-2, -1])
    def fwd4(x): return torch.rot90(x, 2, [-2, -1])
    def inv4(x): return torch.rot90(x, -2, [-2, -1])
    def fwd5(x): return torch.rot90(x, 3, [-2, -1])
    def inv5(x): return torch.rot90(x, -3, [-2, -1])
    def fwd6(x): return torch.flip(torch.rot90(x, 1, [-2, -1]), [-1])
    def inv6(x): return torch.rot90(torch.flip(x, [-1]), -1, [-2, -1])
    def fwd7(x): return torch.flip(torch.rot90(x, 1, [-2, -1]), [-2])
    def inv7(x): return torch.rot90(torch.flip(x, [-2]), -1, [-2, -1])

    tta_transforms = [
        (fwd0, inv0), (fwd1, inv1), (fwd2, inv2), (fwd3, inv3),
        (fwd4, inv4), (fwd5, inv5), (fwd6, inv6), (fwd7, inv7),
    ]

    results = []
    for fwd, inv in tta_transforms:
        out = patch_inference(model, fwd(img), tile, overlap, device)
        results.append(inv(out))

    return torch.stack(results, dim=0).mean(dim=0).clamp(-1., 1.)


# =============================================================================
# FEAT-5: 二维高斯权重拼接
# =============================================================================
_GAUSSIAN_WEIGHT_CACHE = {}


def _gaussian_tile_weight(tile, device, dtype=torch.float32, sigma_scale=0.125):
    key = (tile, sigma_scale)
    if key not in _GAUSSIAN_WEIGHT_CACHE:
        coords = torch.arange(tile, dtype=torch.float32) - (tile - 1) / 2
        sigma = tile * sigma_scale
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        w2d = g1d.outer(g1d)
        w2d = w2d / w2d.max().clamp(min=1e-8)
        w2d = w2d.clamp(min=1e-3)
        _GAUSSIAN_WEIGHT_CACHE[key] = w2d
    return _GAUSSIAN_WEIGHT_CACHE[key].to(device=device, dtype=dtype)


# =============================================================================
# Patch Inference
# =============================================================================
@torch.no_grad()
def patch_inference(model, img, tile=256, overlap=64, device='cuda',
                     blend='gaussian'):
    _, C, H, W = img.shape
    tile = (tile // 16) * 16
    margin = overlap // 2
    step = tile - overlap

    pad_h = (tile - H % tile) % tile if H % tile != 0 else 0
    pad_w = (tile - W % tile) % tile if W % tile != 0 else 0
    img_p = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    oH, oW = img_p.shape[2], img_p.shape[3]

    out_full = torch.zeros(1, 1, oH, oW)
    cnt_full = torch.zeros(1, 1, oH, oW)

    ys = list(range(0, max(oH - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(oW - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < oH: ys.append(oH - tile)
    if xs[-1] + tile < oW: xs.append(oW - tile)

    if blend == 'gaussian':
        w2d = _gaussian_tile_weight(tile, device='cpu').view(1, 1, tile, tile)

    for y in ys:
        for x in xs:
            patch = img_p[:, :, y:y + tile, x:x + tile].to(device)
            pred = model(patch).cpu()

            if blend == 'gaussian':
                out_full[:, :, y:y + tile, x:x + tile] += pred * w2d
                cnt_full[:, :, y:y + tile, x:x + tile] += w2d
            else:
                py1 = margin if y > 0 else 0
                py2 = tile - margin if y + tile < oH else tile
                px1 = margin if x > 0 else 0
                px2 = tile - margin if x + tile < oW else tile

                cy1, cy2 = y + py1, y + py2
                cx1, cx2 = x + px1, x + px2

                out_full[:, :, cy1:cy2, cx1:cx2] += pred[:, :, py1:py2, px1:px2]
                cnt_full[:, :, cy1:cy2, cx1:cx2] += 1.0

    out_full = out_full / cnt_full.clamp(min=1e-6)

    mask = (cnt_full < 1e-6)
    if mask.any():
        fallback = model(img_p.to(device)).cpu()
        out_full[mask] = fallback[mask]

    return out_full[:, :, :H, :W].clamp(-1., 1.)


# =============================================================================
# Metric helpers
# =============================================================================
def to_hu(t):
    return ((t.cpu().float().squeeze().numpy() + 1) / 2 * 2500 - 1000)


def compute_metrics(pred_hu, target_hu):
    ps = psnr_sk(target_hu, pred_hu, data_range=2500)
    ss = ssim_sk(target_hu, pred_hu, data_range=2500)
    ms = mse_sk(target_hu, pred_hu)
    return ps, ss, ms


# =============================================================================
# Progressive patch schedule
# =============================================================================
def get_patch_size(epoch):
    if epoch <= 30:   return 128
    if epoch <= 80:   return 192
    if epoch <= 150:  return 256
    return 320


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, save_dir,
             tile=256, overlap=64, tag='val', use_tta=False,
             method_name='HCT-UNet'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    psnrs, ssims, mses = [], [], []
    per_slice_rows = []

    infer_fn = tta_inference if use_tta else patch_inference
    if use_tta:
        print(f"  [evaluate] TTA 已启用（8 种变换）")

    for i, (ldct, ndct) in enumerate(loader):
        pred = infer_fn(model, ldct, tile, overlap, device)
        ph = to_hu(pred[0, 0])
        gh = to_hu(ndct[0, 0])
        lc = ldct.shape[1] // 2
        lh = to_hu(ldct[0, lc])
        ps, ss, ms = compute_metrics(ph, gh)
        psnrs.append(ps)
        ssims.append(ss)
        mses.append(ms)

        per_slice_rows.append({
            'slice_index': i,
            'method': method_name,
            'psnr': ps,
            'ssim': ss,
            'mse': ms,
        })

        if i % 500 == 0:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            smart_imshow(ax[0], lh, "LDCT")
            smart_imshow(ax[1], ph, f"Denoised\nPSNR:{ps:.2f} | SSIM:{ss:.4f} | MSE:{ms:.1f}")
            smart_imshow(ax[2], gh, "NDCT")
            plt.savefig(os.path.join(save_dir, f"{tag}_case_{i:03d}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [{tag}] {i}/{len(loader.dataset)}  "
                  f"PSNR:{ps:.2f}  SSIM:{ss:.4f}  MSE:{ms:.1f}")

    avg_p = float(np.mean(psnrs)); std_p = float(np.std(psnrs))
    avg_s = float(np.mean(ssims)); std_s = float(np.std(ssims))
    avg_m = float(np.mean(mses));  std_m = float(np.std(mses))
    tta_tag = "+TTA" if use_tta else ""
    print(f"\n{'=' * 60}")
    print(f"[{tag.upper()}]{tta_tag}  N={len(psnrs)}")
    print(f"  PSNR : {avg_p:.2f} ± {std_p:.2f} dB")
    print(f"  SSIM : {avg_s:.4f} ± {std_s:.4f}")
    print(f"  MSE  : {avg_m:.2f} ± {std_m:.2f}")
    print(f"{'=' * 60}\n")

    with open(os.path.join(save_dir, f"{tag}_results.txt"),
              'w', encoding='utf-8') as f:
        f.write(f"=== V4-Final (单卡, 单次训练) {tag.upper()}{tta_tag} ===\n\n")
        f.write(f"PSNR : {avg_p:.2f} ± {std_p:.2f} dB\n")
        f.write(f"SSIM : {avg_s:.4f} ± {std_s:.4f}\n")
        f.write(f"MSE  : {avg_m:.2f} ± {std_m:.2f}\n")
        f.write(f"N    : {len(psnrs)}\n")
        f.write(f"torch: {torch.__version__} | device: {device}\n")

    csv_path = os.path.join(save_dir, f"{tag}_{method_name}_per_slice.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['slice_index', 'method', 'psnr', 'ssim', 'mse'])
        writer.writeheader()
        writer.writerows(per_slice_rows)
    print(f"  [per-slice] saved {len(per_slice_rows)} rows → {csv_path}")

    return avg_p, avg_s, avg_m


def plot_history(history, save_dir):
    keys = [k for k in history if k != 'train_loss']
    n = len(keys) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    ep = range(1, len(history['train_loss']) + 1)
    axes[0].plot(ep, history['train_loss'], 'b-o', ms=3, lw=1.5)
    axes[0].set_title('Total loss')
    axes[0].grid(True)
    for ax, k in zip(axes[1:], keys):
        ax.plot(ep, history[k], ms=3, lw=1.5)
        ax.set_title(k)
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


# =============================================================================
# Fine-tune 组件
# =============================================================================
def build_param_groups(model, weight_decay=5e-5):
    decay_p = [p for n, p in model.named_parameters()
               if p.requires_grad and p.ndim > 1
               and 'rpb' not in n and 'pos' not in n]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and (p.ndim <= 1 or 'rpb' in n or 'pos' in n)]
    return [
        {'params': decay_p, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.},
    ]


def build_finetune_components(model, device):
    optimizer_ft = optim.AdamW(
        build_param_groups(model, weight_decay=0.),
        lr=FINETUNE_LR,
        betas=(0.9, 0.999),
    )
    scheduler_ft = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_ft,
        T_0=FINETUNE_T0,
        T_mult=FINETUNE_T_MULT,
        eta_min=FINETUNE_ETA_MIN,
    )
    criterion_ft = CompositeLoss(
        lc=0.05, ls=0.0, lp=0.0,
        lf=0.0, lw=0.0, ln=0.0, le=0.0,
        lmse=1.0,
    ).to(device)
    return optimizer_ft, scheduler_ft, criterion_ft


# =============================================================================
# FEAT-6: SWA
# =============================================================================
def swa_average_checkpoints(ckpt_paths, device, key='model'):
    assert len(ckpt_paths) > 0
    avg_state = None
    n = len(ckpt_paths)
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=device)
        state = {k.replace('module.', ''): v for k, v in ckpt[key].items()}
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()
    for k in avg_state:
        avg_state[k] /= n
    return avg_state


# =============================================================================
# 单次训练函数
# =============================================================================
def train_one_run(train_patients, val_patients, test_patients,
                  device, BASE_DIR, LDCT_ROOT, NDCT_ROOT, NUM_WORKERS,
                  in_ch=IN_CH):
    print(f"\n{'#' * 70}")
    print(f"#  Training / Validation / Test")
    print(f"{'#' * 70}")
    print(f"  Train: {train_patients}")
    print(f"  Val  : {val_patients}")
    print(f"  Test : {test_patients}\n")

    SAVE_DIR = f"{BASE_DIR}/output/checkpoints/v1011s"
    os.makedirs(SAVE_DIR, exist_ok=True)

    NUM_EPOCHS = FINETUNE_START + FINETUNE_EPOCHS

    train_ds = LDCTDataset(LDCT_ROOT, NDCT_ROOT, train_patients,
                           mode='train', patch_size=128, in_ch=in_ch)
    val_ds   = LDCTDataset(LDCT_ROOT, NDCT_ROOT, val_patients,
                           mode='val', patch_size=0, in_ch=in_ch)
    test_ds  = LDCTDataset(LDCT_ROOT, NDCT_ROOT, test_patients,
                           mode='test', patch_size=0, in_ch=in_ch)

    cur_ps = get_patch_size(1)
    cur_batch = get_batch_size(cur_ps)
    train_loader = make_loader(train_ds, cur_batch, NUM_WORKERS, shuffle=True)
    val_loader   = make_loader(val_ds, batch_size=1, num_workers=0, shuffle=False)
    test_loader  = make_loader(test_ds, batch_size=1, num_workers=0, shuffle=False)

    print(f"[初始] patch={cur_ps}  batch={cur_batch}  in_ch(2.5D)={in_ch}")

    model = LDCTDenoiserV4(
        in_ch=in_ch, bc=88, growth=32,
        bot_depth=4, bot_heads=8, ws=8,
        dec_depths=(2, 2, 2, 2),
        drop=0., attn_drop=0.,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    ema = EMA(model, decay=0.999)

    WARMUP = 10
    optimizer = optim.AdamW(build_param_groups(model, weight_decay=5e-5),
                            lr=1e-4, betas=(0.9, 0.999))

    def lr_lambda(ep):
        if ep < WARMUP:
            return (ep + 1) / WARMUP
        if 80 <= ep < 85:
            return 0.05 + 0.25 * (ep - 80) / 5
        if 150 <= ep < 155:
            return 0.02 + 0.08 * (ep - 150) / 5
        stages = [
            (WARMUP, 30, 1.00),
            (30, 80, 0.60),
            (85, 150, 0.30),
            (155, FINETUNE_START, 0.10),
        ]
        for s_start, s_end, peak in stages:
            if s_start <= ep < s_end:
                t = (ep - s_start) / max(1, s_end - s_start)
                return peak * 0.5 * (1 + math.cos(math.pi * t))
        return 0.02

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = CompositeLoss(lc=1.0, ls=0.8, lp=0.05,
                              lf=0.1, lw=0.2, ln=0.4, le=0.2).to(device)
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss': []}
    best_psnr = 0.
    best_psnr_raw = 0.
    start_epoch = 1
    recent_ckpts = []
    recent_best_ckpts = []
    best_ever_path = None
    finetune_ckpt_paths = []

    optimizer_ft = None
    scheduler_ft = None
    criterion_ft = None
    in_finetune = False

    def find_best_ckpt(save_dir):
        best_files = glob.glob(os.path.join(save_dir, 'best_P*.pth'))
        ckpt_files = glob.glob(os.path.join(save_dir, 'ckpt_ep*.pth'))

        def extract_psnr(path):
            m = re.search(r'_P(\d+(?:\.\d+)?)', os.path.basename(path))
            if not m:
                return 0.0
            try:
                return float(m.group(1))
            except ValueError:
                return 0.0

        candidates = best_files if best_files else ckpt_files
        if not candidates:
            return None, 0.0
        best_path = max(candidates, key=extract_psnr)
        return best_path, extract_psnr(best_path)

    def rebuild_recent_best_ckpts(save_dir):
        files = glob.glob(os.path.join(save_dir, 'best_P*.pth'))

        def extract_epoch(path):
            m = re.search(r'_ep(\d+)', os.path.basename(path))
            return int(m.group(1)) if m else 0

        return sorted(files, key=extract_epoch)

    ckpt_path, ckpt_psnr = find_best_ckpt(SAVE_DIR)
    if ckpt_path is not None:
        print(f"\n[Resume] {os.path.basename(ckpt_path)}  PSNR={ckpt_psnr:.2f}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(state)
        if 'ema' in ckpt:
            ema.shadow.load_state_dict(ckpt['ema'])
        if 'ema_step' in ckpt:
            ema.load_step(ckpt['ema_step'])
        history = ckpt.get('history', history)
        best_psnr = ckpt.get('best_psnr', ckpt_psnr)
        best_psnr_raw = ckpt.get('best_psnr_raw', best_psnr_raw)
        start_epoch = ckpt['epoch'] + 1
        in_finetune = ckpt.get('in_finetune', False)

        if in_finetune:
            optimizer_ft, scheduler_ft, criterion_ft = \
                build_finetune_components(model, device)
            if 'opt' in ckpt:
                optimizer_ft.load_state_dict(ckpt['opt'])
            if 'sched' in ckpt:
                scheduler_ft.load_state_dict(ckpt['sched'])
            scaler = torch.amp.GradScaler('cuda')
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
        else:
            if 'opt' in ckpt:
                optimizer.load_state_dict(ckpt['opt'])
            if 'sched' in ckpt:
                scheduler.load_state_dict(ckpt['sched'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])

        recent_best_ckpts = rebuild_recent_best_ckpts(SAVE_DIR)
        if recent_best_ckpts:
            def _psnr_of(p):
                m = re.search(r'best_P(\d+(?:\.\d+)?)_', os.path.basename(p))
                if not m:
                    return -1.0
                try:
                    return float(m.group(1))
                except ValueError:
                    return -1.0
            best_ever_path = max(recent_best_ckpts, key=_psnr_of)

        if in_finetune:
            ft_ckpts = glob.glob(os.path.join(SAVE_DIR, 'ckpt_ep*.pth'))
            def _ep_of(p):
                m = re.search(r'_ep(\d+)', os.path.basename(p))
                return int(m.group(1)) if m else -1
            finetune_ckpt_paths = sorted(
                [p for p in ft_ckpts if _ep_of(p) > FINETUNE_START],
                key=_ep_of)

        print(f"[Resume] 从 epoch {ckpt['epoch']} 恢复，将从 epoch {start_epoch} 继续  "
              f"(in_finetune={in_finetune})")
        print(f"[Resume] 已找到 {len(recent_best_ckpts)} 个历史 best_P*.pth 文件\n")
    else:
        print("\n[Resume] 未找到 checkpoint，从头训练\n")

    while len(recent_best_ckpts) > 3:
        oldest_best = recent_best_ckpts.pop(0)
        if oldest_best == best_ever_path:
            recent_best_ckpts.append(oldest_best)
            break
        if os.path.exists(oldest_best):
            os.remove(oldest_best)
            print(f"  [清理best/resume] {os.path.basename(oldest_best)}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        if epoch > FINETUNE_START and optimizer_ft is None:
            in_finetune = True
            optimizer_ft, scheduler_ft, criterion_ft = \
                build_finetune_components(model, device)
            scaler = torch.amp.GradScaler('cuda')
            print(f"\n{'=' * 60}")
            print(f"[OPT-2] 切换至 Fine-tune 阶段")
            print(f"  optimizer : AdamW  lr={FINETUNE_LR}  weight_decay=0 (分组)")
            print(f"  scheduler : CosineAnnealingWarmRestarts "
                  f"T_0={FINETUNE_T0}  T_mult={FINETUNE_T_MULT}")
            print(f"  criterion : lc=0.05 lmse=1.0 (纯 MSE 主导, FEAT-4)")
            print(f"{'=' * 60}\n")

        cur_optimizer = optimizer_ft if in_finetune else optimizer
        cur_scheduler = scheduler_ft if in_finetune else scheduler
        cur_criterion = criterion_ft if in_finetune else criterion

        if in_finetune:
            target_ps = 256
            target_batch = get_batch_size(target_ps)
        else:
            target_ps = get_patch_size(epoch)
            target_batch = get_batch_size(target_ps)

        if target_ps != cur_ps or target_batch != cur_batch:
            train_ds.set_patch_size(target_ps)
            train_loader = make_loader(train_ds, target_batch, NUM_WORKERS, shuffle=True)
            print(f"[Epoch {epoch}] patch {cur_ps}→{target_ps}  "
                  f"batch {cur_batch}→{target_batch}")
            cur_ps, cur_batch = target_ps, target_batch

        model.train()
        total, subs_acc = 0., {}

        for bi, (ldct, ndct) in enumerate(train_loader):
            ldct, ndct = ldct.to(device), ndct.to(device)
            cur_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred = model(ldct)
                loss, subs = cur_criterion(pred, ndct, ldct)

            scaler.scale(loss).backward()
            scaler.unscale_(cur_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(cur_optimizer)
            scaler.update()
            ema.update(model)

            total += loss.item()
            for k, v in subs.items():
                subs_acc[k] = subs_acc.get(k, 0.) + v

            if (bi + 1) % 500 == 0:
                model.eval()
                with torch.no_grad():
                    pred_vis = model(ldct[[0]])
                model.train()
                ph = to_hu(pred_vis[0, 0])
                gh = to_hu(ndct[0, 0])
                lc_ = ldct.shape[1] // 2
                lh = to_hu(ldct[0, lc_])
                ps, ss, ms = compute_metrics(ph, gh)
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                smart_imshow(ax[0], lh, "LDCT")
                smart_imshow(ax[1], ph,
                             f"Denoised\nPSNR:{ps:.2f} | SSIM:{ss:.4f} | MSE:{ms:.1f}")
                smart_imshow(ax[2], gh, "NDCT")
                ft_tag = "_ft" if in_finetune else ""
                plt.savefig(os.path.join(SAVE_DIR,
                                         f"ep{epoch:03d}{ft_tag}_b{bi + 1:05d}_P{ps:.2f}_S{ss:.4f}.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  [snap] ep{epoch}{ft_tag} b{bi + 1}  "
                      f"PSNR:{ps:.2f}  SSIM:{ss:.4f}  MSE:{ms:.1f}")

            if bi % 200 == 0:
                mem = torch.cuda.memory_reserved(device) / 1e9 \
                    if device.type == 'cuda' else 0
                sub_str = "  ".join(f"{k}:{v:.4f}" for k, v in subs.items())
                print(f"  ep{epoch} [{bi}/{len(train_loader)}]  "
                      f"loss:{loss.item():.4f}  {sub_str}  VRAM:{mem:.1f}GB")

        avg_loss = total / len(train_loader)
        history['train_loss'].append(avg_loss)
        for k in subs_acc:
            history.setdefault(k, []).append(subs_acc[k] / len(train_loader))
        print(f"{'=' * 60}\nEpoch {epoch}  avg_loss:{avg_loss:.4f}\n{'=' * 60}\n")

        cur_scheduler.step()

        if epoch % 10 == 0:
            ema.eval()
            use_tta_eval = in_finetune

            avg_p, avg_s, avg_m = evaluate(
                ema.shadow, val_loader, device,
                SAVE_DIR, tile=256, overlap=64,
                tag=f'val_ep{epoch}_ema',
                use_tta=use_tta_eval,
                method_name='HCT-UNet_ema')

            model.eval()
            avg_p_raw, avg_s_raw, avg_m_raw = evaluate(
                model, val_loader, device,
                SAVE_DIR, tile=256, overlap=64,
                tag=f'val_ep{epoch}_raw',
                use_tta=use_tta_eval,
                method_name='HCT-UNet_raw')
            model.train()

            use_ema_for_ckpt = avg_p >= avg_p_raw
            print(f"  [ep{epoch}] EMA PSNR={avg_p:.2f}  raw PSNR={avg_p_raw:.2f}  "
                  f"→ 使用 {'EMA' if use_ema_for_ckpt else 'raw'} 作为本次候选")

            best_psnr_raw = max(best_psnr_raw, avg_p_raw)

            save_data = {
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_psnr_raw': best_psnr_raw,
                'model': model.state_dict(),
                'ema': ema.shadow.state_dict(),
                'ema_step': ema.state_dict_step(),
                'opt': cur_optimizer.state_dict(),
                'sched': cur_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'history': history,
                'in_finetune': in_finetune,
            }
            ckpt_save_path = os.path.join(
                SAVE_DIR, f'ckpt_ep{epoch:03d}_P{avg_p:.2f}.pth')
            torch.save(save_data, ckpt_save_path)
            recent_ckpts.append(ckpt_save_path)

            if in_finetune:
                finetune_ckpt_paths.append(ckpt_save_path)

            while len(recent_ckpts) > 2:
                oldest = recent_ckpts.pop(0)
                if os.path.exists(oldest) and 'best' not in oldest \
                        and oldest not in finetune_ckpt_paths:
                    os.remove(oldest)
                    print(f"  [清理] {os.path.basename(oldest)}")

            best_candidate = max(avg_p, avg_p_raw)
            if best_candidate > best_psnr:
                best_psnr = best_candidate
                best_save_path = os.path.join(
                    SAVE_DIR, f'best_P{best_candidate:.2f}_ep{epoch:03d}.pth')
                save_data['best_psnr'] = best_psnr
                torch.save(save_data, best_save_path)
                print(f"  [best] PSNR={best_candidate:.2f} dB → {best_save_path}")

                best_ever_path = best_save_path
                recent_best_ckpts.append(best_save_path)

                while len(recent_best_ckpts) > 3:
                    oldest_best = recent_best_ckpts.pop(0)
                    if oldest_best == best_ever_path:
                        recent_best_ckpts.append(oldest_best)
                        break
                    if os.path.exists(oldest_best):
                        os.remove(oldest_best)
                        print(f"  [清理best] {os.path.basename(oldest_best)}")

            print(f"  [ckpt] ep{epoch:03d}  "
                  f"PSNR={best_candidate:.2f}  (best={best_psnr:.2f})")

    # FEAT-6: SWA
    swa_state = None
    if len(finetune_ckpt_paths) >= 2:
        n_use = min(SWA_LAST_N, len(finetune_ckpt_paths))
        swa_paths = finetune_ckpt_paths[-n_use:]
        print(f"\n[SWA] 对最后 {n_use} 个 fine-tune checkpoint 做权重平均:")
        for p in swa_paths:
            print(f"  - {os.path.basename(p)}")
        swa_state = swa_average_checkpoints(swa_paths, device, key='model')

        swa_model = LDCTDenoiserV4(
            in_ch=in_ch, bc=88, growth=32,
            bot_depth=4, bot_heads=8, ws=8,
            dec_depths=(2, 2, 2, 2),
            drop=0., attn_drop=0.,
        ).to(device)
        swa_model.load_state_dict(swa_state)

        print("\n[SWA] 验证集评估 SWA 平均权重...")
        swa_p, swa_s, swa_m = evaluate(
            swa_model, val_loader, device, SAVE_DIR,
            tile=256, overlap=64, tag='val_swa', use_tta=True,
            method_name='HCT-UNet_swa')
        torch.save({'model': swa_model.state_dict(), 'psnr': swa_p, 'ssim': swa_s},
                   os.path.join(SAVE_DIR, f'swa_P{swa_p:.2f}.pth'))
    else:
        print("\n[SWA] fine-tune 阶段 checkpoint 不足 2 个，跳过 SWA")

    print("\n训练完成！最终验证集评估 (EMA + TTA)...")
    ema.eval()
    evaluate(ema.shadow, val_loader, device, SAVE_DIR,
             tile=256, overlap=64, tag='final_val', use_tta=True,
             method_name='HCT-UNet')
    plot_history(history, SAVE_DIR)

    print("\n测试集最终评估 (EMA + TTA)...")
    ema_p, ema_s, ema_m = evaluate(
        ema.shadow, test_loader, device, SAVE_DIR,
        tile=256, overlap=64, tag='test_final_ema', use_tta=True,
        method_name='HCT-UNet_ema')

    print("\n测试集最终评估 (raw + TTA)...")
    raw_p, raw_s, raw_m = evaluate(
        model, test_loader, device, SAVE_DIR,
        tile=256, overlap=64, tag='test_final_raw', use_tta=True,
        method_name='HCT-UNet_raw')

    candidates = [('ema', ema_p, ema_s, ema_m)]
    candidates.append(('raw', raw_p, raw_s, raw_m))

    if swa_state is not None:
        print("\n测试集最终评估 (SWA + TTA)...")
        swa_model_test = LDCTDenoiserV4(
            in_ch=in_ch, bc=88, growth=32,
            bot_depth=4, bot_heads=8, ws=8,
            dec_depths=(2, 2, 2, 2),
            drop=0., attn_drop=0.,
        ).to(device)
        swa_model_test.load_state_dict(swa_state)
        swa_p2, swa_s2, swa_m2 = evaluate(
            swa_model_test, test_loader, device, SAVE_DIR,
            tile=256, overlap=64, tag='test_final_swa', use_tta=True,
            method_name='HCT-UNet_swa')
        candidates.append(('swa', swa_p2, swa_s2, swa_m2))

    best_tag, avg_p, avg_s, avg_m = max(candidates, key=lambda c: c[1])
    print(f"\n测试集最终选择: {best_tag}  "
          f"PSNR={avg_p:.2f}  SSIM={avg_s:.4f}  MSE={avg_m:.2f}")

    print(f"\n所有文件保存至: {SAVE_DIR}")

    return {
        'psnr': avg_p,
        'ssim': avg_s,
        'mse': avg_m,
        'source': best_tag,
        'test_patients': test_patients,
        'best_psnr': best_psnr,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    BASE_DIR = "/home/user/joshua82/LDCT_Project"
    LDCT_ROOT = f"{BASE_DIR}/dataset/QD301mm"
    NDCT_ROOT = f"{BASE_DIR}/dataset/FD301mm"
    NUM_WORKERS = 4

    print(f"\n患者划分:")
    print(f"  Train: {TRAIN_PATIENTS}")
    print(f"  Val  : {VAL_PATIENTS}")
    print(f"  Test : {TEST_PATIENTS}")
    print(f"\n2.5D 输入窗口 in_ch = {IN_CH}\n")

    result = train_one_run(
        TRAIN_PATIENTS, VAL_PATIENTS, TEST_PATIENTS,
        device, BASE_DIR, LDCT_ROOT, NDCT_ROOT, NUM_WORKERS,
        in_ch=IN_CH,
    )

    print("\n" + "=" * 70)
    print("Final Summary (Test set, best of EMA/raw/SWA + TTA)")
    print("=" * 70)
    print(f"PSNR : {result['psnr']:.2f} dB")
    print(f"SSIM : {result['ssim']:.4f}")
    print(f"MSE  : {result['mse']:.2f}")
    print(f"source = {result['source']}")
    print(f"Test patients = {result['test_patients']}")
    print(f"(val-best PSNR = {result['best_psnr']:.2f})")
    print("=" * 70)

    summary_path = f"{BASE_DIR}/output/checkpoints/v101s_summary.txt"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== Single-Run Summary ===\n\n")
        f.write(f"PSNR : {result['psnr']:.2f} dB\n")
        f.write(f"SSIM : {result['ssim']:.4f}\n")
        f.write(f"MSE  : {result['mse']:.2f}\n")
        f.write(f"source = {result['source']}\n")
        f.write(f"Test patients = {result['test_patients']}\n")
    print(f"\n汇总结果已保存至: {summary_path}")


# =============================================================================
# 入口
# =============================================================================
if __name__ == '__main__':
    assert torch.cuda.is_available(), "未检测到 CUDA 设备，请确认环境配置"
    print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
    main()