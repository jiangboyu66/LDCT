"""
Transformer-Based Attention Framework for
Noise Reduction and Detail Preservation in Low-Dose CT Imaging — V4-MultiKernel
================================================================================
修复清单:
  FIX-1  BatchNorm2d → GroupNorm
  FIX-2  patch_inference 中心裁剪拼接
  FIX-3  PerceptualLoss mean/std 动态对齐 device/dtype
  FIX-4  patch_size 切换时重建 DataLoader
  FIX-5  CSAS 显式传入 H, W
  FIX-6  EdgeAwareLoss Sobel kernel 动态对齐 device/dtype
  FIX-7  safe_heads() 确保 num_heads 能整除 dim，修复 bc=88 reshape 错误
  FIX-8  GradScaler / autocast 使用新 API
  FIX-9  TTA 逆变换顺序修正（第6、7种变换）

新增优化:
  OPT-1  tta_inference() —— 8 种几何变换 TTA（逆变换已修正）
  OPT-2  Fine-tune 阶段 (epoch 301–400)：保守 lr=1e-6, CosineAnnealingLR
  OPT-3  多核联合训练 —— 同时使用 FD301mm / FD451mm / FD303mm / FD453mm
          四种重建核的 FD+QD 配对数据，训练数据量 ×4
          数据集划分保持不变（以患者 ID 区分），测试集仍为 L506

实际数据目录结构:
  BASE_DIR/dataset/
    FD301mm/<patient>/*.npy   ← Full Dose,    B30, 1mm
    FD303mm/<patient>/*.npy   ← Full Dose,    B30, 3mm
    FD451mm/<patient>/*.npy   ← Full Dose,    D45, 1mm
    FD453mm/<patient>/*.npy   ← Full Dose,    D45, 3mm
    QD301mm/<patient>/*.npy   ← Quarter Dose, B30, 1mm
    QD303mm/<patient>/*.npy   ← Quarter Dose, B30, 3mm
    QD451mm/<patient>/*.npy   ← Quarter Dose, D45, 1mm
    QD453mm/<patient>/*.npy   ← Quarter Dose, D45, 3mm

核配对映射 (QD → FD):
    QD301mm ↔ FD301mm
    QD303mm ↔ FD303mm
    QD451mm ↔ FD451mm
    QD453mm ↔ FD453mm

数据集患者划分（10 个患者）:
  训练集 (7): L067, L096, L109, L143, L192, L286, L291
  验证集 (2): L310, L333
  测试集 (1): L506

  ※ 验证/测试 使用 1mmB30 单核（QD301mm/FD301mm），以便和历史结果对比。
    若需多核验证，将 VAL_KERNEL_PAIRS / TEST_KERNEL_PAIRS 改为多核列表即可。

训练轮次:
  主训练: epoch  1 – 300
  精调:   epoch 301 – 400  (共 400 轮)
"""


import os, random, math, glob, re, certifi
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
from copy import deepcopy


# =============================================================================
# 数据集划分 & 多核配置  (OPT-3)
# =============================================================================
TRAIN_PATIENTS = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291']
VAL_PATIENTS   = ['L310', 'L333']
TEST_PATIENTS  = ['L506']

# 核配对：(QD目录名, FD目录名)
# 训练使用全部 4 种重建核配对
TRAIN_KERNEL_PAIRS = [
    ('QD301mm', 'FD301mm'),
    ('QD303mm', 'FD303mm'),
    ('QD451mm', 'FD451mm'),
    ('QD453mm', 'FD453mm'),
]

# 验证/测试沿用 1mmB30 单核，方便与历史结果比较
VAL_KERNEL_PAIRS  = [('QD301mm', 'FD301mm')]
TEST_KERNEL_PAIRS = [('QD301mm', 'FD301mm')]


# =============================================================================
# Fine-tune 超参  (OPT-2 保守版)
# =============================================================================
MAIN_EPOCHS      = 300   # 主训练轮数
FINETUNE_EPOCHS  = 100   # 精调轮数
NUM_EPOCHS       = MAIN_EPOCHS + FINETUNE_EPOCHS  # 400 总轮数

FINETUNE_LR      = 1e-6  # 保守：比主训练低 100×
FINETUNE_ETA_MIN = 1e-8


# =============================================================================
# 工具函数
# =============================================================================
def _infer_hw(L, hint_H=None, hint_W=None):
    if hint_H is not None and hint_W is not None:
        assert hint_H * hint_W == L, f"hint_H*hint_W={hint_H*hint_W} ≠ L={L}"
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
    v_std  = float(np.std(img_hu))
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
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def eval(self):
        self.shadow.eval()
        return self.shadow


# =============================================================================
# RRDB Encoder
# =============================================================================
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth, 3, padding=1, bias=False)
        self.norm = GN(growth)
        self.act  = nn.GELU()

    def forward(self, x):
        return torch.cat([x, self.act(self.norm(self.conv(x)))], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth=32):
        super().__init__()
        self.layers = nn.Sequential(
            DenseLayer(in_ch,            growth),
            DenseLayer(in_ch + growth,   growth),
            DenseLayer(in_ch + growth*2, growth),
            DenseLayer(in_ch + growth*3, growth),
        )
        self.proj = nn.Conv2d(in_ch + growth*4, in_ch, 1, bias=False)
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
        self.stem  = nn.Sequential(
            nn.Conv2d(in_ch, bc, 3, padding=1, bias=False),
            GN(bc), nn.GELU())
        self.enc1  = nn.Sequential(RRDB(bc, growth), RRDB(bc, growth))
        self.down1 = nn.MaxPool2d(2)
        self.enc2  = nn.Sequential(
            nn.Conv2d(bc, bc*2, 1, bias=False), GN(bc*2), nn.GELU(),
            RRDB(bc*2, growth), RRDB(bc*2, growth))
        self.down2 = nn.MaxPool2d(2)
        self.enc3  = nn.Sequential(
            nn.Conv2d(bc*2, bc*4, 1, bias=False), GN(bc*4), nn.GELU(),
            RRDB(bc*4, growth), RRDB(bc*4, growth))
        self.down3 = nn.MaxPool2d(2)
        self.enc4  = nn.Sequential(
            nn.Conv2d(bc*4, bc*8, 1, bias=False), GN(bc*8), nn.GELU(),
            RRDB(bc*8, growth), RRDB(bc*8, growth))
        self.down4 = nn.MaxPool2d(2)

    def forward(self, x):
        e1  = self.enc1(self.stem(x))
        e2  = self.enc2(self.down1(e1))
        e3  = self.enc3(self.down2(e2))
        e4  = self.enc4(self.down3(e3))
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
        self.dim       = dim
        self.ws        = ws
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5

        self.rpb = nn.Parameter(torch.zeros((2*ws-1)**2, num_heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), indexing='ij'))
        cf  = coords.flatten(1)
        rel = cf[:, :, None] - cf[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2*ws - 1
        self.register_buffer('rpi', rel.sum(-1))

        self.qkv       = nn.Linear(dim, dim*3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        rpb  = self.rpb[self.rpi.view(-1)].view(N, N, self.num_heads)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# =============================================================================
# SwinBlock
# =============================================================================
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, ws=8, shift=False,
                 mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.ws    = ws
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        hid        = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, dim), nn.Dropout(drop))
        self.attn  = WindowAttention(dim, ws, num_heads, attn_drop, drop)

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
        mw   = window_partition(img_mask, ws).view(-1, ws*ws)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.).masked_fill(mask == 0, 0.)
        return mask

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        ws    = min(self.ws, H, W)
        shift = ws // 2 if self.shift and min(H, W) > ws else 0
        sc = x
        x  = self.norm1(x).view(B, H, W, C)
        if shift > 0:
            x = torch.roll(x, (-shift, -shift), (1, 2))
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x.permute(0, 3, 1, 2),
                      (0, pad_r, 0, pad_b)).permute(0, 2, 3, 1)
        _, pH, pW, _ = x.shape
        mask = self._build_mask(pH, pW, ws, shift > 0, x.device)
        xw   = window_partition(x, ws)
        xw   = self.attn(xw.view(-1, ws*ws, C), mask)
        xw   = xw.view(-1, ws, ws, C)
        x    = window_reverse(xw, ws, pH, pW)
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()
        if shift > 0:
            x = torch.roll(x, (shift, shift), (1, 2))
        x = x.view(B, H*W, C) + sc
        return x + self.mlp(self.norm2(x))


# =============================================================================
# DualScaleBlock
# =============================================================================
class DualScaleBlock(nn.Module):
    def __init__(self, dim, num_heads, ws=8, shift=False, drop=0., attn_drop=0.):
        super().__init__()
        self.local_attn = SwinBlock(dim, num_heads, ws, shift,
                                    drop=drop, attn_drop=attn_drop)
        dil            = 4
        self.glob_dw   = nn.Conv2d(dim, dim, 3, padding=dil, dilation=dil,
                                   groups=dim, bias=False)
        self.glob_pw   = nn.Conv2d(dim, dim, 1, bias=False)
        self.glob_norm = nn.LayerNorm(dim)
        self.gate      = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.out_norm  = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        xl   = self.local_attn(x, H, W)
        x2d  = x.transpose(1, 2).view(B, C, H, W)
        xg   = self.glob_pw(self.glob_dw(x2d)).flatten(2).transpose(1, 2)
        xg   = self.glob_norm(xg + x)
        gate = self.gate(torch.cat([xl, xg], dim=-1))
        return self.out_norm(gate * xl + (1 - gate) * xg)


# =============================================================================
# CSAS
# =============================================================================
class CSAS(nn.Module):
    def __init__(self, dim, pool_grid=8):
        super().__init__()
        self.pool_grid = pool_grid
        nh             = safe_heads(dim, target_div=64)
        self.nh        = nh
        self.scale     = (dim // nh) ** -0.5
        self.nq        = nn.LayerNorm(dim)
        self.nk        = nn.LayerNorm(dim)
        self.q         = nn.Linear(dim, dim, bias=False)
        self.k         = nn.Linear(dim, dim, bias=False)
        self.v         = nn.Linear(dim, dim, bias=False)
        self.o         = nn.Linear(dim, dim, bias=False)

    def forward(self, query, kv, H=None, W=None):
        B, Lq, C = query.shape
        H, W     = _infer_hw(Lq, H, W)
        g        = min(self.pool_grid, H, W)
        kv_2d   = kv.transpose(1, 2).view(B, C, H, W)
        kv_pool = F.adaptive_avg_pool2d(kv_2d, (g, g))
        kv_flat = kv_pool.flatten(2).transpose(1, 2)
        nh, D = self.nh, C // self.nh
        q = self.q(self.nq(query  )).view(B, Lq,  nh, D).transpose(1, 2)
        k = self.k(self.nk(kv_flat)).view(B, g*g, nh, D).transpose(1, 2)
        v = self.v(kv_flat          ).view(B, g*g, nh, D).transpose(1, 2)
        attn = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, Lq, C)
        return query + self.o(out)


# =============================================================================
# CSG
# =============================================================================
class CSG(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        rd           = max(dim // r, 4)
        self.ch_fc   = nn.Sequential(
            nn.Linear(dim, rd), nn.ReLU(True), nn.Linear(rd, dim), nn.Sigmoid())
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x, H, W):
        B, L, C = x.shape
        ch  = self.ch_fc(x.mean(dim=1))
        x   = x * ch.unsqueeze(1)
        x2d = x.transpose(1, 2).view(B, C, H, W)
        sp  = self.sp_conv(torch.cat(
            [x2d.mean(1, keepdim=True), x2d.max(1, keepdim=True).values], 1))
        return (x2d * sp).flatten(2).transpose(1, 2)


# =============================================================================
# TransformerBottleneck
# =============================================================================
class TransformerBottleneck(nn.Module):
    def __init__(self, dim, train_grid=16, num_heads=8,
                 depth=4, ws=8, drop=0., attn_drop=0.):
        super().__init__()
        self.dim        = dim
        self.train_grid = train_grid
        self.pos        = nn.Parameter(
            torch.zeros(1, train_grid*train_grid, dim))
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
        return pe.permute(0, 2, 3, 1).reshape(1, H*W, self.dim)

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
        self.expand = nn.Linear(in_dim, in_dim*4, bias=False)
        self.norm   = nn.LayerNorm(in_dim)
        self.proj   = nn.Linear(in_dim, out_dim, bias=False) \
                      if in_dim != out_dim else nn.Identity()

    def forward(self, x, H, W):
        x        = self.expand(x)
        B, L, C4 = x.shape
        C        = C4 // 4
        x        = x.view(B, H, W, 2, 2, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x        = self.norm(x.view(B, 2*H*2*W, C))
        return self.proj(x), 2*H, 2*W


# =============================================================================
# DecoderStage
# =============================================================================
class DecoderStage(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, num_heads,
                 ws=8, depth=2, drop=0., attn_drop=0.):
        super().__init__()
        self.expand    = PatchExpand(in_dim, out_dim)
        self.skip_proj = nn.Linear(skip_dim, out_dim, bias=False) \
                         if skip_dim != out_dim else nn.Identity()
        self.csas      = CSAS(out_dim)
        num_heads      = safe_heads(out_dim, target_div=out_dim // max(1, num_heads))
        self.blocks    = nn.ModuleList([
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
        self.norm   = nn.LayerNorm(dim)
        self.main   = nn.Sequential(
            nn.Conv2d(dim,    dim,    3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim,    dim//2, 1, bias=False), nn.GELU(),
            nn.Conv2d(dim//2, in_ch,  1))
        self.detail = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(32,    in_ch, 3, padding=1))
        self.alpha  = nn.Parameter(torch.tensor(0.1))

    def forward(self, tokens, H, W, x_in):
        B, L, C = tokens.shape
        feat   = self.norm(tokens).transpose(1, 2).view(B, C, H, W)
        main   = self.main(feat)
        lp     = F.avg_pool2d(x_in, 3, stride=1, padding=1)
        detail = self.detail(x_in - lp)
        return (x_in + main + self.alpha.clamp(0, 1) * detail).clamp(-1., 1.)


# =============================================================================
# Full Model
# =============================================================================
class LDCTDenoiserV4(nn.Module):
    def __init__(self, in_ch=1, bc=88, growth=32,
                 bot_depth=4, bot_heads=8, ws=8,
                 dec_depths=(2, 2, 2, 2),
                 drop=0., attn_drop=0.):
        super().__init__()
        self.encoder    = RRDBEncoder(in_ch, bc, growth)
        self.bottleneck = TransformerBottleneck(
            dim=bc*8, train_grid=16, num_heads=bot_heads,
            depth=bot_depth, ws=ws, drop=drop, attn_drop=attn_drop)
        self.dec4 = DecoderStage(bc*8, bc*8, bc*4, safe_heads(bc*4),
                                 ws, dec_depths[0], drop, attn_drop)
        self.dec3 = DecoderStage(bc*4, bc*4, bc*2, safe_heads(bc*2),
                                 ws, dec_depths[1], drop, attn_drop)
        self.dec2 = DecoderStage(bc*2, bc*2, bc,   safe_heads(bc),
                                 ws, dec_depths[2], drop, attn_drop)
        self.dec1 = DecoderStage(bc,   bc,   bc,   safe_heads(bc),
                                 ws, dec_depths[3], drop, attn_drop)
        self.head = MultiScaleHead(bc, in_ch)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        skips, bot  = self.encoder(x)
        bot         = self.bottleneck(bot)
        bH, bW      = H // 16, W // 16
        t           = bot.flatten(2).transpose(1, 2)
        t, h, w = self.dec4(t, skips[3], bH, bW)
        t, h, w = self.dec3(t, skips[2], h,  w)
        t, h, w = self.dec2(t, skips[1], h,  w)
        t, h, w = self.dec1(t, skips[0], h,  w)
        return self.head(t, h, w, x)


# =============================================================================
# Loss Functions
# =============================================================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, data_range=2.0, levels=3):
        super().__init__()
        self.dr     = data_range
        self.levels = levels
        self.ws     = window_size
        g   = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g   = torch.exp(-(g**2) / (2*1.5**2)); g /= g.sum()
        self.register_buffer('win', g.outer(g).unsqueeze(0).unsqueeze(0))

    def _ssim(self, x, y):
        C1, C2 = (0.01*self.dr)**2, (0.03*self.dr)**2
        pad = self.ws // 2
        w   = self.win.to(x.device, x.dtype)
        mx  = F.conv2d(x,   w, padding=pad)
        my  = F.conv2d(y,   w, padding=pad)
        mxx = F.conv2d(x*x, w, padding=pad) - mx**2
        myy = F.conv2d(y*y, w, padding=pad) - my**2
        mxy = F.conv2d(x*y, w, padding=pad) - mx*my
        return ((2*mx*my+C1)*(2*mxy+C2) /
                ((mx**2+my**2+C1)*(mxx+myy+C2))).mean()

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
        self.eps2 = eps**2

    def forward(self, pred, target):
        return torch.sqrt((pred - target)**2 + self.eps2).mean()


class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        fp = torch.fft.rfft2(pred.float(),   norm='ortho')
        ft = torch.fft.rfft2(target.float(), norm='ortho')
        return F.l1_loss(fp.abs(), ft.abs())


class HaarWaveletLoss(nn.Module):
    @staticmethod
    def _dwt(x):
        a = x[:, :, 0::2, 0::2]; b = x[:, :, 1::2, 0::2]
        c = x[:, :, 0::2, 1::2]; d = x[:, :, 1::2, 1::2]
        return ((a+b+c+d)*0.25, (a-b+c-d)*0.25,
                (a+b-c-d)*0.25, (a-b-c+d)*0.25)

    def forward(self, pred, target):
        _, lhp, hlp, hhp = self._dwt(pred)
        _, lht, hlt, hht = self._dwt(target)
        return (F.l1_loss(lhp, lht) + F.l1_loss(hlp, hlt) +
                0.5*F.l1_loss(hhp, hht)) / 2.5


class NoiseAwareLoss(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.k = k

    def forward(self, pred, target, ldct):
        k, p = self.k, self.k // 2
        mu  = F.avg_pool2d(ldct,     k, stride=1, padding=p)
        var = (F.avg_pool2d(ldct**2, k, stride=1, padding=p) - mu**2).clamp(0)
        w   = (var / (var.mean() + 1e-6)).clamp(0.5, 3.0)
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
        ep = torch.sqrt(F.conv2d(p, sx, padding=1)**2 +
                        F.conv2d(p, sy, padding=1)**2 + 1e-6)
        et = torch.sqrt(F.conv2d(t, sx, padding=1)**2 +
                        F.conv2d(t, sy, padding=1)**2 + 1e-6)
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
            t3   = t.float().repeat(1, 3, 1, 1)
            mean = self.mean.to(t3.device)
            std  = self.std.to(t3.device)
            return ((t3 + 1) / 2 - mean) / std
        self.feats.clear(); self.resnet(prep(x)); xf = self.feats.copy()
        self.feats.clear(); self.resnet(prep(y)); yf = dict(self.feats)
        return sum(F.mse_loss(xf[l], yf[l]) for l in self.layers) / 3

    def __del__(self):
        for h in self.hooks:
            try: h.remove()
            except: pass


class CompositeLoss(nn.Module):
    def __init__(self, lc=1.0, ls=0.8, lp=0.05, lf=0.1, lw=0.2, ln=0.4, le=0.2):
        super().__init__()
        self.charb  = CharbonnierLoss()
        self.ssim   = SSIMLoss(data_range=2.0, levels=3)
        self.perc   = PerceptualLoss()
        self.freq   = FrequencyLoss()
        self.wav    = HaarWaveletLoss()
        self.noise  = NoiseAwareLoss()
        self.edge   = EdgeAwareLoss()
        self.lc, self.ls, self.lp = lc, ls, lp
        self.lf, self.lw, self.ln = lf, lw, ln
        self.le = le

    def forward(self, pred, target, ldct=None):
        lc = self.charb(pred, target)
        ls = self.ssim(pred, target)
        lp = self.perc(pred, target)
        lf = self.freq(pred, target)
        lw = self.wav(pred, target)
        le = self.edge(pred, target)
        ln = self.noise(pred, target, ldct) if ldct is not None \
             else torch.zeros(1, device=pred.device)
        total = (self.lc*lc + self.ls*ls + self.lp*lp +
                 self.lf*lf + self.lw*lw + self.ln*ln + self.le*le)
        subs  = dict(charb=lc.item(), ssim=ls.item(), perc=lp.item(),
                     freq=lf.item(),  wav=lw.item(),  edge=le.item(),
                     noise=ln.item() if ldct is not None else 0.)
        return total, subs


# =============================================================================
# Dataset  (OPT-3: 多核联合，适配扁平目录结构)
# ─────────────────────────────────────────────────────────────────────────────
# 实际目录结构（所有核目录直接在 dataset/ 下）:
#   dataset/
#     QD301mm/<patient>/*.npy   ← 低剂量  B30 1mm
#     FD301mm/<patient>/*.npy   ← 全剂量  B30 1mm
#     QD303mm/<patient>/*.npy
#     FD303mm/<patient>/*.npy
#     QD451mm/<patient>/*.npy
#     FD451mm/<patient>/*.npy
#     QD453mm/<patient>/*.npy
#     FD453mm/<patient>/*.npy
#
# kernel_pairs: list of (qd_dir_name, fd_dir_name)  e.g. ('QD301mm','FD301mm')
# =============================================================================
class LDCTDataset(Dataset):
    """
    多核联合数据集（扁平目录版）。

    参数
    ----
    dataset_root : str
        包含所有核目录的根目录，例如 BASE_DIR/dataset
    patients     : list[str]
        患者 ID 列表
    kernel_pairs : list[tuple[str, str]]
        核配对列表，每项为 (qd_dir_name, fd_dir_name)
        例如 [('QD301mm', 'FD301mm'), ('QD451mm', 'FD451mm')]
    mode         : 'train' | 'val' | 'test'
    patch_size   : int，0 表示全图不裁剪
    """
    def __init__(self, dataset_root, patients, kernel_pairs,
                 mode='train', patch_size=128):
        self.pairs      = []   # (ld_path, nd_path, kernel_tag)
        self.patch_size = patch_size
        self.is_train   = (mode == 'train')
        avail = []

        for qd_name, fd_name in kernel_pairs:
            qd_root = os.path.join(dataset_root, qd_name)
            fd_root = os.path.join(dataset_root, fd_name)
            for p in patients:
                ld_dir = os.path.join(qd_root, p)
                nd_dir = os.path.join(fd_root, p)
                if not (os.path.isdir(ld_dir) and os.path.isdir(nd_dir)):
                    # 静默跳过找不到的核/患者组合
                    continue
                lf = sorted(f for f in os.listdir(ld_dir) if f.endswith('.npy'))
                nf = sorted(f for f in os.listdir(nd_dir) if f.endswith('.npy'))
                n  = min(len(lf), len(nf))
                if n == 0:
                    continue
                tag = f"{qd_name}↔{fd_name}/{p}"
                avail.append(tag)
                for i in range(n):
                    self.pairs.append((
                        os.path.join(ld_dir, lf[i]),
                        os.path.join(nd_dir, nf[i]),
                        f"{qd_name}↔{fd_name}",
                    ))

        print(f"[{mode.upper()}] {len(avail)} 个(核/患者)组合  "
              f"共 {len(self.pairs)} 个切片对")
        if len(self.pairs) == 0:
            print(f"  ⚠️  警告：{mode} 集为空！请检查路径:")
            print(f"     dataset_root={dataset_root}")
            print(f"     kernel_pairs={kernel_pairs}")
            print(f"     patients={patients}")

    def set_patch_size(self, ps):
        self.patch_size = ps

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _load_slice(path):
        arr = np.load(path).astype(np.float32)
        arr = arr.squeeze()
        if arr.ndim == 1:
            # 展平的 1D 数组，自动 reshape 为最接近的正方形
            n = arr.size
            h = int(math.isqrt(n))
            if h * h != n:
                raise ValueError(f"1D 数组长度 {n} 不是完全平方数，无法 reshape，路径={path}")
            arr = arr.reshape(h, h)
        if arr.ndim != 2:
            raise ValueError(f"squeeze 后仍不是 2D，shape={arr.shape}，路径={path}")
        return arr
    def __getitem__(self, idx):
        lp, np_, kernel_tag = self.pairs[idx]
        ld = np.clip(self._load_slice(lp), -1000, 1500)  # ← 改这两行
        nd = np.clip(self._load_slice(np_), -1000, 1500)  # ← 改这两行
        # 归一化到 [-1, 1]
        ld = (ld + 1000) / 2500 * 2 - 1
        nd = (nd + 1000) / 2500 * 2 - 1
        ld = torch.from_numpy(ld)[None]
        nd = torch.from_numpy(nd)[None]

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
# DataLoader helpers
# =============================================================================
def get_batch_size(patch_size):
    if patch_size <= 128: return 72
    if patch_size <= 192: return 32
    if patch_size <= 256: return 20
    return 12


def get_patch_size(epoch):
    """
    主训练 epoch 1–300 的 patch 调度:
      1–30   → 128
      31–80  → 192
      81–150 → 256
      151+   → 320
    精调阶段固定 256（在训练循环中单独处理）。
    """
    if epoch <= 30:  return 128
    if epoch <= 150: return 256
    return 320


# =============================================================================
# OPT-1: TTA Inference（FIX-9：逆变换顺序已修正）
# =============================================================================
@torch.no_grad()
def tta_inference(model, img, tile=256, overlap=64, device='cuda'):
    """
    D4 对称群 8 种变换取均值。
    逆变换规则：复合变换的逆 = 各步骤逆变换的逆序。
    """
    def fwd0(x): return x
    def inv0(x): return x

    def fwd1(x): return torch.flip(x, [-1])
    def inv1(x): return torch.flip(x, [-1])

    def fwd2(x): return torch.flip(x, [-2])
    def inv2(x): return torch.flip(x, [-2])

    def fwd3(x): return torch.rot90(x,  1, [-2, -1])
    def inv3(x): return torch.rot90(x, -1, [-2, -1])

    def fwd4(x): return torch.rot90(x,  2, [-2, -1])
    def inv4(x): return torch.rot90(x, -2, [-2, -1])

    def fwd5(x): return torch.rot90(x,  3, [-2, -1])
    def inv5(x): return torch.rot90(x, -3, [-2, -1])

    # 正向: rot90(1) → flip(-1)   逆向: flip(-1) → rot90(-1)  (FIX-9)
    def fwd6(x): return torch.flip(torch.rot90(x,  1, [-2, -1]), [-1])
    def inv6(x): return torch.rot90(torch.flip(x, [-1]), -1, [-2, -1])

    # 正向: rot90(1) → flip(-2)   逆向: flip(-2) → rot90(-1)  (FIX-9)
    def fwd7(x): return torch.flip(torch.rot90(x,  1, [-2, -1]), [-2])
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
# Patch Inference
# =============================================================================
@torch.no_grad()
def patch_inference(model, img, tile=256, overlap=64, device='cuda'):
    _, C, H, W = img.shape
    tile   = (tile // 16) * 16
    margin = overlap // 2
    step   = tile - overlap

    pad_h = (tile - H % tile) % tile if H % tile != 0 else 0
    pad_w = (tile - W % tile) % tile if W % tile != 0 else 0
    img_p = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    oH, oW = img_p.shape[2], img_p.shape[3]

    out_full = torch.zeros(1, C, oH, oW)
    cnt_full = torch.zeros(1, C, oH, oW)

    ys = list(range(0, max(oH - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(oW - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < oH: ys.append(oH - tile)
    if xs[-1] + tile < oW: xs.append(oW - tile)

    for y in ys:
        for x in xs:
            patch = img_p[:, :, y:y+tile, x:x+tile].to(device)
            pred  = model(patch).cpu()

            py1 = margin if y > 0 else 0
            py2 = tile - margin if y + tile < oH else tile
            px1 = margin if x > 0 else 0
            px2 = tile - margin if x + tile < oW else tile

            cy1, cy2 = y + py1, y + py2
            cx1, cx2 = x + px1, x + px2

            out_full[:, :, cy1:cy2, cx1:cx2] += pred[:, :, py1:py2, px1:px2]
            cnt_full[:, :, cy1:cy2, cx1:cx2] += 1.0

    out_full = out_full / cnt_full.clamp(min=1)
    mask = (cnt_full == 0)
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
    return ps, ss


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, save_dir,
             tile=256, overlap=64, tag='val', use_tta=False):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    psnrs, ssims = [], []
    infer_fn = tta_inference if use_tta else patch_inference
    if use_tta:
        print(f"  [evaluate] TTA 已启用（8 种变换）")

    for i, (ldct, ndct) in enumerate(loader):
        pred   = infer_fn(model, ldct, tile, overlap, device)
        ph     = to_hu(pred[0, 0])
        gh     = to_hu(ndct[0, 0])
        lh     = to_hu(ldct[0, 0])
        ps, ss = compute_metrics(ph, gh)
        psnrs.append(ps); ssims.append(ss)

        if i % 500 == 0:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            smart_imshow(ax[0], lh, "LDCT")
            smart_imshow(ax[1], ph, f"Denoised\nPSNR:{ps:.2f} dB | SSIM:{ss:.4f}")
            smart_imshow(ax[2], gh, "NDCT")
            plt.savefig(os.path.join(save_dir, f"{tag}_case_{i:03d}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [{tag}] {i}/{len(loader.dataset)}  "
                  f"PSNR:{ps:.2f}  SSIM:{ss:.4f}")

    avg_p = float(np.mean(psnrs)); std_p = float(np.std(psnrs))
    avg_s = float(np.mean(ssims)); std_s = float(np.std(ssims))
    tta_tag = "+TTA" if use_tta else ""
    print(f"\n{'='*60}")
    print(f"[{tag.upper()}]{tta_tag}  N={len(psnrs)}")
    print(f"  PSNR : {avg_p:.2f} ± {std_p:.2f} dB")
    print(f"  SSIM : {avg_s:.4f} ± {std_s:.4f}")
    print(f"{'='*60}\n")

    with open(os.path.join(save_dir, f"{tag}_results.txt"),
              'w', encoding='utf-8') as f:
        f.write(f"=== V4-MultiKernel {tag.upper()}{tta_tag} ===\n\n")
        f.write(f"PSNR : {avg_p:.2f} ± {std_p:.2f} dB\n")
        f.write(f"SSIM : {avg_s:.4f} ± {std_s:.4f}\n")
        f.write(f"N    : {len(psnrs)}\n")
        f.write(f"torch: {torch.__version__} | device: {device}\n")
    return avg_p, avg_s


def plot_history(history, save_dir):
    keys  = [k for k in history if k != 'train_loss']
    n     = len(keys) + 1
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    ep = range(1, len(history['train_loss']) + 1)
    axes[0].plot(ep, history['train_loss'], 'b-o', ms=3, lw=1.5)
    axes[0].set_title('Total loss'); axes[0].grid(True)
    for ax, k in zip(axes[1:], keys):
        ax.plot(ep, history[k], ms=3, lw=1.5)
        ax.set_title(k); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


# =============================================================================
# OPT-2: 保守 Fine-tune 组件 (epoch 301–400)
# =============================================================================
def build_finetune_components(model_ddp, device):
    """
    保守精调：lr=1e-6（比主训练低 100×），纯 CosineAnnealingLR，
    loss 权重只做小幅调整（±30% 以内），避免灾难性遗忘。
    """
    optimizer_ft = optim.AdamW(
        model_ddp.parameters(),
        lr=5e-6,
        weight_decay=0,
        betas=(0.9, 0.999),
    )
    scheduler_ft = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_ft,
        T_0=20,
        T_mult=2,
        eta_min=1e-7,
    )
    criterion_ft = CompositeLoss(
        lc=1.5,  # ↑↑ 大幅提升，PSNR 最直接的驱动力
        ls=1.2,  # ↑  适度保留，结构对齐间接帮助 PSNR
        lp=0.02,  # ↓↓ 大幅降低，感知损失会干扰像素精度
        lf=0.15,  # ↑  频域约束压噪声，利于 PSNR
        lw=0.15,  # ↑  小幅提升
        ln=0.5,  # ↑↑ 噪声加权像素误差，对高噪声区 PSNR 提升明显
        le=0.1,  # ↓↓ 大幅降低，边缘锐化对 PSNR 贡献有限
    ).to(device)
    return optimizer_ft, scheduler_ft, criterion_ft


# =============================================================================
# Checkpoint 搜索
# =============================================================================
def find_best_ckpt(save_dir):
    best_files = glob.glob(os.path.join(save_dir, 'best_P*.pth'))
    ckpt_files = glob.glob(os.path.join(save_dir, 'ckpt_ep*.pth'))
    def extract_psnr(path):
        m = re.search(r'_P([\d.]+)', os.path.basename(path))
        return float(m.group(1)) if m else 0.0
    candidates = best_files if best_files else ckpt_files
    if not candidates:
        return None, 0.0
    best_path = max(candidates, key=extract_psnr)
    return best_path, extract_psnr(best_path)


# =============================================================================
# DDP main
# =============================================================================
def main():
    device  = torch.device('cuda:0')
    is_main = True   # 单卡永远是 main rank

    BASE_DIR     = "/home/user/joshua82/LDCT_Project"
    SAVE_DIR     = f"{BASE_DIR}/output/checkpoints/v102s"
    DATASET_ROOT = f"{BASE_DIR}/dataset"
    NUM_WORKERS  = 4

    print(f"Device: {device}")
    print(f"GPU-0 : {torch.cuda.get_device_name(0)}")
    print(f"\n[OPT-3] 多核联合训练（扁平目录结构）")
    print(f"  DATASET_ROOT : {DATASET_ROOT}")
    print(f"  训练核配对  : {TRAIN_KERNEL_PAIRS}")
    print(f"  验证核配对  : {VAL_KERNEL_PAIRS}")
    print(f"  测试核配对  : {TEST_KERNEL_PAIRS}")
    print(f"  训练患者    : {TRAIN_PATIENTS}")
    print(f"  验证患者    : {VAL_PATIENTS}")
    print(f"  测试患者    : {TEST_PATIENTS}")
    print(f"\n训练计划: epoch 1–{MAIN_EPOCHS}（主训练）"
          f" + epoch {MAIN_EPOCHS+1}–{NUM_EPOCHS}（fine-tune）\n")

    # ── Dataset ───────────────────────────────────────────────────
    train_ds = LDCTDataset(
        DATASET_ROOT, TRAIN_PATIENTS, TRAIN_KERNEL_PAIRS,
        mode='train', patch_size=128)
    val_ds = LDCTDataset(
        DATASET_ROOT, VAL_PATIENTS, VAL_KERNEL_PAIRS,
        mode='val', patch_size=0)
    test_ds = LDCTDataset(
        DATASET_ROOT, TEST_PATIENTS, TEST_KERNEL_PAIRS,
        mode='test', patch_size=0)

    # ── DataLoader（单卡，无 DistributedSampler）─────────────────
    def make_loader(dataset, batch_size, num_workers, shuffle=True):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

    cur_ps    = get_patch_size(1)
    cur_batch = get_batch_size(cur_ps)
    train_loader = make_loader(train_ds, cur_batch, NUM_WORKERS)
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"[初始] patch={cur_ps}  batch={cur_batch}")

    # ── 模型 ──────────────────────────────────────────────────────
    model = LDCTDenoiserV4(
        in_ch=1, bc=88, growth=32,
        bot_depth=4, bot_heads=8, ws=8,
        dec_depths=(2, 2, 2, 2),
        drop=0.05, attn_drop=0.05,
    ).to(device)

    # 单卡不需要 DDP；直接用 model
    model_ddp = model   # 保持变量名兼容，避免大量改动

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    ema = EMA(model, decay=0.9999)

    # ── 主训练优化器 ──────────────────────────────────────────────
    WARMUP    = 10
    optimizer = optim.AdamW(model.parameters(), lr=1e-4,
                            weight_decay=5e-5, betas=(0.9, 0.999))

    def lr_lambda(ep):
        if ep < WARMUP:
            return (ep + 1) / WARMUP
        if 80 <= ep < 85:
            return 0.05 + 0.25 * (ep - 80) / 5
        if 150 <= ep < 155:
            return 0.02 + 0.08 * (ep - 150) / 5
        stages = [
            (WARMUP,  30,  1.00),
            (30,      80,  0.60),
            (85,     150,  0.30),
            (155, MAIN_EPOCHS, 0.10),
        ]
        for s_start, s_end, peak in stages:
            if s_start <= ep < s_end:
                t = (ep - s_start) / max(1, s_end - s_start)
                return peak * 0.5 * (1 + math.cos(math.pi * t))
        return 0.02

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = CompositeLoss(lc=1.0, ls=0.8, lp=0.05,
                              lf=0.1,  lw=0.2, ln=0.4, le=0.2).to(device)
    scaler    = torch.amp.GradScaler('cuda')

    os.makedirs(SAVE_DIR, exist_ok=True)
    history      = {'train_loss': []}
    best_psnr    = 0.
    start_epoch  = 1
    recent_ckpts = []
    optimizer_ft = scheduler_ft = criterion_ft = None
    in_finetune  = False

    # ── Checkpoint 恢复 ───────────────────────────────────────────
    ckpt_path, ckpt_psnr = find_best_ckpt(SAVE_DIR)
    if ckpt_path is None:
        old_save_dir = f"{BASE_DIR}/output/checkpoints/a102"
        ckpt_path, ckpt_psnr = find_best_ckpt(old_save_dir)
        if ckpt_path is not None:
            print(f"[Resume] 从旧目录迁移权重: {old_save_dir}")

    if ckpt_path is not None:
        print(f"\n[Resume] {os.path.basename(ckpt_path)}  PSNR={ckpt_psnr:.2f}")
        ckpt  = torch.load(ckpt_path, map_location=device)
        state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(state)
        if 'ema' in ckpt:
            ema.shadow.load_state_dict(ckpt['ema'])
        if 'opt' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['opt'])
            except Exception:
                print("  [Resume] optimizer 状态不兼容，跳过")
        history     = ckpt.get('history', history)
        best_psnr   = ckpt.get('best_psnr', ckpt_psnr)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"[Resume] 从 epoch {ckpt.get('epoch',0)} 恢复，"
              f"将从 epoch {start_epoch} 继续\n")
    else:
        print("\n[Resume] 未找到 checkpoint，从头训练\n")

    # ── Training Loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        if epoch == MAIN_EPOCHS + 1 and not in_finetune:
            in_finetune = True
            optimizer_ft, scheduler_ft, criterion_ft = \
                build_finetune_components(model_ddp, device)
            scaler = torch.amp.GradScaler('cuda')
            print(f"\n{'='*60}")
            print(f"[OPT-2] 切换至保守 Fine-tune 阶段 "
                  f"(epoch {MAIN_EPOCHS+1}–{NUM_EPOCHS})")
            print(f"{'='*60}\n")

        cur_optimizer = optimizer_ft  if in_finetune else optimizer
        cur_scheduler = scheduler_ft  if in_finetune else scheduler
        cur_criterion = criterion_ft  if in_finetune else criterion

        if in_finetune:
            target_ps    = 256
            target_batch = get_batch_size(target_ps)
        else:
            target_ps    = get_patch_size(epoch)
            target_batch = get_batch_size(target_ps)

        # patch size 切换时重建 DataLoader（FIX-4）
        if target_ps != cur_ps or target_batch != cur_batch:
            train_ds.set_patch_size(target_ps)
            train_loader = make_loader(train_ds, target_batch, NUM_WORKERS)
            print(f"[Epoch {epoch}] patch {cur_ps}→{target_ps}  "
                  f"batch {cur_batch}→{target_batch}")
            cur_ps, cur_batch = target_ps, target_batch

        # ── 单 epoch 训练 ─────────────────────────────────────────
        model.train()
        total, subs_acc = 0., {}

        for bi, (ldct, ndct) in enumerate(train_loader):
            ldct, ndct = ldct.to(device), ndct.to(device)
            cur_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred       = model(ldct)
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
                lh = to_hu(ldct[0, 0])
                ps, ss = compute_metrics(ph, gh)
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                smart_imshow(ax[0], lh, "LDCT")
                smart_imshow(ax[1], ph,
                    f"Denoised\nPSNR:{ps:.2f} dB | SSIM:{ss:.4f}")
                smart_imshow(ax[2], gh, "NDCT")
                ft_tag = "_ft" if in_finetune else ""
                plt.savefig(os.path.join(SAVE_DIR,
                    f"ep{epoch:03d}{ft_tag}_b{bi+1:05d}"
                    f"_P{ps:.2f}_S{ss:.4f}.png"),
                    dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  [snap] ep{epoch}{ft_tag} b{bi+1}  "
                      f"PSNR:{ps:.2f}  SSIM:{ss:.4f}")

            if bi % 200 == 0:
                mem     = torch.cuda.memory_reserved(device) / 1e9
                sub_str = "  ".join(f"{k}:{v:.4f}" for k, v in subs.items())
                print(f"  ep{epoch} [{bi}/{len(train_loader)}]  "
                      f"loss:{loss.item():.4f}  {sub_str}  VRAM:{mem:.1f}GB")

        avg_loss = total / len(train_loader)
        history['train_loss'].append(avg_loss)
        for k in subs_acc:
            history.setdefault(k, []).append(subs_acc[k] / len(train_loader))
        print(f"{'='*60}\nEpoch {epoch}  avg_loss:{avg_loss:.4f}\n{'='*60}\n")

        cur_scheduler.step()

        # ── 验证 & 保存 ──────────────────────────────────────────
        if epoch % 10 == 0:
            ema.eval()
            use_tta_eval = in_finetune
            avg_p, avg_s = evaluate(
                ema.shadow, val_loader, device, SAVE_DIR,
                tile=256, overlap=64,
                tag=f'val_ep{epoch}', use_tta=use_tta_eval)
            model.train()

            save_data = {
                'epoch':              epoch,
                'best_psnr':          best_psnr,
                'model':              model.state_dict(),
                'ema':                ema.shadow.state_dict(),
                'opt':                cur_optimizer.state_dict(),
                'sched':              cur_scheduler.state_dict(),
                'history':            history,
                'in_finetune':        in_finetune,
                'train_kernel_pairs': TRAIN_KERNEL_PAIRS,
            }
            ckpt_save_path = os.path.join(
                SAVE_DIR, f'ckpt_ep{epoch:03d}_P{avg_p:.2f}.pth')
            torch.save(save_data, ckpt_save_path)
            recent_ckpts.append(ckpt_save_path)

            while len(recent_ckpts) > 2:
                oldest = recent_ckpts.pop(0)
                if os.path.exists(oldest) and 'best' not in oldest:
                    os.remove(oldest)
                    print(f"  [清理] {os.path.basename(oldest)}")

            if avg_p > best_psnr:
                best_psnr = avg_p
                best_path = os.path.join(
                    SAVE_DIR, f'best_P{avg_p:.2f}_ep{epoch:03d}.pth')
                save_data['best_psnr'] = best_psnr
                torch.save(save_data, best_path)
                print(f"  [best] PSNR={avg_p:.2f} dB → {best_path}")

            print(f"  [ckpt] ep{epoch:03d}  "
                  f"PSNR={avg_p:.2f}  (best={best_psnr:.2f})")

    # ── 训练结束 ──────────────────────────────────────────────────
    print("\n训练完成！最终验证集评估 (EMA + TTA)...")
    ema.eval()
    evaluate(ema.shadow, val_loader, device, SAVE_DIR,
             tile=256, overlap=64, tag='final_val', use_tta=True)
    plot_history(history, SAVE_DIR)

    print("\n测试集最终评估 (EMA + TTA)...")
    evaluate(ema.shadow, test_loader, device, SAVE_DIR,
             tile=256, overlap=64, tag='test_final', use_tta=True)
    print(f"\n所有文件保存至: {SAVE_DIR}")

# =============================================================================
# torchrun 入口
# =============================================================================
if __name__ == '__main__':
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
    main()