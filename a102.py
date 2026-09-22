"""
Transformer-Based Attention Framework for
Noise Reduction and Detail Preservation in Low-Dose CT Imaging — V5-Fixed-CT-MultiKernel-A103
=============================================================================
基于 A102 的修复版本

【A103 改动（相对 A102）】
  FIX-A  GRAD_LOSS_MAX: 0.15 → 0.05（主因修复，避免与 SSIM/Charb 争优化方向）
  FIX-B  验证 & 保存频率: 每 10 epoch → 每 5 epoch（更细粒度监控）
  FIX-C  只保留最近 3 个非-best ckpt（与原逻辑一致，但配合 FIX-B 更及时清理）
  FIX-D  Resume 逻辑: 优先识别 ep240 附近 best ckpt，跳过已完成的 240 轮直接续训
  FIX-E  Fine-tune LR: 5e-6 → 2e-5，给模型从局部最优爬出的空间
  FIX-F  多核训练加权采样: QD301/FD301 采样权重 ×2，缓解验证域分布漂移
  FIX-G  finetune criterion: lg 权重从 GRAD_LOSS_MAX 显式设为 0.05
  FIX-H  主训练阶段 epoch 155-300 新增一次 LR warm restart（ep240 重置到 3e-5）

【保留 A102 全部内容】
  MOD-1~6 全部保留
  FIX-1~9 全部保留
  UPG-1~13 全部保留
  CHANGE-1~8 全部保留

数据集划分（10个患者）:
  训练集 (7): L067, L096, L109, L143, L192, L286, L291
  验证集 (2): L310, L333
  测试集 (1): L506

核配对映射 (训练使用全部 4 核):
  QD301mm ↔ FD301mm  (采样权重 ×2)
  QD303mm ↔ FD303mm  (采样权重 ×1)
  QD451mm ↔ FD451mm  (采样权重 ×1)
  QD453mm ↔ FD453mm  (采样权重 ×1)

验证/测试使用 QD301mm/FD301mm（保持历史可比性）
"""

import os, random, math, certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from torchvision import models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk
from copy import deepcopy


# =============================================================================
# 数据集划分 & 多核配置
# =============================================================================
TRAIN_PATIENTS = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291']
VAL_PATIENTS   = ['L310', 'L333']
TEST_PATIENTS  = ['L506']

# 训练使用全部 4 种重建核配对
# FIX-F: 采样权重列表与 TRAIN_KERNEL_PAIRS 对应
TRAIN_KERNEL_PAIRS = [
    ('QD301mm', 'FD301mm'),
    ('QD303mm', 'FD303mm'),
    ('QD451mm', 'FD451mm'),
    ('QD453mm', 'FD453mm'),
]
# QD301/FD301 验证域采样权重 ×2，其余 ×1
KERNEL_SAMPLE_WEIGHTS = [2, 1, 1, 1]

# 验证/测试保持单核（历史可比性）
VAL_KERNEL_PAIRS  = [('QD301mm', 'FD301mm')]
TEST_KERNEL_PAIRS = [('QD301mm', 'FD301mm')]


# =============================================================================
# Fine-tune 阶段超参
# =============================================================================
FINETUNE_START   = 300
FINETUNE_EPOCHS  = 150
FINETUNE_LR      = 2e-5        # FIX-E: 5e-6 → 2e-5
FINETUNE_ETA_MIN = 1e-7
FINETUNE_T0      = 20
FINETUNE_T_MULT  = 2


# =============================================================================
# FIX-A: GradientConsistencyLoss 渐进系数上限修正
# =============================================================================
GRAD_LOSS_START = 100
GRAD_LOSS_END   = 300
GRAD_LOSS_MAX   = 0.05          # FIX-A: 0.15 → 0.05


def get_grad_loss_weight(epoch):
    if epoch < GRAD_LOSS_START:
        return 0.0
    if epoch >= GRAD_LOSS_END:
        return GRAD_LOSS_MAX
    t = (epoch - GRAD_LOSS_START) / max(1, GRAD_LOSS_END - GRAD_LOSS_START)
    return GRAD_LOSS_MAX * t


# =============================================================================
# FIX-H: 主训练阶段 epoch 240 附近的 LR warm restart
# =============================================================================
WARM_RESTART_EPOCH = 240        # 在此 epoch 将 LR 重置到 WARM_RESTART_LR
WARM_RESTART_LR    = 3e-5       # 重置目标 LR（相对 BASE_LR=1e-4 的倍率 = 0.30）
WARM_RESTART_DECAY = 30         # 重置后余弦衰减到 eta_min 所需的 epoch 数


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
# UPG-6: DropPath
# =============================================================================
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        survival = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.bernoulli(
            torch.full(shape, survival, device=x.device, dtype=x.dtype)
        ) / survival
        return x * noise


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
    def __init__(self, in_ch=1, bc=96, growth=32):
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
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.ws        = ws
        self.shift     = shift
        self.norm1     = nn.LayerNorm(dim)
        self.norm2     = nn.LayerNorm(dim)
        hid            = int(dim * mlp_ratio)
        self.mlp       = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, dim), nn.Dropout(drop))
        self.attn      = WindowAttention(dim, ws, num_heads, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

        x = sc + self.drop_path(x.view(B, H*W, C) - sc)
        x = x.view(B, H*W, C) if x.ndim == 4 else x
        return x + self.drop_path(self.mlp(self.norm2(x)))


# =============================================================================
# DualScaleBlock
# =============================================================================
class DualScaleBlock(nn.Module):
    def __init__(self, dim, num_heads, ws=8, shift=False,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.local_attn = SwinBlock(dim, num_heads, ws, shift,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path)
        dil            = 4
        self.glob_dw   = nn.Conv2d(dim, dim, 3, padding=dil, dilation=dil,
                                   groups=dim, bias=False)
        self.glob_pw   = nn.Conv2d(dim, dim, 1, bias=False)
        self.glob_norm = nn.LayerNorm(dim)
        self.gate      = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.out_norm  = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        B, L, C = x.shape
        xl  = self.local_attn(x, H, W)
        x2d = x.transpose(1, 2).view(B, C, H, W)
        xg  = self.glob_pw(self.glob_dw(x2d)).flatten(2).transpose(1, 2)
        xg  = self.glob_norm(xg + x)
        gate = self.gate(torch.cat([xl, xg], dim=-1))
        out  = gate * xl + (1 - gate) * xg
        return self.out_norm(x + self.drop_path(out - x))


# =============================================================================
# DCAT（双向交叉注意力）
# =============================================================================
class DCAT(nn.Module):
    def __init__(self, dim, pool_grid=8):
        super().__init__()
        self.pool_grid = pool_grid
        nh             = safe_heads(dim, target_div=dim // 8)
        self.nh        = nh
        self.scale     = (dim // nh) ** -0.5

        self.norm_q1   = nn.LayerNorm(dim)
        self.norm_k1   = nn.LayerNorm(dim)
        self.norm_q2   = nn.LayerNorm(dim)
        self.norm_k2   = nn.LayerNorm(dim)

        self.q1 = nn.Linear(dim, dim, bias=False)
        self.k1 = nn.Linear(dim, dim, bias=False)
        self.v1 = nn.Linear(dim, dim, bias=False)
        self.q2 = nn.Linear(dim, dim, bias=False)
        self.k2 = nn.Linear(dim, dim, bias=False)
        self.v2 = nn.Linear(dim, dim, bias=False)

        self.gate     = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, -2.0)

        self.out_norm = nn.LayerNorm(dim)
        self.o1       = nn.Linear(dim, dim, bias=False)
        self.o2       = nn.Linear(dim, dim, bias=False)

    def _cross_attn(self, q_src, kv_src, q_lin, k_lin, v_lin,
                    nq_ln, nk_ln, H, W):
        B, Lq, C = q_src.shape
        g = min(self.pool_grid, H, W)
        kv_2d   = kv_src.transpose(1, 2).view(B, C, H, W)
        kv_pool = F.adaptive_avg_pool2d(kv_2d, (g, g))
        kv_flat = kv_pool.flatten(2).transpose(1, 2)

        nh, D = self.nh, C // self.nh
        q = q_lin(nq_ln(q_src)).view(B, Lq,  nh, D).transpose(1, 2)
        k = k_lin(nk_ln(kv_flat)).view(B, g*g, nh, D).transpose(1, 2)
        v = v_lin(kv_flat).view(B, g*g, nh, D).transpose(1, 2)

        attn = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, Lq, C)

    def forward(self, x, skip, H=None, W=None):
        B, Lq, C = x.shape
        H, W = _infer_hw(Lq, H, W)

        out1 = self.o1(self._cross_attn(
            x, skip, self.q1, self.k1, self.v1,
            self.norm_q1, self.norm_k1, H, W))

        skip_sg = skip.detach()
        out2 = self.o2(self._cross_attn(
            skip_sg, x, self.q2, self.k2, self.v2,
            self.norm_q2, self.norm_k2, H, W))

        gate = self.gate(torch.cat([out1, out2], dim=-1))
        return self.out_norm(x + gate * out1 + (1 - gate) * out2)


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
                 depth=6, ws=8, drop=0., attn_drop=0.,
                 drop_path_rate=0.2):
        super().__init__()
        self.dim        = dim
        self.train_grid = train_grid
        self.pos        = nn.Parameter(
            torch.zeros(1, train_grid*train_grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            DualScaleBlock(dim, num_heads, ws,
                           shift=(i % 2 == 1),
                           drop=drop, attn_drop=attn_drop,
                           drop_path=dpr[i])
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
                 ws=8, depth=2, drop=0., attn_drop=0.,
                 drop_path_rate=0.1):
        super().__init__()
        self.expand    = PatchExpand(in_dim, out_dim)
        self.skip_proj = nn.Linear(skip_dim, out_dim, bias=False) \
                         if skip_dim != out_dim else nn.Identity()
        self.dcat      = DCAT(out_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks    = nn.ModuleList([
            DualScaleBlock(out_dim, num_heads, ws,
                           shift=(i % 2 == 1),
                           drop=drop, attn_drop=attn_drop,
                           drop_path=dpr[i])
            for i in range(depth)])
        self.csg = CSG(out_dim)

    def forward(self, x, skip, H_in, W_in):
        x, H, W = self.expand(x, H_in, W_in)

        B, C_s, Hs, Ws = skip.shape
        if Hs != H or Ws != W:
            skip = F.interpolate(skip.float(), (H, W),
                                 mode='bilinear', align_corners=False)
        skip_t = self.skip_proj(skip.flatten(2).transpose(1, 2))

        x = self.dcat(x, skip_t, H=H, W=W)
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
        self.hfe = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=2, dilation=2, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, in_ch, 1))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta  = nn.Parameter(torch.tensor(0.05))

    def forward(self, tokens, H, W, x_in):
        B, L, C = tokens.shape
        feat   = self.norm(tokens).transpose(1, 2).view(B, C, H, W)
        main   = self.main(feat)
        lp     = F.avg_pool2d(x_in, 3, stride=1, padding=1)
        hp     = x_in - lp
        detail = self.detail(hp)
        hfe    = self.hfe(hp)
        alpha  = self.alpha.clamp(0, 1)
        beta   = self.beta.clamp(0, 0.5)
        return (x_in + main + alpha * detail + beta * hfe).clamp(-1., 1.)


# =============================================================================
# AuxHead
# =============================================================================
class AuxHead(nn.Module):
    def __init__(self, dim, in_ch=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim,    dim // 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 2, in_ch, 1),
        )

    def forward(self, tokens, H, W, x_in):
        B, L, C = tokens.shape
        feat = self.norm(tokens).transpose(1, 2).view(B, C, H, W)
        out  = self.proj(feat)
        _, _, Hin, Win = x_in.shape
        if out.shape[-2:] != (Hin, Win):
            out = F.interpolate(out.float(), (Hin, Win),
                                mode='bilinear', align_corners=False)
        return (x_in + out).clamp(-1., 1.)


# =============================================================================
# Full Model
# =============================================================================
class LDCTDenoiserV5Fixed(nn.Module):
    def __init__(self, in_ch=1, bc=96, growth=32,
                 bot_depth=6, bot_heads=8, ws=8,
                 dec_depths=(3, 3, 2, 2),
                 drop=0., attn_drop=0.,
                 drop_path_rate=0.2):
        super().__init__()
        self.encoder    = RRDBEncoder(in_ch, bc, growth)
        self.bottleneck = TransformerBottleneck(
            dim=bc*8, train_grid=16, num_heads=bot_heads,
            depth=bot_depth, ws=ws, drop=drop, attn_drop=attn_drop,
            drop_path_rate=drop_path_rate)

        self.dec4 = DecoderStage(bc*8, bc*8, bc*4, 8, ws, dec_depths[0],
                                 drop, attn_drop, drop_path_rate * 0.5)
        self.dec3 = DecoderStage(bc*4, bc*4, bc*2, 8, ws, dec_depths[1],
                                 drop, attn_drop, drop_path_rate * 0.5)
        self.dec2 = DecoderStage(bc*2, bc*2, bc,   8, ws, dec_depths[2],
                                 drop, attn_drop, drop_path_rate * 0.3)
        self.dec1 = DecoderStage(bc,   bc,   bc,   8, ws, dec_depths[3],
                                 drop, attn_drop, drop_path_rate * 0.3)
        self.head = MultiScaleHead(bc, in_ch)

        self.aux_head3 = AuxHead(bc*2, in_ch)
        self.aux_head2 = AuxHead(bc,   in_ch)

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

        t, h, w     = self.dec4(t, skips[3], bH, bW)
        t3, h3, w3  = self.dec3(t, skips[2], h,  w)
        t2, h2, w2  = self.dec2(t3, skips[1], h3, w3)
        t1, h1, w1  = self.dec1(t2, skips[0], h2, w2)
        main_out    = self.head(t1, h1, w1, x)

        if self.training:
            aux3 = self.aux_head3(t3, h3, w3, x)
            aux2 = self.aux_head2(t2, h2, w2, x)
            return main_out, aux3, aux2

        return main_out


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
    def __init__(self, phase_weight=0.1):
        super().__init__()
        self.phase_weight = phase_weight

    def forward(self, pred, target):
        fp = torch.fft.rfft2(pred.float(),   norm='ortho')
        ft = torch.fft.rfft2(target.float(), norm='ortho')

        loss_amp = F.l1_loss(fp.abs(), ft.abs())

        if self.phase_weight > 0:
            amp_mask = (ft.abs() > ft.abs().mean()).float()
            phase_diff = torch.angle(fp) - torch.angle(ft)
            phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
            loss_phase = (phase_diff.abs() * amp_mask).mean()
            return loss_amp + self.phase_weight * loss_phase

        return loss_amp


class HaarWaveletLoss(nn.Module):
    @staticmethod
    def _dwt(x):
        a = x[:, :, 0::2, 0::2]; b = x[:, :, 1::2, 0::2]
        c = x[:, :, 0::2, 1::2]; d = x[:, :, 1::2, 1::2]
        ll = (a + b + c + d) * 0.25
        lh = (a - b + c - d) * 0.25
        hl = (a + b - c - d) * 0.25
        hh = (a - b - c + d) * 0.25
        return ll, lh, hl, hh

    def forward(self, pred, target, levels=3):
        hf_weights = [0.5, 1.0, 1.5]
        loss = 0.
        p, t = pred, target

        for lvl in range(levels):
            if p.shape[-1] < 2 or p.shape[-2] < 2:
                break

            ll_p, lhp, hlp, hhp = self._dwt(p)
            ll_t, lht, hlt, hht = self._dwt(t)

            w = hf_weights[min(lvl, len(hf_weights) - 1)]
            loss += w * (F.l1_loss(lhp, lht) +
                         F.l1_loss(hlp, hlt) +
                         0.5 * F.l1_loss(hhp, hht))

            p, t = ll_p, ll_t

        total_w = sum(hf_weights[:levels]) * 1.5
        return loss / total_w


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


class GradientConsistencyLoss(nn.Module):
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
        gxp = F.conv2d(p, sx, padding=1)
        gyp = F.conv2d(p, sy, padding=1)
        gxt = F.conv2d(t, sx, padding=1)
        gyt = F.conv2d(t, sy, padding=1)

        mag_p = torch.sqrt(gxp**2 + gyp**2 + 1e-6)
        mag_t = torch.sqrt(gxt**2 + gyt**2 + 1e-6)
        loss_mag  = F.l1_loss(mag_p, mag_t)
        edge_mask = (mag_t > mag_t.mean()).float()
        cos_sim   = (gxp * gxt + gyp * gyt) / (mag_p * mag_t + 1e-6)
        loss_dir  = ((1 - cos_sim) * edge_mask).sum() / (edge_mask.sum() + 1e-6)
        return loss_mag + 0.3 * loss_dir


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()

        self.feats, self.hooks = {}, []
        for name in ('layer1', 'layer2', 'layer3', 'layer4'):
            h = dict(resnet.named_modules())[name].register_forward_hook(
                lambda m, i, o, n=name: self.feats.update({n: o}))
            self.hooks.append(h)

        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = resnet.to(dev)
        for p in self.resnet.parameters():
            p.requires_grad_(False)

        self.layers = ('layer1', 'layer2', 'layer3', 'layer4')
        self.layer_weights = {
            'layer1': 0.5,
            'layer2': 1.0,
            'layer3': 1.0,
            'layer4': 0.3,
        }
        self.weight_sum = sum(self.layer_weights.values())

        self.register_buffer('mean',
            torch.tensor([0.2135, 0.2135, 0.2135]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.1932, 0.1932, 0.1932]).view(1, 3, 1, 1))

    def forward(self, x, y):
        def prep(t):
            t3   = t.float().repeat(1, 3, 1, 1)
            mean = self.mean.to(t3.device)
            std  = self.std.to(t3.device)
            return ((t3 + 1) / 2 - mean) / std

        self.feats.clear(); self.resnet(prep(x)); xf = self.feats.copy()
        self.feats.clear(); self.resnet(prep(y)); yf = dict(self.feats)

        loss = sum(
            self.layer_weights[l] * F.mse_loss(xf[l], yf[l])
            for l in self.layers
        ) / self.weight_sum
        return loss

    def __del__(self):
        for h in self.hooks:
            try: h.remove()
            except: pass


class CompositeLoss(nn.Module):
    def __init__(self, lc=1.0, ls=1.5, lp=0.05, lf=0.15,
                 lw=0.4, ln=0.3, le=0.3, lg=0.0):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ssim  = SSIMLoss(data_range=2.0, levels=3)
        self.perc  = PerceptualLoss()
        self.freq  = FrequencyLoss(phase_weight=0.1)
        self.wav   = HaarWaveletLoss()
        self.noise = NoiseAwareLoss()
        self.edge  = EdgeAwareLoss()
        self.grad  = GradientConsistencyLoss()
        self.lc, self.ls, self.lp = lc, ls, lp
        self.lf, self.lw, self.ln = lf, lw, ln
        self.le,  self.lg         = le, lg

    def forward(self, pred, target, ldct=None, lg_override=None):
        lc = self.charb(pred, target)
        ls = self.ssim(pred,  target)
        lp = self.perc(pred,  target)
        lf = self.freq(pred,  target)
        lw = self.wav(pred,   target)
        le = self.edge(pred,  target)
        lg = self.grad(pred,  target)
        ln = self.noise(pred, target, ldct) if ldct is not None \
             else torch.zeros(1, device=pred.device)

        lg_w = lg_override if lg_override is not None else self.lg

        total = (self.lc*lc + self.ls*ls + self.lp*lp +
                 self.lf*lf + self.lw*lw + self.ln*ln +
                 self.le*le + lg_w*lg)
        subs  = dict(charb=lc.item(), ssim=ls.item(), perc=lp.item(),
                     freq=lf.item(),  wav=lw.item(),  edge=le.item(),
                     grad=lg.item(),
                     noise=ln.item() if ldct is not None else 0.)
        return total, subs


# =============================================================================
# Dataset（多核版 + FIX-F 加权采样支持）
# =============================================================================
class LDCTDataset(Dataset):
    def __init__(self, dataset_root, patients, kernel_pairs,
                 mode='train', patch_size=128,
                 kernel_sample_weights=None):
        """
        kernel_sample_weights: list[int/float], 与 kernel_pairs 等长。
            训练集使用 WeightedRandomSampler 时传入；
            验证/测试集无需传入。
        """
        self.pairs = []
        self.patch_size = patch_size
        self.is_train = (mode == 'train')
        # 记录每条样本对应的核权重（用于 WeightedRandomSampler）
        self.sample_weights = []
        avail = []

        for ki, (qd_name, fd_name) in enumerate(kernel_pairs):
            w = (kernel_sample_weights[ki]
                 if kernel_sample_weights is not None else 1)
            qd_root = os.path.join(dataset_root, qd_name)
            fd_root = os.path.join(dataset_root, fd_name)
            for p in patients:
                ld_dir = os.path.join(qd_root, p)
                nd_dir = os.path.join(fd_root, p)
                if not (os.path.isdir(ld_dir) and os.path.isdir(nd_dir)):
                    continue
                lf = sorted(f for f in os.listdir(ld_dir) if f.endswith('.npy'))
                nf = sorted(f for f in os.listdir(nd_dir) if f.endswith('.npy'))
                n = min(len(lf), len(nf))
                if n == 0:
                    continue
                avail.append(f"{qd_name}↔{fd_name}/{p}")
                for i in range(n):
                    self.pairs.append((
                        os.path.join(ld_dir, lf[i]),
                        os.path.join(nd_dir, nf[i]),
                    ))
                    self.sample_weights.append(float(w))

        print(f"[{mode.upper()}] {len(avail)} 个(核/患者)组合  "
              f"共 {len(self.pairs)} 个切片对")

    def set_patch_size(self, ps):
        self.patch_size = ps

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _load_and_normalize(path):
        arr = np.load(path).astype(np.float32)
        arr = np.squeeze(arr)

        if arr.ndim == 1:
            n = arr.size
            h = int(math.isqrt(n))
            if h * h == n:
                arr = arr.reshape(h, h)
            else:
                for h in range(int(math.sqrt(n)), 1, -1):
                    if n % h == 0:
                        arr = arr.reshape(h, n // h)
                        break
                else:
                    raise ValueError(
                        f"无法 reshape 1D 数组: {path}, shape={arr.shape}")

        if arr.ndim != 2:
            raise ValueError(
                f"加载后不是 2D: {path}, shape={arr.shape}")

        arr = np.clip(arr, -1000, 1500)
        arr = (arr + 1000) / 2500 * 2 - 1   # [-1, 1]
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx):
        lp, np_ = self.pairs[idx]

        ld = self._load_and_normalize(lp)
        nd = self._load_and_normalize(np_)

        if self.is_train and self.patch_size > 0:
            _, h, w = ld.shape
            ps = (min(self.patch_size, h, w) // 16) * 16
            ps = max(ps, 64)
            i = random.randint(0, h - ps)
            j = random.randint(0, w - ps)
            ld = TF.crop(ld, i, j, ps, ps)
            nd = TF.crop(nd, i, j, ps, ps)

            if random.random() > 0.5:
                ld, nd = TF.hflip(ld), TF.hflip(nd)
            if random.random() > 0.5:
                ld, nd = TF.vflip(ld), TF.vflip(nd)
            k = random.randint(0, 3)
            if k:
                ld, nd = torch.rot90(ld, k, [1, 2]), torch.rot90(nd, k, [1, 2])

            if random.random() > 0.7:
                f = random.uniform(0.95, 1.05)
                ld = (ld * f).clamp(-1, 1)
                nd = (nd * f).clamp(-1, 1)

            if random.random() > 0.8:
                extra_noise = torch.randn_like(ld) * random.uniform(0.005, 0.02)
                ld = (ld + extra_noise).clamp(-1, 1)

            if random.random() > 0.85:
                gamma = random.uniform(0.9, 1.1)
                ld_01 = ((ld + 1) / 2).clamp(0, 1)
                ld_01 = torch.pow(ld_01, gamma)
                ld = (ld_01 * 2 - 1).clamp(-1, 1)

            if random.random() > 0.9:
                _, h2, w2 = ld.shape
                mh = random.randint(h2 // 16, h2 // 8)
                mw = random.randint(w2 // 16, w2 // 8)
                y0 = random.randint(0, h2 - mh)
                x0 = random.randint(0, w2 - mw)
                lmean = ld[:, y0:y0 + mh, x0:x0 + mw].mean()
                ld[:, y0:y0 + mh, x0:x0 + mw] = lmean

        return ld, nd


# =============================================================================
# Batch size & Patch size 调度
# =============================================================================
def get_batch_size(patch_size):
    if patch_size <= 128: return 48
    if patch_size <= 192: return 24
    if patch_size <= 256: return 12
    if patch_size <= 320: return 8
    return 6


def get_patch_size(epoch):
    if epoch <= 20:   return 128
    if epoch <= 60:   return 192
    if epoch <= 120:  return 256
    if epoch <= 240:  return 320
    return 384


# =============================================================================
# TTA & Patch Inference
# =============================================================================
@torch.no_grad()
def _tta_8(model, img, tile, overlap, device):
    fwds = [
        (lambda x: x,                              lambda x: x),
        (lambda x: torch.flip(x, [-1]),            lambda x: torch.flip(x, [-1])),
        (lambda x: torch.flip(x, [-2]),            lambda x: torch.flip(x, [-2])),
        (lambda x: torch.rot90(x, 1, [-2,-1]),     lambda x: torch.rot90(x,-1,[-2,-1])),
        (lambda x: torch.rot90(x, 2, [-2,-1]),     lambda x: torch.rot90(x,-2,[-2,-1])),
        (lambda x: torch.rot90(x, 3, [-2,-1]),     lambda x: torch.rot90(x,-3,[-2,-1])),
        (lambda x: torch.flip(torch.rot90(x, 1,[-2,-1]),[-1]),
         lambda x: torch.rot90(torch.flip(x,[-1]),-1,[-2,-1])),
        (lambda x: torch.flip(torch.rot90(x, 1,[-2,-1]),[-2]),
         lambda x: torch.rot90(torch.flip(x,[-2]),-1,[-2,-1])),
    ]
    results = []
    for fwd, inv in fwds:
        out = patch_inference(model, fwd(img), tile, overlap, device)
        results.append(inv(out))
    return torch.stack(results, 0).mean(0).clamp(-1., 1.)


@torch.no_grad()
def tta_inference(model, img, tile=256, overlap=64, device='cuda'):
    brightness_scales = [0.98, 1.0, 1.02]
    all_results = []
    for scale in brightness_scales:
        img_aug = (img * scale).clamp(-1., 1.)
        result  = _tta_8(model, img_aug, tile, overlap, device)
        all_results.append((result / scale).clamp(-1., 1.))
    return torch.stack(all_results, 0).mean(0).clamp(-1., 1.)


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
# 显存预检
# =============================================================================
def vram_check(model, device):
    configs = [(128, 8), (192, 4), (256, 2)]
    model.train()
    print("\n[显存预检 — A100 80GB]")
    for patch, batch in configs:
        torch.cuda.empty_cache()
        try:
            dummy = torch.randn(batch, 1, patch, patch).to(device)
            with torch.amp.autocast('cuda'):
                out = model(dummy)
                if isinstance(out, tuple):
                    out = out[0]
            loss = out.mean()
            loss.backward()
            used = torch.cuda.memory_reserved(device) / 1e9
            status = "✅" if used < 60 else ("⚠️ 偏高" if used < 72 else "❌ 危险")
            print(f"  patch={patch:<4} batch={batch}  峰值≈{used:.1f}GB  {status}")
            del dummy, out, loss
        except RuntimeError as e:
            print(f"  patch={patch:<4} batch={batch}  ❌ OOM: {e}")
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    print()


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
        print(f"  [evaluate] TTA 已启用（24 种变换：8 几何 × 3 亮度）")

    for i, (ldct, ndct) in enumerate(loader):
        pred   = infer_fn(model, ldct, tile, overlap, device)
        ph     = to_hu(pred[0, 0])
        gh     = to_hu(ndct[0, 0])
        lh     = to_hu(ldct[0, 0])
        ps, ss = compute_metrics(ph, gh)
        psnrs.append(ps); ssims.append(ss)

        if i % 350 == 0:
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
    tta_tag = "+TTA24" if use_tta else ""
    print(f"\n{'='*60}")
    print(f"[{tag.upper()}]{tta_tag}  N={len(psnrs)}")
    print(f"  PSNR : {avg_p:.2f} ± {std_p:.2f} dB")
    print(f"  SSIM : {avg_s:.4f} ± {std_s:.4f}")
    print(f"{'='*60}\n")

    with open(os.path.join(save_dir, f"{tag}_results.txt"),
              'w', encoding='utf-8') as f:
        f.write(f"=== V5-Fixed-CT-MultiKernel-A103 {tag.upper()}{tta_tag} ===\n\n")
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
# Fine-tune 阶段组件（FIX-E: LR 提升; FIX-G: lg=0.05 显式设定）
# =============================================================================
def build_finetune_components(model, device):
    optimizer_ft = optim.AdamW(
        model.parameters(),
        lr=FINETUNE_LR,           # FIX-E: 2e-5
        weight_decay=0,
        betas=(0.9, 0.999),
    )
    scheduler_ft = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_ft,
        T_0=FINETUNE_T0,
        T_mult=FINETUNE_T_MULT,
        eta_min=FINETUNE_ETA_MIN,
    )
    criterion_ft = CompositeLoss(
        lc=1.2, ls=1.2, lp=0.05,   # FIX-I: lc 0.8→1.2, ls 2.0→1.2，把优化方向更多拉向像素级PSNR
        lf=0.05, lw=0.1, ln=0.2,
        le=0.5,
        lg=GRAD_LOSS_MAX,         # FIX-G: 显式 0.05（而非 A102 的 0.15）
    ).to(device)
    return optimizer_ft, scheduler_ft, criterion_ft

# =============================================================================
# 主训练函数
# =============================================================================
def main():
    assert torch.cuda.is_available(), "未检测到 CUDA 设备"
    device = torch.device('cuda:0')

    print(f"[V5-Fixed-CT-MultiKernel-A103] Device: {device}")
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"显存: {total_vram:.1f} GB")

    # ── 路径 ──────────────────────────────────────────────────────
    BASE_DIR     = "/home/user/joshua82/LDCT_Project"
    SAVE_DIR     = f"{BASE_DIR}/output/checkpoints/a102"
    DATASET_ROOT = f"{BASE_DIR}/dataset"
    NUM_WORKERS  = 8

    NUM_EPOCHS = FINETUNE_START + FINETUNE_EPOCHS   # 450

    # ── 验证 / 保存频率 ────────────────────────────────────────────
    # FIX-B: 每 5 epoch 做一次验证和保存
    VAL_EVERY = 10

    print(f"\n数据集划分:")
    print(f"  训练集: {TRAIN_PATIENTS}")
    print(f"  验证集: {VAL_PATIENTS}")
    print(f"  测试集: {TEST_PATIENTS}")
    print(f"\n【多核配置】")
    print(f"  训练核配对: {TRAIN_KERNEL_PAIRS}")
    print(f"  核采样权重: {KERNEL_SAMPLE_WEIGHTS}  (FIX-F)")
    print(f"  验证/测试核: {VAL_KERNEL_PAIRS}")
    print(f"\n【A103 修复列表】")
    print(f"  FIX-A  GRAD_LOSS_MAX: 0.15 → {GRAD_LOSS_MAX}")
    print(f"  FIX-B  验证频率: 每 {VAL_EVERY} epoch")
    print(f"  FIX-C  保留最近 3 个非-best ckpt")
    print(f"  FIX-D  Resume 自动跳过已完成轮次（优先 ep240 best）")
    print(f"  FIX-E  Finetune LR: {FINETUNE_LR:.0e}")
    print(f"  FIX-F  多核采样权重: QD301×2 其余×1")
    print(f"  FIX-G  Finetune criterion lg={GRAD_LOSS_MAX}")
    print(f"  FIX-H  主训练 ep{WARM_RESTART_EPOCH} LR warm restart → {WARM_RESTART_LR:.0e}")
    print(f"\n训练阶段: epoch 1–{FINETUNE_START}（主训练）"
          f" + epoch {FINETUNE_START+1}–{NUM_EPOCHS}（fine-tune）\n")

    # ── Dataset（多核 + 加权采样） ─────────────────────────────────
    train_ds = LDCTDataset(
        DATASET_ROOT, TRAIN_PATIENTS, TRAIN_KERNEL_PAIRS,
        mode='train', patch_size=128,
        kernel_sample_weights=KERNEL_SAMPLE_WEIGHTS)   # FIX-F
    val_ds   = LDCTDataset(DATASET_ROOT, VAL_PATIENTS, VAL_KERNEL_PAIRS,
                           mode='val',   patch_size=0)
    test_ds  = LDCTDataset(DATASET_ROOT, TEST_PATIENTS, TEST_KERNEL_PAIRS,
                           mode='test',  patch_size=0)

    # FIX-F: 构建 WeightedRandomSampler
    def make_train_loader(dataset, batch_size, num_workers):
        weights = torch.tensor(dataset.sample_weights, dtype=torch.float32)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )

    def make_loader(dataset, batch_size, num_workers, shuffle=True):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )

    cur_ps    = get_patch_size(1)
    cur_batch = get_batch_size(cur_ps)
    train_loader = make_train_loader(train_ds, cur_batch, NUM_WORKERS)
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"[初始] patch={cur_ps}  batch={cur_batch}")

    # ── 模型 ──────────────────────────────────────────────────────
    model = LDCTDenoiserV5Fixed(
        in_ch=1, bc=96, growth=32,
        bot_depth=6, bot_heads=8, ws=8,
        dec_depths=(3, 3, 2, 2),
        drop=0.05, attn_drop=0.05,
        drop_path_rate=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[V5-Fixed-CT-MultiKernel-A103] Parameters: {n_params:.1f}M")
    vram_check(model, device)

    ema = EMA(model, decay=0.9999)

    # ── 主训练阶段优化器 ──────────────────────────────────────────
    BASE_LR  = 1e-4
    WARMUP   = 5
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR,
                            weight_decay=1e-4,
                            betas=(0.9, 0.999))

    # FIX-H: ep240 warm restart 注入到 lr_lambda
    def lr_lambda(ep):
        # FIX-H: warm restart 段（ep240 开始，持续 WARM_RESTART_DECAY epoch）
        if WARM_RESTART_EPOCH <= ep < WARM_RESTART_EPOCH + WARM_RESTART_DECAY:
            t = (ep - WARM_RESTART_EPOCH) / WARM_RESTART_DECAY
            peak = WARM_RESTART_LR / BASE_LR
            return peak * 0.5 * (1 + math.cos(math.pi * t))

        if ep < WARMUP:
            return 1e-7 / BASE_LR + (1.0 - 1e-7 / BASE_LR) * (ep + 1) / WARMUP
        if 80 <= ep < 85:
            return 0.05 + 0.25 * (ep - 80) / 5
        if 150 <= ep < 155:
            return 0.02 + 0.08 * (ep - 150) / 5
        stages = [
            (WARMUP,  30,  1.00),
            (30,      80,  0.60),
            (85,     150,  0.30),
            (155, WARM_RESTART_EPOCH, 0.20),
            (WARM_RESTART_EPOCH + WARM_RESTART_DECAY, FINETUNE_START, 0.05),
        ]
        for s_start, s_end, peak in stages:
            if s_start <= ep < s_end:
                t = (ep - s_start) / max(1, s_end - s_start)
                return peak * 0.5 * (1 + math.cos(math.pi * t))
        return 0.01

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = CompositeLoss(
        lc=1.0, ls=0.8, lp=0.05,
        lf=0.1,  lw=0.2, ln=0.4,
        le=0.2,  lg=0.0,
    ).to(device)

    edge_criterion = EdgeAwareLoss().to(device)

    scaler = torch.amp.GradScaler('cuda')

    os.makedirs(SAVE_DIR, exist_ok=True)
    history = {'train_loss': []}
    best_psnr = 0.
    start_epoch = 1
    recent_ckpts = []
    recent_best_ckpts = []  # FIX-J: 最近3个 best checkpoint（不含全场最高）
    global_best_path = None  # FIX-J: 历史最高 PSNR 对应的 checkpoint 路径，永久保留
    global_best_psnr = 0.

    optimizer_ft = None
    scheduler_ft = None
    criterion_ft = None
    in_finetune  = False

    # ── FIX-D: Checkpoint 恢复（优先找 a102 的 ep240 best，再找 a103 自身） ──
    def find_latest_ckpt(primary_dir, fallback_dir=None):
        """
        优先在 primary_dir 中搜索；找不到则在 fallback_dir 中搜索。
        返回 (path, epoch, psnr)
        """
        import glob, re

        def _scan(d):
            if d is None or not os.path.isdir(d):
                return []
            return (glob.glob(os.path.join(d, 'ckpt_ep*.pth')) +
                    glob.glob(os.path.join(d, 'best_P*.pth')))

        def _parse(path):
            base = os.path.basename(path)
            em = re.search(r'ep(\d+)', base)
            pm = re.search(r'_P([\d.]+?)(?=_|\.pth)', base)
            ep   = int(em.group(1)) if em else 0
            psnr = float(pm.group(1)) if pm else 0.0
            return ep, psnr

        for d in [primary_dir, fallback_dir]:
            files = _scan(d)
            if not files:
                continue
            best = max(files, key=lambda p: _parse(p)[0])
            ep, psnr = _parse(best)
            return best, ep, psnr

        return None, 0, 0.0

    # 先找 a103 自身，再 fallback 到 a102
    A102_DIR = f"{BASE_DIR}/output/checkpoints/a102"
    ckpt_path, ckpt_epoch, ckpt_psnr = find_latest_ckpt(SAVE_DIR, A102_DIR)

    if ckpt_path is not None:
        print(f"\n[Resume] {os.path.basename(ckpt_path)}  "
              f"ep={ckpt_epoch}  PSNR={ckpt_psnr:.2f}")
        ckpt  = torch.load(ckpt_path, map_location=device)
        state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}

        cur_state = model.state_dict()
        filtered  = {k: v for k, v in state.items() if k in cur_state
                     and cur_state[k].shape == v.shape}
        missing   = [k for k in cur_state if k not in filtered]
        if missing:
            print(f"  [Resume] 新增层（随机初始化）: "
                  f"{missing[:5]}{'...' if len(missing)>5 else ''}")
        cur_state.update(filtered)
        model.load_state_dict(cur_state)

        if 'ema' in ckpt:
            ema_state = ema.shadow.state_dict()
            filtered_ema = {k: v for k, v in ckpt['ema'].items()
                            if k in ema_state and ema_state[k].shape == v.shape}
            ema_state.update(filtered_ema)
            ema.shadow.load_state_dict(ema_state)

        history     = ckpt.get('history', history)
        best_psnr   = ckpt.get('best_psnr', ckpt_psnr)
        start_epoch = ckpt_epoch + 1
        import glob as _glob
        _existing_best = _glob.glob(os.path.join(SAVE_DIR, 'best_P*.pth'))
        if _existing_best:
            def _extract_psnr(p):
                m = re.search(r'best_P([\d.]+?)_ep', os.path.basename(p))
                return float(m.group(1)) if m else 0.0

            def _extract_ep(p):
                m = re.search(r'_ep(\d+)', os.path.basename(p))
                return int(m.group(1)) if m else 0

            _existing_best.sort(key=_extract_ep)  # 按 epoch 顺序排列
            recent_best_ckpts = _existing_best[-3:]  # 最近3个
            _global = max(_existing_best, key=_extract_psnr)
            global_best_path = _global
            global_best_psnr = _extract_psnr(_global)
            print(f"  [Resume] 重建 best ckpt 清单：最近3个={len(recent_best_ckpts)}个，"
                  f"全场最高={os.path.basename(global_best_path)}(PSNR={global_best_psnr:.2f})")
        in_finetune = ckpt.get('in_finetune', False)
        if in_finetune:
            print(f"  [Resume] 检测到 fine-tune 阶段，恢复 fine-tune 组件")
            optimizer_ft, scheduler_ft, criterion_ft = \
                build_finetune_components(model, device)
            ema.decay = 0.99995
            if 'opt' in ckpt:
                try:
                    optimizer_ft.load_state_dict(ckpt['opt'])
                    print(f"  [Resume] optimizer_ft 状态已恢复")
                except Exception as e:
                    print(f"  [Resume] optimizer_ft 状态不兼容，重新初始化: {e}")
            if 'sched' in ckpt:
                try:
                    scheduler_ft.load_state_dict(ckpt['sched'])
                    print(f"  [Resume] scheduler_ft 状态已恢复")
                except Exception as e:
                    print(f"  [Resume] scheduler_ft 状态不兼容，重新初始化: {e}")
        else:
            if 'opt' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['opt'])
                    # FIX-D: 强制将 LR 调整到 start_epoch 对应的值（避免继承旧 LR）
                    for pg in optimizer.param_groups:
                        pg['lr'] = BASE_LR * lr_lambda(start_epoch)
                    print(f"  [Resume] optimizer 状态已恢复，"
                          f"LR 重置为 {BASE_LR * lr_lambda(start_epoch):.2e}")
                except Exception as e:
                    print(f"  [Resume] optimizer 状态不兼容，重新初始化: {e}")

            # 将 scheduler 步进到 start_epoch
            for _ in range(start_epoch - 1):
                scheduler.step()

        print(f"[Resume] 从 ep{ckpt_epoch} 恢复，"
              f"将从 ep{start_epoch} 继续\n")
    else:
        print("\n[Resume] 未找到 checkpoint，从头训练\n")

    # ── Training Loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        # ----------------------------------------------------------
        # 切换至 fine-tune 阶段
        # ----------------------------------------------------------
        if epoch == FINETUNE_START + 1 and not in_finetune:
            in_finetune  = True
            optimizer_ft, scheduler_ft, criterion_ft = \
                build_finetune_components(model, device)
            scaler  = torch.amp.GradScaler('cuda')
            ema.decay = 0.99995
            print(f"\n{'='*60}")
            print(f"[A103] 切换至 Fine-tune 阶段 (epoch {epoch})")
            print(f"  optimizer : AdamW  lr={FINETUNE_LR:.0e}  wd=0  (FIX-E)")
            print(f"  scheduler : CosineAnnealingWarmRestarts "
                  f"T_0={FINETUNE_T0}  T_mult={FINETUNE_T_MULT}")
            print(f"  EMA decay : {ema.decay}")
            print(f"  criterion lg={GRAD_LOSS_MAX}  (FIX-G)")
            print(f"{'='*60}\n")

        cur_optimizer = optimizer_ft if in_finetune else optimizer
        cur_scheduler = scheduler_ft if in_finetune else scheduler
        cur_criterion = criterion_ft if in_finetune else criterion

        # FIX-A: 主训练阶段 grad loss 权重上限为 0.05
        lg_weight = get_grad_loss_weight(epoch) if not in_finetune else None

        # ----------------------------------------------------------
        # Patch size / batch size 切换
        # ----------------------------------------------------------
        if in_finetune:
            target_ps    = 256
            target_batch = get_batch_size(target_ps)
        else:
            target_ps    = get_patch_size(epoch)
            target_batch = get_batch_size(target_ps)

        target_batch = max(target_batch, 1)

        if target_ps != cur_ps or target_batch != cur_batch:
            train_ds.set_patch_size(target_ps)
            train_loader = make_train_loader(train_ds, target_batch, NUM_WORKERS)
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
                outputs = model(ldct)
                main_pred, aux3_pred, aux2_pred = outputs

                loss_main, subs = cur_criterion(main_pred, ndct, ldct,
                                                lg_override=lg_weight)

                loss_aux3 = (CharbonnierLoss()(aux3_pred, ndct) +
                             0.5 * SSIMLoss()(aux3_pred, ndct) +
                             0.2 * edge_criterion(aux3_pred, ndct))

                loss_aux2 = (CharbonnierLoss()(aux2_pred, ndct) +
                             0.5 * SSIMLoss()(aux2_pred, ndct) +
                             0.1 * edge_criterion(aux2_pred, ndct))

                loss = loss_main + 0.3 * loss_aux3 + 0.15 * loss_aux2

            scaler.scale(loss).backward()
            scaler.unscale_(cur_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(cur_optimizer)
            scaler.update()
            ema.update(model)

            total += loss.item()
            for k, v in subs.items():
                subs_acc[k] = subs_acc.get(k, 0.) + v

            if (bi + 1) % 900 == 0:
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
                    f"ep{epoch:03d}{ft_tag}_b{bi+1:05d}_P{ps:.2f}_S{ss:.4f}.png"),
                    dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  [snap] ep{epoch}{ft_tag} b{bi+1}  "
                      f"PSNR:{ps:.2f}  SSIM:{ss:.4f}")

            if bi % 200 == 0:
                mem = torch.cuda.memory_reserved(device) / 1e9
                sub_str = "  ".join(f"{k}:{v:.4f}" for k, v in subs.items())
                lg_info = f"  lg_w:{lg_weight:.3f}" if lg_weight is not None else ""
                cur_lr = cur_optimizer.param_groups[0]['lr']
                print(f"  ep{epoch} [{bi}/{len(train_loader)}]  "
                      f"loss:{loss.item():.4f}  {sub_str}{lg_info}"
                      f"  lr:{cur_lr:.2e}  VRAM:{mem:.1f}GB")

        avg_loss = total / len(train_loader)
        history['train_loss'].append(avg_loss)
        for k in subs_acc:
            history.setdefault(k, []).append(subs_acc[k] / len(train_loader))
        print(f"{'='*60}\nEpoch {epoch}  avg_loss:{avg_loss:.4f}\n{'='*60}\n")

        cur_scheduler.step()

        # ── FIX-B: 每 VAL_EVERY epoch 验证 & 保存 ─────────────────
        if epoch % VAL_EVERY == 0:
            ema.eval()
            avg_p, avg_s = evaluate(
                ema.shadow, val_loader, device,
                SAVE_DIR, tile=256, overlap=64,
                tag=f'val_ep{epoch}',
                use_tta=False)
            if in_finetune:
                model.eval()
                avg_p_raw, avg_s_raw = evaluate(
                    model, val_loader, device, SAVE_DIR,
                    tile=256, overlap=64,
                    tag=f'val_raw_ep{epoch}',
                    use_tta=False)
                print(f"  [诊断] EMA vs Raw: EMA={avg_p:.4f}  Raw={avg_p_raw:.4f}  "
                      f"差值={avg_p - avg_p_raw:.4f}")
            model.train()

            save_data = {
                'epoch':       epoch,
                'best_psnr':   best_psnr,
                'model':       model.state_dict(),
                'ema':         ema.shadow.state_dict(),
                'opt':         cur_optimizer.state_dict(),
                'sched':       cur_scheduler.state_dict(),
                'history':     history,
                'in_finetune': in_finetune,
            }
            ckpt_save_path = os.path.join(
                SAVE_DIR, f'ckpt_ep{epoch:03d}_P{avg_p:.2f}.pth')
            torch.save(save_data, ckpt_save_path)
            recent_ckpts.append(ckpt_save_path)

            # FIX-C: 只保留最近 3 个非-best ckpt
            while len(recent_ckpts) > 3:
                oldest = recent_ckpts.pop(0)
                if os.path.exists(oldest) and 'best' not in oldest:
                    os.remove(oldest)
                    print(f"  [清理] {os.path.basename(oldest)}")

            if avg_p > best_psnr:
                best_psnr = avg_p
                best_save_path = os.path.join(
                    SAVE_DIR, f'best_P{avg_p:.2f}_ep{epoch:03d}.pth')
                save_data['best_psnr'] = best_psnr
                torch.save(save_data, best_save_path)
                print(f"  [best] PSNR={avg_p:.2f} dB → {best_save_path}")

                # FIX-J: 更新历史最高纪录（永久保留，不参与滚动清理）
                if avg_p > global_best_psnr:
                    global_best_psnr = avg_p
                    global_best_path = best_save_path

                # FIX-J: 维护最近3个 best ckpt 的滚动清理，跳过 global_best_path
                recent_best_ckpts.append(best_save_path)
                while len(recent_best_ckpts) > 3:
                    oldest_best = recent_best_ckpts.pop(0)
                    if oldest_best == global_best_path:
                        # 这个是全场最高，不能删——放回列表最前面继续占位保留
                        # （说明当前列表里 global best 恰好排在最旧的位置，
                        #  跳过它、改删下一个真正可以删的）
                        recent_best_ckpts.insert(0, oldest_best)
                        if len(recent_best_ckpts) > 3:
                            # 找列表里除 global_best_path 外最旧的一个来删
                            for cand in list(recent_best_ckpts):
                                if cand != global_best_path:
                                    recent_best_ckpts.remove(cand)
                                    if os.path.exists(cand):
                                        os.remove(cand)
                                        print(f"  [清理-best] {os.path.basename(cand)}"
                                              f"（保留全场最高: {os.path.basename(global_best_path)}）")
                                    break
                        break
                    else:
                        if os.path.exists(oldest_best):
                            os.remove(oldest_best)
                            print(f"  [清理-best] {os.path.basename(oldest_best)}")

    # ── 训练结束 ──────────────────────────────────────────────────
    print("\n训练完成！最终验证集评估 (EMA + TTA-24)...")
    ema.eval()
    evaluate(ema.shadow, val_loader, device, SAVE_DIR,
             tile=256, overlap=64, tag='final_val', use_tta=True)
    plot_history(history, SAVE_DIR)

    print("\n测试集最终评估 (EMA + TTA-24)...")
    evaluate(ema.shadow, test_loader, device, SAVE_DIR,
             tile=256, overlap=64, tag='test_final', use_tta=True)
    print(f"\n所有文件保存至: {SAVE_DIR}")


# =============================================================================
# 启动入口
# =============================================================================
if __name__ == '__main__':
    assert torch.cuda.is_available(), "需要至少 1 张 GPU"
    print(f"[V5-Fixed-CT-MultiKernel-A103] "
          f"检测到 {torch.cuda.device_count()} 张 GPU，使用 cuda:0")
    main()