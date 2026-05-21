"""
LDCT Denoiser V7 — bc=96 双 A100 80GB 稳定版
=============================================
目标:  PSNR > 47 dB  |  SSIM > 0.98
硬件:  2× A100 80 GB（NVLink DDP）

相对 V6-bc160 的核心变更:
  ✅ bc=96 (~95M 参数) — 匹配 7 患者数据集规模，防止过拟合
  ✅ growth=32 — 恢复原始值，bc=96 密集连接强度已充分
  ✅ bot_depth=6 — 补偿 bc 缩小的全局建模容量
  ✅ dec_depths=(2,2,2,2) — 对称解码器，小数据集不需要更深
  ✅ lr=2e-4 — 模型变小，可用更大 lr 加速收敛
  ✅ warmup=10 epoch — 参数量小，预热需求降低
  ✅ drop_path_rate=0.10 — 过拟合风险降低，正则化适当放松
  ✅ layer_scale_init=1e-4 — 小模型不需要极保守的初始化
  ✅ weight_decay=5e-4 — 小数据集需要更强 L2 正则
  ✅ batch/卡(128px)=20~24 — VRAM 充裕后大 batch 稳定梯度
  ✅ 梯度裁剪改为固定阈值 1.0 — 防止初期动态阈值过松
  ✅ 感知 loss 延迟到 epoch 15 启用 — 防初期方向错误
  ✅ loss 各项 NaN 单独检测 — 定位具体哪个 loss 发散
  ✅ cudnn.benchmark=True — bc=96 模型小，可开 benchmark 加速

数据集划分（10个患者）:
  训练集 (7): L067, L096, L109, L143, L192, L286, L291
  验证集 (2): L310, L333
  测试集 (1): L506

调试建议（见文末注释）
"""

# ─────────────────────────────────────────────────────────────────
# 0. 环境 & 导入
# ─────────────────────────────────────────────────────────────────
import os, sys, math, random, argparse
from copy import deepcopy

import certifi
os.environ['SSL_CERT_FILE']            = certifi.where()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import torchvision.transforms.functional as TF
from torchvision import models

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity  as ssim_sk

# ─────────────────────────────────────────────────────────────────
# 1. 全局超参 — bc=96 甜点配置
# ─────────────────────────────────────────────────────────────────
TRAIN_PATIENTS = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291']
VAL_PATIENTS   = ['L310', 'L333']
TEST_PATIENTS  = ['L506']

BC              = 96           # ✅ 甜点: ~95M 参数，匹配 7 患者数据规模
GROWTH          = 32           # 恢复原始值，bc=96 已充分
BOT_DIM         = BC * 8       # 768
BOT_DEPTH       = 6            # ↑ 补偿 bc 缩小的全局建模容量
DROP_PATH_RATE  = 0.10         # ↓ 过拟合风险低，放松正则
LAYER_SCALE_INI = 1e-2         # ↑↑ 提高到 1e-3：gnorm>100 说明 1e-4 仍太小
                               # LayerScale gamma 梯度 ∝ 1/gamma，gamma 越小梯度越大
USE_GRAD_CKPT   = True         # 依然开启，节省显存给更大 batch
USE_BF16        = True         # A100 原生 bf16

# ─────────────────────────────────────────────────────────────────
# 2. 工具函数
# ─────────────────────────────────────────────────────────────────
def GN(ch):
    """GroupNorm，自适应 groups，最大 32"""
    g = min(32, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


def safe_heads(dim, prefer=8):
    """找能整除 dim 且最接近 prefer 的 head 数"""
    nh = prefer
    while nh > 1 and dim % nh != 0:
        nh -= 1
    return nh


def _infer_hw(L, H=None, W=None):
    if H is not None and W is not None:
        return H, W
    h = int(math.isqrt(L))
    while h > 1 and L % h != 0:
        h -= 1
    return h, L // h


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*a):
    if is_main():
        print(*a, flush=True)


def smart_imshow(ax, img_hu, title=''):
    vm, vs = float(np.mean(img_hu)), float(np.std(img_hu))
    vmin, vmax = (-160, 240) if -360 <= vm <= 440 and vs >= 1 \
                 else (vm - 2*vs - 1, vm + 2*vs + 1)
    ax.imshow(img_hu, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.axis('off')


# ─────────────────────────────────────────────────────────────────
# 3. LayerScale
# ─────────────────────────────────────────────────────────────────
class LayerScale(nn.Module):
    """
    可学习的逐通道缩放。
    bc=96 用 init=1e-4（比 bc=160 的 1e-5 宽松），
    小模型梯度天然稳定，不需要极保守的初始化。
    """
    def __init__(self, dim, init=LAYER_SCALE_INI):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x):
        if x.dim() == 3:
            return x * self.gamma
        return x * self.gamma.view(1, -1, 1, 1)


# ─────────────────────────────────────────────────────────────────
# 4. Stochastic Depth (DropPath)
# ─────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, device=x.device).bernoulli_(keep) / keep
        return x * noise


def build_drop_path_rates(total_blocks, max_rate=DROP_PATH_RATE):
    """线性增大的 drop_path_rate，浅层小、深层大"""
    return [max_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)]


# ─────────────────────────────────────────────────────────────────
# 5. EMA（自适应 decay）
# ─────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = deepcopy(
            model.module if hasattr(model, 'module') else model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def set_decay(self, d):
        self.decay = d

    @torch.no_grad()
    def update(self, model):
        m = model.module if hasattr(model, 'module') else model
        for s, p in zip(self.shadow.parameters(), m.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def eval_model(self):
        self.shadow.eval()
        return self.shadow


# ─────────────────────────────────────────────────────────────────
# 6. Lookahead
# ─────────────────────────────────────────────────────────────────
class Lookahead(optim.Optimizer):
    def __init__(self, base_opt, alpha=0.5, k=20):
        self.optimizer = base_opt
        self.alpha = alpha; self.k = k; self._step = 0
        self.slow = [p.clone().detach()
                     for pg in base_opt.param_groups for p in pg['params']]
        super().__init__(base_opt.param_groups, {})
        self._warmup_epochs = 0

    def set_warmup(self, n_warmup_steps):
        self._warmup_epochs = n_warmup_steps
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1
        if self._step > self._warmup_epochs and self._step % self.k == 0:
            idx = 0
            for pg in self.optimizer.param_groups:
                for p in pg['params']:
                    s = self.slow[idx]
                    s.add_(p.data - s, alpha=self.alpha)
                    p.data.copy_(s); idx += 1
        return loss

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {'base': self.optimizer.state_dict(),
                'slow': [s.clone() for s in self.slow],
                'step': self._step,'warmup': self._warmup_epochs}

    def load_state_dict(self, d):
        self.optimizer.load_state_dict(d['base'])
        self.slow  = [s.clone() for s in d['slow']]
        self._step = d['step']
        self._warmup_epochs = d.get('warmup', 0)


# ─────────────────────────────────────────────────────────────────
# 7. SimpleGate (NAFNet)
# ─────────────────────────────────────────────────────────────────
class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1 if x.dim() == 3 else 1)
        return a * b


# ─────────────────────────────────────────────────────────────────
# 8. RRDB Encoder
# ─────────────────────────────────────────────────────────────────
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth=GROWTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, growth, 3, padding=1, bias=False),
            GN(growth), nn.GELU())

    def forward(self, x):
        return torch.cat([x, self.net(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, ch, growth=GROWTH, n_layers=4):
        super().__init__()
        layers = []
        c = ch
        for _ in range(n_layers):
            layers.append(DenseLayer(c, growth))
            c += growth
        self.layers = nn.Sequential(*layers)
        self.proj   = nn.Conv2d(c, ch, 1, bias=False)
        self.norm   = GN(ch)

    def _fwd(self, x):
        return self.norm(self.proj(self.layers(x))) * 0.2 + x

    def forward(self, x):
        if USE_GRAD_CKPT and self.training:
            return grad_checkpoint(self._fwd, x, use_reentrant=False)
        return self._fwd(x)


class RRDB(nn.Module):
    def __init__(self, ch, growth=GROWTH):
        super().__init__()
        self.b1 = DenseBlock(ch, growth)
        self.b2 = DenseBlock(ch, growth)
        self.b3 = DenseBlock(ch, growth)

    def forward(self, x):
        return self.b3(self.b2(self.b1(x))) * 0.2 + x


class RRDBEncoder(nn.Module):
    def __init__(self, in_ch=1, bc=BC, growth=GROWTH):
        super().__init__()
        self.stem  = nn.Sequential(
            nn.Conv2d(in_ch, bc, 3, padding=1, bias=False), GN(bc), nn.GELU())
        self.enc1  = nn.Sequential(RRDB(bc, growth), RRDB(bc, growth))
        self.down1 = nn.Sequential(nn.Conv2d(bc, bc, 2, stride=2, bias=False), GN(bc))
        self.enc2  = nn.Sequential(
            nn.Conv2d(bc, bc*2, 1, bias=False), GN(bc*2), nn.GELU(),
            RRDB(bc*2, growth), RRDB(bc*2, growth))
        self.down2 = nn.Sequential(nn.Conv2d(bc*2, bc*2, 2, stride=2, bias=False), GN(bc*2))
        self.enc3  = nn.Sequential(
            nn.Conv2d(bc*2, bc*4, 1, bias=False), GN(bc*4), nn.GELU(),
            RRDB(bc*4, growth), RRDB(bc*4, growth))
        self.down3 = nn.Sequential(nn.Conv2d(bc*4, bc*4, 2, stride=2, bias=False), GN(bc*4))
        self.enc4  = nn.Sequential(
            nn.Conv2d(bc*4, bc*8, 1, bias=False), GN(bc*8), nn.GELU(),
            RRDB(bc*8, growth), RRDB(bc*8, growth))
        self.down4 = nn.Sequential(nn.Conv2d(bc*8, bc*8, 2, stride=2, bias=False), GN(bc*8))

    def forward(self, x):
        e1  = self.enc1(self.stem(x))
        e2  = self.enc2(self.down1(e1))
        e3  = self.enc3(self.down2(e2))
        e4  = self.enc4(self.down3(e3))
        bot = self.down4(e4)
        return [e1, e2, e3, e4], bot


# ─────────────────────────────────────────────────────────────────
# 9. Window Attention helpers
# ─────────────────────────────────────────────────────────────────
def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H//ws, ws, W//ws, ws, C)
    return x.permute(0,1,3,2,4,5).contiguous().view(-1, ws*ws, C)


def window_reverse(wins, ws, H, W):
    nH, nW = H//ws, W//ws
    B = wins.shape[0] // (nH * nW)
    x = wins.view(B, nH, nW, ws, ws, -1)
    return x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)


# ─────────────────────────────────────────────────────────────────
# 10. WindowAttention
# ─────────────────────────────────────────────────────────────────
class WindowAttention(nn.Module):
    def __init__(self, dim, ws, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        num_heads      = safe_heads(dim, num_heads)
        self.dim       = dim
        self.ws        = ws
        self.nh        = num_heads
        self.scale     = (dim // num_heads) ** -0.5

        self.rpb = nn.Parameter(torch.zeros((2*ws-1)**2, num_heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), indexing='ij'))
        cf  = coords.flatten(1)
        rel = (cf[:, :, None] - cf[:, None, :]).permute(1,2,0).contiguous()
        rel[:,:,0] += ws-1;  rel[:,:,1] += ws-1
        rel[:,:,0] *= 2*ws-1
        self.register_buffer('rpi', rel.sum(-1))

        self.qkv       = nn.Linear(dim, dim*3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        nh, D    = self.nh, C // self.nh
        qkv = self.qkv(x).reshape(B_, N, 3, nh, D).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2,-1)
        rpb  = self.rpb[self.rpi.view(-1)].view(N, N, nh)
        attn = attn + rpb.permute(2,0,1).unsqueeze(0)
        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_//nW, nW, nh, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, nh, N, N)
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        return self.proj_drop(self.proj(
               (attn @ v).transpose(1,2).reshape(B_, N, C)))


# ─────────────────────────────────────────────────────────────────
# 11. SwinBlock（LayerScale + DropPath）
# ─────────────────────────────────────────────────────────────────
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, ws=8, shift=False,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.ws    = ws
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        hid            = dim * 2
        self.ffn_up    = nn.Linear(dim, hid*2)
        self.ffn_gate  = SimpleGate()
        self.ffn_dn    = nn.Linear(hid, dim)
        self.drop      = nn.Dropout(drop)
        self.attn      = WindowAttention(dim, ws, num_heads, attn_drop, drop)
        self.ls1       = LayerScale(dim)
        self.ls2       = LayerScale(dim)
        self.dp        = DropPath(drop_path)

    def _build_mask(self, H, W, ws, shift, device):
        if not shift or min(H, W) <= ws:
            return None
        img = torch.zeros(1, H, W, 1, device=device)
        for i, hs in enumerate((slice(0,-ws), slice(-ws,-shift), slice(-shift,None))):
            for j, ws_ in enumerate((slice(0,-ws), slice(-ws,-shift), slice(-shift,None))):
                img[:, hs, ws_, :] = i*3+j
        mw   = window_partition(img, ws).squeeze(-1)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask!=0, -100.).masked_fill(mask==0, 0.)

    def forward(self, x, H, W):
        B, L, C = x.shape
        ws    = min(self.ws, H, W)
        shift = ws//2 if self.shift and min(H,W) > ws else 0

        sc = x
        x  = self.norm1(x).view(B, H, W, C)
        if shift: x = torch.roll(x, (-shift,-shift), (1,2))
        pb = (ws - H%ws)%ws;  pr = (ws - W%ws)%ws
        if pb or pr:
            x = F.pad(x.permute(0,3,1,2), (0,pr,0,pb)).permute(0,2,3,1)
        _, pH, pW, _ = x.shape
        mask = self._build_mask(pH, pW, ws, shift>0, x.device)
        xw   = window_partition(x, ws)
        xw   = self.attn(xw, mask)
        x    = window_reverse(xw.view(-1,ws,ws,C), ws, pH, pW)
        if pb or pr: x = x[:,:H,:W,:].contiguous()
        if shift:    x = torch.roll(x, (shift,shift), (1,2))
        x = sc + self.dp(self.ls1(x.view(B,L,C)))

        sc = x
        ffn_out = self.drop(self.ffn_dn(self.ffn_gate(self.ffn_up(self.norm2(x)))))
        x  = sc + self.dp(self.ls2(ffn_out))
        return x


# ─────────────────────────────────────────────────────────────────
# 12. DualScaleBlock
# ─────────────────────────────────────────────────────────────────
class DualScaleBlock(nn.Module):
    def __init__(self, dim, num_heads, ws=8, shift=False,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.local  = SwinBlock(dim, num_heads, ws, shift, drop, attn_drop, drop_path)
        self.dw     = nn.Conv2d(dim, dim, 3, padding=4, dilation=4, groups=dim, bias=False)
        self.pw     = nn.Conv2d(dim, dim, 1, bias=False)
        self.gnorm  = nn.LayerNorm(dim)
        self.gate   = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.onorm  = nn.LayerNorm(dim)
        self.ls     = LayerScale(dim)
        self.dp     = DropPath(drop_path)

    def _fwd(self, x, H, W):
        B, L, C = x.shape
        xl  = self.local(x, H, W)
        xg  = self.pw(self.dw(x.transpose(1,2).view(B,C,H,W))).flatten(2).transpose(1,2)
        xg  = self.gnorm(xg + x)
        g   = self.gate(torch.cat([xl, xg], -1))
        out = self.onorm(g*xl + (1-g)*xg)
        return x + self.dp(self.ls(out))

    def forward(self, x, H, W):
        if USE_GRAD_CKPT and self.training:
            return grad_checkpoint(self._fwd, x, H, W, use_reentrant=False)
        return self._fwd(x, H, W)

# ─────────────────────────────────────────────────────────────────
# 13. NAFBlock
# ─────────────────────────────────────────────────────────────────
class NAFBlock(nn.Module):
    def __init__(self, ch, ffn_ratio=2, drop_path=0.):
        super().__init__()
        hid        = int(ch * ffn_ratio)
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.dw    = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw1   = nn.Conv2d(ch, hid*2, 1, bias=False)
        self.gate  = SimpleGate()
        self.pw2   = nn.Conv2d(hid, ch, 1, bias=False)
        # ca 作用于 gate 之后的 hid 通道（不是 ch），修复通道不匹配 bug
        # 用 Linear 替代 Conv2d(1x1) 序列，彻底消除 AdaptiveAvgPool2d 的 stride 警告
        self.ca_fc = nn.Sequential(
            nn.Linear(hid, max(hid//8, 4)),
            nn.ReLU(True),
            nn.Linear(max(hid//8, 4), hid),
            nn.Sigmoid())
        self.ffn1  = nn.Linear(ch, hid*2)
        self.ffn2  = nn.Linear(hid, ch)
        self.ls1   = LayerScale(ch)
        self.ls2   = LayerScale(ch)
        self.dp    = DropPath(drop_path)

    def _fwd(self, x, H, W):
        B, L, C = x.shape
        x2d = self.norm1(x).transpose(1,2).contiguous().view(B,C,H,W)
        x2d = self.gate(self.pw1(self.dw(x2d)))             # [B, hid, H, W]
        # 通道注意力：全局平均 → Linear → Sigmoid，无 AdaptiveAvgPool2d stride 问题
        ca_w = self.ca_fc(x2d.mean(dim=[2,3]))               # [B, hid]
        x2d  = x2d * ca_w.unsqueeze(-1).unsqueeze(-1)        # [B, hid, H, W]
        x2d = self.pw2(x2d)
        sc1 = x + self.dp(self.ls1(x2d.flatten(2).transpose(1,2)))
        sc2 = sc1 + self.dp(self.ls2(self.ffn2(self.gate(self.ffn1(self.norm2(sc1))))))
        return sc2

    def forward(self, x, H=None, W=None):
        B, L, C = x.shape
        H, W = _infer_hw(L, H, W)
        if USE_GRAD_CKPT and self.training:
            return grad_checkpoint(self._fwd, x, H, W, use_reentrant=False)
        return self._fwd(x, H, W)


# ─────────────────────────────────────────────────────────────────
# 14. CSAS（跨尺度注意力采样）
# ─────────────────────────────────────────────────────────────────
class CSAS(nn.Module):
    def __init__(self, dim, pool_grid=8):
        super().__init__()
        self.g  = pool_grid
        nh      = safe_heads(dim, 8)
        self.nh = nh
        self.sc = (dim // nh) ** -0.5
        self.nq = nn.LayerNorm(dim);  self.nk = nn.LayerNorm(dim)
        self.q  = nn.Linear(dim, dim, bias=False)
        self.k  = nn.Linear(dim, dim, bias=False)
        self.v  = nn.Linear(dim, dim, bias=False)
        self.o  = nn.Linear(dim, dim, bias=False)
        self.ls = LayerScale(dim)

    def forward(self, query, kv, H=None, W=None):
        B, Lq, C = query.shape
        H, W     = _infer_hw(Lq, H, W)
        g        = min(self.g, H, W)
        kv_p     = F.adaptive_avg_pool2d(
            kv.transpose(1,2).view(B,C,H,W), (g,g)).flatten(2).transpose(1,2)
        nh, D    = self.nh, C // self.nh
        q = self.q(self.nq(query)).view(B, Lq, nh, D).transpose(1,2)
        k = self.k(self.nk(kv_p )).view(B, g*g, nh, D).transpose(1,2)
        v = self.v(kv_p           ).view(B, g*g, nh, D).transpose(1,2)
        attn = torch.softmax((q*self.sc) @ k.transpose(-2,-1), dim=-1)
        out  = (attn @ v).transpose(1,2).reshape(B, Lq, C)
        return query + self.ls(self.o(out))


# ─────────────────────────────────────────────────────────────────
# 15. CSG（通道空间门控）
# ─────────────────────────────────────────────────────────────────
class CSG(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        rd           = max(dim//r, 4)
        self.ch_fc   = nn.Sequential(
            nn.Linear(dim, rd), nn.ReLU(True), nn.Linear(rd, dim), nn.Sigmoid())
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x, H, W):
        B, L, C = x.shape
        ch  = self.ch_fc(x.mean(1))
        x   = x * ch.unsqueeze(1)
        x2d = x.transpose(1,2).view(B,C,H,W)
        sp  = self.sp_conv(torch.cat(
            [x2d.mean(1,keepdim=True), x2d.max(1,keepdim=True).values], 1))
        return (x2d*sp).flatten(2).transpose(1,2)


# ─────────────────────────────────────────────────────────────────
# 16. TransformerBottleneck（bot_depth=6 补偿容量）
# ─────────────────────────────────────────────────────────────────
class TransformerBottleneck(nn.Module):
    def __init__(self, dim=BOT_DIM, train_grid=16, num_heads=8,
                 depth=BOT_DEPTH, ws=8, drop=0., attn_drop=0.,
                 drop_path_rates=None):
        super().__init__()
        self.dim        = dim
        self.train_grid = train_grid
        self.pos        = nn.Parameter(torch.zeros(1, train_grid**2, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        rates = drop_path_rates or [0.] * (depth + 2)
        num_heads = safe_heads(dim, num_heads)
        self.blocks = nn.ModuleList([
            DualScaleBlock(dim, num_heads, ws, shift=(i%2==1),
                           drop=drop, attn_drop=attn_drop, drop_path=rates[i])
            for i in range(depth)])
        self.naf1 = NAFBlock(dim, drop_path=rates[-2])
        self.naf2 = NAFBlock(dim, drop_path=rates[-1])
        self.norm = nn.LayerNorm(dim)

    def _get_pos(self, H, W):
        G = self.train_grid
        if H == G and W == G:
            return self.pos
        pe = self.pos.reshape(1, G, G, self.dim).permute(0,3,1,2)
        pe = F.interpolate(pe.float(), (H,W), mode='bilinear', align_corners=False)
        return pe.permute(0,2,3,1).reshape(1, H*W, self.dim).to(self.pos.dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1,2) + self._get_pos(H, W)
        for blk in self.blocks:
            if USE_GRAD_CKPT and self.training:
                t = grad_checkpoint(blk, t, H, W, use_reentrant=False)
            else:
                t = blk(t, H, W)
        t = self.naf1(t, H, W)
        t = self.naf2(t, H, W)
        return self.norm(t).transpose(1,2).view(B, C, H, W)


# ─────────────────────────────────────────────────────────────────
# 17. PatchExpand + DecoderStage
# ─────────────────────────────────────────────────────────────────
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
        x        = x.view(B, H, W, 2, 2, C).permute(0,1,3,2,4,5).contiguous()
        x        = self.norm(x.view(B, 2*H*2*W, C))
        return self.proj(x), 2*H, 2*W


class DecoderStage(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, num_heads=8,
                 ws=8, depth=2, drop=0., attn_drop=0., drop_path_rates=None):
        super().__init__()
        self.expand    = PatchExpand(in_dim, out_dim)
        self.skip_proj = nn.Linear(skip_dim, out_dim, bias=False) \
                         if skip_dim != out_dim else nn.Identity()
        self.csas      = CSAS(out_dim)
        num_heads      = safe_heads(out_dim, num_heads)
        rates          = drop_path_rates or [0.] * (depth + 1)
        self.blocks    = nn.ModuleList([
            DualScaleBlock(out_dim, num_heads, ws, shift=(i%2==1),
                           drop=drop, attn_drop=attn_drop, drop_path=rates[i])
            for i in range(depth)])
        self.naf = NAFBlock(out_dim, drop_path=rates[-1])
        self.csg = CSG(out_dim)

    def forward(self, x, skip, H_in, W_in):
        x, H, W = self.expand(x, H_in, W_in)
        B, C_s, Hs, Ws = skip.shape
        if Hs != H or Ws != W:
            skip = F.interpolate(skip.float(), (H,W), mode='bilinear', align_corners=False)
        skip_t = self.skip_proj(skip.flatten(2).transpose(1,2))
        x = self.csas(x, skip_t, H=H, W=W)
        for blk in self.blocks:
            if USE_GRAD_CKPT and self.training:
                x = grad_checkpoint(blk, x, H, W, use_reentrant=False)
            else:
                x = blk(x, H, W)
        x = self.naf(x, H, W)
        return self.csg(x, H, W), H, W


# ─────────────────────────────────────────────────────────────────
# 18. MultiScaleHead（多尺度细节增强）
# ─────────────────────────────────────────────────────────────────
class MultiScaleHead(nn.Module):
    def __init__(self, dim, in_ch=1):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.main  = nn.Sequential(
            nn.Conv2d(dim,    dim,    3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim,    dim//2, 1, bias=False), nn.GELU(),
            nn.Conv2d(dim//2, in_ch,  1))
        self.det1  = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(32, in_ch, 1))
        self.det2  = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=2, dilation=2, bias=False), nn.GELU(),
            nn.Conv2d(32, in_ch, 1))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta  = nn.Parameter(torch.tensor(0.05))
        self.gamma = nn.Parameter(torch.tensor(0.03))

    def forward(self, tokens, H, W, x_in):
        B, L, C = tokens.shape
        feat   = self.norm(tokens).transpose(1,2).view(B, C, H, W)
        main   = self.main(feat)
        lp     = F.avg_pool2d(x_in, 3, stride=1, padding=1)
        hp     = x_in - lp
        d1     = self.det1(hp)
        d2     = self.det2(hp)
        a = self.alpha.clamp(0,1)
        b = self.beta.clamp(0,1)
        g = self.gamma.clamp(0,1)
        return (x_in + main + a*d1 + b*d2 + g*hp).clamp(-1., 1.)


# ─────────────────────────────────────────────────────────────────
# 19. 完整模型 LDCTDenoiserV7 (bc=96)
# ─────────────────────────────────────────────────────────────────
class LDCTDenoiserV7(nn.Module):
    """
    bc=96, growth=32, bot_depth=6
    参数量约 95M，双 A100 80GB，每卡约 28~38GB
    """
    def __init__(self, in_ch=1, bc=BC, growth=GROWTH,
                 bot_depth=BOT_DEPTH, ws=8,
                 dec_depths=(2,2,2,2),
                 drop=0.05, attn_drop=0.05,
                 drop_path_rate=DROP_PATH_RATE):
        super().__init__()

        total_blks = bot_depth + 2 + sum(d+1 for d in dec_depths)
        all_rates  = build_drop_path_rates(total_blks, drop_path_rate)
        idx        = [0]
        def _next_rates(n):
            r = all_rates[idx[0]: idx[0]+n]; idx[0] += n; return r

        self.encoder    = RRDBEncoder(in_ch, bc, growth)
        self.bottleneck = TransformerBottleneck(
            dim=bc*8, train_grid=16, num_heads=8,   # bc=96 → dim=768, heads=8
            depth=bot_depth, ws=ws, drop=drop, attn_drop=attn_drop,
            drop_path_rates=_next_rates(bot_depth+2))

        self.dec4 = DecoderStage(bc*8, bc*8, bc*4, 8, ws, dec_depths[0],
                                 drop, attn_drop, _next_rates(dec_depths[0]+1))
        self.dec3 = DecoderStage(bc*4, bc*4, bc*2, 8, ws, dec_depths[1],
                                 drop, attn_drop, _next_rates(dec_depths[1]+1))
        self.dec2 = DecoderStage(bc*2, bc*2, bc,   8, ws, dec_depths[2],
                                 drop, attn_drop, _next_rates(dec_depths[2]+1))
        self.dec1 = DecoderStage(bc,   bc,   bc,   8, ws, dec_depths[3],
                                 drop, attn_drop, _next_rates(dec_depths[3]+1))
        self.head = MultiScaleHead(bc, in_ch)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight);  nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W  = x.shape
        skips, bot  = self.encoder(x)
        bot         = self.bottleneck(bot)
        bH, bW      = H//16, W//16
        t           = bot.flatten(2).transpose(1,2)
        t, h, w     = self.dec4(t, skips[3], bH, bW)
        t, h, w     = self.dec3(t, skips[2], h,  w)
        t, h, w     = self.dec2(t, skips[1], h,  w)
        t, h, w     = self.dec1(t, skips[0], h,  w)
        return self.head(t, h, w, x)


# ─────────────────────────────────────────────────────────────────
# 20. Loss Functions
# ─────────────────────────────────────────────────────────────────
class SSIMLoss(nn.Module):
    def __init__(self, ws=11, dr=2.0, levels=3):
        super().__init__()
        self.dr = dr; self.levels = levels; self.ws = ws
        g = torch.arange(ws, dtype=torch.float32) - ws//2
        g = torch.exp(-(g**2)/(2*1.5**2)); g /= g.sum()
        self.register_buffer('win', g.outer(g).unsqueeze(0).unsqueeze(0))

    def _ssim(self, x, y):
        C1, C2 = (0.01*self.dr)**2, (0.03*self.dr)**2
        p = self.ws//2;  w = self.win.to(x.device, x.dtype)
        mx  = F.conv2d(x,   w, padding=p);  my  = F.conv2d(y,   w, padding=p)
        mxx = F.conv2d(x*x, w, padding=p) - mx**2
        myy = F.conv2d(y*y, w, padding=p) - my**2
        mxy = F.conv2d(x*y, w, padding=p) - mx*my
        return ((2*mx*my+C1)*(2*mxy+C2)/((mx**2+my**2+C1)*(mxx+myy+C2))).mean()

    def forward(self, x, y):
        loss = 0.
        for i in range(self.levels):
            loss += 1. - self._ssim(x, y)
            if i < self.levels-1:
                x = F.avg_pool2d(x, 2); y = F.avg_pool2d(y, 2)
        return loss / self.levels


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps**2

    def forward(self, p, t):
        return torch.sqrt((p-t)**2 + self.eps2).mean()


class FreqFocalLoss(nn.Module):
    def __init__(self, gamma=2.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        fp = torch.fft.rfft2(pred.float(),   norm='ortho')
        ft = torch.fft.rfft2(target.float(), norm='ortho')
        diff = (fp - ft).abs()
        diff = diff.clamp(max=10.0)
        B, C, H, W2 = diff.shape
        yf = torch.fft.fftfreq(H,  device=diff.device).abs()
        xf = torch.fft.rfftfreq(H,  device=diff.device).abs()
        w  = (1 + self.gamma*(yf.view(H,1)+xf.view(1,W2)).clamp(0,1)).to(diff.dtype)
        return (diff * w).mean()


class HaarWaveletLoss(nn.Module):
    @staticmethod
    def _dwt(x):
        a=x[:,:,0::2,0::2]; b=x[:,:,1::2,0::2]
        c=x[:,:,0::2,1::2]; d=x[:,:,1::2,1::2]
        return ((a+b+c+d)*.25,(a-b+c-d)*.25,(a+b-c-d)*.25,(a-b-c+d)*.25)

    def forward(self, p, t):
        _,lhp,hlp,hhp = self._dwt(p)
        _,lht,hlt,hht = self._dwt(t)
        return (F.l1_loss(lhp,lht)+F.l1_loss(hlp,hlt)+.5*F.l1_loss(hhp,hht))/2.5


class NoiseAwareLoss(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.k = k

    def forward(self, pred, target, ldct):
        k, p = self.k, self.k//2
        mu  = F.avg_pool2d(ldct, k, stride=1, padding=p)
        var = (F.avg_pool2d(ldct**2, k, stride=1, padding=p)-mu**2).clamp(0)
        w   = (var/(var.mean()+1e-6)).clamp(0.5, 3.0)
        return (F.l1_loss(pred,target,reduction='none')*w).mean()


class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32).view(1,1,3,3)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx', sx); self.register_buffer('sy', sy)

    def forward(self, p, t):
        sx=self.sx.to(p.device,p.dtype); sy=self.sy.to(p.device,p.dtype)
        ep=torch.sqrt(F.conv2d(p,sx,padding=1)**2+F.conv2d(p,sy,padding=1)**2+1e-6)
        et=torch.sqrt(F.conv2d(t,sx,padding=1)**2+F.conv2d(t,sy,padding=1)**2+1e-6)
        return F.l1_loss(ep, et)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
        self.feats, self.hooks = {}, []
        for nm in ('layer1','layer2','layer3'):
            h = dict(resnet.named_modules())[nm].register_forward_hook(
                lambda m,i,o,n=nm: self.feats.update({n:o}))
            self.hooks.append(h)
        self.resnet = resnet
        for p in self.resnet.parameters(): p.requires_grad_(False)
        self.layers = ('layer1','layer2','layer3')
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def _prep(self, t):
        t3 = t.float().repeat(1,3,1,1)
        return ((t3+1)/2 - self.mean.to(t3.device)) / self.std.to(t3.device)

    def forward(self, x, y):
        self.feats.clear(); self.resnet(self._prep(x)); xf = dict(self.feats)
        self.feats.clear(); self.resnet(self._prep(y)); yf = dict(self.feats)
        return sum(F.mse_loss(xf[l],yf[l]) for l in self.layers)/3

    def __del__(self):
        for h in self.hooks:
            try: h.remove()
            except: pass


class CompositeLoss(nn.Module):
    """
    bc=96 loss 配置:
      - 感知 loss 延迟到 epoch 15 启用（防初期方向错误）
      - SSIM 权重 1.5 → 1.2（避免 SSIM 数值不稳定主导训练）
      - 频率 loss 权重略提升（补偿 bc=96 对高频细节的学习能力）
    """
    def __init__(self):
        super().__init__()
        self.charb  = CharbonnierLoss()
        self.ssim   = SSIMLoss(dr=2.0, levels=3)
        self.perc   = PerceptualLoss()
        self.freq   = FreqFocalLoss(gamma=2.5)
        self.wav    = HaarWaveletLoss()
        self.noise  = NoiseAwareLoss()
        self.edge   = EdgeAwareLoss()
        # bc=96 初始权重（极保守）：
        #   edge/freq 初期极小（0.01/0.02），随 epoch 线性升入
        #   实测 edge loss 初期 ~3.0，远大于 charb ~0.7，直接主导 gnorm>8
        #   延迟升入后 epoch 1 的 gnorm 可从 8+ 降至 1~2
        self.w      = dict(c=1.0, s=0.0, p=0.0, f=0.0, wv=0.0, n=0.0, e=0.0)

    def set_weights(self, **kw): self.w.update(kw)

    def forward(self, pred, target, ldct=None):
        w  = self.w

        # 逐项计算，单独检测 NaN（便于调试定位）
        lc = self.charb(pred, target)
        ls = self.ssim(pred,  target)
        lf = self.freq(pred,  target)
        lw = self.wav(pred,   target)
        le = self.edge(pred,  target)
        ln = self.noise(pred, target, ldct) if ldct is not None \
             else pred.new_zeros(1)

        # 感知 loss 单独保护：若 p_w=0 则跳过前向（节省算力）
        if w['p'] > 0:
            lp = self.perc(pred, target)
            if not torch.isfinite(lp):
                lp = pred.new_zeros(1)
        else:
            lp = pred.new_zeros(1)

        # 安全汇总：各项 NaN 替换为 0（不终止训练，但记录日志）
        def safe(t): return t if torch.isfinite(t) else t.new_zeros(1)
        lc, ls, lf, lw, le, ln = map(safe, [lc, ls, lf, lw, le, ln])

        total = (w['c']*lc + w['s']*ls + w['p']*lp +
                 w['f']*lf + w['wv']*lw + w['n']*ln + w['e']*le)

        subs = dict(
            charb = lc.item(), ssim  = ls.item(), perc  = lp.item(),
            freq  = lf.item(), wav   = lw.item(), edge  = le.item(),
            noise = ln.item() if ldct is not None else 0.)
        return total, subs

    def set_weights(self, **kw): self.w.update(kw)


def schedule_loss_weights(crit, epoch, total, current_psnr=0.0):
    c=1.0; s=0.0; p=0.0; f=0.0; wv=0.0; n=0.0; e=0.0

    # 先用纯 Charbonnier 把 PSNR 推过 30 dB
    if current_psnr < 27.0:
        crit.set_weights(c=1.0, s=0, p=0, f=0, wv=0, n=0, e=0)
        return

    # PSNR 30~35：引入 noise + freq
    if current_psnr >= 27.0:
        n = min(0.2, (current_psnr - 30) / 5 * 0.2)
        f = min(0.05, (current_psnr - 30) / 5 * 0.05)

    # PSNR 35+：引入 ssim
    if current_psnr >= 32.0:
        s = min(0.5, (current_psnr - 35) / 5 * 0.5)
        wv = min(0.15, (current_psnr - 35) / 5 * 0.15)

    # PSNR 40+：引入 edge
    if current_psnr >= 37.0:
        e = min(0.03, (current_psnr - 40) / 5 * 0.03)

    crit.set_weights(c=c, s=s, p=p, f=f, wv=wv, n=n, e=e)

# ─────────────────────────────────────────────────────────────────
# 21. Dataset
# ─────────────────────────────────────────────────────────────────
class LDCTDataset(Dataset):
    def __init__(self, ldct_root, ndct_root, patients, mode='train', patch_size=128,repeat=10):
        self.pairs = []
        avail = []
        for p in patients:
            ld = os.path.join(ldct_root, p)
            nd = os.path.join(ndct_root, p)
            if not (os.path.isdir(ld) and os.path.isdir(nd)):
                log(f"  [警告] {p} 目录不存在，跳过"); continue
            avail.append(p)
            lf = sorted(f for f in os.listdir(ld) if f.endswith('.npy'))
            nf = sorted(f for f in os.listdir(nd) if f.endswith('.npy'))
            for i in range(min(len(lf), len(nf))):
                self.pairs.append((os.path.join(ld,lf[i]), os.path.join(nd,nf[i])))
        self.patch_size = patch_size
        self.is_train   = (mode == 'train')
        self.repeat = repeat
        log(f"[{mode.upper()}] 患者={avail}  共 {len(self.pairs)} 切片对")

    def set_patch_size(self, ps): self.patch_size = ps

    def __len__(self): return len(self.pairs) * self.repeat

    def __getitem__(self, idx):
        lp, np_ = self.pairs[idx % len(self.pairs)]  # ← 唯一改动
        try:
            ld = np.clip(np.load(lp).astype(np.float32), -1000, 1500)
            nd = np.clip(np.load(np_).astype(np.float32), -1000, 1500)
        except Exception as e:
            log(f"  [DATA-ERR] {lp}: {e}")
            dummy = torch.zeros(1, 512, 512)
            return dummy, dummy

        ld = (ld + 1000) / 2500 * 2 - 1
        nd = (nd + 1000) / 2500 * 2 - 1
        ld = torch.from_numpy(ld)[None]
        nd = torch.from_numpy(nd)[None]

        if self.is_train and self.patch_size > 0:
            _, h, w = ld.shape
            ps = max((min(self.patch_size, h, w) // 16) * 16, 64)
            i = random.randint(0, h - ps)
            j = random.randint(0, w - ps)
            ld = TF.crop(ld, i, j, ps, ps)
            nd = TF.crop(nd, i, j, ps, ps)
            if random.random() > .5:
                ld, nd = TF.hflip(ld), TF.hflip(nd)
            if random.random() > .5:
                ld, nd = TF.vflip(ld), TF.vflip(nd)
            k = random.randint(0, 3)
            if k:
                ld, nd = torch.rot90(ld, k, [1, 2]), torch.rot90(nd, k, [1, 2])
            if random.random() > .7:
                f = random.uniform(0.95, 1.05)
                ld = (ld * f).clamp(-1, 1)
                nd = (nd * f).clamp(-1, 1)
            if random.random() > .8:
                sigma = random.uniform(0.005, 0.02)
                ld = (ld + torch.randn_like(ld) * sigma).clamp(-1, 1)

        return ld, nd
# ─────────────────────────────────────────────────────────────────
# 22. DataLoader helpers
# ─────────────────────────────────────────────────────────────────
def get_patch_size(epoch, total=200):
    """
    bc=96 渐进 patch 策略（基于压测结果）:
      epoch   1~50:  128（大 batch=128/卡，稳定梯度）
      epoch  51~110: 192（batch=64/卡）
      epoch 111+:    256（batch=32/卡）
    注: 320px 在 bc=96 下 OOM，不使用
    """
    if epoch <= 50:   return 128
    if epoch <= 110:  return 192
    return 256


def get_batch_size(patch, world_size=2):
    """
    基于实测压测结果（A100 80GB × 2，bc=96，grad_ckpt=ON，bf16）：
      patch=128 → 128/卡（alloc=57.4GB）  全局 256
      patch=192 →  64/卡（alloc=64.5GB）  全局 128
      patch=256 →  32/卡（alloc=57.4GB）  全局  64
      patch=320 → OOM，不使用
    """
    if patch <= 128:  bs = 256
    elif patch <= 192: bs = 128
    else:              bs = 64
    return max(bs // world_size, 1)


def make_loader(ds, batch, nw, rank, world_size, train):
    sampler = DistributedSampler(ds, num_replicas=world_size,
                                 rank=rank, shuffle=train) if world_size>1 else None
    return DataLoader(ds, batch_size=batch,
                      shuffle=(sampler is None and train),
                      sampler=sampler, num_workers=nw,
                      pin_memory=True, drop_last=train,
                      persistent_workers=(nw>0),
                      prefetch_factor=2 if nw>0 else None)


# ─────────────────────────────────────────────────────────────────
# 23. Patch Inference（Hann 窗融合）
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def patch_inference(model, img, tile=256, overlap=64, device='cuda'):
    _, C, H, W = img.shape
    tile  = max((tile//16)*16, 64)
    step  = max(tile-overlap, 16)
    ph = (tile-H%tile)%tile if H%tile else 0
    pw = (tile-W%tile)%tile if W%tile else 0
    img_p = F.pad(img, (0,pw,0,ph), mode='reflect')
    oH, oW = img_p.shape[2], img_p.shape[3]

    out = torch.zeros(1,C,oH,oW); cnt = torch.zeros(1,C,oH,oW)
    hw  = torch.hann_window(tile, periodic=False)
    blend = (hw.unsqueeze(1)*hw.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    ys = list(range(0, max(oH-tile,0)+1, step)) or [0]
    xs = list(range(0, max(oW-tile,0)+1, step)) or [0]
    if ys[-1]+tile < oH: ys.append(oH-tile)
    if xs[-1]+tile < oW: xs.append(oW-tile)

    for y in ys:
        for x in xs:
            patch  = img_p[:,:,y:y+tile,x:x+tile].to(device)
            pred   = model(patch).cpu().float()
            b      = blend.expand_as(pred)
            out[:,:,y:y+tile,x:x+tile] += pred*b
            cnt[:,:,y:y+tile,x:x+tile] += b

    out = out / cnt.clamp(min=1e-8)
    return out[:,:,:H,:W].clamp(-1., 1.)


# ─────────────────────────────────────────────────────────────────
# 24. Metrics
# ─────────────────────────────────────────────────────────────────
def to_hu(t):
    return ((t.cpu().float().squeeze().numpy()+1)/2*2500-1000)


def compute_metrics(ph, gh):
    return (psnr_sk(gh, ph, data_range=2500),
            ssim_sk(gh, ph, data_range=2500))


# ─────────────────────────────────────────────────────────────────
# 25. 显存预检
# ─────────────────────────────────────────────────────────────────
# def vram_check(model, device, save_dir):
#     """
#     显存压测 — 结果写入 vram_check.txt，之后每次启动自动跳过。
#     手动删除 vram_check.txt 可强制重新压测。
#     """
#     cache_file = os.path.join(save_dir, 'vram_check.txt')
#
#     # ── 已有缓存则直接跳过 ──────────────────────────────────
#     if os.path.exists(cache_file):
#         log(f"[显存压测] 发现缓存 {cache_file}，跳过（删除该文件可重新压测）")
#         with open(cache_file) as f:
#             log(f.read())
#         return
#
#     patch_sizes = [128, 192, 256, 320]
#     max_safe_gb = 68.0
#     amp_dtype   = torch.bfloat16 if USE_BF16 else torch.float16
#
#     model.train()
#     log("\n[显存压测 — 首次运行，结果将缓存到磁盘]")
#     log(f"  目标: 每卡峰值 < {max_safe_gb:.0f}GB\n")
#
#     results = {}
#     lines   = ["=== vram_check 压测结果 ===\n"]
#
#     for patch in patch_sizes:
#         best_batch = 0
#         candidates = [2,4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128]
#         for batch in candidates:
#             torch.cuda.empty_cache()
#             torch.cuda.reset_peak_memory_stats(device)
#             try:
#                 dummy = torch.randn(batch, 1, patch, patch,
#                                     device=device, dtype=amp_dtype)
#                 with torch.amp.autocast('cuda', dtype=amp_dtype):
#                     out = model(dummy)
#                 out.mean().backward()
#                 peak = torch.cuda.max_memory_allocated(device) / 1e9
#                 resv = torch.cuda.memory_reserved(device) / 1e9
#                 model.zero_grad(set_to_none=True)
#                 del dummy, out
#                 torch.cuda.empty_cache()
#
#                 if peak < max_safe_gb:
#                     best_batch = batch
#                     log(f"  patch={patch:<4}  batch={batch:>4}  alloc={peak:.1f}GB  ✅")
#                 else:
#                     log(f"  patch={patch:<4}  batch={batch:>4}  alloc={peak:.1f}GB  ⚠️ 停止")
#                     break
#             except torch.cuda.OutOfMemoryError:
#                 model.zero_grad(set_to_none=True)
#                 torch.cuda.empty_cache()
#                 log(f"  patch={patch:<4}  batch={batch:>4}  ❌ OOM")
#                 break
#
#         results[patch] = best_batch
#         line = f"  patch={patch:<4}  max_batch/卡={best_batch:>4}  全局={best_batch*2:>4}"
#         lines.append(line)
#         log(f"  → patch={patch}  推荐 batch/卡 = {best_batch}\n")
#
#     summary = "\n".join(lines)
#     log("=" * 50)
#     log(summary)
#     log("=" * 50)
#
#     # ── 写入缓存 ────────────────────────────────────────────
#     os.makedirs(save_dir, exist_ok=True)
#     with open(cache_file, 'w') as f:
#         f.write(summary + "\n")
#     log(f"  结果已缓存至 {cache_file}，下次启动自动跳过压测\n")
#     return results


# ─────────────────────────────────────────────────────────────────
# 26. Training（梯度爆炸防护升级版）
# ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, ema, loader, optimizer, criterion,
                    scaler, device, epoch, history, save_dir,
                    rank=0, world_size=1):
    model.train()
    total, subs_acc = 0., {}
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    gnorm_ema = 1.0
    GRAD_CLIP = 1.0
    skip_count = 0
    nan_loss_count = 0
    oom_count = 0          # 累计 OOM 次数，超过阈值说明 batch 设置偏高
    amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16

    for bi, (ldct, ndct) in enumerate(loader):
        ldct, ndct = ldct.to(device), ndct.to(device)
        optimizer.zero_grad(set_to_none=True)

        # OOM fallback：按 2/4/8 分段重试，每次 OOM 记录一次
        # 若单 epoch OOM 超过 5 次，说明 get_batch_size 设置过激，建议调低
        oom_this_batch = False
        cur_ldct, cur_ndct = ldct, ndct
        for oom_retry in range(3):   # 最多减半 3 次（原 → 1/2 → 1/4 → 1/8）
            try:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    pred       = model(cur_ldct)
                    loss, subs = criterion(pred, cur_ndct, cur_ldct)
                break   # 成功则跳出重试循环
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                half = max(cur_ldct.shape[0] // 2, 1)
                if oom_retry == 0:
                    oom_count += 1
                    log(f"  [OOM×{oom_count}] ep{epoch} b{bi}  "
                        f"batch {cur_ldct.shape[0]}→{half}  "
                        f"(超过5次请调低 get_batch_size)")
                cur_ldct  = cur_ldct[:half]
                cur_ndct  = cur_ndct[:half]
                oom_this_batch = True
                if half == 1 and oom_retry == 2:
                    log(f"  [OOM-FATAL] batch=1 仍 OOM，跳过本 batch")
                    skip_count += 1
                    optimizer.zero_grad(set_to_none=True)
                    break
        else:
            # for-else: 3 次重试全部 OOM
            skip_count += 1
            continue

        # NaN/Inf 守卫（逐项检查 subs，定位发散源）
        if not torch.isfinite(loss):
            nan_loss_count += 1
            bad = [k for k,v in subs.items() if not math.isfinite(v)]
            log(f"  [SKIP-NaN] ep{epoch} b{bi}  loss={loss.item():.4f}"
                f"  发散项={bad}  累计={nan_loss_count}")
            optimizer.zero_grad(set_to_none=True)
            if nan_loss_count > 30:
                log("  [ABORT] NaN 过多，终止本 epoch！建议降低 lr 或检查数据")
                break
            continue
        nan_loss_count = 0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # 固定阈值梯度裁剪（bc=96 模型稳定，1.0 已充分）

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        gnorm_val = float(gnorm)
        gnorm_ema = 0.95 * gnorm_ema + 0.05 * min(gnorm_val, 10.0)
        skip_threshold = max(gnorm_ema * 15, 20.0)  # 自适应：15倍
        if gnorm_val > skip_threshold:
            log(f"  [SKIP-EXPLODE] ep{epoch} b{bi}  gnorm={gnorm_val:.1f}"
                f"  threshold={skip_threshold:.1f}，跳过")
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            continue

        if gnorm_val > 10.0 and is_main():
            log(f"  [WARN] ep{epoch} b{bi}  gnorm={gnorm_val:.2f} > 10.0")
        # 梯度异常预警（gnorm > 10 说明某层出现问题）


        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        total += loss.item()
        for k, v in subs.items():
            subs_acc[k] = subs_acc.get(k,0.) + v

        if is_main() and (bi+1) % 200 == 0:
            _save_snap(model, cur_ldct, cur_ndct, epoch, bi+1, save_dir, device, amp_dtype)

        if is_main() and bi % 50 == 0:
            mem     = torch.cuda.memory_allocated(device)/1e9   # 用 allocated 比 reserved 更精确
            sub_str = "  ".join(f"{k}:{v:.4f}" for k,v in subs.items())
            log(f"  ep{epoch} [{bi}/{len(loader)}]  "
                f"loss:{loss.item():.4f}  gnorm:{float(gnorm):.3f}  "
                f"{sub_str}  VRAM_alloc:{mem:.1f}GB")

    n   = max(len(loader) - skip_count, 1)
    avg = total / n
    if is_main():
        history['train_loss'].append(avg)
        for k in subs_acc:
            history.setdefault(k,[]).append(subs_acc[k]/n)
        oom_hint = (f"  ⚠️  本 epoch OOM {oom_count} 次，建议调低 get_batch_size"
                    if oom_count > 5 else
                    f"  OOM 次数: {oom_count}（正常）")
        log(f"{'='*60}\nEpoch {epoch}  avg_loss:{avg:.4f}\n{oom_hint}\n{'='*60}\n")
    return avg


def _save_snap(model, ldct, ndct, epoch, step, save_dir, device, dtype):
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        pred = model(ldct[[0]])
    model.train()
    ph = to_hu(pred[0,0]); gh = to_hu(ndct[0,0]); lh = to_hu(ldct[0,0])
    ps, ss = compute_metrics(ph, gh)
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    smart_imshow(ax[0],lh,"LDCT")
    smart_imshow(ax[1],ph,f"Denoised\nPSNR:{ps:.2f} | SSIM:{ss:.4f}")
    smart_imshow(ax[2],gh,"NDCT")
    plt.savefig(os.path.join(save_dir,
        f"ep{epoch:03d}_b{step:05d}_P{ps:.2f}_S{ss:.4f}.png"),
        dpi=150, bbox_inches='tight'); plt.close()
    log(f"  [snap] ep{epoch} b{step}  PSNR:{ps:.2f}  SSIM:{ss:.4f}")


# ─────────────────────────────────────────────────────────────────
# 27. Evaluation
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, save_dir, tile=256, overlap=64, tag='val',max_samples=None):
    os.makedirs(save_dir, exist_ok=True,)
    model.eval(); psnrs, ssims = [], []
    for i, (ldct, ndct) in enumerate(loader):
        if max_samples is not None and i >= max_samples:  # ← 加这两行
            break
        pred = patch_inference(model, ldct, tile, overlap, device)
        ph, gh, lh = to_hu(pred[0,0]), to_hu(ndct[0,0]), to_hu(ldct[0,0])
        ps, ss = compute_metrics(ph, gh)
        psnrs.append(ps); ssims.append(ss)
        if i % 200 == 0:
            fig, ax = plt.subplots(1,3,figsize=(15,5))
            smart_imshow(ax[0],lh,"LDCT")
            smart_imshow(ax[1],ph,f"Denoised\nPSNR:{ps:.2f} | SSIM:{ss:.4f}")
            smart_imshow(ax[2],gh,"NDCT")
            plt.savefig(os.path.join(save_dir,f"{tag}_case_{i:03d}.png"),
                        dpi=150, bbox_inches='tight'); plt.close()
            log(f"  [{tag}] {i}/{len(loader.dataset)}  PSNR:{ps:.2f}  SSIM:{ss:.4f}")

    ap=float(np.mean(psnrs)); sp=float(np.std(psnrs))
    as_=float(np.mean(ssims)); ss_=float(np.std(ssims))
    log(f"\n{'='*60}\n[{tag.upper()}]  N={len(psnrs)}")
    log(f"  PSNR : {ap:.2f} ± {sp:.2f} dB")
    log(f"  SSIM : {as_:.4f} ± {ss_:.4f}\n{'='*60}\n")
    with open(os.path.join(save_dir,f"{tag}_results.txt"),'w') as f:
        f.write(f"=== V7-bc96 {tag.upper()} ===\n\n")
        f.write(f"PSNR : {ap:.2f} ± {sp:.2f} dB\n")
        f.write(f"SSIM : {as_:.4f} ± {ss_:.4f}\n")
        f.write(f"N    : {len(psnrs)}\n")
    return ap, as_



def plot_history(history, save_dir):
    keys = [k for k in history if k != 'train_loss']
    n    = len(keys)+1
    fig, axes = plt.subplots(1,n,figsize=(4*n,4))
    if n == 1: axes = [axes]
    ep = range(1,len(history['train_loss'])+1)
    axes[0].plot(ep,history['train_loss'],'b-o',ms=3,lw=1.5)
    axes[0].set_title('Total loss'); axes[0].grid(True)
    for ax,k in zip(axes[1:],keys):
        ax.plot(ep,history[k],ms=3,lw=1.5); ax.set_title(k); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'training_curves.png'),dpi=150); plt.close()


# ─────────────────────────────────────────────────────────────────
# 28. 主训练函数（DDP worker）
# ─────────────────────────────────────────────────────────────────
import datetime
def get_repeat(patch):
    if patch <= 128:  return 10
    if patch <= 192:  return 6
    return 4
def main_worker(rank, world_size, cfg):
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30 分钟
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'
    dist.init_process_group('nccl', init_method='env://',
                            world_size=world_size, rank=rank,
                            timeout=datetime.timedelta(minutes=120))
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    BASE_DIR    = cfg['base_dir']
    SAVE_DIR    = cfg['save_dir']
    LDCT_ROOT   = cfg['ldct_root']
    NDCT_ROOT   = cfg['ndct_root']
    NUM_EPOCHS  = cfg.get('epochs', 200)
    WARMUP      = cfg.get('warmup', 10)
    NUM_WORKERS = cfg.get('num_workers', 6)

    os.makedirs(SAVE_DIR, exist_ok=True)
    if is_main():
        log(f"\n{'='*60}")
        log(f"V7-bc96  |  bc={BC}  growth={GROWTH}  bot_depth={BOT_DEPTH}")
        log(f"LayerScale={LAYER_SCALE_INI}  DropPath={DROP_PATH_RATE}")
        log(f"GradCkpt={USE_GRAD_CKPT}  bf16={USE_BF16}")
        log(f"世界大小: {world_size}  设备: {device}")
        log(f"{'='*60}\n")

    # ── 数据集 ────────────────────────────────────────────────
    train_ds = LDCTDataset(LDCT_ROOT, NDCT_ROOT, TRAIN_PATIENTS, 'train', 128, repeat=10)
    val_ds   = LDCTDataset(LDCT_ROOT, NDCT_ROOT, VAL_PATIENTS,   'val',   0)
    test_ds  = LDCTDataset(LDCT_ROOT, NDCT_ROOT, TEST_PATIENTS,  'test',  0)

    cur_ps    = get_patch_size(1, NUM_EPOCHS)
    cur_batch = get_batch_size(cur_ps, world_size)
    train_loader = make_loader(train_ds, cur_batch, NUM_WORKERS, rank, world_size, True)
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    log(f"[初始] patch={cur_ps}  batch_per_gpu={cur_batch}  全局batch={cur_batch*world_size}")

    # ── 模型 ──────────────────────────────────────────────────
    model = LDCTDenoiserV7(
        in_ch=1, bc=BC, growth=GROWTH,
        bot_depth=BOT_DEPTH, ws=8,
        dec_depths=(2,2,2,2),
        drop=0.05, attn_drop=0.05,
        drop_path_rate=DROP_PATH_RATE,
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # vram_check 只在 Rank 0 执行，之后 barrier 同步
    # if is_main():
    #     n_params = sum(p.numel() for p in model.parameters()) / 1e6
    #     log(f"Parameters: {n_params:.1f}M")
    #     vram_check(model.module, device, SAVE_DIR)
    # dist.barrier()  # 等 Rank 0 压测完再继续

    ema = EMA(model, decay=0.9999)

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'gamma' not in n],
         'lr': 2e-4},
        {'params': [p for n, p in model.named_parameters() if 'gamma' in n],
         'lr': 2e-5},
    ], weight_decay=5e-4, betas=(0.9, 0.95))

    # ── LR Scheduler（绑定底层 AdamW，Lookahead 包装后仍有效）──
    def lr_lambda(ep):
        if ep < WARMUP:
            return (ep + 1) / WARMUP
        restart_ep = int(NUM_EPOCHS * 0.6)
        if ep < restart_ep:
            t = (ep - WARMUP) / max(restart_ep - WARMUP, 1)
        else:
            t = (ep - restart_ep) / max(NUM_EPOCHS - restart_ep, 1)
        return max(0.5 * (1 + math.cos(math.pi * t)), 1e-2)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = CompositeLoss().to(device)
    scaler    = torch.amp.GradScaler('cuda', enabled=not USE_BF16)

    history    = {'train_loss': []}
    best_psnr  = 0.
    start_epoch = 1

    # ── 续训 ──────────────────────────────────────────────────
    best_path   = os.path.join(SAVE_DIR, 'best.pth')
    resume_path = os.path.join(SAVE_DIR, 'resume.pth')

    ckpt_to_load = None
    if os.path.exists(best_path):
        ckpt_to_load = best_path
        log(f"[Resume] 发现 best.pth，从最佳检查点继续训练")
    elif os.path.exists(resume_path):
        ckpt_to_load = resume_path
        log(f"[Resume] 未找到 best.pth，从 resume.pth 继续训练")

    if ckpt_to_load is not None:
        ck = torch.load(ckpt_to_load, map_location='cpu')
        model.module.load_state_dict(ck['model'])
        ema.shadow.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['opt'])
        scheduler.load_state_dict(ck['sched'])
        history     = ck.get('history', history)
        start_epoch = ck['epoch'] + 1
        best_psnr   = ck.get('best_psnr', 0.)

        if start_epoch > 60:
            log(f"[Resume] 从 epoch {ck['epoch']} 继续，已过 ep60，立即启用 Lookahead")
            optimizer = Lookahead(optimizer, alpha=0.5, k=20)
            optimizer.set_warmup(60 * len(train_loader))
        else:
            log(f"[Resume] 从 epoch {ck['epoch']} 继续")

    # ── Training Loop ─────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if epoch == 60 and not isinstance(optimizer, Lookahead):
            log("[Epoch 60] 启用 Lookahead (k=20, alpha=0.5)")
            optimizer = Lookahead(optimizer, alpha=0.5, k=20)
            optimizer.set_warmup(60 * len(train_loader))

        # if epoch == 76:
        #     inner = optimizer.optimizer if isinstance(optimizer, Lookahead) \
        #         else optimizer
        #     inner.param_groups[0]['lr'] = 3e-4
        #     inner.param_groups[1]['lr'] = 3e-5
        #
        #     # 重置 scheduler，从 ep76 开始重新 cosine 退火到 ep200
        #     remaining = NUM_EPOCHS - epoch
        #     scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #         inner, T_max=remaining, eta_min=2e-5)
        #     log(f"  [lr-reset] ep76  lr=3e-4，scheduler 重置，退火到 ep{NUM_EPOCHS}")
        # ep60：把 AdamW 包进 Lookahead（只做一次）

        target_ps    = get_patch_size(epoch, NUM_EPOCHS)
        target_batch = get_batch_size(target_ps, world_size)
        if target_ps != cur_ps or target_batch != cur_batch:
            train_ds.set_patch_size(target_ps)
            train_ds.repeat = get_repeat(target_ps)
            train_loader = make_loader(train_ds, target_batch, NUM_WORKERS,
                                       rank, world_size, True)
            log(f"[Epoch {epoch}] patch {cur_ps}→{target_ps}  "
                f"batch/gpu {cur_batch}→{target_batch}  "
                f"全局 {cur_batch*world_size}→{target_batch*world_size}")
            cur_ps, cur_batch = target_ps, target_batch

        ema.set_decay(min(0.9995 + 5e-5 * (epoch / NUM_EPOCHS), 0.9999))
        schedule_loss_weights(criterion, epoch, NUM_EPOCHS, current_psnr=best_psnr)

        if is_main():
            inner  = optimizer.optimizer if isinstance(optimizer, Lookahead) \
                     else optimizer
            cur_lr = inner.param_groups[0]['lr']
            p_w    = criterion.w['p']
            log(f"[Epoch {epoch}] lr={cur_lr:.2e}  perc_w={p_w:.4f}  "
                f"patch={cur_ps}  batch/gpu={cur_batch}")

        train_one_epoch(model, ema, train_loader, optimizer, criterion,
                        scaler, device, epoch, history, SAVE_DIR, rank, world_size)
        scheduler.step()

        # ── 每 15 epoch 验证 ──────────────────────────────────
        if epoch % 5 == 0:
            if rank == 0:
                avg_p, avg_s = evaluate(ema.eval_model(), val_loader, device,
                                        SAVE_DIR, tile=256, overlap=64,
                                        tag=f'val_ep{epoch}', max_samples=300)
            # Rank 1 不做任何推理，直接等
            dist.barrier()
            model.train()

            if rank == 0:
                ck_data = dict(
                    epoch=epoch, best_psnr=best_psnr,
                    model=model.module.state_dict(),
                    ema=ema.shadow.state_dict(),
                    opt=optimizer.optimizer.state_dict()
                    if isinstance(optimizer, Lookahead)
                    else optimizer.state_dict(),
                    sched=scheduler.state_dict(),
                    history=history,
                )
                torch.save(ck_data, resume_path)

                if avg_p > best_psnr:
                    best_psnr = avg_p
                    ck_data['best_psnr'] = best_psnr
                    torch.save(ck_data, best_path)
                    log(f"  [★ BEST] PSNR={avg_p:.2f} dB  SSIM={avg_s:.4f}"
                        f"  → best.pth（已覆盖）")
                else:
                    log(f"  [ckpt] ep{epoch:03d}  PSNR={avg_p:.2f}  SSIM={avg_s:.4f}"
                        f"  (best={best_psnr:.2f}，未更新 best.pth)")

    # ── 最终评估（两卡都跑，只有 Rank 0 保存）────────────────
    if rank == 0:
        log("\n训练完成！最终验证集评估 (EMA)...")
        evaluate(ema.eval_model(), val_loader, device, SAVE_DIR,
                 tile=256, overlap=64, tag='final_val')
        plot_history(history, SAVE_DIR)
    dist.barrier()

    if rank == 0:
        log("\n测试集最终评估 (EMA)...")
        evaluate(ema.eval_model(), test_loader, device, SAVE_DIR,
                 tile=256, overlap=64, tag='test_final')
        log(f"\n所有文件保存至: {SAVE_DIR}")
    dist.barrier()

    dist.destroy_process_group()

# ─────────────────────────────────────────────────────────────────
# 29. 入口
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='LDCT V7 bc=96')
    ap.add_argument('--base_dir',    default='/home/user/joshua82/LDCT_Project')
    ap.add_argument('--save_dir',    default=None)
    ap.add_argument('--ldct_root',   default=None)
    ap.add_argument('--ndct_root',   default=None)
    ap.add_argument('--epochs',      type=int, default=200)
    ap.add_argument('--warmup',      type=int, default=10)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--gpus',        type=int, default=2)
    ap.add_argument('--port',        type=int, default=29500)
    args = ap.parse_args()

    BASE = args.base_dir
    cfg  = dict(
        base_dir    = BASE,
        save_dir    = args.save_dir    or f"{BASE}/output/checkpoints/a100",
        ldct_root   = args.ldct_root   or f"{BASE}/dataset/QD301mm",
        ndct_root   = args.ndct_root   or f"{BASE}/dataset/FD301mm",
        epochs      = args.epochs,
        warmup      = args.warmup,
        num_workers = args.num_workers,
    )

    world_size = min(args.gpus, torch.cuda.device_count())
    log(f"启动 DDP  world_size={world_size}")
    log(f"GPU: {[torch.cuda.get_device_name(i) for i in range(world_size)]}")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    if world_size > 1:
        mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', str(args.port))
        dist.init_process_group('nccl', init_method='env://',
                                world_size=world_size, rank=rank,
                                timeout=datetime.timedelta(hours=4))
        main_worker(0, 1, cfg)


# =================================================================
# 后期调试建议（Troubleshooting Guide）
# =================================================================
#
# ── 问题1：训练初期 loss 激增 / NaN ────────────────────────────
#
#   原因A: 感知 loss 过早启用
#   → schedule_loss_weights 中 epoch<15 时 p_w=0，已处理
#   → 若仍出现，临时将感知延迟阈值从 15 调大到 25~30
#
#   原因B: lr 过高
#   → 将 lr 从 2e-4 降至 1e-4，重新从 epoch 1 训练
#   → 若是 resume 时突然 NaN，检查 checkpoint 是否完整
#
#   原因C: 数据异常值（某张切片 HU 范围极端）
#   → 在 LDCTDataset.__getitem__ 加打印：
#     print(f"max={ld.max():.3f}  min={ld.min():.3f}")
#   → 找到异常切片后从 pairs 中过滤掉
#
# ── 问题2：验证 PSNR 在 epoch 50 后不再提升（过拟合） ──────────
#
#   方案A: 提高 drop_path_rate（从 0.10 → 0.15）
#   方案B: 提高 weight_decay（从 5e-4 → 1e-3）
#   方案C: 在 LDCTDataset 中提高噪声增广概率（0.8 → 0.6）
#   方案D: 加入 CutMix/Mixup 切片级增广（对 CT 去噪有效）
#
# ── 问题3：PSNR 到 46~47 后平台期 ─────────────────────────────
#
#   方案A: 在 epoch 150 手动降 lr 到 5e-5，持续训练 50 epoch
#     base_opt.param_groups[0]['lr'] = 5e-5
#   方案B: 检查频率 loss 权重是否已足够（f_w 末期应达到 0.25）
#   方案C: 临时切换到 patch=512 推理（overlap=128），看指标是否提升
#     → 若 512 推理明显更好，说明感受野不足，考虑增大 ws 到 16
#   方案D: 加入 Test Time Augmentation（TTA）:
#     8 次 flip/rotate 推理取平均，PSNR 通常 +0.2~0.5 dB
#
# ── 问题4：gnorm 持续 > 3.0 ───────────────────────────────────
#
#   方案A: 降低 lr（2e-4 → 1e-4）
#   方案B: 检查 LayerScale gamma 是否正常增长
#     for n,p in model.named_parameters():
#         if 'gamma' in n: print(n, p.data.mean().item())
#   方案C: 检查 SSIM loss 是否因 patch 尺寸太小而产生异常梯度
#     → 确保 patch_size >= 64，SSIM 窗口 ws=11 需要至少 11px
#
# ── 问题5：VRAM 超出预期 ───────────────────────────────────────
#
#   → 降低 batch size：get_batch_size 中各级别 -4
#   → 确认 USE_GRAD_CKPT=True
#   → 关闭 cudnn.benchmark（可能缓存多个卷积算法占用额外显存）
#
# ── 关键监控指标（建议每 5 epoch 记录）────────────────────────
#
#   1. 验证 PSNR（EMA 模型 vs 即时模型的差距，差距大=EMA 在帮忙）
#   2. gnorm（正常范围 0.1~1.5，持续 >3.0 = 问题信号）
#   3. charb loss（应该单调下降，反弹 = 灾难性遗忘风险）
#   4. ssim loss（正常范围 0.01~0.08，突增 = SSIM 数值不稳）
#   5. VRAM 峰值（bc=96 应 < 40GB/卡）
#
# =================================================================