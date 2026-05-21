"""
LDCTDenoiser V5 — 2×A100 80GB  最终稳定版
==========================================
相比上一版的核心修复:
  STABLE-1  每个loss子项单独clamp上限，从源头阻断spike传播
  STABLE-2  HaarWaveletLoss改用更稳定的实现，避免高频爆炸
  STABLE-3  res_scale改为固定0.2，不再可学习（消除动态不稳定）
  STABLE-4  异常batch跳过逻辑前移到loss计算后、backward之前
  STABLE-5  scaler初始化scale从默认65536降到1024，减少bf16溢出
  STABLE-6  梯度裁剪阈值从1.0降到0.5

  FIX-1  EMA decay warmup + buffer同步，修复EMA早期失效
  FIX-2  CompositeLoss.forward wav loss早期(epoch<=50)关闭
  FIX-3  ssim权重早期从0.8降到0.3，让charb主导早期收敛
  FIX-4  CharbonnierLoss eps可在后期调小至5e-4

启动命令（单节点2卡）:
  torchrun --nproc_per_node=2 train_a100_v5_stable.py
"""

import random, math, os
import glob as _glob

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_DEBUG'] = 'WARN'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint
import torchvision.transforms.functional as TF
from torchvision import models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk
from copy import deepcopy

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR  = "/home/user/joshua82/LDCT_Project"
SAVE_DIR  = f"{BASE_DIR}/output/checkpoints/a102"
LDCT_ROOT = f"{BASE_DIR}/dataset/QD1mm"
NDCT_ROOT = f"{BASE_DIR}/dataset/FD1mm"

TRAIN_PATIENTS = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291']
VAL_PATIENTS   = ['L310', 'L333']
TEST_PATIENTS  = ['L506']

HU_SHIFT   = 1024.0
HU_NORM    = 4096.0
DATA_RANGE = 1.0
ACCUM_STEPS = 4

# 每个loss子项的最大允许值，超过直接clamp（STABLE-1）
# FIX-2: wav cap从0.05放宽到0.1（epoch>50才启用）
LOSS_CAPS = dict(ssim=1.0, perc=0.1, freq=0.2, wav=0.1, edge=0.5)


def preprocess(img: np.ndarray) -> np.ndarray:
    return np.clip((img.astype(np.float32) + HU_SHIFT) / HU_NORM, 0., 1.)


def deprocess(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.cpu().float().squeeze().numpy()
    return np.asarray(x, dtype=np.float32) * HU_NORM - HU_SHIFT


# =============================================================================
# DDP
# =============================================================================
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def log(msg):
    if is_main():
        print(msg, flush=True)


# =============================================================================
# 工具
# =============================================================================
def _infer_hw(L, hint_H=None, hint_W=None):
    if hint_H is not None and hint_W is not None:
        assert hint_H * hint_W == L
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
        vmin = v_mean - 2*v_std - 1
        vmax = v_mean + 2*v_std + 1
    else:
        vmin, vmax = -160, 240
    ax.imshow(img_hu, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.axis('off')


# =============================================================================
# LR 调度（手动，无scheduler）
# =============================================================================
LR_SCALES = {'encoder': 0.3, 'bottleneck': 0.7, 'decoder': 1.0, 'other': 1.0}

def adjust_lr(optimizer, epoch, base_lr=5e-5,
              warmup_epochs=15, total_epochs=200, min_lr=5e-6):
    if epoch < warmup_epochs:
        factor = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor   = (min_lr + (base_lr - min_lr) * cosine) / base_lr
    for pg in optimizer.param_groups:
        scale = LR_SCALES.get(pg.get('name', 'other'), 1.0)
        pg['lr'] = base_lr * factor * scale
    return base_lr * factor


# =============================================================================
# EMA
# FIX-1: decay warmup避免早期EMA失效 + buffer同步修复归一化统计
# =============================================================================
class EMA:
    def __init__(self, model, decay=0.99995):
        self.max_decay = decay          # FIX-1: 改为max_decay
        self.step = 0                   # FIX-1: 新增step计数
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        # FIX-1: warmup decay公式
        # step=1   → decay≈0.167  (快速跟上RAW模型)
        # step=100 → decay≈0.917
        # step=2000→ decay≈0.995
        # step足够大后趋近max_decay=0.99995
        decay = min(self.max_decay, (1 + self.step) / (10 + self.step))

        m = model.module if hasattr(model, 'module') else model

        # parameters: EMA滑动平均
        for s, p in zip(self.shadow.parameters(), m.parameters()):
            s.data.mul_(decay).add_(p.data, alpha=1 - decay)

        # FIX-1: buffers直接复制（GroupNorm running stats等）
        # 不做EMA，确保归一化统计与RAW模型保持一致
        for s, b in zip(self.shadow.buffers(), m.buffers()):
            if s.dtype.is_floating_point:
                s.data.copy_(b.data)

    def eval(self):
        self.shadow.eval()
        return self.shadow


# =============================================================================
# 模型
# =============================================================================
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth=40):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth, 3, padding=1, bias=False)
        self.norm = GN(growth); self.act = nn.GELU()
    def forward(self, x):
        return torch.cat([x, self.act(self.norm(self.conv(x)))], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth=40):
        super().__init__()
        self.layers = nn.Sequential(
            DenseLayer(in_ch,          growth), DenseLayer(in_ch+growth,   growth),
            DenseLayer(in_ch+growth*2, growth), DenseLayer(in_ch+growth*3, growth))
        self.proj = nn.Conv2d(in_ch+growth*4, in_ch, 1, bias=False)
        self.norm = GN(in_ch)
    def forward(self, x):
        return self.norm(self.proj(self.layers(x))) * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, in_ch, growth=40):
        super().__init__()
        self.db1 = DenseBlock(in_ch, growth)
        self.db2 = DenseBlock(in_ch, growth)
        self.db3 = DenseBlock(in_ch, growth)
    def forward(self, x):
        return self.db3(self.db2(self.db1(x))) * 0.2 + x

class RRDBEncoder(nn.Module):
    def __init__(self, in_ch=1, bc=160, growth=40):
        super().__init__()
        self.stem  = nn.Sequential(nn.Conv2d(in_ch,bc,3,padding=1,bias=False),GN(bc),nn.GELU())
        self.enc1  = nn.Sequential(RRDB(bc,growth), RRDB(bc,growth))
        self.down1 = nn.MaxPool2d(2)
        self.enc2  = nn.Sequential(nn.Conv2d(bc,bc*2,1,bias=False),GN(bc*2),nn.GELU(),
                                   RRDB(bc*2,growth),RRDB(bc*2,growth))
        self.down2 = nn.MaxPool2d(2)
        self.enc3  = nn.Sequential(nn.Conv2d(bc*2,bc*4,1,bias=False),GN(bc*4),nn.GELU(),
                                   RRDB(bc*4,growth),RRDB(bc*4,growth))
        self.down3 = nn.MaxPool2d(2)
        self.enc4  = nn.Sequential(nn.Conv2d(bc*4,bc*8,1,bias=False),GN(bc*8),nn.GELU(),
                                   RRDB(bc*8,growth),RRDB(bc*8,growth))
        self.down4 = nn.MaxPool2d(2)
    def forward(self, x):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        return [e1,e2,e3,e4], self.down4(e4)

def window_partition(x, ws):
    B,H,W,C = x.shape
    return (x.view(B,H//ws,ws,W//ws,ws,C).permute(0,1,3,2,4,5).contiguous().view(-1,ws,ws,C))

def window_reverse(wins, ws, H, W):
    B = wins.shape[0]//(H//ws*W//ws)
    return (wins.view(B,H//ws,W//ws,ws,ws,-1).permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1))

class WindowAttention(nn.Module):
    def __init__(self, dim, ws, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        while num_heads > 1 and dim % num_heads != 0: num_heads -= 1
        self.dim=dim; self.ws=ws; self.num_heads=num_heads
        self.scale = (dim//num_heads)**-0.5
        self.rpb = nn.Parameter(torch.zeros((2*ws-1)**2, num_heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(ws),torch.arange(ws),indexing='ij'))
        cf  = coords.flatten(1)
        rel = cf[:,:,None]-cf[:,None,:]
        rel = rel.permute(1,2,0).contiguous()
        rel[:,:,0]+=ws-1; rel[:,:,1]+=ws-1; rel[:,:,0]*=2*ws-1
        self.register_buffer('rpi', rel.sum(-1))
        self.qkv = nn.Linear(dim,dim*3,bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, mask=None):
        B_,N,C = x.shape
        qkv = self.qkv(x).reshape(B_,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        attn = (q*self.scale)@k.transpose(-2,-1)
        rpb  = self.rpb[self.rpi.view(-1)].view(N,N,self.num_heads)
        attn = attn + rpb.permute(2,0,1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_//nW,nW,self.num_heads,N,N)
                    +mask.unsqueeze(1).unsqueeze(0)).view(-1,self.num_heads,N,N)
        attn = self.attn_drop(torch.softmax(attn,dim=-1))
        return self.proj_drop(self.proj((attn@v).transpose(1,2).reshape(B_,N,C)))

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, ws=8, shift=False, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.ws=ws; self.shift=shift
        self.norm1=nn.LayerNorm(dim); self.norm2=nn.LayerNorm(dim)
        hid=int(dim*mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim,hid),nn.GELU(),nn.Dropout(drop),
                                 nn.Linear(hid,dim),nn.Dropout(drop))
        self.attn = WindowAttention(dim,ws,num_heads,attn_drop,drop)
    def _build_mask(self,H,W,ws,shift,device):
        if not shift or min(H,W)<=ws: return None
        img_mask=torch.zeros(1,H,W,1,device=device)
        cnt=0
        for hs in (slice(0,-ws),slice(-ws,-shift),slice(-shift,None)):
            for ws_ in (slice(0,-ws),slice(-ws,-shift),slice(-shift,None)):
                img_mask[:,hs,ws_,:]=cnt; cnt+=1
        mw=window_partition(img_mask,ws).view(-1,ws*ws)
        mask=mw.unsqueeze(1)-mw.unsqueeze(2)
        return mask.masked_fill(mask!=0,-100.).masked_fill(mask==0,0.)
    def forward(self, x, H, W):
        B,L,C=x.shape
        ws=min(self.ws,H,W)
        shift=ws//2 if self.shift and min(H,W)>ws else 0
        sc=x; x=self.norm1(x).view(B,H,W,C)
        if shift>0: x=torch.roll(x,(-shift,-shift),(1,2))
        pad_b=(ws-H%ws)%ws; pad_r=(ws-W%ws)%ws
        if pad_b>0 or pad_r>0:
            x=F.pad(x.permute(0,3,1,2),(0,pad_r,0,pad_b)).permute(0,2,3,1)
        _,pH,pW,_=x.shape
        mask=self._build_mask(pH,pW,ws,shift>0,x.device)
        xw=window_partition(x,ws)
        xw=self.attn(xw.view(-1,ws*ws,C),mask).view(-1,ws,ws,C)
        x=window_reverse(xw,ws,pH,pW)
        if pad_b>0 or pad_r>0: x=x[:,:H,:W,:].contiguous()
        if shift>0: x=torch.roll(x,(shift,shift),(1,2))
        x=x.view(B,H*W,C)+sc
        return x+self.mlp(self.norm2(x))

class DualScaleBlock(nn.Module):
    def __init__(self, dim, num_heads, ws=8, shift=False, drop=0., attn_drop=0.):
        super().__init__()
        self.local_attn=SwinBlock(dim,num_heads,ws,shift,drop=drop,attn_drop=attn_drop)
        self.glob_dw=nn.Conv2d(dim,dim,3,padding=4,dilation=4,groups=dim,bias=False)
        self.glob_pw=nn.Conv2d(dim,dim,1,bias=False)
        self.glob_norm=nn.LayerNorm(dim)
        self.gate=nn.Sequential(nn.Linear(dim*2,dim),nn.Sigmoid())
        self.out_norm=nn.LayerNorm(dim)
    def forward(self, x, H, W):
        B,L,C=x.shape
        xl=self.local_attn(x,H,W)
        xg=self.glob_pw(self.glob_dw(x.transpose(1,2).view(B,C,H,W))).flatten(2).transpose(1,2)
        xg=self.glob_norm(xg+x)
        gate=self.gate(torch.cat([xl,xg],dim=-1))
        return self.out_norm(gate*xl+(1-gate)*xg)

class CSAS(nn.Module):
    def __init__(self, dim, pool_grid=8):
        super().__init__()
        self.pool_grid=pool_grid
        nh=safe_heads(dim,target_div=64)
        self.nh=nh; self.scale=(dim//nh)**-0.5
        self.nq=nn.LayerNorm(dim); self.nk=nn.LayerNorm(dim)
        self.q=nn.Linear(dim,dim,bias=False); self.k=nn.Linear(dim,dim,bias=False)
        self.v=nn.Linear(dim,dim,bias=False); self.o=nn.Linear(dim,dim,bias=False)
    def forward(self, query, kv, H=None, W=None):
        B,Lq,C=query.shape; H,W=_infer_hw(Lq,H,W)
        g=min(self.pool_grid,H,W)
        kv_flat=F.adaptive_avg_pool2d(kv.transpose(1,2).view(B,C,H,W),(g,g)).flatten(2).transpose(1,2)
        nh,D=self.nh,C//self.nh
        q=self.q(self.nq(query  )).view(B,Lq, nh,D).transpose(1,2)
        k=self.k(self.nk(kv_flat)).view(B,g*g,nh,D).transpose(1,2)
        v=self.v(kv_flat          ).view(B,g*g,nh,D).transpose(1,2)
        attn=torch.softmax((q*self.scale)@k.transpose(-2,-1),dim=-1)
        return query+self.o((attn@v).transpose(1,2).reshape(B,Lq,C))

class CSG(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        rd=max(dim//r,4)
        self.ch_fc=nn.Sequential(nn.Linear(dim,rd),nn.ReLU(True),nn.Linear(rd,dim),nn.Sigmoid())
        self.sp_conv=nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),nn.Sigmoid())
    def forward(self, x, H, W):
        B,L,C=x.shape
        x=x*self.ch_fc(x.mean(1)).unsqueeze(1)
        x2d=x.transpose(1,2).view(B,C,H,W)
        sp=self.sp_conv(torch.cat([x2d.mean(1,keepdim=True),x2d.max(1,keepdim=True).values],1))
        return (x2d*sp).flatten(2).transpose(1,2)

class TransformerBottleneck(nn.Module):
    def __init__(self, dim, train_grid=16, num_heads=8, depth=12, ws=8, drop=0., attn_drop=0.):
        super().__init__()
        self.dim=dim; self.train_grid=train_grid
        self.pos=nn.Parameter(torch.zeros(1,train_grid*train_grid,dim))
        nn.init.trunc_normal_(self.pos,std=0.02)
        num_heads=safe_heads(dim,target_div=dim//max(1,num_heads))
        self.blocks=nn.ModuleList([
            DualScaleBlock(dim,num_heads,ws,shift=(i%2==1),drop=drop,attn_drop=attn_drop)
            for i in range(depth)])
        self.norm=nn.LayerNorm(dim)
    def _get_pos(self,H,W):
        G=self.train_grid
        if H==G and W==G: return self.pos
        pe=self.pos.reshape(1,G,G,self.dim).permute(0,3,1,2)
        pe=F.interpolate(pe.float(),(H,W),mode='bilinear',align_corners=False)
        return pe.permute(0,2,3,1).reshape(1,H*W,self.dim)
    def forward(self, x):
        B,C,H,W=x.shape
        t=x.flatten(2).transpose(1,2)+self._get_pos(H,W)
        for blk in self.blocks:
            if self.training: t=checkpoint(blk,t,H,W,use_reentrant=False)
            else: t=blk(t,H,W)
        return self.norm(t).transpose(1,2).view(B,C,H,W)

class PatchExpand(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.expand=nn.Linear(in_dim,in_dim*4,bias=False)
        self.norm=nn.LayerNorm(in_dim)
        self.proj=nn.Linear(in_dim,out_dim,bias=False) if in_dim!=out_dim else nn.Identity()
    def forward(self, x, H, W):
        B,L,C=x.shape
        x=self.expand(x); C4=x.shape[-1]; C=C4//4
        x=x.view(B,H,W,2,2,C).permute(0,1,3,2,4,5).contiguous()
        x=x.view(B,2*H*2*W,C); x=self.norm(x)
        return self.proj(x),2*H,2*W

class DecoderStage(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, num_heads,
                 ws=8, depth=2, drop=0., attn_drop=0.):
        super().__init__()
        self.expand=PatchExpand(in_dim,out_dim)
        self.skip_proj=nn.Linear(skip_dim,out_dim,bias=False) if skip_dim!=out_dim else nn.Identity()
        self.csas=CSAS(out_dim)
        num_heads=safe_heads(out_dim,target_div=out_dim//max(1,num_heads))
        self.blocks=nn.ModuleList([
            DualScaleBlock(out_dim,num_heads,ws,shift=(i%2==1),drop=drop,attn_drop=attn_drop)
            for i in range(depth)])
        self.csg=CSG(out_dim)
    def forward(self, x, skip, H_in, W_in):
        x,H,W=self.expand(x,H_in,W_in)
        B,C_s,Hs,Ws=skip.shape
        if Hs!=H or Ws!=W:
            skip=F.interpolate(skip.float(),(H,W),mode='bilinear',align_corners=False)
        skip_t=self.skip_proj(skip.flatten(2).transpose(1,2))
        x=self.csas(x,skip_t,H=H,W=W)
        for blk in self.blocks:
            if self.training: x=checkpoint(blk,x,H,W,use_reentrant=False)
            else: x=blk(x,H,W)
        return self.csg(x,H,W),H,W


class MultiScaleHead(nn.Module):
    def __init__(self, dim, in_ch=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim,    3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim//2, 1, bias=False), nn.GELU(),
            nn.Conv2d(dim//2, in_ch, 1))
        self.detail = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(32, in_ch, 3, padding=1))
        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.register_buffer('res_scale', torch.tensor(0.1))

    def forward(self, tokens, H, W, x_in):
        B, L, C = tokens.shape
        feat   = self.norm(tokens).transpose(1,2).view(B,C,H,W)
        main = torch.tanh(self.main(feat)) * self.res_scale
        detail = self.detail(x_in - F.avg_pool2d(x_in, 3, stride=1, padding=1))
        detail = torch.tanh(detail) * self.alpha.abs().clamp(max=0.05)
        out    = x_in + main + detail
        return out.clamp(0., 1.)


class LDCTDenoiserV5(nn.Module):
    def __init__(self, in_ch=1, bc=160, growth=40,
                 bot_depth=12, bot_heads=10, ws=8,
                 dec_depths=(3,3,2,2), drop=0., attn_drop=0.):
        super().__init__()
        self.encoder    = RRDBEncoder(in_ch, bc, growth)
        self.bottleneck = TransformerBottleneck(
            dim=bc*8, train_grid=16, num_heads=bot_heads,
            depth=bot_depth, ws=ws, drop=drop, attn_drop=attn_drop)
        self.dec4 = DecoderStage(bc*8,bc*8,bc*4,safe_heads(bc*4),ws,dec_depths[0],drop,attn_drop)
        self.dec3 = DecoderStage(bc*4,bc*4,bc*2,safe_heads(bc*2),ws,dec_depths[1],drop,attn_drop)
        self.dec2 = DecoderStage(bc*2,bc*2,bc,  safe_heads(bc),  ws,dec_depths[2],drop,attn_drop)
        self.dec1 = DecoderStage(bc,  bc,  bc,  safe_heads(bc),  ws,dec_depths[3],drop,attn_drop)
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
        B,C,H,W=x.shape
        skips,bot=self.encoder(x)
        bot=self.bottleneck(bot)
        bH,bW=H//16,W//16
        t=bot.flatten(2).transpose(1,2)
        t,h,w=self.dec4(t,skips[3],bH,bW)
        t,h,w=self.dec3(t,skips[2],h,w)
        t,h,w=self.dec2(t,skips[1],h,w)
        t,h,w=self.dec1(t,skips[0],h,w)
        return self.head(t,h,w,x)


# =============================================================================
# 损失函数
# =============================================================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, data_range=1.0, levels=3):
        super().__init__()
        self.dr=data_range; self.levels=levels; self.ws=window_size
        g=torch.arange(window_size,dtype=torch.float32)-window_size//2
        g=torch.exp(-(g**2)/(2*1.5**2)); g/=g.sum()
        self.register_buffer('win',g.outer(g).unsqueeze(0).unsqueeze(0))
    def _ssim(self,x,y):
        C1=(0.01*self.dr)**2; C2=(0.03*self.dr)**2
        pad=self.ws//2; w=self.win.to(x.device,x.dtype)
        mx=F.conv2d(x,w,padding=pad); my=F.conv2d(y,w,padding=pad)
        mxx=F.conv2d(x*x,w,padding=pad)-mx**2
        myy=F.conv2d(y*y,w,padding=pad)-my**2
        mxy=F.conv2d(x*y,w,padding=pad)-mx*my
        return ((2*mx*my+C1)*(2*mxy+C2)/((mx**2+my**2+C1)*(mxx+myy+C2))).mean()
    def forward(self,x,y):
        loss=0.
        for i in range(self.levels):
            loss+=1.-self._ssim(x,y)
            if i<self.levels-1: x=F.avg_pool2d(x,2); y=F.avg_pool2d(y,2)
        return loss/self.levels

# FIX-4: eps保留1e-3，epoch>100后可在main()里替换为5e-4
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__(); self.eps2=eps**2
    def forward(self,pred,target):
        return torch.sqrt((pred-target)**2+self.eps2).mean()

class FrequencyLoss(nn.Module):
    def forward(self,pred,target):
        fp=torch.fft.rfft2(pred.float(),norm='ortho')
        ft=torch.fft.rfft2(target.float(),norm='ortho')
        return F.l1_loss(fp.abs(),ft.abs())

# 直接删掉 HaarWaveletLoss，用这个替换
class WaveletLoss(nn.Module):
    """用多尺度 avg_pool 模拟小波，数值完全稳定"""
    def forward(self, pred, target):
        loss = torch.tensor(0., device=pred.device, requires_grad=True)
        p, t = pred, target
        for _ in range(3):
            # 低频残差
            p_low = F.avg_pool2d(p, 2, stride=2)
            t_low = F.avg_pool2d(t, 2, stride=2)
            # 高频 = 原图 - 上采样(低频)
            p_high = p - F.interpolate(p_low, size=p.shape[-2:], mode='nearest')
            t_high = t - F.interpolate(t_low, size=t.shape[-2:], mode='nearest')
            loss = loss + F.l1_loss(p_high, t_high)
            p, t = p_low, t_low
        return loss / 3.0
class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32).view(1,1,3,3)
        sy=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx',sx); self.register_buffer('sy',sy)
    def forward(self,p,t):
        sx=self.sx.to(p.device,p.dtype); sy=self.sy.to(p.device,p.dtype)
        ep=torch.sqrt(F.conv2d(p,sx,padding=1)**2+F.conv2d(p,sy,padding=1)**2+1e-6)
        et=torch.sqrt(F.conv2d(t,sx,padding=1)**2+F.conv2d(t,sy,padding=1)**2+1e-6)
        return F.l1_loss(ep,et)

class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        resnet=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
        self.feats,self.hooks={},[]
        for name in ('layer1','layer2'):
            h=dict(resnet.named_modules())[name].register_forward_hook(
                lambda m,i,o,n=name: self.feats.update({n:o}))
            self.hooks.append(h)
        self.resnet=resnet.to(device)
        for p in self.resnet.parameters(): p.requires_grad_(False)
        self.layers=('layer1','layer2'); self.MAX_SIDE=128
        self.register_buffer('mean',torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def _prep(self,t):
        if t.shape[-1]>self.MAX_SIDE or t.shape[-2]>self.MAX_SIDE:
            t=F.interpolate(t,(self.MAX_SIDE,self.MAX_SIDE),mode='bilinear',align_corners=False)
        t3=t.float().repeat(1,3,1,1)
        return (t3-self.mean.to(t3.device))/self.std.to(t3.device)
    def forward(self,x,y):
        with torch.no_grad():
            self.feats.clear(); self.resnet(self._prep(y))
            yf={k:v.detach() for k,v in self.feats.items()}
        losses=[]
        for i in range(x.shape[0]):
            self.feats.clear(); self.resnet(self._prep(x[i:i+1]))
            xf=dict(self.feats)
            losses.append(sum(F.mse_loss(xf[l],yf[l][i:i+1]) for l in self.layers)/len(self.layers))
        return torch.stack(losses).mean()
    def __del__(self):
        for h in self.hooks:
            try: h.remove()
            except: pass


# FIX-3: ssim早期权重从0.8降到0.3，避免不稳定的ssim主导早期收敛
def get_loss_weights(epoch):
    if epoch <= 30:  # 前 30 个 epoch 用最保守的配置
        return dict(lc=1.0, ls=0.1, lp=0.01, lf=0.05, lw=0.0, le=0.1)
    elif epoch <= 50:
        return dict(lc=1.0, ls=0.3, lp=0.03, lf=0.08, lw=0.0, le=0.20)
    elif epoch <= 100:
        return dict(lc=1.0, ls=1.0, lp=0.06, lf=0.12, lw=0.20, le=0.35)
    elif epoch <= 150:
        return dict(lc=1.0, ls=2.0, lp=0.10, lf=0.15, lw=0.30, le=0.50)
    else:
        return dict(lc=0.8, ls=3.0, lp=0.12, lf=0.20, lw=0.40, le=0.60)

class CompositeLoss(nn.Module):
    def __init__(self, lc=1.0, ls=0.3, lp=0.03,          # FIX-3: 默认ls改为0.3
                 lf=0.08, lw=0.10, le=0.20, device='cuda'):
        super().__init__()
        self.charb=CharbonnierLoss(); self.ssim=SSIMLoss(data_range=DATA_RANGE,levels=3)
        self.perc=PerceptualLoss(device=device); self.freq=FrequencyLoss()
        self.wav=WaveletLoss(); self.edge=EdgeAwareLoss()
        self.lc,self.ls,self.lp=lc,ls,lp
        self.lf,self.lw,self.le=lf,lw,le
        self.current_epoch = 1          # FIX-2: 记录当前epoch，控制wav启用时机

    def set_epoch(self, epoch: int):    # FIX-2: 由train_one_epoch调用
        self.current_epoch = epoch

    def forward(self, pred, target):
        lc = self.charb(pred,target)
        ls = self.ssim (pred,target).clamp(max=LOSS_CAPS['ssim'])
        lp = self.perc (pred,target).clamp(max=LOSS_CAPS['perc'])
        lf = self.freq (pred,target).clamp(max=LOSS_CAPS['freq'])
        le = self.edge (pred,target).clamp(max=LOSS_CAPS['edge'])

        # FIX-2: epoch<=50时关闭wav loss，避免早期高频系数不稳定污染梯度
        # epoch>50后重新启用，cap放宽到0.1（见LOSS_CAPS）
        if self.current_epoch <= 50:
            lw = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        else:
            lw_raw = self.wav(pred, target)
            # wav 出现 NaN/Inf 时直接用 0 替代，不污染梯度
            lw = lw_raw.clamp(max=LOSS_CAPS['wav']) if torch.isfinite(lw_raw) else \
                torch.zeros(1, device=pred.device)
        mean_penalty = (pred.mean(dim=[1, 2, 3]) - target.mean(dim=[1, 2, 3])).abs().mean()
        # 仅在均值偏差>0.05时生效（正常训练不影响）
        mean_penalty = F.relu(mean_penalty - 0.10) * 2.0

        total = (self.lc*lc + self.ls*ls + self.lp*lp +
                 self.lf*lf + self.lw*lw + self.le*le + mean_penalty)
        return total, dict(
            charb=lc.item(), ssim=ls.item(), perc=lp.item(),
            freq=lf.item(),  wav=lw.item(),  edge=le.item(),mean_penalty=mean_penalty.item())


# =============================================================================
# Dataset
# =============================================================================
class LDCTDataset(Dataset):
    def __init__(self, ldct_root, ndct_root, patients, mode='train', patch_size=256):
        self.pairs=[]; avail=[]
        for p in patients:
            ld=os.path.join(ldct_root,p); nd=os.path.join(ndct_root,p)
            if not (os.path.isdir(ld) and os.path.isdir(nd)):
                log(f"  [警告] {p} 不存在，跳过"); continue
            avail.append(p)
            lf=sorted(f for f in os.listdir(ld) if f.endswith('.npy'))
            nf=sorted(f for f in os.listdir(nd) if f.endswith('.npy'))
            for i in range(min(len(lf),len(nf))):
                self.pairs.append((os.path.join(ld,lf[i]),os.path.join(nd,nf[i])))
        self.patch_size=patch_size; self.is_train=(mode=='train')
        log(f"[{mode.upper()}] 患者={avail}  共 {len(self.pairs)} 个切片对")
    def set_patch_size(self,ps): self.patch_size=ps
    def __len__(self): return len(self.pairs)
    def __getitem__(self,idx):
        lp,np_=self.pairs[idx]
        ld=torch.from_numpy(preprocess(np.load(lp)))[None]
        nd=torch.from_numpy(preprocess(np.load(np_)))[None]
        if self.is_train:
            _,h,w=ld.shape; ps=min(self.patch_size,h,w); ps=(ps//16)*16
            if ps<h or ps<w:
                i=random.randint(0,h-ps); j=random.randint(0,w-ps)
                ld=TF.crop(ld,i,j,ps,ps); nd=TF.crop(nd,i,j,ps,ps)
            if random.random()>0.5: ld,nd=TF.hflip(ld),TF.hflip(nd)
            if random.random()>0.5: ld,nd=TF.vflip(ld),TF.vflip(nd)
            k=random.randint(0,3)
            if k: ld=torch.rot90(ld,k,[1,2]); nd=torch.rot90(nd,k,[1,2])
            if random.random()>0.8:
                f=random.uniform(0.98,1.02)
                ld=(ld*f).clamp(0.,1.); nd=(nd*f).clamp(0.,1.)
        return ld,nd


# =============================================================================
# 推理
# =============================================================================
@torch.no_grad()
def patch_inference(model, img, tile=512, overlap=64, device='cuda'):
    _,C,H,W=img.shape; tile=(tile//16)*16; step=tile-overlap
    pad_h=(tile-H%tile)%tile if H%tile else 0
    pad_w=(tile-W%tile)%tile if W%tile else 0
    img_p=F.pad(img,(0,pad_w,0,pad_h),mode='reflect')
    oH,oW=img_p.shape[2],img_p.shape[3]
    out=torch.zeros(1,C,oH,oW); cnt=torch.zeros(1,C,oH,oW)
    wy=torch.hann_window(tile,periodic=False).view(-1,1).expand(tile,tile)
    wx=torch.hann_window(tile,periodic=False).view(1,-1).expand(tile,tile)
    weight=(wy*wx).unsqueeze(0).unsqueeze(0)
    ys=list(range(0,oH-tile+1,step)) or [0]
    xs=list(range(0,oW-tile+1,step)) or [0]
    if not ys or ys[-1]+tile<oH: ys.append(max(0,oH-tile))
    if not xs or xs[-1]+tile<oW: xs.append(max(0,oW-tile))
    for y in ys:
        for x in xs:
            pred=model(img_p[:,:,y:y+tile,x:x+tile].to(device)).cpu()
            out[:,:,y:y+tile,x:x+tile]+=pred*weight
            cnt[:,:,y:y+tile,x:x+tile]+=weight
    return (out/cnt.clamp(min=1e-6))[:,:,:H,:W].clamp(0.,1.)

@torch.no_grad()
def tta_inference_16x(model, img, tile=512, overlap=96, device='cuda'):
    preds=[]
    for fh in [False,True]:
        for fv in [False,True]:
            for rk in [0,1,2,3]:
                x=img.clone()
                if fh: x=torch.flip(x,[3])
                if fv: x=torch.flip(x,[2])
                if rk: x=torch.rot90(x,rk,[2,3])
                pred=patch_inference(model,x,tile=tile,overlap=overlap,device=device)
                if rk:  pred=torch.rot90(pred,-rk,[2,3])
                if fv:  pred=torch.flip(pred,[2])
                if fh:  pred=torch.flip(pred,[3])
                preds.append(pred)
    return torch.stack(preds).mean(0).clamp(0.,1.)


# =============================================================================
# 指标
# =============================================================================
def to_norm(t):
    if isinstance(t,torch.Tensor): return t.cpu().float().squeeze().numpy()
    return np.asarray(t).squeeze()

def compute_metrics(pred_norm,target_norm):
    return (psnr_sk(target_norm,pred_norm,data_range=DATA_RANGE),
            ssim_sk(target_norm,pred_norm,data_range=DATA_RANGE))


# =============================================================================
# Patch/Batch schedule
# =============================================================================
def get_patch_size(epoch):
    if epoch<=40:  return 256
    if epoch<=80:  return 320
    if epoch<=120: return 384
    if epoch<=160: return 448
    return 512

def get_batch_size(patch_size):
    if patch_size<=256: return 6
    if patch_size<=320: return 4
    if patch_size<=384: return 3
    if patch_size<=448: return 2
    return 2


# =============================================================================
# 参数组
# =============================================================================
def build_param_groups(model, base_lr):
    m=model.module if hasattr(model,'module') else model
    enc_params=list(m.encoder.parameters())
    bot_params=list(m.bottleneck.parameters())
    dec_params=(list(m.dec4.parameters())+list(m.dec3.parameters())+
                list(m.dec2.parameters())+list(m.dec1.parameters())+
                list(m.head.parameters()))
    all_ids=({id(p) for p in enc_params}|{id(p) for p in bot_params}|{id(p) for p in dec_params})
    other_params=[p for p in m.parameters() if id(p) not in all_ids]
    groups=[
        {'params':enc_params, 'lr':base_lr*LR_SCALES['encoder'],    'name':'encoder'},
        {'params':bot_params, 'lr':base_lr*LR_SCALES['bottleneck'],  'name':'bottleneck'},
        {'params':dec_params, 'lr':base_lr*LR_SCALES['decoder'],     'name':'decoder'},
    ]
    if other_params:
        groups.append({'params':other_params,'lr':base_lr*LR_SCALES['other'],'name':'other'})
    return groups


# =============================================================================
# ckpt保存
# =============================================================================
def save_ckpt(ckpt_data, save_dir, epoch, avg_p, best_psnr, max_keep=3):
    old_ckpts=sorted(_glob.glob(os.path.join(save_dir,"ckpt_ep*.pth")))
    while len(old_ckpts)>=max_keep:
        os.remove(old_ckpts.pop(0))
    ckpt_path=os.path.join(save_dir,f'ckpt_ep{epoch:03d}_P{avg_p:.2f}.pth')
    tmp=ckpt_path+'.tmp'; torch.save(ckpt_data,tmp); os.replace(tmp,ckpt_path)
    log(f"  [ckpt] ep{epoch:03d}  PSNR={avg_p:.2f}  best={best_psnr:.2f}")
    if avg_p>best_psnr:
        best_path=os.path.join(save_dir,f'best_P{avg_p:.2f}_ep{epoch:03d}.pth')
        tmp2=best_path+'.tmp'; torch.save(ckpt_data,tmp2); os.replace(tmp2,best_path)
        log(f"  [best] PSNR={avg_p:.2f}  ← 新最优")
        return avg_p
    return best_psnr


# =============================================================================
# 训练一个epoch
# =============================================================================
def train_one_epoch(model, ema, loader, optimizer, criterion,
                    scaler, device, epoch, history, save_dir, base_lr, rank=0,use_ddp=False):
    model.train()
    weights=get_loss_weights(epoch)
    criterion.lc=weights['lc']; criterion.ls=weights['ls']
    criterion.lp=weights['lp']; criterion.lf=weights['lf']
    criterion.lw=weights['lw']; criterion.le=weights['le']
    criterion.set_epoch(epoch)  # FIX-2: 通知criterion当前epoch，控制wav启用

    if rank==0:
        cur_lr=optimizer.param_groups[2]['lr']
        log(f"  [ep{epoch}] lr(decoder)={cur_lr:.2e}  "
            +"  ".join(f"{k}:{v}" for k,v in weights.items()))
        os.makedirs(save_dir,exist_ok=True)

    total,subs_acc=0.,{}
    optimizer.zero_grad(set_to_none=True)

    for bi,(ldct,ndct) in enumerate(loader):
        ldct=ldct.to(device,non_blocking=True)
        ndct=ndct.to(device,non_blocking=True)
        # with torch.no_grad():
        #     with model.no_sync():  # 不触发梯度同步
        #         probe = model(ldct[[0]])
        #     probe_mean = torch.tensor(probe.mean().item(), device=device)
        #     probe_std = torch.tensor(probe.std().item(), device=device)
        # if use_ddp:
        #     dist.all_reduce(probe_mean, op=dist.ReduceOp.MIN)
        #     dist.all_reduce(probe_std, op=dist.ReduceOp.MIN)
        # if probe_std < 0.01 or probe_mean > 0.95 or probe_mean < 0.05:
        #     if rank == 0:
        #         log(f"  [COLLAPSE] ep{epoch} b{bi} mean={probe.mean():.4f} std={probe.std():.4f}，两卡同步跳过")
        #     optimizer.zero_grad(set_to_none=True)
        #     torch.cuda.empty_cache()  # 强制释放显存碎片
        #     continue
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            pred=model(ldct)
            pred_mean = pred.float().mean().item()
            pred_std = pred.float().std().item()
            if pred_std < 0.005 or pred_mean > 0.92 or pred_mean < 0.05:
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5  # 崩溃时减半lr
                scaler._scale = max(scaler._scale / 2, 64.0)
                if rank == 0:
                    log(f"  [COLLAPSE-SKIP] ep{epoch} b{bi} "
                        f"mean={pred_mean:.4f} std={pred_std:.4f}，跳过并重置优化器")
                optimizer.zero_grad(set_to_none=True)
                # FIX-C: 崩溃时降低scaler scale，防止bf16溢出继续扩大
                scaler._scale = max(scaler._scale / 2, 64.0)
                torch.cuda.empty_cache()
                continue
            loss,subs=criterion(pred.float(),ndct.float())
            loss=loss/ACCUM_STEPS
            loss_ok = torch.tensor(1.0 if torch.isfinite(loss) else 0.0, device=device)
        if use_ddp:
            dist.all_reduce(loss_ok, op=dist.ReduceOp.MIN)
        if loss_ok.item() < 0.5:
            if rank == 0:
                log(f"  [SKIP] ep{epoch} b{bi} loss异常，两卡同步跳过")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            scaler.update()
            continue
        if not torch.isfinite(loss):
            log(f"  [SKIP] ep{epoch} b{bi} loss不有限({loss.item():.4f})，跳过")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            scaler.update()
            continue

        scaler.scale(loss).backward()

        if (bi+1)%ACCUM_STEPS==0 or (bi+1)==len(loader):
            scaler.unscale_(optimizer)
            clip_val = 0.1 if epoch <= 50 else 0.25
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            grad_ok = torch.tensor(1.0 if torch.isfinite(grad_norm) else 0.0, device=device)
            if use_ddp:
                dist.all_reduce(grad_ok, op=dist.ReduceOp.MIN)
            # has_bad_params = False
            # for name, param in model.named_parameters():
            #     if param.grad is not None and not torch.isfinite(param.grad).all():
            #         has_bad_params = True
            #         log(f"  [ALERT] {name} 梯度异常")
            #         break
            #     if not torch.isfinite(param).all():
            #         has_bad_params = True
            #         log(f"  [ALERT] {name} 权重异常")
            #         break
            if grad_ok.item() < 0.5:
                if rank == 0:
                    log(f"  [SKIP] ep{epoch} b{bi} grad异常，两卡同步跳过")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                scaler.update()
                continue
            if not torch.isfinite(grad_norm):
                log(f"  [SKIP] ep{epoch} b{bi} grad异常，跳过更新")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                scaler.update()
                continue
            scaler.step(optimizer); scaler.update()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        real_loss=loss.item()*ACCUM_STEPS
        total+=real_loss
        for k,v in subs.items():
            subs_acc[k]=subs_acc.get(k,0.)+v

        if rank==0 and (bi+1)%200==0:
            with torch.no_grad():
                m=model.module if hasattr(model,'module') else model
                m.eval(); pred_vis=m(ldct[[0]])
                log(f"  pred均值:{pred_vis.mean():.4f}  标准差:{pred_vis.std():.4f}")
                m.train()
            pn=to_norm(pred_vis[0,0]); gn=to_norm(ndct[0,0]); ln=to_norm(ldct[0,0])
            ps,ss=compute_metrics(pn,gn)
            fig,ax=plt.subplots(1,3,figsize=(15,5))
            smart_imshow(ax[0],deprocess(ln),"LDCT")
            smart_imshow(ax[1],deprocess(pn),f"Pred PSNR:{ps:.2f} SSIM:{ss:.4f}")
            smart_imshow(ax[2],deprocess(gn),"NDCT")
            plt.savefig(os.path.join(save_dir,f"ep{epoch:03d}_b{bi+1:05d}_P{ps:.2f}.png"),
                        dpi=100,bbox_inches='tight')
            plt.close()
            log(f"  [snap] ep{epoch} b{bi+1}/{len(loader)}  PSNR:{ps:.2f}  SSIM:{ss:.4f}")

        if rank==0 and bi%50==0:
            mem=torch.cuda.memory_reserved(device)/1e9
            sub_str="  ".join(f"{k}:{v:.4f}" for k,v in subs.items())
            log(f"  ep{epoch} [{bi}/{len(loader)}]  loss:{real_loss:.4f}  {sub_str}  VRAM:{mem:.1f}GB")

    n=len(loader); avg=total/n
    if rank==0:
        history['train_loss'].append(avg)
        for k in subs_acc: history.setdefault(k,[]).append(subs_acc[k]/n)
        log(f"{'='*60}\nEpoch {epoch}  avg_loss:{avg:.4f}\n{'='*60}\n")

    # FIX-4: epoch>100时自动将CharbonnierLoss的eps调小至5e-4，提高对小误差的敏感度
    if epoch == 100 and rank == 0:
        criterion.charb = CharbonnierLoss(eps=5e-4)
        log("  [FIX-4] epoch=100，CharbonnierLoss eps: 1e-3 → 5e-4")

    return avg


# =============================================================================
# 验证
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, save_dir, tile=512, overlap=64, tag='val', use_tta=False):
    os.makedirs(save_dir,exist_ok=True)
    model.eval(); psnrs,ssims=[],[]
    for i,(ldct,ndct) in enumerate(loader):
        if use_tta:
            pred=tta_inference_16x(model,ldct,tile=tile,overlap=overlap,device=device)
        else:
            pred=patch_inference(model,ldct,tile=tile,overlap=overlap,device=device)
        pn=to_norm(pred[0,0]); gn=to_norm(ndct[0,0]); ln=to_norm(ldct[0,0])
        ps,ss=compute_metrics(pn,gn)
        psnrs.append(ps); ssims.append(ss)
        if i==0:
            log(f"  [DEBUG] pred:{pn.min():.4f}~{pn.max():.4f}  gt:{gn.min():.4f}~{gn.max():.4f}")
        if i%100==0:
            fig,ax=plt.subplots(1,3,figsize=(15,5))
            smart_imshow(ax[0],deprocess(ln),"LDCT")
            smart_imshow(ax[1],deprocess(pn),f"Pred\nPSNR:{ps:.2f} SSIM:{ss:.4f}")
            smart_imshow(ax[2],deprocess(gn),"NDCT")
            plt.savefig(os.path.join(save_dir,f"{tag}_case_{i:03d}.png"),dpi=100,bbox_inches='tight')
            plt.close()
            log(f"  [{tag}] {i}/{len(loader.dataset)}  PSNR:{ps:.2f}  SSIM:{ss:.4f}")
    avg_p=float(np.mean(psnrs)); std_p=float(np.std(psnrs))
    avg_s=float(np.mean(ssims)); std_s=float(np.std(ssims))
    tta_str=" (TTA×16)" if use_tta else ""
    log(f"\n{'='*60}\n[{tag.upper()}]{tta_str}  N={len(psnrs)}")
    log(f"  PSNR : {avg_p:.2f} ± {std_p:.2f} dB")
    log(f"  SSIM : {avg_s:.4f} ± {std_s:.4f}\n{'='*60}\n")
    with open(os.path.join(save_dir,f"{tag}_results.txt"),'w',encoding='utf-8') as f:
        f.write(f"=== V5-STABLE {tag.upper()}{tta_str} ===\n\n")
        f.write(f"PSNR : {avg_p:.2f} ± {std_p:.2f} dB\n")
        f.write(f"SSIM : {avg_s:.4f} ± {std_s:.4f}\n")
        f.write(f"N    : {len(psnrs)}\n")
        f.write(f"torch: {torch.__version__} | device: {device}\n")
    return avg_p,avg_s

def plot_history(history, save_dir):
    keys=[k for k in history if k!='train_loss']
    n=len(keys)+1
    fig,axes=plt.subplots(1,n,figsize=(4*n,4))
    ep=range(1,len(history['train_loss'])+1)
    axes[0].plot(ep,history['train_loss'],'b-o',ms=2,lw=1.5)
    axes[0].set_title('Total loss'); axes[0].grid(True)
    for ax,k in zip(axes[1:],keys):
        ax.plot(ep,history[k],ms=2,lw=1.5); ax.set_title(k); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'training_curves.png'),dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    use_ddp='LOCAL_RANK' in os.environ
    if use_ddp:
        local_rank=setup_ddp()
        device=torch.device(f'cuda:{local_rank}')
        world_size=dist.get_world_size(); rank=dist.get_rank()
    else:
        local_rank=0; rank=0; world_size=1
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log(f"Device: {device}  rank={rank}/{world_size}")
    if rank==0:
        log(f"GPU  : {torch.cuda.get_device_name(local_rank)}")
        log(f"VRAM : {torch.cuda.get_device_properties(local_rank).total_memory/1e9:.1f} GB")

    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    os.makedirs(SAVE_DIR,exist_ok=True)

    train_ds=LDCTDataset(LDCT_ROOT,NDCT_ROOT,TRAIN_PATIENTS,mode='train',patch_size=256)
    val_ds  =LDCTDataset(LDCT_ROOT,NDCT_ROOT,VAL_PATIENTS,  mode='val',  patch_size=0)
    test_ds =LDCTDataset(LDCT_ROOT,NDCT_ROOT,TEST_PATIENTS, mode='test', patch_size=0)
    train_sampler=DistributedSampler(train_ds,shuffle=True) if use_ddp else None

    model=LDCTDenoiserV5(
        in_ch=1,bc=160,growth=40,bot_depth=12,bot_heads=10,ws=8,
        dec_depths=(3,3,2,2),drop=0.02,attn_drop=0.02).to(device)
    n_params=sum(p.numel() for p in model.parameters())/1e6
    log(f"Parameters: {n_params:.1f}M")

    ema=EMA(model,decay=0.99995)  # FIX-1: EMA内部自动处理warmup
    if use_ddp:
        model=DDP(model,device_ids=[local_rank],
                  find_unused_parameters=False,gradient_as_bucket_view=True)

    NUM_EPOCHS=200; BATCH_PER_GPU=8
    TOTAL_BATCH=BATCH_PER_GPU*world_size*ACCUM_STEPS
    scaled_lr = 5e-5
    log(f"total_batch(等效)={TOTAL_BATCH}  scaled_lr={scaled_lr:.2e}")

    optimizer=optim.AdamW(build_param_groups(model,scaled_lr),
                          weight_decay=1e-4,betas=(0.9,0.999))
    criterion=CompositeLoss(device=str(device))

    scaler=torch.amp.GradScaler('cuda', init_scale=128.0, growth_interval=2000)

    history={'train_loss':[]}; best_psnr=0.; start_epoch=1

    search_dirs=[SAVE_DIR]+[
        SAVE_DIR.replace('v5_stable', d)
        for d in ['v5_a100_fixed','v5_a100_opt']]
    all_ckpts=sorted(sum(
        [_glob.glob(os.path.join(d,"ckpt_ep*.pth")) for d in search_dirs],[]))

    if all_ckpts and rank==0:
        resume_path=all_ckpts[-1]
        log(f"\n[续训] 加载: {resume_path}")
        ckpt_data=torch.load(resume_path,map_location=device)
        m=model.module if hasattr(model,'module') else model
        m.load_state_dict(ckpt_data['model'])
        ema.shadow.load_state_dict(ckpt_data['ema'])
        history=ckpt_data.get('history',{'train_loss':[]})
        best_psnr=ckpt_data.get('best_psnr',0.)
        start_epoch=ckpt_data['epoch']+1
        # FIX-1: 续训时恢复EMA的step计数，避免warmup重置
        resumed_steps = start_epoch * 214  # 约估每epoch的batch数
        ema.step = resumed_steps
        log(f"[续训] 从 epoch {start_epoch} 继续  最优PSNR={best_psnr:.2f}  ema.step≈{resumed_steps}")

    if use_ddp:
        t=torch.tensor(start_epoch,device=device)
        dist.broadcast(t,src=0); start_epoch=int(t.item())

    cur_ps=get_patch_size(start_epoch)
    cur_batch=get_batch_size(cur_ps)
    train_ds.set_patch_size(cur_ps)

    def make_loader(bs):
        if use_ddp:
            return DataLoader(train_ds,batch_size=bs,sampler=train_sampler,
                              num_workers=8,pin_memory=True,
                              persistent_workers=True,prefetch_factor=2)
        return DataLoader(train_ds,batch_size=bs,shuffle=True,
                          num_workers=8,pin_memory=True,
                          persistent_workers=True,prefetch_factor=2)

    train_loader=make_loader(cur_batch)
    if rank==0:
        val_loader =DataLoader(val_ds, batch_size=1,shuffle=False,num_workers=4)
        test_loader=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=4)

    log(f"[初始化] patch={cur_ps}  batch/gpu={cur_batch}  "
        f"accum={ACCUM_STEPS}  等效total={cur_batch*world_size*ACCUM_STEPS}  "
        f"start_ep={start_epoch}\n")

    for epoch in range(start_epoch, NUM_EPOCHS+1):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        cur_lr=adjust_lr(optimizer,epoch,base_lr=scaled_lr,
                         warmup_epochs=15,total_epochs=NUM_EPOCHS,min_lr=5e-6)
        if rank==0:
            log(f"[lr] ep{epoch}  decoder={cur_lr:.2e}  "
                f"encoder={cur_lr*LR_SCALES['encoder']:.2e}  "
                f"bottleneck={cur_lr*LR_SCALES['bottleneck']:.2e}")

        if epoch==150:
            log("[OPT] epoch=150，关闭 Dropout")
            m=model.module if hasattr(model,'module') else model
            for mod in m.modules():
                if isinstance(mod,nn.Dropout): mod.p=0.0

        target_ps=get_patch_size(epoch); target_batch=get_batch_size(target_ps)
        if target_ps!=cur_ps or target_batch!=cur_batch:
            train_ds.set_patch_size(target_ps)
            train_loader=make_loader(target_batch)
            log(f"[Epoch {epoch}] patch {cur_ps}→{target_ps}  "
                f"batch/gpu {cur_batch}→{target_batch}")
            cur_ps,cur_batch=target_ps,target_batch

        train_one_epoch(model,ema,train_loader,optimizer,criterion,
                        scaler,device,epoch,history,SAVE_DIR,
                        base_lr=scaled_lr,rank=rank,use_ddp=use_ddp)


        if epoch % 15 == 0 and rank == 0:
            use_tta=(epoch>150)
            ema.eval()
            avg_p_ema,avg_s_ema=evaluate(ema.shadow,val_loader,device,SAVE_DIR,
                                          tile=512,overlap=64,
                                          tag=f'val_ema_ep{epoch}',use_tta=use_tta)
            raw_m=model.module if hasattr(model,'module') else model
            raw_m.eval()
            avg_p_raw,avg_s_raw=evaluate(raw_m,val_loader,device,SAVE_DIR,
                                          tile=512,overlap=64,
                                          tag=f'val_raw_ep{epoch}',use_tta=use_tta)
            raw_m.train()

            if avg_p_ema>=avg_p_raw:
                avg_p,avg_s,src=avg_p_ema,avg_s_ema,'EMA'
            else:
                avg_p,avg_s,src=avg_p_raw,avg_s_raw,'RAW'
            log(f"  EMA:{avg_p_ema:.2f}  RAW:{avg_p_raw:.2f}  取最优={avg_p:.2f} ({src})")

            disk=os.statvfs(SAVE_DIR)
            free_gb=disk.f_bavail*disk.f_frsize/1e9
            if free_gb<5.0: log(f"  [警告] 磁盘剩余 {free_gb:.1f} GB！")

            m_save=model.module if hasattr(model,'module') else model
            ckpt_data={'epoch':epoch,'model':m_save.state_dict(),
                       'ema':ema.shadow.state_dict(),'opt':optimizer.state_dict(),
                       'history':history,'best_psnr':best_psnr,'best_ssim':avg_s,
                       'ema_step':ema.step}  # FIX-1: 保存ema.step供续训恢复
            best_psnr=save_ckpt(ckpt_data,SAVE_DIR,epoch,avg_p,best_psnr,max_keep=3)
            model.train()

    if rank==0:
        log("\n训练完成！最终评估 (EMA + TTA×16)...")
        ema.eval()
        evaluate(ema.shadow,val_loader,device,SAVE_DIR,tile=512,overlap=96,tag='final_val',use_tta=True)
        evaluate(ema.shadow,test_loader,device,SAVE_DIR,tile=512,overlap=96,tag='test_final',use_tta=True)
        plot_history(history,SAVE_DIR)
        log(f"\n所有文件保存至: {SAVE_DIR}")

    if use_ddp: cleanup_ddp()


if __name__=='__main__':
    main()