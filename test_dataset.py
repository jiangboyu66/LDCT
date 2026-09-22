"""
CT Slice Alignment Verifier
验证 LDCT/NDCT .npy 数据集是否完美对齐
"""

import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. 基础统计指标
# ──────────────────────────────────────────────

def compute_basic_stats(ldct: np.ndarray, ndct: np.ndarray) -> Dict:
    """形状、HU范围、均值/方差"""
    return {
        "shape_match": ldct.shape == ndct.shape,
        "ldct_shape": list(ldct.shape),
        "ndct_shape": list(ndct.shape),
        "ldct_mean": float(np.mean(ldct)),
        "ndct_mean": float(np.mean(ndct)),
        "ldct_std": float(np.std(ldct)),
        "ndct_std": float(np.std(ndct)),
        "mean_diff": float(abs(np.mean(ldct) - np.mean(ndct))),
        "ldct_range": [float(ldct.min()), float(ldct.max())],
        "ndct_range": [float(ndct.min()), float(ndct.max())],
    }

# ──────────────────────────────────────────────
# 2. 结构相似性 (SSIM)
# ──────────────────────────────────────────────

def compute_ssim(ldct: np.ndarray, ndct: np.ndarray) -> float:
    """手动实现 SSIM，不依赖 scikit-image"""
    C1 = (0.01 * (ndct.max() - ndct.min())) ** 2
    C2 = (0.03 * (ndct.max() - ndct.min())) ** 2

    mu_x = np.mean(ldct)
    mu_y = np.mean(ndct)
    sigma_x = np.std(ldct)
    sigma_y = np.std(ndct)
    sigma_xy = np.mean((ldct - mu_x) * (ndct - mu_y))

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))
    return float(ssim)

# ──────────────────────────────────────────────
# 3. 互信息对齐检测
# ──────────────────────────────────────────────

def compute_mutual_information(ldct: np.ndarray, ndct: np.ndarray, bins: int = 64) -> float:
    """互信息：对齐图像 MI 应显著高于错位图像"""
    # 归一化到 [0, bins-1]
    def normalize(arr):
        arr = arr.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr, dtype=np.int32)
        return ((arr - mn) / (mx - mn) * (bins - 1)).astype(np.int32)

    x = normalize(ldct).ravel()
    y = normalize(ndct).ravel()

    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    px_py = px[:, None] * py[None, :]
    nonzero = pxy > 0
    mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
    return float(mi)

# ──────────────────────────────────────────────
# 4. 平移错位检测（互相关）
# ──────────────────────────────────────────────

def detect_shift(ldct: np.ndarray, ndct: np.ndarray) -> Dict:
    """用互相关检测像素级平移错位"""
    try:
        from numpy.fft import fft2, ifft2, fftshift

        # 归一化
        def norm(arr):
            a = arr.astype(np.float32)
            return (a - a.mean()) / (a.std() + 1e-8)

        f1 = fft2(norm(ldct))
        f2 = fft2(norm(ndct))
        cross_power = f1 * np.conj(f2)
        denom = np.abs(cross_power)
        denom[denom < 1e-10] = 1e-10
        phase_corr = ifft2(cross_power / denom)
        phase_corr = np.abs(fftshift(phase_corr))

        cy, cx = np.array(phase_corr.shape) // 2
        peak_y, peak_x = np.unravel_index(phase_corr.argmax(), phase_corr.shape)
        shift_y = peak_y - cy
        shift_x = peak_x - cx
        peak_val = float(phase_corr.max())

        return {
            "shift_y_px": int(shift_y),
            "shift_x_px": int(shift_x),
            "shift_magnitude_px": float(np.sqrt(shift_y**2 + shift_x**2)),
            "correlation_peak": peak_val,
            "perfectly_aligned": abs(shift_y) <= 1 and abs(shift_x) <= 1
        }
    except Exception as e:
        return {"error": str(e)}

# ──────────────────────────────────────────────
# 5. 解剖结构一致性（骨骼/软组织 mask 重叠）
# ──────────────────────────────────────────────

def compute_mask_overlap(ldct: np.ndarray, ndct: np.ndarray) -> Dict:
    """
    对特定 HU 阈值区域做 Dice 系数
    骨骼 >400 HU，软组织 -100~100 HU，肺 <-500 HU
    """
    regions = {
        "bone_HU_gt400": (400, 3000),
        "soft_tissue_HU_m100_100": (-100, 100),
        "lung_HU_lt_m500": (-1500, -500),
    }
    results = {}
    for name, (lo, hi) in regions.items():
        mask_l = (ldct >= lo) & (ldct <= hi)
        mask_n = (ndct >= lo) & (ndct <= hi)
        intersection = np.logical_and(mask_l, mask_n).sum()
        union_sum = mask_l.sum() + mask_n.sum()
        dice = float(2 * intersection / (union_sum + 1e-8))
        iou = float(intersection / (np.logical_or(mask_l, mask_n).sum() + 1e-8))
        results[name] = {"dice": round(dice, 4), "iou": round(iou, 4)}
    return results

# ──────────────────────────────────────────────
# 6. 综合判断
# ──────────────────────────────────────────────

def grade_alignment(report: Dict) -> Tuple[str, List[str]]:
    """给出对齐质量等级和警告列表"""
    warnings_list = []
    score = 100

    stats = report.get("basic_stats", {})
    if not stats.get("shape_match", True):
        warnings_list.append("❌ 形状不匹配！")
        score -= 40

    if stats.get("mean_diff", 0) > 50:
        warnings_list.append(f"⚠️  均值差异过大: {stats['mean_diff']:.1f} HU（预期 <50）")
        score -= 15

    ssim = report.get("ssim", 0)
    if ssim < 0.5:
        warnings_list.append(f"❌ SSIM 过低: {ssim:.3f}（对齐图像通常 >0.5）")
        score -= 20
    elif ssim < 0.7:
        warnings_list.append(f"⚠️  SSIM 偏低: {ssim:.3f}（建议 >0.7）")
        score -= 10

    mi = report.get("mutual_information", 0)
    if mi < 0.3:
        warnings_list.append(f"⚠️  互信息偏低: {mi:.3f}（可能存在错位）")
        score -= 10

    shift = report.get("shift_detection", {})
    mag = shift.get("shift_magnitude_px", 0)
    if mag > 5:
        warnings_list.append(f"❌ 检测到显著平移: {mag:.1f} px (dy={shift.get('shift_y_px')}, dx={shift.get('shift_x_px')})")
        score -= 25
    elif mag > 2:
        warnings_list.append(f"⚠️  轻微平移: {mag:.1f} px")
        score -= 10

    mask = report.get("mask_overlap", {})
    bone_dice = mask.get("bone_HU_gt400", {}).get("dice", 1.0)
    if bone_dice < 0.7:
        warnings_list.append(f"❌ 骨骼区域 Dice 过低: {bone_dice:.3f}（预期 >0.7）")
        score -= 20
    elif bone_dice < 0.85:
        warnings_list.append(f"⚠️  骨骼区域 Dice 偏低: {bone_dice:.3f}（建议 >0.85）")
        score -= 8

    lung_dice = mask.get("lung_HU_lt_m500", {}).get("dice", 1.0)
    if lung_dice < 0.75:
        warnings_list.append(f"⚠️  肺区域 Dice 偏低: {lung_dice:.3f}")
        score -= 5

    score = max(0, score)
    if score >= 90:
        grade = "✅ PERFECT（完美对齐）"
    elif score >= 75:
        grade = "✅ GOOD（良好）"
    elif score >= 55:
        grade = "⚠️  SUSPECT（可疑，建议人工复查）"
    else:
        grade = "❌ MISALIGNED（错位，会造成性能损失）"

    return grade, warnings_list, score

# ──────────────────────────────────────────────
# 7. 主验证函数
# ──────────────────────────────────────────────

def verify_pair(ldct_path: str, ndct_path: str, verbose: bool = True) -> Dict:
    """验证一对 LDCT/NDCT .npy 切片"""
    ldct = np.load(ldct_path).astype(np.float32)
    ndct = np.load(ndct_path).astype(np.float32)

    # 处理多维数组（如 (1, H, W) 或 (H, W, 1)）
    if ldct.ndim == 3 and ldct.shape[0] == 1:
        ldct = ldct[0]
    if ndct.ndim == 3 and ndct.shape[0] == 1:
        ndct = ndct[0]
    if ldct.ndim == 3 and ldct.shape[-1] == 1:
        ldct = ldct[..., 0]
    if ndct.ndim == 3 and ndct.shape[-1] == 1:
        ndct = ndct[..., 0]

    report = {
        "ldct_path": ldct_path,
        "ndct_path": ndct_path,
        "basic_stats": compute_basic_stats(ldct, ndct),
    }

    if ldct.shape == ndct.shape and ldct.ndim == 2:
        report["ssim"] = compute_ssim(ldct, ndct)
        report["mutual_information"] = compute_mutual_information(ldct, ndct)
        report["shift_detection"] = detect_shift(ldct, ndct)
        report["mask_overlap"] = compute_mask_overlap(ldct, ndct)
    else:
        report["error"] = "形状不匹配，跳过高级指标"

    grade, warnings_list, score = grade_alignment(report)
    report["alignment_score"] = score
    report["grade"] = grade
    report["warnings"] = warnings_list

    if verbose:
        print(f"\n{'='*60}")
        print(f"LDCT: {os.path.basename(ldct_path)}")
        print(f"NDCT: {os.path.basename(ndct_path)}")
        print(f"{'─'*60}")
        stats = report["basic_stats"]
        print(f"形状匹配:    {'✅' if stats['shape_match'] else '❌'}  {stats['ldct_shape']} vs {stats['ndct_shape']}")
        print(f"SSIM:        {report.get('ssim', 'N/A'):.4f}" if 'ssim' in report else "SSIM:        N/A")
        print(f"互信息:      {report.get('mutual_information', 'N/A'):.4f}" if 'mutual_information' in report else "互信息:      N/A")
        if 'shift_detection' in report:
            s = report['shift_detection']
            print(f"平移偏移:    dy={s.get('shift_y_px',0)}px  dx={s.get('shift_x_px',0)}px  magnitude={s.get('shift_magnitude_px',0):.2f}px")
        if 'mask_overlap' in report:
            for region, vals in report['mask_overlap'].items():
                print(f"  {region:35s} Dice={vals['dice']:.4f}  IoU={vals['iou']:.4f}")
        print(f"{'─'*60}")
        print(f"对齐分数:    {score}/100")
        print(f"对齐等级:    {grade}")
        if warnings_list:
            print("警告:")
            for w in warnings_list:
                print(f"  {w}")

    return report


def verify_dataset(root_dir: str, ldct_subdir: str = "LDCT", ndct_subdir: str = "NDCT",
                   output_json: str = None) -> Dict:
    """
    批量验证整个数据集
    目录结构：root_dir/LDCT/*.npy  和  root_dir/NDCT/*.npy
    """
    ldct_dir = Path(root_dir) / ldct_subdir
    ndct_dir = Path(root_dir) / ndct_subdir

    ldct_files = sorted(ldct_dir.glob("*.npy"))
    ndct_files = sorted(ndct_dir.glob("*.npy"))

    print(f"\n📂 数据集根目录: {root_dir}")
    print(f"   LDCT 切片数: {len(ldct_files)}")
    print(f"   NDCT 切片数: {len(ndct_files)}")

    if len(ldct_files) != len(ndct_files):
        print(f"❌ 文件数量不匹配！LDCT={len(ldct_files)} vs NDCT={len(ndct_files)}")

    results = []
    scores = []
    perfect = 0
    suspect = 0
    misaligned = 0

    for lf, nf in zip(ldct_files, ndct_files):
        if lf.stem != nf.stem:
            print(f"⚠️  文件名不匹配: {lf.stem} vs {nf.stem}")
        rep = verify_pair(str(lf), str(nf), verbose=True)
        results.append(rep)
        scores.append(rep["alignment_score"])
        if rep["alignment_score"] >= 90:
            perfect += 1
        elif rep["alignment_score"] >= 55:
            suspect += 1
        else:
            misaligned += 1

    summary = {
        "total_pairs": len(results),
        "perfect_or_good": perfect,
        "suspect": suspect,
        "misaligned": misaligned,
        "mean_score": float(np.mean(scores)) if scores else 0,
        "min_score": float(np.min(scores)) if scores else 0,
        "max_score": float(np.max(scores)) if scores else 0,
    }

    print(f"\n{'='*60}")
    print(f"📊 数据集汇总")
    print(f"{'─'*60}")
    print(f"总切片对:      {summary['total_pairs']}")
    print(f"✅ 完美/良好:  {summary['perfect_or_good']} ({100*summary['perfect_or_good']/max(1,summary['total_pairs']):.1f}%)")
    print(f"⚠️  可疑:      {summary['suspect']}")
    print(f"❌ 错位:       {summary['misaligned']}")
    print(f"平均对齐分:    {summary['mean_score']:.1f}/100")
    print(f"最低对齐分:    {summary['min_score']:.1f}/100")

    full_report = {"summary": summary, "pairs": results}
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 详细报告已保存: {output_json}")

    return full_report


# ──────────────────────────────────────────────
# 8. CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT切片对齐验证工具")
    sub = parser.add_subparsers(dest="mode")

    # 单对验证
    sp = sub.add_parser("pair", help="验证单对 LDCT/NDCT")
    sp.add_argument("ldct", help="LDCT .npy 路径")
    sp.add_argument("ndct", help="NDCT .npy 路径")
    sp.add_argument("--json", default=None, help="输出 JSON 路径")

    # 批量验证
    ds = sub.add_parser("dataset", help="批量验证数据集")
    ds.add_argument("root", help="数据集根目录")
    ds.add_argument("--ldct", default="LDCT", help="LDCT 子目录名")
    ds.add_argument("--ndct", default="NDCT", help="NDCT 子目录名")
    ds.add_argument("--json", default="alignment_report.json", help="输出 JSON 路径")

    args = parser.parse_args()

    if args.mode == "pair":
        rep = verify_pair(args.ldct, args.ndct)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rep, f, indent=2)
    elif args.mode == "dataset":
        verify_dataset(args.root, args.ldct, args.ndct, args.json)
    else:
        # Demo with synthetic data
        print("🔬 生成合成演示数据...")
        np.random.seed(42)
        H, W = 512, 512

        # 模拟真实 CT（ndct），添加噪声得到 ldct
        ndct_demo = np.random.randn(H, W).astype(np.float32) * 150 + 50
        ldct_demo = ndct_demo + np.random.randn(H, W).astype(np.float32) * 80  # 噪声

        np.save("/tmp/ldct_demo.npy", ldct_demo)
        np.save("/tmp/ndct_demo.npy", ndct_demo)

        print("\n--- 演示1：完美对齐（LDCT = NDCT + 噪声）---")
        verify_pair("/tmp/ldct_demo.npy", "/tmp/ndct_demo.npy")

        # 模拟错位（平移 15 px）
        ndct_shifted = np.roll(ndct_demo, shift=(15, 10), axis=(0, 1))
        np.save("/tmp/ndct_shifted.npy", ndct_shifted)

        print("\n--- 演示2：错位对齐（NDCT 平移 15px）---")
        verify_pair("/tmp/ldct_demo.npy", "/tmp/ndct_shifted.npy")