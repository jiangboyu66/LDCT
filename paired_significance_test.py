"""
Paired statistical significance testing: HCT-UNet vs. each baseline.

Consumes the per-slice CSVs produced by the patched evaluate() function
(see evaluate_patch.md). Run this AFTER you have generated one CSV per
method on the SAME test slices in the SAME order.

Usage:
    python paired_significance_test.py --save_dir /path/to/SAVE_DIR --tag test_final

Outputs:
    - Printed table: mean diff, Wilcoxon p, paired-t p, Cohen's d_z, for each
      baseline vs HCT-UNet, for both PSNR and SSIM.
    - {save_dir}/significance_results.csv (machine-readable)
    - {save_dir}/significance_results.md   (paste-ready markdown table)

Why Wilcoxon signed-rank (primary) + paired t-test (secondary):
    This is a matched-pairs design (same slice, two methods) evaluated on a
    single held-out test patient, so slices are NOT independent draws from a
    homogeneous population — they cluster by anatomical region and local
    noise level. Wilcoxon only assumes the paired differences are symmetric
    about their median, which is a much weaker and more defensible
    assumption here than the paired t-test's normality-of-differences
    assumption. Both are reported so you can show agreement; if they
    disagree, that disagreement itself is worth reporting rather than
    hiding, and a Shapiro-Wilk normality check on the differences is
    printed to help you judge which test to trust more.

    Multiple-comparisons correction: comparing HCT-UNet against 5 baselines
    on 2 metrics is 10 tests. Holm-Bonferroni correction is applied across
    all 10 so a single significant p-value isn't a multiple-comparisons
    artifact. This matters directly for your advisor's ask — an uncorrected
    p < 0.05 against 5 baselines is a much weaker claim than a corrected one.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d_z(diff):
    """Effect size for paired differences: mean diff / std of diff."""
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def load_method_csvs(save_dir, tag):
    pattern = os.path.join(save_dir, f'{tag}_*_per_slice.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No per-slice CSVs found matching {pattern}. "
            f"Did you run the patched evaluate() for each method with this tag?"
        )
    frames = {}
    for f in files:
        m = re.match(rf'{re.escape(tag)}_(.+)_per_slice\.csv', os.path.basename(f))
        method = m.group(1) if m else os.path.basename(f)
        df = pd.read_csv(f)
        frames[method] = df.sort_values('slice_index').reset_index(drop=True)
    return frames


def align_and_check(frames, proposed_name):
    """Verify every method has identical slice_index sets before pairing."""
    if proposed_name not in frames:
        raise ValueError(
            f"'{proposed_name}' not found among methods: {list(frames.keys())}. "
            f"Pass --proposed_name matching your CSV filename exactly."
        )
    ref_idx = set(frames[proposed_name]['slice_index'])
    problems = []
    for name, df in frames.items():
        idx = set(df['slice_index'])
        if idx != ref_idx:
            missing = ref_idx - idx
            extra = idx - ref_idx
            problems.append(
                f"  {name}: {len(idx)} slices vs {proposed_name}'s {len(ref_idx)}. "
                f"Missing {len(missing)}, extra {len(extra)}."
            )
    if problems:
        raise ValueError(
            "Slice alignment mismatch — DO NOT proceed with a position-based "
            "paired test until every method covers the exact same slices:\n"
            + "\n".join(problems)
        )
    return sorted(ref_idx)


def run_tests(frames, proposed_name, metric, alpha=0.05):
    shared_idx = align_and_check(frames, proposed_name)
    proposed_vals = (
        frames[proposed_name].set_index('slice_index').loc[shared_idx, metric].values
    )

    rows = []
    for name, df in frames.items():
        if name == proposed_name:
            continue
        baseline_vals = df.set_index('slice_index').loc[shared_idx, metric].values
        diff = proposed_vals - baseline_vals  # positive = proposed better

        # Wilcoxon signed-rank (primary). Falls back gracefully if all diffs
        # are zero (degenerate, but avoids a crash on toy data).
        if np.allclose(diff, 0):
            w_stat, w_p = np.nan, 1.0
        else:
            w_stat, w_p = stats.wilcoxon(diff, alternative='two-sided',
                                          zero_method='wilcox')

        # Paired t-test (secondary, for comparison).
        t_stat, t_p = stats.ttest_rel(proposed_vals, baseline_vals)

        # Normality check on the differences, to help judge which test to trust.
        if len(diff) >= 3:
            sh_stat, sh_p = stats.shapiro(diff)
        else:
            sh_stat, sh_p = np.nan, np.nan

        d_z = cohens_d_z(diff)

        rows.append({
            'baseline': name,
            'metric': metric,
            'n_slices': len(shared_idx),
            'mean_diff': float(np.mean(diff)),
            'median_diff': float(np.median(diff)),
            'wilcoxon_stat': float(w_stat) if not np.isnan(w_stat) else np.nan,
            'wilcoxon_p_raw': float(w_p),
            'paired_t_stat': float(t_stat),
            'paired_t_p_raw': float(t_p),
            'shapiro_p_on_diffs': float(sh_p) if not np.isnan(sh_p) else np.nan,
            'cohens_d_z': d_z,
        })

    result_df = pd.DataFrame(rows)

    # Holm-Bonferroni correction across ALL tests passed in (caller controls
    # whether that's per-metric or the full PSNR+SSIM x baselines family —
    # see main() where both metrics are corrected together).
    return result_df


def holm_bonferroni(pvals, alpha=0.05):
    """Returns adjusted p-values and reject/accept booleans."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    reject = adjusted < alpha
    return adjusted, reject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', required=True,
                     help='Directory containing {tag}_{method}_per_slice.csv files')
    ap.add_argument('--tag', default='test_final',
                     help='Tag prefix used when evaluate() wrote the CSVs')
    ap.add_argument('--proposed_name', default='HCT-UNet',
                     help='method_name string used for your proposed model')
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    frames = load_method_csvs(args.save_dir, args.tag)
    print(f"Found methods: {list(frames.keys())}")
    for name, df in frames.items():
        print(f"  {name}: {len(df)} slices")

    all_results = []
    for metric in ['psnr', 'ssim']:
        res = run_tests(frames, args.proposed_name, metric, args.alpha)
        all_results.append(res)
    combined = pd.concat(all_results, ignore_index=True)

    # Holm-Bonferroni across the FULL family (all baselines x both metrics),
    # applied separately to the Wilcoxon p-values and the t-test p-values.
    combined['wilcoxon_p_holm'], combined['wilcoxon_significant'] = \
        holm_bonferroni(combined['wilcoxon_p_raw'].values, args.alpha)
    combined['paired_t_p_holm'], combined['paired_t_significant'] = \
        holm_bonferroni(combined['paired_t_p_raw'].values, args.alpha)

    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', None)
    print("\n" + "=" * 100)
    print("PAIRED SIGNIFICANCE TEST: HCT-UNet vs. baselines "
          f"(Holm-Bonferroni corrected, family size = {len(combined)})")
    print("=" * 100)
    display_cols = ['baseline', 'metric', 'n_slices', 'mean_diff',
                     'wilcoxon_p_raw', 'wilcoxon_p_holm', 'wilcoxon_significant',
                     'paired_t_p_raw', 'paired_t_p_holm', 'paired_t_significant',
                     'cohens_d_z', 'shapiro_p_on_diffs']
    print(combined[display_cols].to_string(index=False))

    csv_out = os.path.join(args.save_dir, 'significance_results.csv')
    combined.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    # Paste-ready markdown table for the report.
    md_path = os.path.join(args.save_dir, 'significance_results.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("| Baseline | Metric | N | Mean Δ (proposed − baseline) | "
                "Wilcoxon p (Holm) | Paired t p (Holm) | Cohen's d_z |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in combined.iterrows():
            sig_mark = "**" if r['wilcoxon_significant'] else ""
            f.write(
                f"| {r['baseline']} | {r['metric'].upper()} | {int(r['n_slices'])} | "
                f"{sig_mark}{r['mean_diff']:+.3f}{sig_mark} | "
                f"{r['wilcoxon_p_holm']:.4g} | {r['paired_t_p_holm']:.4g} | "
                f"{r['cohens_d_z']:.3f} |\n"
            )
    print(f"Saved: {md_path}")

    print("\nInterpretation notes:")
    print("  - mean_diff > 0 means HCT-UNet outperforms the baseline on average.")
    print("  - A shapiro_p_on_diffs < 0.05 means the paired differences are NOT")
    print("    normally distributed — trust the Wilcoxon result over the t-test")
    print("    in that row.")
    print("  - |Cohen's d_z| < 0.2 is a small effect even if p is significant;")
    print("    with N in the hundreds of slices, statistical significance and")
    print("    practical/clinical significance can diverge. Report both.")
    print("  - If wilcoxon_significant is False for CT-Mamba specifically, that")
    print("    is a legitimate and reportable finding — your report's own text")
    print("    already states the two methods are 'near-parity' in places; a")
    print("    non-significant result there is consistent with your own")
    print("    qualitative claims, not a failure of the experiment.")


if __name__ == '__main__':
    main()