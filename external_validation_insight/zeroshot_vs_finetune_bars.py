#!/usr/bin/env python3
"""INSIGHT zero-shot vs fine-tuned SPARC-P AUROC per task and site, one bar figure per fine-tuning strategy.
Also provides the result loaders used by zeroshot_vs_finetune_scatter.py.
Usage: python zeroshot_vs_finetune_bars.py [-ci ensemble] [--metric overall]
"""
# put repo root, data/ and models/ on sys.path
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models'), os.path.join(_R, 'results_vis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
RESULTS = sparc_paths.INSIGHT_RESULTS_DIR

import argparse

import os
import numpy as np
import pandas as pd
import matplotlib
from scipy import stats
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator


# =============================================================================
# Font
# =============================================================================

# folder with the Liberation Sans .ttf files; unset -> matplotlib's default font
FONT_DIR = os.environ.get('SPARC_FONT_DIR', '')
for _fname in [
    "LiberationSans-Regular.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSans-Italic.ttf",
    "LiberationSans-BoldItalic.ttf",
]:
    _fp = os.path.join(FONT_DIR, _fname)
    if FONT_DIR and os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)

rcParams["font.family"] = "Liberation Sans"
rcParams["axes.unicode_minus"] = False


# =============================================================================
# Paths
# =============================================================================

# Results from INSIGHT_finetune/res (LT and neutropenia B/C2 retrained from the lr3e-4 SPARC-P).
# positive_counts.csv / training_counts.csv and n_numbers.csv (same data and splits) are copied into res/.
BASE = os.path.join(RESULTS, 'INSIGHT_finetune_lr3e4', 'res_all')

ZEROSHOT_FOLDER = "zeroshot"

# (folder, figure basename, figure title)
# (folder, figure basename, title). The two "from MSK pretrain" variants start
# from the task-naive MSK-CAIA checkpoint; the two "from MSK finetune" variants
# start from the MSK checkpoint already finetuned for that task.
FINETUNE_VARIANTS = [
    ("insight-finetune-mskpretrain",
     "insight_zeroshot_vs_finetune_sitespecific",
     "Site-specific finetuning (from MSK pretrain)"),
    ("insight-finetune-mskfinetuned",
     "insight_zeroshot_vs_finetune_sitespecific_mskfinetuned",
     "Site-specific finetuning (from MSK finetune)"),
    ("insight-finetune-pooled-mskpretrain",
     "insight_zeroshot_vs_finetune_pooled",
     "Pooled finetuning (from MSK pretrain)"),
    ("insight-finetune-pooled-mskfinetuned",
     "insight_zeroshot_vs_finetune_pooled_mskfinetuned",
     "Pooled finetuning (from MSK finetune)"),
]

SAVE_DIR = os.path.join(RESULTS, 'INSIGHT_finetune_lr3e4', 'vis')

_ap = argparse.ArgumentParser()
_ap.add_argument("-ci", "--ci", choices=["folds", "ensemble"], default="folds",
                 help="folds: mean +/- SD over folds with ensemble diamond (default). "
                      "ensemble: ensemble bar with 95%% bootstrap CI, no diamond, "
                      "files suffixed _ensembleCI.")
# --metric overall (or env INSIGHT_AUROC=overall, which the scatter script inherits): side-effect
# tasks use the overall next-infusion AUROC instead of the weighted one; 180-day mortality is
# unchanged. Every output gets _overallauroc in its name.
_ap.add_argument("--metric", choices=["weighted", "overall"],
                 default=os.environ.get("INSIGHT_AUROC", "weighted"))
_args, _ = _ap.parse_known_args()
CI_MODE = _args.ci
METRIC = _args.metric
METRIC_SUFFIX = "_overallauroc" if METRIC == "overall" else ""
METRIC_WORD = "weighted" if METRIC == "weighted" else "overall"

# Reruns of the same models/data that add the ensemble bootstrap CI columns
# (eval_insight_multimodel.py -nb 1000), keyed by the BASE subfolder they reproduce.
CI_BASE = os.path.join(RESULTS, 'INSIGHT_weightedci')
CI_RESULTS = {
    "zeroshot":                             "res_zeroshot_testsplit",
    "insight-finetune-mskfinetuned":        "res_ft_sitespecific",
    "insight-finetune-pooled-mskfinetuned": "res_ft_pooled",
}
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Sites / tasks
# =============================================================================

SITES = list("ABCDEF")

TASKS = [
    # (task_suffix,          display_name)
    ("binary",              "180-day mortality"),
    ("AKI",                 "Acute kidney injury"),
    ("Liver_toxicity",      "Liver toxicity"),
    ("anemia",              "Anemia"),
    ("febrile_neutropenia", "Febrile neutropenia"),
    ("neutropenia",         "Neutropenia"),
]


def _metric_col(task):
    return "auroc" if task == "binary" else f"test_auroc_nextinf_{METRIC}"


def _ci_cols(task):
    return (("auroc_ci_lower", "auroc_ci_upper") if task == "binary"
            else ("auroc_ci_lower", "auroc_ci_upper") if METRIC == "overall"   # auroc_ci_* = overall
            else ("auroc_weighted_ci_lower", "auroc_weighted_ci_upper"))


# =============================================================================
# N counts (from n_numbers.csv; test splits are strategy-agnostic)
# =============================================================================

N_POS   = {}   # (site, task) -> n_with_medication  (task-eligible patients)
N_TOTAL = {}   # site         -> n_test_total

# Site x task cells with fewer than MIN_N task-eligible patients are dropped:
# the weighted AUROC degenerates (one site has <11 for the *penia / anemia tasks,
# where every fold collapses to exactly 0.0 or 1.0).
MIN_N = 20

# True positive-label counts per (site, task), from get_positive_counts.py.
# N_POS above is n_with_medication - task-ELIGIBLE patients, a denominator, not
# an event count - so it must not be labelled as positives.
N_POSITIVE = {}
_pos_csv = os.path.join(BASE, "positive_counts.csv")
if os.path.exists(_pos_csv):
    _dfp = pd.read_csv(_pos_csv)
    for _, _r in _dfp.iterrows():
        N_POSITIVE[(_r["site"], _r["task"])] = int(_r["n_pos"])

_n_csv = os.path.join(BASE, "n_numbers.csv")
if os.path.exists(_n_csv):
    _df_n = pd.read_csv(_n_csv)
    _df_a = _df_n[_df_n["strategy"] == "A_mskpretrain"].dropna(subset=["n_with_medication"])
    for _, _row in _df_a.iterrows():
        N_POS[(_row["site"], _row["task"])] = int(_row["n_with_medication"])
        if pd.notna(_row["n_test_total"]):
            N_TOTAL[_row["site"]] = int(_row["n_test_total"])


# =============================================================================
# Colors
# =============================================================================

COLORS_ZEROSHOT = {
    'MSK-CAIA': '#78B3C4',                # light teal
    'MSK-CAIA\nprospective': '#8EC5D6',   # lighter teal variant
    'DFCI-CAIA': '#F5A89A',               # light coral
    'FHCC-CAIA': '#D4A8D4',               # light mauve
    'JHU-CAIA': '#BEB0DC',                # light lavender
    'MIMICIV-hosp': '#A89CCE',            # light purple
    'EHRSHOT': '#E4C89A',                 # light amber
    'INSIGHT-A': '#8DC3EA',               # light blue
    'INSIGHT-B': '#A1CEEE',               # lighter blue
    'INSIGHT-C': '#B5D9F2',               # lightest blue
    'INSIGHT-D': '#7AA8D0',               # light steel blue
    'INSIGHT-E': '#6895BD',               # lighter dark blue
    'INSIGHT-F': '#5682AA',               # light darkest blue
}

COLORS_FINETUNE = {
    'MSK-CAIA': '#2D6B7A',                # dark teal
    'MSK-CAIA\nprospective': '#407A8C',   # darker teal variant
    'DFCI-CAIA': '#C6654F',               # dark coral
    'FHCC-CAIA': '#8E5A8E',               # dark mauve
    'JHU-CAIA': '#6E5FA0',                # dark lavender
    'MIMICIV-hosp': '#5B4D8A',            # dark purple
    'EHRSHOT': '#B38A4A',                 # dark amber
    'INSIGHT-A': '#3875A8',               # dark blue
    'INSIGHT-B': '#4682B4',               # darker blue
    'INSIGHT-C': '#5B9BD5',               # even darker blue
    'INSIGHT-D': '#2C5C8E',               # darkest steel blue
    'INSIGHT-E': '#1F4D7B',               # even darker blue
    'INSIGHT-F': '#123E68',               # darkest blue
}

ENSEMBLE_COLOR = "#CC0000"

X_MIN, X_MAX = 45, 108   # room on the right for significance labels
SIG_X        = 104       # x position of the significance annotation


# =============================================================================
# Data loading
# =============================================================================

def _csv_path(folder, site, task):
    return os.path.join(
        BASE, folder, site, "eval_across_models_insight",
        f"model_performance_dict_parsimonious_{task}_SPARC-P_insight.csv",
    )


def load_site_task(folder, site, task):
    """Return (fold_vals, ensemble, reason) for one site x task; reason is None if the cell loaded."""
    fpath = _csv_path(folder, site, task)
    if not os.path.exists(fpath):
        return np.array([]), np.nan, "no result"

    df  = pd.read_csv(fpath)
    col = _metric_col(task)
    if col not in df.columns:
        return np.array([]), np.nan, "no result"

    fold_num  = pd.to_numeric(df["val_fold"], errors="coerce")
    folds     = df.assign(_fold=fold_num)[fold_num.notna()].sort_values("_fold")   # paired_sig matches by position
    fold_vals = folds[col].dropna().astype(float).values

    ens_rows = df[df["val_fold"].astype(str) == "ensemble"]
    ens_val  = np.nan
    if not ens_rows.empty and not pd.isna(ens_rows[col].values[0]):
        ens_val = float(ens_rows[col].values[0])

    return fold_vals, ens_val, None


def _fold_stats(fold_vals):
    """Return (mean, std), or (nan, nan) if there are no folds or all folds equal 0, 0.5 or 1 (no events)."""
    valid = fold_vals[~np.isnan(fold_vals)] if len(fold_vals) else fold_vals
    if len(valid) == 0:
        return np.nan, np.nan
    if np.std(valid) < 1e-9 and np.any(np.abs(valid[0] - np.array([0.0, 0.5, 1.0])) < 1e-6):
        return np.nan, np.nan
    return float(np.mean(valid)), float(np.std(valid))


def _empty_cell(reason):
    return {"mean": np.nan, "std": np.nan, "ens": np.nan, "ci": None,
            "folds": np.array([]), "reason": reason}


def load_ci(folder, site, task, ens_val):
    """(lo, hi) of the ensemble bootstrap CI from the INSIGHT_weightedci rerun, or None."""
    sub = CI_RESULTS.get(folder)
    if sub is None or np.isnan(ens_val):
        return None
    fpath = os.path.join(CI_BASE, sub, site, "eval_across_models_insight",
                         f"model_performance_dict_parsimonious_{task}_SPARC-P_insight.csv")
    if not os.path.exists(fpath):
        print(f"  [ci] {folder} {site} {task}: no CI rerun file")
        return None
    df = pd.read_csv(fpath)
    lo_c, hi_c = _ci_cols(task)
    ens = df[df["val_fold"].astype(str) == "ensemble"]
    if ens.empty or lo_c not in df.columns or pd.isna(ens[lo_c].values[0]):
        print(f"  [ci] {folder} {site} {task}: no CI in rerun")
        return None
    rerun_ens = float(ens[_metric_col(task)].values[0])
    if abs(rerun_ens - ens_val) > 1e-4:
        print(f"  [ci] {folder} {site} {task}: rerun ensemble {rerun_ens:.5f} != plotted "
              f"{ens_val:.5f} -- CI not used")
        return None
    return float(ens[lo_c].values[0]), float(ens[hi_c].values[0])


def collect(folder):
    """task -> site -> dict(mean, std, ens, folds, reason)."""
    out = {}
    for task, _ in TASKS:
        out[task] = {}
        for site in SITES:
            n_pos = N_POS.get((site, task))
            if n_pos is not None and n_pos < MIN_N:
                out[task][site] = _empty_cell("insufficient events")
                continue

            folds, ens, reason = load_site_task(folder, site, task)
            if reason is not None:
                out[task][site] = _empty_cell(reason)
                continue

            mean, std = _fold_stats(folds)
            if np.isnan(mean):
                out[task][site] = _empty_cell("no events")
                continue

            out[task][site] = {
                "mean": mean, "std": std, "ens": ens,
                "ci": load_ci(folder, site, task, ens) if CI_MODE == "ensemble" else None,
                "folds": folds, "reason": None,
            }
    return out


def mask_incomplete_pairs(zs_data, ft_data):
    """Return copies of (zs, ft) with any site x task blanked where either side is unusable."""
    zs_out, ft_out = {}, {}
    for task, _ in TASKS:
        zs_out[task], ft_out[task] = {}, {}
        for site in SITES:
            zs, ft = zs_data[task][site], ft_data[task][site]
            if not (np.isnan(zs["mean"]) or np.isnan(ft["mean"])):
                zs_out[task][site], ft_out[task][site] = zs, ft
                continue
            reason = ft["reason"] or zs["reason"] or "no data"
            zs_out[task][site] = _empty_cell(reason)
            ft_out[task][site] = _empty_cell(reason)
    return zs_out, ft_out


# =============================================================================
# Statistics
# =============================================================================

def _sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "N.S."


def paired_sig(zs_folds, ft_folds):
    """Paired Wilcoxon over matched folds; returns a label or None."""
    if len(zs_folds) == 0 or len(ft_folds) == 0:
        return None
    n = min(len(zs_folds), len(ft_folds))
    a, b = np.asarray(zs_folds[:n]), np.asarray(ft_folds[:n])
    if n < 2 or np.all(a == b):
        return None
    try:
        _, p = stats.wilcoxon(a, b, alternative="two-sided")
    except Exception:
        return None
    return _sig_label(p)


# =============================================================================
# Figure
# =============================================================================

def _site_label(site, task):
    """Site label with the event count (eligible-patient count if positive_counts.csv has no row)."""
    n_ev = N_POSITIVE.get((site, task))
    if n_ev is not None:
        return f"INSIGHT-{site}\n({n_ev:,} events)"
    n_elig = N_POS.get((site, task))
    if n_elig is None:
        return f"INSIGHT-{site}"
    return f"INSIGHT-{site}\n({n_elig:,} eligible)"


def _bar_val(cell):
    return cell["ens"] if CI_MODE == "ensemble" else cell["mean"]


def _bar_err(cell):
    """Fold SD (symmetric), or the asymmetric ensemble CI; None draws no error bar."""
    if CI_MODE == "folds":
        return cell["std"] * 100
    if cell["ci"] is None:
        return None
    lo, hi = cell["ci"]
    return [[max(cell["ens"] - lo, 0) * 100], [max(hi - cell["ens"], 0) * 100]]


def make_figure(zs_data, ft_data, out_basename, variant_title):
    ncols, nrows = 3, 2
    bar_h  = 0.34
    offset = bar_h / 2 + 0.02

    # barh puts index 0 at the bottom; reverse so site A sits on top
    site_order = list(reversed(SITES))
    y_pos      = np.arange(len(SITES))

    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 9))
    fig.subplots_adjust(
        left=0.11, right=0.975,
        bottom=0.13, top=0.87,
        hspace=0.42, wspace=0.16,
    )

    for ti, (task, task_name) in enumerate(TASKS):
        row, col = divmod(ti, ncols)
        ax = axes[row][col]

        for yi, site in enumerate(site_order):
            key = f"INSIGHT-{site}"
            zs  = zs_data[task][site]
            ft  = ft_data[task][site]

            if np.isnan(zs["mean"]) and np.isnan(ft["mean"]):
                ax.text(
                    X_MIN + 1.5, yi, ft.get("reason") or "no data",
                    fontsize=6.5, style="italic", color="#999999",
                    ha="left", va="center", zorder=3,
                )
                continue

            # Zero-shot (light) — lower of the pair
            if not np.isnan(zs["mean"]):
                ax.barh(
                    yi - offset, _bar_val(zs) * 100,
                    height=bar_h,
                    xerr=_bar_err(zs),
                    color=COLORS_ZEROSHOT[key], alpha=0.95,
                    edgecolor="black", linewidth=1.1, capsize=2.5,
                    error_kw={"linewidth": 0.9, "capthick": 0.9,
                              "ecolor": "#333333", "zorder": 4},
                    zorder=2,
                )
                if CI_MODE == "folds" and not np.isnan(zs["ens"]):
                    ax.scatter(zs["ens"] * 100, yi - offset, marker="D",
                               color=ENSEMBLE_COLOR, s=22, zorder=6,
                               edgecolors="white", linewidths=0.5)

            # Finetuned (dark) — upper of the pair
            if not np.isnan(ft["mean"]):
                ax.barh(
                    yi + offset, _bar_val(ft) * 100,
                    height=bar_h,
                    xerr=_bar_err(ft),
                    color=COLORS_FINETUNE[key], alpha=0.95,
                    edgecolor="black", linewidth=1.1, capsize=2.5,
                    error_kw={"linewidth": 0.9, "capthick": 0.9,
                              "ecolor": "#333333", "zorder": 4},
                    zorder=2,
                )
                if CI_MODE == "folds" and not np.isnan(ft["ens"]):
                    ax.scatter(ft["ens"] * 100, yi + offset, marker="D",
                               color=ENSEMBLE_COLOR, s=22, zorder=6,
                               edgecolors="white", linewidths=0.5)

            # Significance (paired over folds)
            label = paired_sig(zs["folds"], ft["folds"])
            if label is not None:
                ax.text(
                    SIG_X, yi, label,
                    ha="center", va="center",
                    fontsize=9 if label != "N.S." else 6.5,
                    fontweight="bold" if label != "N.S." else "normal",
                    color="black", zorder=8,
                )

        # -- Y axis --
        ax.set_yticks(y_pos)
        if col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([_site_label(s, task) for s in site_order],
                               fontsize=7)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_ylim(-0.6, len(SITES) - 0.4)

        # -- X axis --
        ax.set_xlabel(f"{METRIC_WORD.capitalize()} AUROC (%)", fontsize=9, labelpad=4)
        ax.set_xlim(X_MIN, X_MAX)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.set_xticks([50, 60, 70, 80, 90, 100])
        ax.tick_params(axis="x", which="major", labelsize=7.5, length=3, width=0.6)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5, which="major")
        ax.set_axisbelow(True)

        ax.set_title(task_name, fontsize=11, fontweight="bold", pad=8)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.7)
            ax.spines[side].set_color("#333333")

    # =========================================================================
    # Legend
    # =========================================================================

    # Colour encodes the site; shade encodes zero-shot (light) vs finetuned
    # (dark).  Show both shades of every site so the pairing is unambiguous.
    legend_handles = []
    for site in SITES:
        key = f"INSIGHT-{site}"
        legend_handles.append(
            mpatches.Patch(facecolor=COLORS_ZEROSHOT[key], alpha=0.95,
                           edgecolor="black", linewidth=0.7,
                           label=f"{key}  zero-shot")
        )
    for site in SITES:
        key = f"INSIGHT-{site}"
        legend_handles.append(
            mpatches.Patch(facecolor=COLORS_FINETUNE[key], alpha=0.95,
                           edgecolor="black", linewidth=0.7,
                           label=f"{key}  finetuned")
        )
    if CI_MODE == "folds":
        legend_handles.append(
            mlines.Line2D([], [], marker="D", linestyle="None",
                          color=ENSEMBLE_COLOR, markersize=6, label="Ensemble")
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    fig.suptitle(
        f"INSIGHT: Zero-shot vs. {variant_title} (SPARC-P), per task and site\n"
        + "Upper bar = finetuned (dark), lower bar = zero-shot (light)  ·  "
        + ("mean ± std across 10 folds  ·  red diamond = ensemble  ·  " if CI_MODE == "folds"
           else "10-fold ensemble ± 95% bootstrap CI  ·  ")
        + "paired Wilcoxon: *** p<0.001, ** p<0.01, * p<0.05",
        fontsize=11.5, fontweight="bold", y=0.965, ha="center",
    )

    for ext in ("pdf", "png"):
        _sfx = "_ensembleCI" if CI_MODE == "ensemble" else ""
        out = os.path.join(SAVE_DIR, f"{out_basename}{_sfx}{METRIC_SUFFIX}.{ext}")
        fig.savefig(out, bbox_inches="tight", facecolor="white",
                    dpi=300 if ext == "png" else None)
        print(f"Saved: {out}")

    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("Loading zero-shot results ...")
    zs_data = collect(ZEROSHOT_FOLDER)

    for folder, basename, variant_title in FINETUNE_VARIANTS:
        if CI_MODE == "ensemble" and folder not in CI_RESULTS:
            print(f"\n=== {variant_title}: no ensemble-CI rerun, skipped in -ci ensemble ===")
            continue
        print(f"\n=== {variant_title}  ({folder}) ===")
        ft_raw = collect(folder)
        zs_plot, ft_plot = mask_incomplete_pairs(zs_data, ft_raw)

        for task, task_name in TASKS:
            print(f"\n{task_name}:")
            for site in SITES:
                zs, ft = zs_plot[task][site], ft_plot[task][site]
                if np.isnan(zs["mean"]) and np.isnan(ft["mean"]):
                    print(f"  INSIGHT-{site}  n/a ({ft.get('reason') or 'no data'})")
                    continue
                sig = paired_sig(zs["folds"], ft["folds"]) or "-"
                print(
                    f"  INSIGHT-{site}  "
                    f"zs={zs['mean']:.4f}±{zs['std']:.4f} (ens {zs['ens']:.4f})  "
                    f"ft={ft['mean']:.4f}±{ft['std']:.4f} (ens {ft['ens']:.4f})  "
                    f"{sig}"
                )

        make_figure(zs_plot, ft_plot, basename, variant_title)

    print("\nDone.")
