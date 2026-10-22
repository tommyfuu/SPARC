#!/usr/bin/env python3
"""Fig. 4b: INSIGHT zero-shot (x) vs fine-tuned (y) SPARC-P ensemble AUROC, one point per site x task, one panel per
fine-tuning strategy. Filled points: significant change (paired bootstrap, BH-FDR within strategy).
Published figure: SCATTER_STYLE=emph SCATTER_SIG=bh SCATTER_PAIR=1 SCATTER_SIZE=total SCATTER_ENC=size
SCATTER_SMIN=20 SCATTER_SMAX=110 SCATTER_ROWS=1 SCATTER_NS=fade SCATTER_MCNEMAR=1.
Usage: python zeroshot_vs_finetune_scatter.py
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

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats

from zeroshot_vs_finetune_bars import (
    BASE, SITES, TASKS, ZEROSHOT_FOLDER, FINETUNE_VARIANTS,
    N_POS, collect, mask_incomplete_pairs, _csv_path, _metric_col,
    METRIC_SUFFIX, METRIC_WORD, METRIC,   # set by INSIGHT_AUROC=overall
)
import pickle
import bootstrap_stats as bs

# _bootstrap: predictions and output folder
BOOT_ROOT = os.path.join(RESULTS, 'INSIGHT_bootstrap')
BOOT_RES = {ZEROSHOT_FOLDER: "res_zeroshot_testsplit",
            "insight-finetune-mskfinetuned": "res_ft_sitespecific",
            "insight-finetune-pooled-mskfinetuned": "res_ft_pooled"}
N_BOOT = int(os.environ.get("SCATTER_NBOOT", "1000"))
TEST = os.environ.get("SCATTER_TEST", "bootstrap")
assert TEST in ("bootstrap", "delong"), TEST
# SCATTER_NS=fade: draw n.s. points filled in their task colour at SCATTER_NS_ALPHA opacity
# (default 0.6) instead of as open rings; significant points stay fully opaque. Adds _fade.
NS_FADE = os.environ.get("SCATTER_NS") == "fade"
NS_ALPHA = float(os.environ.get("SCATTER_NS_ALPHA", "0.6"))
# SCATTER_MCNEMAR=1: print and draw (between the two panels) the exact McNemar test of
# "above the line" (finetuned > zero-shot, no significance required), pooled vs site-specific,
# over the cells both panels show. Adds _mcnemar to the filename.
MCNEMAR = os.environ.get("SCATTER_MCNEMAR") == "1"


def mcnemar_above_line(rows_site, rows_pool):
    """2x2 of above-the-line, pooled (rows) x site-specific (columns), exact McNemar p."""
    s = {(r["site"], r["task"]): r["delta"] > 0 for r in rows_site}
    p = {(r["site"], r["task"]): r["delta"] > 0 for r in rows_pool}
    keys = sorted(set(s) & set(p))
    n11 = sum(p[k] and s[k] for k in keys); n10 = sum(p[k] and not s[k] for k in keys)
    n01 = sum((not p[k]) and s[k] for k in keys); n00 = sum((not p[k]) and (not s[k]) for k in keys)
    pval = stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
    return dict(n=len(keys), n11=n11, n10=n10, n01=n01, n00=n00, p=pval)


TEST_TXT = (f"paired bootstrap, {N_BOOT} resamples" if TEST == "bootstrap" else "paired DeLong")
SAVE_DIR = os.path.join(BOOT_ROOT, "vis")
os.makedirs(SAVE_DIR, exist_ok=True)


def _load_boot_preds(folder, site, task):
    sub = BOOT_RES.get(folder)
    if sub is None:
        return None
    p = os.path.join(BOOT_ROOT, sub, site, "ensemble_predictions",
                     f"{task}_SPARC-P_ensemble_preds.pickle")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _row_keys(d):
    """Content key per row: (patient_id, infusion number, time-to-event, label, k), k numbering duplicates.
    Rows are matched on content because the pooled run can order a patient's rows differently."""
    seen, keys = {}, []
    for pid, ni, t, e in zip(d["patient_ids"], np.asarray(d["num_infusions"]).ravel(),
                             np.asarray(d["time_to_events"]).ravel(), np.asarray(d["y_events"]).ravel()):
        base = (pid, float(ni), float(t), float(e))
        k = seen.get(base, 0)
        seen[base] = k + 1
        keys.append(base + (k,))
    return keys


def _delong_components(y, s):
    """DeLong structural components (midrank form): (auc, V10 over positives, V01 over negatives)."""
    from scipy.stats import rankdata
    y = np.asarray(y).ravel().astype(bool)
    s = np.asarray(s, dtype=np.float64).ravel()
    x, z = s[y], s[~y]
    m, n = x.size, z.size
    tz = rankdata(np.concatenate([x, z]))
    v10 = (tz[:m] - rankdata(x)) / n
    v01 = 1.0 - (tz[m:] - rankdata(z)) / m
    return v10.mean(), v10, v01


def _delong_diff(y, s1, s2):
    """Paired DeLong: (AUC1 - AUC2, its variance) for two scores on the same rows."""
    a1, p1, q1 = _delong_components(y, s1)
    a2, p2, q2 = _delong_components(y, s2)
    d10, d01 = p1 - p2, q1 - q2
    var = ((np.var(d10, ddof=1) / d10.size) if d10.size > 1 else 0.0) + \
          ((np.var(d01, ddof=1) / d01.size) if d01.size > 1 else 0.0)
    return a1 - a2, var


def _delong_p(preds, option, variant):
    """Paired DeLong (p, reason), fine-tuned vs zero-shot; group-size weighted for weighted AUROC."""
    ctx = bs.prepare_context(preds["finetuned"], option)
    if ctx["total_n"] == 0:
        return np.nan, "no usable infusion groups"
    y = ctx["labels"].ravel()
    a = np.asarray(preds["finetuned"]["y_preds"]).ravel()
    b = np.asarray(preds["zeroshot"]["y_preds"]).ravel()
    if variant.startswith("weighted"):
        delta = var = 0.0
        for rows, n_g in zip(ctx["group_row_idx"], ctx["group_sizes"]):
            w = n_g / ctx["total_n"]
            dg, vg = _delong_diff(y[rows], a[rows], b[rows])
            delta += w * dg
            var += w * w * vg
    else:
        rows = np.concatenate(ctx["group_row_idx"])
        delta, var = _delong_diff(y[rows], a[rows], b[rows])
    if var <= 0:
        return np.nan, "zero DeLong variance"
    return float(2 * stats.norm.sf(abs(delta) / np.sqrt(var))), "DeLong"


def paired_bootstrap_p(folder, site, task, zs_ens, ft_ens, cell_index):
    """Two-sided paired-bootstrap p for finetuned - zero-shot on this cell, or (nan, reason)."""
    z = _load_boot_preds(ZEROSHOT_FOLDER, site, task)
    f = _load_boot_preds(folder, site, task)
    if z is None or f is None:
        return np.nan, "no predictions"
    kz, kf = _row_keys(z), _row_keys(f)
    iz = {k: i for i, k in enumerate(kz)}
    common = [k for k in kf if k in iz]
    # rows present in only one arm (a few per cell for pooled) are dropped; refuse if >2%
    if len(common) < 0.98 * max(len(kz), len(kf)):
        return np.nan, f"rows do not match ({len(kz)} vs {len(kf)}, {len(common)} shared)"
    if len(common) < max(len(kz), len(kf)):
        print(f"  [pair] {task}/INSIGHT-{site}: {max(len(kz), len(kf)) - len(common)} unmatched "
              f"rows dropped ({len(kz)} zero-shot, {len(kf)} finetuned, {len(common)} paired)")
    fi = np.array([i for i, k in enumerate(kf) if k in iz])
    zi = np.array([iz[k] for k in kf if k in iz])
    keys = ["y_events", "time_to_events", "num_infusions"]
    for k in keys:
        if not np.array_equal(np.asarray(z[k])[zi], np.asarray(f[k])[fi]):
            return np.nan, f"{k} differ between arms"
    arm = lambda d, idx: {k: np.asarray(d[k])[idx] for k in ["y_preds"] + keys}
    preds = {"finetuned": arm(f, fi), "zeroshot": arm(z, zi)}
    option = task if task == "binary" else f"{task}_risk"
    variant = "overall_auroc" if (task == "binary" or METRIC == "overall") else "weighted_auroc"
    # the saved predictions must reproduce the plotted ensemble values
    for name, ref in (("zeroshot", zs_ens), ("finetuned", ft_ens)):
        pe = bs.point_estimate(preds[name], option, variant)
        if np.isfinite(pe) and abs(pe - ref) > 2e-3:
            print(f"  [warn] {task}/{site} {name}: bootstrap-run AUROC {pe:.4f} != plotted {ref:.4f}")
    if TEST == "delong":
        return _delong_p(preds, option, variant)
    draws, paired = bs.bootstrap_task(preds, option, N_BOOT, (1024, cell_index, 0),
                                      reference_model="finetuned")
    _, p, n_valid, _ = bs.difference_pvalue(draws[variant]["finetuned"], draws[variant]["zeroshot"], paired)
    return p, ("paired" if paired else "UNPAIRED") + f", {n_valid} valid draws"



# =============================================================================
# True positive-label counts (produced by get_positive_counts.py)
#
# NOTE: N_POS imported above is n_with_medication from n_numbers.csv - the count
# of task-ELIGIBLE patients, i.e. a denominator.  It is NOT the event count and
# is carried here only as `n_eligible`.  The actual number of positive labels
# comes from positive_counts.csv.
# =============================================================================

_pos_csv = os.path.join(BASE, "positive_counts.csv")
N_POSITIVE = {}    # (site, task) -> number of positive labels in the test set
N_SAMPLES  = {}    # (site, task) -> number of infusion-level samples
if os.path.exists(_pos_csv):
    _dfp = pd.read_csv(_pos_csv)
    for _, _r in _dfp.iterrows():
        N_POSITIVE[(_r["site"], _r["task"])] = int(_r["n_pos"])
        N_SAMPLES[(_r["site"], _r["task"])] = int(_r["n_samples"])
else:
    raise SystemExit(f"Missing {_pos_csv} - run get_positive_counts.py first.")


# =============================================================================
# Training-set sizes (produced by get_training_counts.py)
#
# These are the sizes of the data each model was actually FINETUNED on, which is
# a different quantity from the test-set counts above: N_POSITIVE / N_SAMPLES
# describe the held-out set the AUROCs are measured on.
#
# Two properties make a single number per cell exact rather than a summary:
#   * the training set is identical across all ten folds (verified: the
#     fold-to-fold spread is exactly zero for every cell), and
#   * it is identical between the two initialisation variants, since "from MSK
#     pretrain" and "from MSK finetune" differ only in the starting checkpoint.
# So one table serves both figures.
#
# Site-specific runs train on one institute, so they have a value per
# (site, task).  Pooled runs train on all six at once, so they have one value
# per task that every site of that task shares - carried under site "ALL".
# =============================================================================

_train_csv = os.path.join(BASE, "training_counts.csv")
N_TRAIN = {}       # (site, task) -> dict(samples, patients, pos)  site-specific
N_TRAIN_POOLED = {}  # task -> dict(samples, patients, pos)
if os.path.exists(_train_csv):
    _dft = pd.read_csv(_train_csv)
    # Either initialisation variant gives the same numbers; take the first.
    def _splits(_g):
        """train / val / test / total sample counts for one cell."""
        return {
            "train": int(_g["n_train_samples"].iloc[0]),
            "val":   int(_g["n_val_samples"].iloc[0]),
            "test":  int(_g["n_test_samples"].iloc[0]),
            "total": int(_g["n_total_samples"].iloc[0]),
            "train_pos": int(_g["n_train_pos"].iloc[0]),
        }

    _site = _dft[~_dft["pooled"] & (_dft["strategy"] == "insight-finetune-mskpretrain")]
    for (_s, _t), _g in _site.groupby(["site", "task"]):
        N_TRAIN[(_s, _t)] = _splits(_g)
    _pool = _dft[_dft["pooled"]
                 & (_dft["strategy"] == "insight-finetune-pooled-mskpretrain")]
    for _t, _g in _pool.groupby("task"):
        N_TRAIN_POOLED[_t] = _splits(_g)
else:
    print(f"[warn] {_train_csv} not found - run get_training_counts.py; "
          f"the figure will be drawn without the training-size table.")


# =============================================================================
# Encoding
# =============================================================================

# Okabe-Ito, assigned to tasks in fixed order (never cycled).
# Validated all-pairs on the light surface (scripts/validate_palette.py):
#   normal-vision worst dE 15.6 (#D55E00 vs #E69F00, floor 15)  -> PASS
#   CVD worst min(protan,deutan) dE 7.6 (#CC79A7 vs #009E73)    -> 6-8 warn band
#
# That CVD warn band is legal only alongside a secondary encoding of the same
# variable.  This figure used to ship one - a per-task marker shape - and no
# longer does, so task identity now rests on colour alone and the two tasks in
# that pair (neutropenia and liver toxicity) are the hard ones to tell apart for
# a deuteranopic reader.  The legend and the training-N table below the panels
# are the remaining non-colour routes to task identity; re-stepping the two
# colours to dE >= 8 would remove the caveat entirely.
TASK_COLOR = {
    "binary":              "#0072B2",   # blue
    "AKI":                 "#E69F00",   # orange
    "Liver_toxicity":      "#009E73",   # bluish green
    "anemia":              "#56B4E9",   # sky blue
    "febrile_neutropenia": "#D55E00",   # vermillion
    "neutropenia":         "#CC79A7",   # reddish purple
}

MARKER = "o"   # every task, every point

# -- two marker sizings, one per figure row -------------------------------------
# Row 1 draws every point at MARKER_SIZE: nothing about a marker's appearance
# depends on the event count, so the panel reads as plain performance.
# Row 2 repeats the same points with area proportional to sqrt(n_pos), so visual
# size tracks magnitude without the largest cell (1,774 events) swamping the
# smallest (3).  A floor keeps every marker big enough to carry its institute
# letter.
# Marker areas in pt^2.  The floor is set by the institute letter drawn inside
# the marker rather than by visibility: below roughly 70 pt^2 (a 8.4 pt circle)
# a bold letter no longer fits legibly, which is what stops these going smaller.
MARKER_SIZE = 88.0
MARKER_LETTER_SIZE = 5.2

S_MIN, S_MAX = 70.0, 430.0
# SCATTER_SMIN / SCATTER_SMAX override the marker-area range (pt^2).
if os.environ.get("SCATTER_SMIN"):
    S_MIN = float(os.environ["SCATTER_SMIN"])
if os.environ.get("SCATTER_SMAX"):
    S_MAX = float(os.environ["SCATTER_SMAX"])
SIZE_LEGEND_VALUES = [10, 100, 1000]


# SCATTER_SIZE: "train" = training-split samples of the run behind the point;
# "total" = total cohort (train+val+test samples) of the institute x task cell.
# SCATTER_ENC: "size" encodes it as marker area, "alpha" as marker opacity.
SIZE_MODE = os.environ.get("SCATTER_SIZE", "")
ENC = os.environ.get("SCATTER_ENC", "size")
SIZE_BY_TRAIN = SIZE_MODE in ("train", "total")
_TR_LO, _TR_HI = np.log10(80.0), np.log10(60000.0)
if SIZE_MODE == "train":
    SIZE_LEGEND_VALUES = [100, 1000, 10000, 50000]
elif SIZE_MODE == "total":
    SIZE_LEGEND_VALUES = [500, 5000, 50000]
    _TR_LO, _TR_HI = np.log10(300.0), np.log10(60000.0)
ALPHA_MODE = SIZE_BY_TRAIN and ENC == "alpha"
PENDING_LINES = []   # (ax, x, y_base, y, marker_area) drawn after layout


def _train_frac(n):
    return float(np.clip((np.log10(max(n, 1)) - _TR_LO) / (_TR_HI - _TR_LO), 0.0, 1.0))


def cohort_alpha(n):
    return 0.22 + 0.78 * _train_frac(n or 1)


def _marker_alpha(n):
    return cohort_alpha(n) if ALPHA_MODE else 0.95


def marker_size(n_pos):
    if ALPHA_MODE:
        return 120.0
    if SIZE_BY_TRAIN:
        return S_MIN + (S_MAX - S_MIN) * _train_frac(n_pos or 1)
    if n_pos is None or n_pos <= 0:
        return S_MIN
    return float(np.clip(13.0 * np.sqrt(n_pos), S_MIN, S_MAX))


def letter_size(n_pos):
    if ALPHA_MODE:
        return 5.6
    if SIZE_BY_TRAIN:
        # scale the letter with the marker diameter so it fits small bubbles
        return float(np.clip(0.40 * np.sqrt(marker_size(n_pos)), 3.4, 6.6))
    if n_pos is None or n_pos <= 0:
        return 4.4
    return float(np.clip(3.1 + 1.5 * np.log10(max(n_pos, 1)), 4.4, 6.6))


def axis_limits(all_rows, pad=3.0, step=5):
    """Shared square range covering every plotted point, snapped to `step`."""
    vals = [v for r in all_rows for v in (r["zeroshot_ensemble"], r["finetuned_ensemble"])]
    lo = step * np.floor((min(vals) - pad) / step)
    hi = step * np.ceil((max(vals) + pad) / step)
    return float(lo), float(min(hi, 100.0))


# =============================================================================
# Significance
#
# Paired Wilcoxon signed-rank, zero-shot vs. finetuned, over the 10 matched
# folds of one site x task cell.
#
# Two properties of this test at n=10 shape how the result may be drawn:
#   * The smallest attainable two-sided p is 2/2**10 = 0.00195, so a "p<0.001"
#     tier can never be reached and must not appear in the legend.  Most cells
#     in fact sit exactly on that floor.
#   * Because the folds share a test set and differ only in the training split,
#     this tests the consistency of the fold-level change, not whether the
#     effect would replicate in a new cohort.  It is a within-cell statement.
# =============================================================================

ALPHA = 0.05


def load_folds_by_label(folder, site, task):
    """Return {fold label: metric value} for one site x task, so folds pair by label, not row order."""
    fpath = _csv_path(folder, site, task)
    if not os.path.exists(fpath):
        return {}
    df = pd.read_csv(fpath)
    col = _metric_col(task)
    if col not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        label = str(r["val_fold"])
        if label == "ensemble" or pd.isna(r[col]):
            continue
        out[label] = float(r[col])
    return out


def paired_wilcoxon_p(zs_map, ft_map):
    """Two-sided paired Wilcoxon over shared folds; returns (p, n_paired), p nan if the test cannot run."""
    labels = sorted(set(zs_map) & set(ft_map), key=lambda x: (len(x), x))
    if len(labels) < 2:
        return np.nan, len(labels)
    a = np.array([zs_map[l] for l in labels], dtype=float)
    b = np.array([ft_map[l] for l in labels], dtype=float)
    if np.all(a == b):
        return np.nan, len(labels)
    try:
        _, p = stats.wilcoxon(a, b, alternative="two-sided")
    except Exception:
        return np.nan, len(labels)
    return float(p), len(labels)


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values; nan inputs stay nan."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    idx = np.flatnonzero(ok)[np.argsort(p[ok])]
    m = len(idx)
    ranked = p[idx] * m / np.arange(1, m + 1)
    q[idx] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], None, 1.0)
    return q


# =============================================================================
# Data
# =============================================================================

def _npos(r):
    if SIZE_MODE == "total":
        return r.get("n_total")
    if SIZE_BY_TRAIN:
        return r.get("n_train")
    """
    Event count for a cell, or None when it is unknown.

    The pooled strategies produce results for site x task cells that the
    site-specific runs skipped (too few positives to train), and those cells
    have no row in positive_counts.csv because get_positive_counts.py found no
    training log for them. They are still worth plotting; they just cannot
    contribute to the event-count correlation.
    """
    v = r.get("n_pos")
    return None if v is None else int(v)


def build_points(ft_folder):
    """Rows (task, task_name, site, zero-shot, fine-tuned ensemble, in percent) for usable site x task cells.
    BH correction is applied within this strategy."""
    zs_raw = collect(ZEROSHOT_FOLDER)
    ft_raw = collect(ft_folder)
    zs, ft = mask_incomplete_pairs(zs_raw, ft_raw)

    rows = []
    for task, task_name in TASKS:
        for site in SITES:
            z, f = zs[task][site], ft[task][site]
            if np.isnan(z["ens"]) or np.isnan(f["ens"]):
                continue
            # _bootstrap: paired bootstrap on the ensemble predictions
            p, n_paired = paired_bootstrap_p(ft_folder, site, task, z["ens"], f["ens"], len(rows))
            if not np.isfinite(p):
                print(f"  [untestable] {task}/INSIGHT-{site}: {n_paired}")
            rows.append({
                "task": task,
                "task_name": task_name,
                "site": site,
                "n_pos": N_POSITIVE.get((site, task)),        # positive labels
                "n_samples": N_SAMPLES.get((site, task)),     # infusion-level samples
                "n_eligible": N_POS.get((site, task)),        # patients on the drug
                "zeroshot_ensemble":  z["ens"] * 100,
                "finetuned_ensemble": f["ens"] * 100,
                "delta": (f["ens"] - z["ens"]) * 100,
                "p_wilcoxon": p,
                "n_folds_paired": n_paired,
                # nan p means the test could not run; such a cell is drawn open,
                # the same as a genuine non-significant one, and is listed in the
                # console output so it is not silently folded in.
                "significant": bool(np.isfinite(p) and p < ALPHA),
                "testable": bool(np.isfinite(p)),
            })

    for row, q in zip(rows, bh_fdr([r["p_wilcoxon"] for r in rows])):
        row["q_bh"] = q
        row["significant_bh"] = bool(np.isfinite(q) and q < ALPHA)
    return rows


# =============================================================================
# Figure
# =============================================================================

def _draw_panel(ax, rows, title, lo, hi, sized, stats_box):
    """One zero-shot (x) vs fine-tuned (y) panel; filled markers have p < ALPHA.
    sized scales marker area with the event count; stats_box adds the above/below/median summary."""
    # -- 45-degree reference: no change from finetuning --
    ax.plot([lo, hi], [lo, hi],
            linestyle=(0, (3, 3)), color="#666666", linewidth=1.2,
            zorder=1)

    # SCATTER_STYLE=emph: shade the helped/hurt half-planes and draw each
    # point's vertical distance to the diagonal (its delta) as a thin segment.
    emph = os.environ.get("SCATTER_STYLE") == "emph"
    if emph:
        ax.fill_between([lo, hi], [lo, hi], [hi, hi], color="#2E7D32", alpha=0.06, zorder=0, linewidth=0)
        ax.fill_between([lo, hi], [lo, lo], [lo, hi], color="#C62828", alpha=0.06, zorder=0, linewidth=0)
        for r in rows:
            x0, y0 = r["zeroshot_ensemble"], r["finetuned_ensemble"]
            PENDING_LINES.append((ax, x0, x0, y0, marker_size(_npos(r) or 0) if sized else MARKER_SIZE))

    # Draw order, outermost key first:
    #   1. significant points before N.S. ones, so the open markers always end
    #      up on top.  Several N.S. cells sit within a marker width of a
    #      significant neighbour, and if the filled neighbour were drawn last it
    #      would hide the only thing marking that point as N.S.  The reverse
    #      overlap is much cheaper: an open marker clipping a filled one still
    #      leaves its position, colour and shape readable.
    #   2. when markers vary in size, the big ones first so small ones stay
    #      visible on top of them.
    if sized:
        order = sorted(rows, key=lambda x: (not x["significant"], -(_npos(x) or 0)))
    else:
        # stable, so points keep their task order within each group
        order = sorted(rows, key=lambda x: not x["significant"])
    for r in order:
        n = _npos(r) or 0
        sig = bool(r["significant"])
        _solid = sig or NS_FADE
        ax.scatter(
            r["zeroshot_ensemble"], r["finetuned_ensemble"],
            marker=MARKER,
            s=marker_size(n) if sized else MARKER_SIZE,
            facecolor=TASK_COLOR[r["task"]] if _solid else "white",
            edgecolor="white" if _solid else TASK_COLOR[r["task"]],
            linewidth=1.4 if _solid else 1.9,
            alpha=(NS_ALPHA if (NS_FADE and not sig) else _marker_alpha(n)), zorder=3,
        )
        # Institute letter as a direct label inside the marker
        ax.text(
            r["zeroshot_ensemble"], r["finetuned_ensemble"], r["site"],
            ha="center", va="center",
            fontsize=letter_size(n) if sized else MARKER_LETTER_SIZE,
            color="white" if (sig or NS_FADE) and _marker_alpha(n) > 0.55 else "#333333",
            fontweight="bold", zorder=4,
        )

    if stats_box:
        n_up   = sum(1 for r in rows if r["delta"] > 0)
        n_down = sum(1 for r in rows if r["delta"] < 0)
        # Split the significant points by direction: "significant" alone would
        # read as good news, and in several panels most of it is not.
        n_sig_up = sum(1 for r in rows if r["significant"] and r["delta"] > 0)
        n_sig_dn = sum(1 for r in rows if r["significant"] and r["delta"] < 0)
        n_ns     = sum(1 for r in rows if not r["significant"])
        ax.text(
            0.035, 0.965,
            f"{n_up}/{len(rows)} above the line, {n_down}/{len(rows)} below\n"
            f"median Δ = {np.median([r['delta'] for r in rows]):+.1f} pts\n"
            f"{TEST_TXT}, {SIG_TXT} < {ALPHA:g}:\n"
            f"    {n_sig_up} better · {n_sig_dn} worse\n"
            f"    {n_ns} n.s. → open markers",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#333333", linespacing=1.5,
        )
    ax.text(
        0.975, 0.03, "finetuning hurt",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, style="italic", color="#999999",
    )
    ax.text(
        0.025, 0.60, "finetuning helped",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=7.5, style="italic", color="#999999",
    )

    ticks = [t for t in range(int(lo), int(hi) + 1, 10) if lo <= t <= hi]
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=8, length=3, width=0.6)
    ax.set_xlabel(f"Zero-shot ensemble {METRIC_WORD} AUROC (%)", fontsize=9.5, labelpad=5)
    ax.set_ylabel(f"Finetuned ensemble {METRIC_WORD} AUROC (%)", fontsize=9.5, labelpad=5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=9)
    ax.grid(alpha=0.28, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color("#333333")


# Bottom-row style: "sized" (the original: same scatter, marker area = event
# count) or "delta" (a waterfall of finetuned - zero-shot per cell).  Both
# panels in the delta row share one y-range so their bars compare directly.
BOTTOM_ROW = os.environ.get("SCATTER_BOTTOM_ROW", "sized")
_BH = os.environ.get("SCATTER_SIG") == "bh"
SIG_TXT = "BH-corrected q" if _BH else "p"


def delta_limit(panels, step=5):
    """Symmetric y-limit shared by every delta panel, rounded up to `step`."""
    m = max(abs(r["delta"]) for _, rows in panels for r in rows)
    return step * np.ceil((m + 1.0) / step)


def _draw_delta_panel(ax, rows, title, ylim):
    """Waterfall of per-cell change (fine-tuned minus zero-shot, AUROC points), largest gain first."""
    order = sorted(rows, key=lambda r: -r["delta"])
    x = np.arange(len(order))
    ax.axhspan(0, ylim, color="#2E7D32", alpha=0.045, zorder=0)
    ax.axhspan(-ylim, 0, color="#C62828", alpha=0.045, zorder=0)
    ax.axhline(0, color="#666666", linewidth=1.2, linestyle=(0, (3, 3)), zorder=1)
    for i, r in zip(x, order):
        c = TASK_COLOR[r["task"]]
        sig = bool(r["significant"])
        ax.vlines(i, 0, r["delta"], color=c, linewidth=2.2, alpha=0.85, zorder=2)
        ax.scatter(i, r["delta"], marker=MARKER, s=MARKER_SIZE,
                   facecolor=c if sig else "white",
                   edgecolor="white" if sig else c,
                   linewidth=1.4 if sig else 1.9, zorder=3)
        ax.text(i, r["delta"], r["site"], ha="center", va="center",
                fontsize=MARKER_LETTER_SIZE, fontweight="bold",
                color="white" if sig else "#333333", zorder=4)
    n = len(order)
    n_up = sum(1 for r in rows if r["delta"] > 0)
    n_sig_up = sum(1 for r in rows if r["significant"] and r["delta"] > 0)
    n_sig_dn = sum(1 for r in rows if r["significant"] and r["delta"] < 0)
    ax.text(0.98, 0.965,
            f"{n_up}/{n} cells improved\n"
            f"{n_sig_up} significantly better · {n_sig_dn} significantly worse\n"
            f"median Δ = {np.median([r['delta'] for r in rows]):+.1f} pts",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color="#222222", linespacing=1.45)
    ax.text(0.02, 0.03, "finetuning hurt", transform=ax.transAxes,
            fontsize=7.5, style="italic", color="#999999")
    ax.text(0.02, 0.965, "finetuning helped", transform=ax.transAxes, va="top",
            fontsize=7.5, style="italic", color="#999999")
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-ylim, ylim)
    ax.set_xticks([])
    ax.set_xlabel("Institute × task cells, sorted by change", fontsize=9.5, labelpad=5)
    ax.set_ylabel(f"Finetuned − zero-shot {METRIC_WORD} AUROC (pts)", fontsize=9.5, labelpad=5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=9)
    ax.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, length=3, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color("#333333")


def _draw_delta_scatter(ax, rows, title, lo, hi, ylim):
    """As _draw_panel, but y is the change (fine-tuned minus zero-shot, AUROC points)."""
    ax.axhspan(0, ylim, color="#2E7D32", alpha=0.06, zorder=0, linewidth=0)
    ax.axhspan(-ylim, 0, color="#C62828", alpha=0.06, zorder=0, linewidth=0)
    ax.axhline(0, color="#666666", linewidth=1.2, linestyle=(0, (3, 3)), zorder=1)
    for r in rows:
        PENDING_LINES.append((ax, r["zeroshot_ensemble"], 0.0, r["delta"], marker_size(_npos(r) or 0)))
    order = sorted(rows, key=lambda x: (not x["significant"], -(_npos(x) or 0)))
    for r in order:
        n = _npos(r) or 0
        sig = bool(r["significant"])
        c = TASK_COLOR[r["task"]]
        ax.scatter(r["zeroshot_ensemble"], r["delta"], marker=MARKER, s=marker_size(n),
                   facecolor=c if sig else "white", edgecolor="white" if sig else c,
                   linewidth=1.4 if sig else 1.9, alpha=_marker_alpha(n), zorder=3)
        ax.text(r["zeroshot_ensemble"], r["delta"], r["site"], ha="center", va="center",
                fontsize=letter_size(n), color="white" if sig and _marker_alpha(n) > 0.55 else "#333333",
                fontweight="bold", zorder=4)
    ax.text(0.025, 0.965, "finetuning helped", transform=ax.transAxes, va="top",
            fontsize=7.5, style="italic", color="#999999")
    ax.text(0.975, 0.03, "finetuning hurt", transform=ax.transAxes, ha="right",
            fontsize=7.5, style="italic", color="#999999")
    ticks = [t for t in range(int(lo), int(hi) + 1, 10) if lo <= t <= hi]
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(f"Zero-shot ensemble {METRIC_WORD} AUROC (%)", fontsize=9.5, labelpad=5)
    ax.set_ylabel(f"Finetuned − zero-shot {METRIC_WORD} AUROC (pts)", fontsize=9.5, labelpad=5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=9)
    ax.grid(alpha=0.28, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, length=3, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color("#333333")


def _fmt_cell(rec):
    """'train / val / test' over 'total N' sample counts (not patients), or a dash if the cell never ran."""
    if rec is None:
        return "\u2014"
    return (f"{rec['train']:,} / {rec['val']:,} / {rec['test']:,}\n"
            f"total {rec['total']:,}")


def draw_training_table(ax, plotted_sites, plotted_pooled):
    """Table of training-set size per task under the scatter panels; cells not plotted are dimmed."""
    ax.axis("off")
    if not N_TRAIN:
        return

    col_labels = [f"INSIGHT-{s}" for s in SITES] + ["Pooled\n(all sites)"]
    row_labels, cell_text, dim = [], [], []
    for task, task_name in TASKS:
        row_labels.append(task_name)
        row, row_dim = [], []
        for site in SITES:
            rec = N_TRAIN.get((site, task))
            row.append(_fmt_cell(rec))
            row_dim.append(rec is None or (site, task) not in plotted_sites)
        rec = N_TRAIN_POOLED.get(task)
        row.append(_fmt_cell(rec))
        row_dim.append(rec is None or task not in plotted_pooled)
        cell_text.append(row)
        dim.append(row_dim)

    table = ax.table(
        cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
        cellLoc="center", rowLoc="right", loc="center",
        colWidths=[0.113] * len(SITES) + [0.135],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    table.scale(1.0, 2.35)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#cccccc")
        if r == 0:                                   # column header
            cell.set_text_props(fontweight="bold", color="#333333")
            cell.set_facecolor("#f2f2f2")
        elif c == -1:                                # row label
            cell.set_text_props(fontweight="bold", color="#333333")
            cell.set_facecolor("#fafafa")
        else:
            if dim[r - 1][c]:
                # Not plotted: present for completeness, greyed so it cannot be
                # mistaken for a point in the panels above.
                cell.set_text_props(color="#aaaaaa", style="italic")
                cell.set_facecolor("#fbfbfb")
            else:
                cell.set_text_props(color="#222222")
            if c == len(SITES):                      # pooled column
                cell.set_facecolor("#f7f9fb" if not dim[r - 1][c] else "#fbfbfb")

    ax.set_title(
        "Cohort size per task, in samples: train / val / test, and their total"
        "  \u00b7  identical across the 10 folds and across both "
        "initialisations\n"
        "institute columns = the site-specific runs (left panels)  \u00b7  "
        "pooled column = the pooled runs (right panels)  \u00b7  "
        "grey italic = cell not plotted in its panel above",
        fontsize=9, fontweight="bold", pad=15, color="#333333",
    )


def make_figure(panels, stem, group_title):
    """Draw and save the figure; panels is a list of (variant_title, rows), stem the output basename."""
    lo, hi = axis_limits([r for _, rows in panels for r in rows])

    # Width scales with the panel count so four finetuning variants stay legible;
    # the panels are square, so height follows from the per-panel width.
    ncol = len(panels)
    per_panel_w = 6.3 if ncol <= 2 else 5.2
    # A third band holds the training-size table.  The scatter panels keep their
    # square aspect, so the extra height is added to the figure rather than taken
    # out of them; height_ratios give the table a fixed share of it.
    has_table = bool(N_TRAIN)
    table_h = 3.5 if has_table else 0.0
    rows_mode = os.environ.get("SCATTER_ROWS", "")
    one_row = rows_mode in ("1", "d", "1d")
    n_plot = 2 if rows_mode == "1d" else 1
    if one_row:
        fig = plt.figure(figsize=(per_panel_w * ncol, 6.4 * n_plot + table_h))
        gs = fig.add_gridspec(
            n_plot + (1 if has_table else 0), ncol,
            height_ratios=([1.0] * n_plot + [table_h / 5.4]) if has_table else None,
            left=0.07 if ncol <= 2 else 0.05, right=0.985,
            bottom=0.15, top=0.86, wspace=0.18, hspace=0.34,
        )
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncol)]
                         for r in range(n_plot)])
        ax_table = fig.add_subplot(gs[n_plot, :]) if has_table else None
    else:
      fig = plt.figure(figsize=(per_panel_w * ncol, 11.6 + table_h))

    if one_row:
        pass
    elif has_table:
        gs = fig.add_gridspec(
            3, ncol, height_ratios=[1.0, 1.0, table_h / 5.0],
            left=0.07 if ncol <= 2 else 0.05, right=0.985,
            bottom=0.105, top=0.905, wspace=0.18, hspace=0.30,
        )
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncol)]
                         for r in range(2)])
        ax_table = fig.add_subplot(gs[2, :])
    else:
        gs = fig.add_gridspec(
            2, ncol,
            left=0.07 if ncol <= 2 else 0.05, right=0.985,
            bottom=0.125, top=0.875, wspace=0.18, hspace=0.30,
        )
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncol)]
                         for r in range(2)])
        ax_table = None

    # Row 1: no event-count encoding.  The above/below/median summary lives here
    # only - row 2 plots exactly the same points, so repeating it would just be
    # the same three numbers twice.
    if one_row:
        ylim_d = delta_limit(panels)
        for ri, kind in enumerate({"1": ["s"], "d": ["d"], "1d": ["s", "d"]}[rows_mode]):
            for ax, (variant_title, rows) in zip(axes[ri], panels):
                if kind == "s":
                    _draw_panel(ax, rows, variant_title, lo, hi, sized=True, stats_box=False)
                else:
                    _draw_delta_scatter(ax, rows, (variant_title if rows_mode == "d"
                                        else f"{variant_title}: change vs. zero-shot"),
                                        lo, hi, ylim_d)
    else:
      for ax, (variant_title, rows) in zip(axes[0], panels):
        _draw_panel(ax, rows, variant_title, lo, hi, sized=False, stats_box=True)

    if MCNEMAR and len(panels) == 2 and panels[0][1] and panels[1][1]:
        m = mcnemar_above_line(panels[0][1], panels[1][1])
        print(f"\n  McNemar, above the line (no significance required), {m['n']} paired cells:")
        print(f"                       site-specific above   site-specific not above")
        print(f"    pooled above       {m['n11']:>8d}              {m['n10']:>8d}")
        print(f"    pooled not above   {m['n01']:>8d}              {m['n00']:>8d}")
        print(f"    pooled {m['n11'] + m['n10']}/{m['n']} vs site-specific {m['n11'] + m['n01']}/{m['n']} above; "
              f"discordant {m['n10']} vs {m['n01']}; exact McNemar p = {m['p']:.4g}")
        _l, _r = axes[0][0].get_position(), axes[0][1].get_position()
        fig.text((_l.x1 + _r.x0) / 2, (_l.y0 + _l.y1) / 2,
                 "Above the line\n(no significance required)\n"
                 f"{m['n']} paired cells\n\n"
                 "              site-specific\n"
                 "              above   not\n"
                 f"pooled above  {m['n11']:>5d}  {m['n10']:>4d}\n"
                 f"       not    {m['n01']:>5d}  {m['n00']:>4d}\n\n"
                 f"pooled {m['n11'] + m['n10']}/{m['n']} vs site {m['n11'] + m['n01']}/{m['n']}\n"
                 f"discordant {m['n10']} vs {m['n01']}\n"
                 f"exact McNemar p = {m['p']:.3g}",
                 ha="center", va="center", fontsize=8.5, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#999999"))

    # Row 2: identical points, marker area scaled by the event count; or, with
    # SCATTER_BOTTOM_ROW=delta, the per-cell change as a sorted waterfall.
    if one_row:
        pass
    elif BOTTOM_ROW == "delta":
        ylim = delta_limit(panels)
        for ax, (variant_title, rows) in zip(axes[1], panels):
            _draw_delta_panel(ax, rows, f"{variant_title}: change vs. zero-shot", ylim)
    else:
        for ax, (variant_title, rows) in zip(axes[1], panels):
            _draw_panel(ax, rows, f"{variant_title}, sized by event count",
                        lo, hi, sized=True, stats_box=False)

    # Row 3: the training-set sizes behind those points.  Which cells count as
    # "plotted" comes from the panels themselves, so the table can never claim a
    # cell the figure did not draw.  Panel 0 is the site-specific strategy and
    # panel 1, when present, the pooled one.
    if ax_table is not None:
        plotted_sites = {(r["site"], r["task"]) for r in panels[0][1]}
        plotted_pooled = ({r["task"] for r in panels[1][1]}
                          if len(panels) > 1 else set())
        draw_training_table(ax_table, plotted_sites, plotted_pooled)

    # -- Legend: task identity (colour + shape together) --
    handles = [
        mlines.Line2D([], [], linestyle="None",
                      marker=MARKER, markersize=7,
                      markerfacecolor=TASK_COLOR[task],
                      markeredgecolor="white", markeredgewidth=1.0,
                      label=task_name)
        for task, task_name in TASKS
    ]
    handles.append(
        mlines.Line2D([], [], color="#666666", linewidth=1.2,
                      linestyle=(0, (3, 3)), label="No change")
    )
    # Fill key, in neutral grey so it cannot be mistaken for a task.  Only the
    # two attainable states are listed: at n=10 the smallest possible two-sided
    # p is 0.00195, so no "p<0.001" tier is shown.
    handles.append(
        mlines.Line2D([], [], linestyle="None", marker="o", markersize=8,
                      markerfacecolor="#666666", markeredgecolor="white",
                      markeredgewidth=1.0, label=f"{SIG_TXT} < {ALPHA:g}")
    )
    handles.append(
        mlines.Line2D([], [], linestyle="None", marker="o", markersize=8,
                      markerfacecolor="#666666" if NS_FADE else "white",
                      markeredgecolor="white" if NS_FADE else "#666666",
                      alpha=NS_ALPHA if NS_FADE else 1.0,
                      markeredgewidth=1.0 if NS_FADE else 1.6, label="n.s.")
    )
    task_legend = fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.055),
        handlelength=1.3, handletextpad=0.5, columnspacing=1.1,
    )
    fig.add_artist(task_legend)

    # -- Size legend: marker area in the bottom row encodes positive labels --
    size_handles = [
        mlines.Line2D([], [], linestyle="None", marker="o",
                      markersize=np.sqrt(marker_size(v)),
                      markerfacecolor="#555555" if ALPHA_MODE else "#9a9a9a",
                      alpha=cohort_alpha(v) if ALPHA_MODE else 1.0,
                      markeredgecolor="white",
                      markeredgewidth=1.2, label=f"{v:,}")
        for v in SIZE_LEGEND_VALUES
    ]
    if BOTTOM_ROW != "delta":
      fig.legend(
        handles=size_handles, loc="lower center", ncol=len(size_handles),
        fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.002),
        title=(("Marker opacity" if ALPHA_MODE else "Marker area")
               + (" - total cohort size of the institute x task (samples)" if SIZE_MODE == "total"
                  else " - training-set size (samples)")
               if SIZE_BY_TRAIN
               else "Bottom row only - positive labels in the site's test set"),
        title_fontsize=8.5,
        handlelength=1.6, handletextpad=0.7, columnspacing=2.2,
        labelspacing=1.0, borderpad=0.9,
    )

    fig.suptitle(
        "INSIGHT: zero-shot vs. finetuned SPARC-P ensemble performance\n"
        "One point per institute × task  ·  letter = institute  ·  "
        "above the dotted line = finetuning improved AUROC\n"
        + ("solid" if NS_FADE else "filled") + f" = change significant at {SIG_TXT} < "
        f"{ALPHA:g}"
        f" ({TEST_TXT} on ensemble predictions)  ·  "
        + (f"faded ({NS_ALPHA:.0%} opacity) = not significant\n" if NS_FADE else "open = not significant\n")
        + ((("marker opacity" if ALPHA_MODE else "marker area")
            + (" scaled by the institute x task total cohort size (train + val + test samples)"
               if SIZE_MODE == "total" else
               " scaled by training-set size (samples; pooled = all sites combined)"))
           if one_row else
           "bottom row: the same cells as change vs. zero-shot, sorted; "
           "both panels on the same scale"
           if BOTTOM_ROW == "delta" else
           "top row: every marker the same size  ·  "
           "bottom row: the same points, marker area scaled by the event count"),
        fontsize=12, fontweight="bold", y=0.985, ha="center",
    )

    # Drop lines: drawn now that the layout is final, stopping at each marker's
    # edge so the bubble never covers its own line.
    fig.canvas.draw()
    for _ax, _x, _yb, _y, _s in PENDING_LINES:
        _T = _ax.transData
        _p = np.asarray(_T.transform((_x, _y))); _b = np.asarray(_T.transform((_x, _yb)))
        _d = _b - _p; _L = float(np.hypot(*_d))
        _r = (np.sqrt(_s) / 2.0 + 1.2) * fig.dpi / 72.0
        if _L <= _r:
            continue
        _e = _T.inverted().transform(_p + _d / _L * _r)
        _ax.plot([_x, _x], [_yb, _e[1]], color="#2E7D32" if _y > _yb else "#C62828",
                 linewidth=1.3, alpha=0.75, zorder=2, solid_capstyle="butt")
    PENDING_LINES.clear()

    _tag = {"total": "_cohort" + ("alpha" if ALPHA_MODE else "size")}.get(SIZE_MODE, "")
    if os.environ.get("SCATTER_SMIN") or os.environ.get("SCATTER_SMAX"):
        _tag += f"_s{S_MIN:g}-{S_MAX:g}"
    for ext in ("pdf", "png"):
        out = os.path.join(SAVE_DIR, f"{stem}{_tag}{'_deltarow' if BOTTOM_ROW == 'delta' else ''}{'_bh' if _BH else ''}{'_emph' if os.environ.get('SCATTER_STYLE') == 'emph' else ''}{('_' + {'1': 'onerow', 'd': 'deltascatter', '1d': 'orig_plus_delta'}[rows_mode]) if one_row else ''}{METRIC_SUFFIX}_{TEST}{'_fade' if NS_FADE else ''}{'_mcnemar' if MCNEMAR else ''}.{ext}")
        fig.savefig(out, bbox_inches="tight", facecolor="white",
                    dpi=300 if ext == "png" else None)
        print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

# Finetuning runs are grouped by the checkpoint they start from, because that is
# the comparison that matters: "from MSK pretrain" begins task-naive, while
# "from MSK finetune" begins from a checkpoint already trained on the same task
# at MSK. Each figure holds the two strategies (site-specific and pooled) that
# share an initialisation.
GROUPS = [   # _bootstrap: only the MSK-finetuned strategies have prediction runs
    ("zeroshot_vs_finetune_scatter_mskfinetuned", "from the MSK task-finetuned checkpoint",
     ["insight-finetune-mskfinetuned", "insight-finetune-pooled-mskfinetuned"]),
]
_TITLE = {folder: title for folder, _, title in FINETUNE_VARIANTS}


if __name__ == "__main__":
    frames = []
    for stem, group_title, folders in GROUPS:
        panels = []
        print(f"\n########## {group_title} ##########")
        for folder in folders:
            variant_title = _TITLE.get(folder, folder)
            rows = build_points(folder)
            # Training-set size behind each point: the site's own train split for
            # site-specific runs, the task's pooled train split for pooled runs.
            for r in rows:
                rec = (N_TRAIN_POOLED.get(r["task"]) if "pooled" in folder
                       else N_TRAIN.get((r["site"], r["task"])))
                r["n_train"] = rec["train"] if rec else None
                own = N_TRAIN.get((r["site"], r["task"]))
                r["n_total"] = own["total"] if own else None
            # SCATTER_SIG=bh: fill markers and count "significant" by the BH-corrected
            # q < ALPHA (per strategy), as in the summary table.
            if os.environ.get("SCATTER_SIG") == "bh":
                for r in rows:
                    r["significant"] = r["significant_bh"]
            # Pair the panels (2026-09-29): drop pooled cells with no site-specific
            # counterpart (e.g. D-FN, where site D had too few training positives for a
            # site-specific model), so both panels show the same site x task cells, as in
            # test_pooled_vs_sitespecific_above_diagonal.py.  SCATTER_PAIR=0 keeps them.
            if os.environ.get("SCATTER_PAIR", "1") != "0" and panels and panels[0][1]:
                _keep = {(r["site"], r["task"]) for r in panels[0][1]}
                _drop = [(r["site"], r["task"]) for r in rows if (r["site"], r["task"]) not in _keep]
                if _drop:
                    print(f"  [pair] dropping pooled cells without a site-specific result: {_drop}")
                rows = [r for r in rows if (r["site"], r["task"]) in _keep]
            panels.append((variant_title, rows))
            if not rows:
                print(f"\n=== {variant_title} ===\n  no usable points (eval incomplete?)")
                continue

            df = pd.DataFrame(rows)
            df.insert(0, "strategy", variant_title)
            frames.append(df)

            up = sum(1 for r in rows if r["delta"] > 0)
            _known = [r for r in rows if _npos(r) is not None]
            n = np.array([max(_npos(r), 1) for r in _known], dtype=float)
            d = np.array([r["delta"] for r in _known], dtype=float)
            rho, pval = stats.spearmanr(n, d)
            print(f"\n=== {variant_title} ===")
            print(f"  {len(rows)} points, {up} above the line, median delta {np.median(d):+.2f} pts")
            print(f"  Spearman rho(n_pos, delta) = {rho:+.3f}  p={pval:.4f}")

            sig_up = [r for r in rows if r["significant"] and r["delta"] > 0]
            sig_dn = [r for r in rows if r["significant"] and r["delta"] < 0]
            ns     = [r for r in rows if not r["significant"]]
            bh_up  = sum(1 for r in rows if r["significant_bh"] and r["delta"] > 0)
            bh_dn  = sum(1 for r in rows if r["significant_bh"] and r["delta"] < 0)
            print(f"  {TEST_TXT}: "
                  f"{len(sig_up) + len(sig_dn)}/{len(rows)} at p<{ALPHA:g}  "
                  f"({len(sig_up)} better, {len(sig_dn)} worse), {len(ns)} n.s.")
            print(f"  after BH-FDR across this strategy's {len(rows)} cells: "
                  f"{bh_up + bh_dn}/{len(rows)} at q<{ALPHA:g}  "
                  f"({bh_up} better, {bh_dn} worse)")
            _floor = sum(1 for r in rows
                         if np.isfinite(r["p_wilcoxon"]) and r["p_wilcoxon"] == 0)
            if TEST == "bootstrap":
                print(f"  {_floor} cells with no resample reversing the sign (p < {2 / N_BOOT:g})")
            _untestable = [r for r in rows if not r["testable"]]
            if _untestable:
                print(f"  [warn] {len(_untestable)} cells could not be tested "
                      f"(drawn open): "
                      + ", ".join(f"{r['task_name']}/INSIGHT-{r['site']}"
                                  for r in _untestable))
            if ns:
                print("  n.s. cells (open markers):")
                for r in sorted(ns, key=lambda x: -abs(x["delta"])):
                    _p = r["p_wilcoxon"]
                    print(f"    {r['task_name']:<20s} INSIGHT-{r['site']}  "
                          f"delta={r['delta']:+6.1f}  "
                          f"p={'n/a' if not np.isfinite(_p) else f'{_p:.4f}'}")
            if (n < 100).any() and (n >= 100).any():
                print(f"  <100 events: median {np.median(d[n < 100]):+.2f} (n={(n < 100).sum()})   "
                      f">=100 events: median {np.median(d[n >= 100]):+.2f} (n={(n >= 100).sum()})")
            print("  five largest drops:")
            for r in sorted(rows, key=lambda x: x["delta"])[:5]:
                print(f"    {r['task_name']:<20s} INSIGHT-{r['site']}  n_pos={(_npos(r) if _npos(r) is not None else -1):>5d}  "
                      f"zs={r['zeroshot_ensemble']:5.1f}  ft={r['finetuned_ensemble']:5.1f}  "
                      f"delta={r['delta']:+6.1f}  "
                      f"{'p<' + format(ALPHA, 'g') if r['significant'] else 'n.s.'}")

        if any(rows for _, rows in panels):
            make_figure(panels, stem, group_title)
        else:
            print(f"  [skip] {stem}: no data in either strategy")

    if frames:
        table_out = os.path.join(SAVE_DIR, f"zeroshot_vs_finetune_scatter_data{METRIC_SUFFIX}_{TEST}.csv")
        pd.concat(frames, ignore_index=True).to_csv(table_out, index=False)
        print(f"\nSaved: {table_out}")
    print("Done.")
