#!/usr/bin/env python3
"""SPARC-P AUROC by cohort, one panel per task: MSK-CAIA as a reference line; MSK-CAIA prospective, CAIA partner
sites, INSIGHT sites A-F (zero-shot), MIMIC-IV and EHRSHOT as bars.
Published figure: SPARC_FIG_SUFFIX=_unitfix_minN with -ci ensemble.
Usage: python auc_by_cohort.py [-ci ensemble] [--metric overall] [--grouped]
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


# MIMIC / EHRSHOT.
EXT_DIR = os.path.join(sparc_paths.SPARC_RESULTS_VIS_ROOT, 'eval_across_models_valthresh_disabled')

# MSK-CAIA (retrospective) and MSK-CAIA-timeshift (prospective). Same file
# schema as EXT_DIR, from the batch_091426 rerun.
MSK_DIR = EXT_DIR

# All CAIA-site results (DFCI / FHCC / JHU) live in one workbook,
# one sheet per task × site, with the same columns as the eval CSVs.
# batch_first=True CAIA-site results (2026-09-29). Sheets are named "{task}_{site}_batchfirst"
# with short task names (FN, livtox, neut). Was os.path.join(BASE, "SPARC-P_CAIA_results.xlsx").
CAIA_XLSX = os.path.join(RESULTS, 'CAIA', 'SPARC-P_batchfirst_results_FINAL.xlsx')
# same sheets and ensemble values, plus ensemble bootstrap CIs from the partner sites
CAIA_CI_XLSX = os.path.join(RESULTS, 'CAIA', 'SPARC-P_batchfirst_CI_results.xlsx')
CAIA_SHEET_SUFFIX = "_batchfirst"

# INSIGHT per-site.
#
# The lab-unit / admission-ordering fixes live in the INSIGHT preprocessing
# pipeline, so the INSIGHT rows must be read from the UNIT-FIXED tree, not from
# BASE (= SPARC-regularBCELoss-ensemble), which predates them. Per that tree's
# PROVENANCE.md, only the INSIGHT rows were recomputed; MSK-CAIA, mskshift, the
# CAIA sites, MIMIC-IV and EHRSHOT are carried over verbatim, so every other
# cohort in this figure is unaffected.
#
# This tree prefers the '*_multi_insight.csv' filenames. It is not the zero-shot
# reference used for the finetuning scatter, which disagrees with it (e.g. site E
# anemia comes back a degenerate all-1.0 there but is a healthy 0.846-0.938 here).
INSIGHT_ROOT = os.path.join(RESULTS, 'INSIGHT_zeroshot', 'res')

# -ci ensemble: bars = SPARC-P 10-fold ensemble with its 95% bootstrap CI (1000 resamples;
# AUROC CI for 180-day mortality, weighted-AUROC CI for the side-effect tasks), no diamond,
# files suffixed _ensembleCI. -ci folds (default) is the original figure.
_ap = argparse.ArgumentParser()
_ap.add_argument("-ci", "--ci", choices=["folds", "ensemble"], default="folds")
# --metric overall: next-infusion AUROC pooled over infusion groups instead of the
# group-size-weighted mean; files get _overallauroc. 180-day mortality is the same either way.
_ap.add_argument("--metric", choices=["weighted", "overall"], default="weighted")
# --grouped: one colour per cohort type instead of per cohort, public-dataset labels in grey;
# files get _grouped.
_ap.add_argument("--grouped", action="store_true")
_args = _ap.parse_args()
CI_MODE = _args.ci
METRIC = _args.metric
SE_METRIC_COL = f"test_auroc_nextinf_{METRIC}"
METRIC_LABEL = "Weighted AUROC" if METRIC == "weighted" else "Overall AUROC"

# INSIGHT zero-shot rerun that adds the CI columns; same models/data as INSIGHT_ROOT
# (verified: all 42 CSVs match to 2e-6). eval_1 CSVs (MSK_DIR / EXT_DIR) carry them already.
# The CAIA-site workbook has no CI columns (run at the partner sites), so those bars have none.
INSIGHT_CI_ROOT = os.path.join(RESULTS, 'INSIGHT_weightedci', 'res_zeroshot_fullsite')


def _ci_cols(task_suffix):
    # auroc_ci_* is the overall next-infusion AUROC CI (AUROC CI for 180-day mortality)
    return (("auroc_ci_lower", "auroc_ci_upper") if task_suffix == "binary" or METRIC == "overall"
            else ("auroc_weighted_ci_lower", "auroc_weighted_ci_upper"))


def load_ensemble_ci(ds_name, task_suffix, metric_col, ens_val):
    """(lo, hi) of SPARC-P's ensemble bootstrap CI, or None if its ensemble value differs from the plotted one."""
    if CI_MODE != "ensemble" or np.isnan(ens_val):
        return None
    if ds_name in CAIA_SITE_SHEETS:
        if not os.path.exists(CAIA_CI_XLSX):
            return None
        sheet = f"{CAIA_TASK_SHEETS.get(task_suffix)}_{CAIA_SITE_SHEETS[ds_name]}{CAIA_SHEET_SUFFIX}"
        df = pd.read_excel(CAIA_CI_XLSX, sheet_name=sheet)
        lo_c, hi_c = (("auroc_wt_ci_lo", "auroc_wt_ci_hi")
                      if task_suffix != "binary" and metric_col.endswith("weighted")
                      else ("auroc_ci_lo", "auroc_ci_hi"))
        row = df[(df["model"] == "SPARC-P_ensemble") & (df["val_fold"].astype(str) == "ensemble")]
        if row.empty or lo_c not in df.columns or pd.isna(row[lo_c].values[0]):
            print(f"  [ci] {ds_name} {task_suffix}: no CI in {os.path.basename(CAIA_CI_XLSX)}")
            return None
        if abs(float(row[metric_col].values[0]) - ens_val) > 1e-4:
            print(f"  [ci] {ds_name} {task_suffix}: CI workbook ensemble "
                  f"{float(row[metric_col].values[0]):.5f} != plotted {ens_val:.5f} -- not used")
            return None
        return float(row[lo_c].values[0]), float(row[hi_c].values[0])
    if ds_name.startswith("INSIGHT-"):
        fpath = os.path.join(INSIGHT_CI_ROOT, ds_name.split("-")[1], "eval_across_models_insight",
                             f"model_performance_dict_parsimonious_{task_suffix}_SPARC-P_insight.csv")
    else:
        dsid = {MSK_PROSPECTIVE: "MSK_CAIA_timeshift", "MIMICIV-hosp": "MIMICIV_hosp",
                "EHRSHOT": "stanfordEHRSHOT"}.get(ds_name)
        if dsid is None:
            return None          # CAIA sites: no CI available
        fpath = os.path.join(MSK_DIR if ds_name == MSK_PROSPECTIVE else EXT_DIR,
                             f"model_performance_dict_across_models_parsimonious_"
                             f"{task_suffix}_{dsid}_ensemble.csv")
    if not os.path.exists(fpath):
        print(f"  [ci] {ds_name} {task_suffix}: no CI file")
        return None
    df = pd.read_csv(fpath)
    lo_c, hi_c = _ci_cols(task_suffix)
    row = df[(df["model"] == "SPARC-P_ensemble") & (df["val_fold"].astype(str) == "ensemble")]
    if row.empty or lo_c not in df.columns or pd.isna(row[lo_c].values[0]):
        print(f"  [ci] {ds_name} {task_suffix}: no CI in file")
        return None
    if abs(float(row[metric_col].values[0]) - ens_val) > 1e-4:
        print(f"  [ci] {ds_name} {task_suffix}: CI file ensemble "
              f"{float(row[metric_col].values[0]):.5f} != plotted {ens_val:.5f} -- not used")
        return None
    return float(row[lo_c].values[0]), float(row[hi_c].values[0])


# Display name → sheet-name suffix
CAIA_SITE_SHEETS = {
    "DFCI-CAIA": "DFCI-CAIA",
    "FHCC-CAIA": "FH-CAIA",
    "JHU-CAIA":  "JHU-CAIA",
}

# task_suffix → sheet-name prefix
CAIA_TASK_SHEETS = {
    "binary":              "binary",
    "AKI":                 "AKI",
    "Liver_toxicity":      "livtox",
    "anemia":              "anemia",
    "febrile_neutropenia": "FN",
    "neutropenia":         "neut",
}

SAVE_DIR = os.path.join(RESULTS, 'vis')
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Tasks
# =============================================================================

TASKS = [
    # (task_suffix,          display_name,          metric_col)
    ("binary",              "180-day mortality",    "auroc"),
    ("AKI",                 "Acute kidney injury",  SE_METRIC_COL),
    ("Liver_toxicity",      "Liver toxicity",       SE_METRIC_COL),
    ("anemia",              "Anemia",               SE_METRIC_COL),
    ("febrile_neutropenia", "Febrile neutropenia",  SE_METRIC_COL),
    ("neutropenia",         "Neutropenia",          SE_METRIC_COL),
]
# --grouped: side-effect tasks first, 180-day mortality last
if _args.grouped:
    TASKS = [t for t in TASKS if t[0] != "binary"] + [t for t in TASKS if t[0] == "binary"]

# Which tasks have data in each external dataset.
# neutropenia MIMIC excluded: all folds below chance (N/label issue).
# Liver_toxicity MIMIC excluded: all folds at exactly 0.5 (filename: *_unincluded).
MIMIC_TASKS     = {"binary", "AKI", "anemia"}
EHRSHOT_TASKS   = {"binary", "AKI", "anemia"}
MSK_CAIA_TASKS  = {"binary", "AKI", "Liver_toxicity", "anemia", "febrile_neutropenia", "neutropenia"}


# =============================================================================
# Colors  (user-specified palette)
# =============================================================================

COHORT_COLORS = {
    "MSK-CAIA":               "#4A90A4",   # teal
    "MSK-CAIA prospective":   "#5FA8BD",   # lighter teal
    "DFCI-CAIA":              "#E8866F",   # coral
    "FHCC-CAIA":              "#B87FB8",   # mauve/purple-pink
    "JHU-CAIA":               "#9B8BC4",   # lavender
    "MIMICIV-hosp":           "#8172B2",   # purple
    "EHRSHOT":                "#D4A76A",   # amber
    "INSIGHT-A":              "#5B9BD5",   # blue (colorblind-friendly)
    "INSIGHT-B":              "#70AADE",   # light blue
    "INSIGHT-C":              "#85B9E7",   # lighter blue
    "INSIGHT-D":              "#4682B4",   # steel blue
    "INSIGHT-E":              "#3A6FA1",   # darker blue
    "INSIGHT-F":              "#2C5C8E",   # darkest blue
}

# Ensemble diamond, matching vis_eval_1_across_tasks_t1e_controlled_bar_
# 060826_horizontal_succinct_wilcoxon_all.py: red face, darkred edge.
ENSEMBLE_COLOR  = "red"
ENSEMBLE_EDGE   = "darkred"
ENSEMBLE_SIZE   = 6
ENSEMBLE_EDGEW  = 1.1
MSKCAIA_COLOR   = COHORT_COLORS["MSK-CAIA"]     # teal — MSK-CAIA vertical dotted line

# --grouped: muted colours per cohort type
GROUP_COLORS = {
    "MSK": "#3E9A97",                                    # muted teal
    "MSK prospective": "#C27C5D",                        # muted terracotta
    "Tertiary cancer centers": "#D4B651",                # muted mustard yellow
    "Comprehensive hospitals (INSIGHT)": "#5B84B1",      # muted blue
    "Comprehensive hospitals (public)": "#A7BED8",       # lighter blue, same hue
}
COHORT_GROUP = {
    "MSK-CAIA": "MSK",
    "MSK-CAIA prospective": "MSK prospective",
    "DFCI-CAIA": "Tertiary cancer centers",
    "FHCC-CAIA": "Tertiary cancer centers",
    "JHU-CAIA": "Tertiary cancer centers",
    **{f"INSIGHT-{c}": "Comprehensive hospitals (INSIGHT)" for c in "ABCDEF"},
    "MIMICIV-hosp": "Comprehensive hospitals (public)",
    "EHRSHOT": "Comprehensive hospitals (public)",
}
PUBLIC_DATASETS = {"MIMICIV-hosp", "EHRSHOT"}
PUBLIC_LABEL_COLOR = "#8C8C8C"
if _args.grouped:
    COHORT_COLORS = {ds: GROUP_COLORS[g] for ds, g in COHORT_GROUP.items()}
    MSKCAIA_COLOR = GROUP_COLORS["MSK"]

# Set True to show y-tick labels only on the leftmost subplot of each row
SHARE_Y_LABELS = True

# Display order: top → bottom in each subplot (barh reverses this)
MSK_PROSPECTIVE = "MSK-CAIA prospective"

DATASET_ORDER = [
    MSK_PROSPECTIVE,
    "DFCI-CAIA",
    "FHCC-CAIA",
    "JHU-CAIA",
    "INSIGHT-A",
    "INSIGHT-B",
    "INSIGHT-C",
    "INSIGHT-D",
    "INSIGHT-E",
    "INSIGHT-F",
    "MIMICIV-hosp",
    "EHRSHOT",
]

# MSK-CAIA prospective (internal, time-shifted) is set off from the external
# cohorts by a dashed rule directly beneath it. All bars share one outline
# weight: the rule alone carries the distinction.
BAR_LINEWIDTH   = 1.5
SEPARATOR_BELOW = MSK_PROSPECTIVE


# =============================================================================
# Data loading
# =============================================================================

def _load_df(fpath, metric_col):
    """Return DataFrame or None if file missing / column absent."""
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    if metric_col not in df.columns:
        return None
    return df


def load_insight_sparcp(data_root, task_suffix, metric_col):
    """Return (fold_vals, ens_val) for SPARC-P from an eval_insight.py results folder."""
    # The unit-fixed tree names these '*_multi_insight.csv'; older trees use
    # '*_SPARC-P_insight.csv'. Prefer the former, fall back to the latter.
    _dir = os.path.join(data_root, "eval_across_models_insight")
    _multi = os.path.join(
        _dir, f"model_performance_dict_parsimonious_{task_suffix}_multi_insight.csv")
    fpath = _multi if os.path.exists(_multi) else os.path.join(
        _dir, f"model_performance_dict_parsimonious_{task_suffix}_SPARC-P_insight.csv")
    df = _load_df(fpath, metric_col)
    if df is None:
        return np.array([]), np.nan

    fold_mask = (df["model"] == "SPARC-P") & (df["val_fold"].astype(str) != "ensemble")
    fold_vals = df.loc[fold_mask, metric_col].dropna().values.astype(float)

    ens_mask = (df["model"] == "SPARC-P_ensemble") & (df["val_fold"].astype(str) == "ensemble")
    ens_val  = float(df.loc[ens_mask, metric_col].iloc[0]) if ens_mask.any() else np.nan

    return fold_vals, ens_val


def load_external_sparcp(task_suffix, dataset_name, metric_col, data_dir=None):
    """Return (fold_vals, ens_val) for SPARC-P from the eval_1 results (EXT_DIR, or MSK_DIR for the MSK cohorts)."""
    fname = (
        f"model_performance_dict_across_models_parsimonious_"
        f"{task_suffix}_{dataset_name}_ensemble.csv"
    )
    df = _load_df(os.path.join(data_dir or EXT_DIR, fname), metric_col)
    if df is None:
        return np.array([]), np.nan

    fold_mask = (df["model"] == "SPARC-P") & (df["val_fold"].astype(str) != "ensemble")
    fold_vals = df.loc[fold_mask, metric_col].dropna().values.astype(float)

    ens_mask = (df["model"] == "SPARC-P_ensemble") & (df["val_fold"].astype(str) == "ensemble")
    ens_val  = float(df.loc[ens_mask, metric_col].iloc[0]) if ens_mask.any() else np.nan

    return fold_vals, ens_val


_CAIA_SHEETS = None


def _caia_sheets():
    """Read (and cache) every sheet of SPARC-P_CAIA_results.xlsx."""
    global _CAIA_SHEETS
    if _CAIA_SHEETS is None:
        _CAIA_SHEETS = (
            pd.read_excel(CAIA_XLSX, sheet_name=None)
            if os.path.exists(CAIA_XLSX) else {}
        )
    return _CAIA_SHEETS


def load_caia_sparcp(task_suffix, site_name, metric_col):
    """Return (fold_vals, ens_val) for one CAIA site (SPARC-P_CAIA_results.xlsx, sheet "{task}_{site}")."""
    task_sheet = CAIA_TASK_SHEETS.get(task_suffix)
    site_sheet = CAIA_SITE_SHEETS.get(site_name)
    if task_sheet is None or site_sheet is None:
        return np.array([]), np.nan

    df = _caia_sheets().get(f"{task_sheet}_{site_sheet}{CAIA_SHEET_SUFFIX}")
    if df is None or metric_col not in df.columns:
        return np.array([]), np.nan

    fold_mask = (df["model"] == "SPARC-P") & (df["val_fold"].astype(str) != "ensemble")
    fold_vals = df.loc[fold_mask, metric_col].dropna().values.astype(float)

    ens_mask = (df["model"] == "SPARC-P_ensemble") & (df["val_fold"].astype(str) == "ensemble")
    ens_val  = float(df.loc[ens_mask, metric_col].iloc[0]) if ens_mask.any() else np.nan

    return fold_vals, ens_val


# Minimum cohort size for an INSIGHT site x task cell.
#
# Rule used for the zero-shot vs finetune comparison:
#     "Site x task cells with fewer than MIN_N task-eligible patients are
#      dropped: the weighted AUROC degenerates (one site has <11 for the
#      *penia / anemia tasks, where every fold collapses to exactly 0.0 or 1.0)."
#     MIN_N = 20
#
# The gate is on n_with_medication -- task-ELIGIBLE PATIENTS, a denominator.
# It is NOT an event count; positive_counts.csv is a different quantity and is
# scraped from finetuning eval logs, so it must not be used for exclusion.
#
# Without this, sites with <20 eligible patients would be plotted for some tasks.
#
# n_numbers.csv records the strategy-agnostic test splits; set SPARC_INSIGHT_N_NUMBERS
# to its location. The unit fix corrected lab units, not medication eligibility, so
# these counts carry over.
MIN_N = 20
_N_NUMBERS_CSV = os.environ.get('SPARC_INSIGHT_N_NUMBERS')
if not _N_NUMBERS_CSV:
    print('WARNING: SPARC_INSIGHT_N_NUMBERS is not set; the MIN_N gate on INSIGHT cells is disabled')


def _insight_n_eligible(site, task_suffix):
    """Task-eligible patient count for an INSIGHT cell, or None if unknown."""
    if not hasattr(_insight_n_eligible, "_cache"):
        cache = {}
        try:
            _dn = pd.read_csv(_N_NUMBERS_CSV)  # None -> TypeError -> empty cache
            _dn = _dn[_dn["strategy"] == "A_mskpretrain"].dropna(
                subset=["n_with_medication"])
            cache = {(r.site, r.task): int(r.n_with_medication)
                     for r in _dn.itertuples()}
        except Exception:
            cache = {}
        _insight_n_eligible._cache = cache
    return _insight_n_eligible._cache.get((site, task_suffix))


# NOTE: do NOT gate INSIGHT cells on positive_counts.csv.
# That file is scraped from the FINETUNING eval logs, so
# a missing (site, task) row means only that no finetuning run covered the cell --
# not that the site has no events. Site E has anemia events and real zero-shot
# numbers in the pre-unit-fix tree, but no anemia row in positive_counts.csv.
# Degenerate cells are identified by _fold_stats below instead.


def _fold_stats(fold_vals):
    """Return (mean, std), or (nan, nan) if there are no valid folds or all folds equal 0, 0.5 or 1 (no events).
    Below-chance means are kept."""
    valid = fold_vals[~np.isnan(fold_vals)] if len(fold_vals) else fold_vals
    if len(valid) == 0:
        return np.nan, np.nan
    if np.std(valid) < 1e-9 and np.any(np.abs(valid[0] - np.array([0.0, 0.5, 1.0])) < 1e-6):
        return np.nan, np.nan
    return float(np.mean(valid)), float(np.std(valid))


def collect_all_data():
    """Return data[task][ds_name] = (mean, std, ens_val) and mskcaia_ens[task] = ensemble AUROC (or nan)."""
    data = {}
    mskcaia_ens = {}
    for task_suffix, _, metric_col in TASKS:
        data[task_suffix] = {}
        for ds_name in DATASET_ORDER:
            if ds_name == MSK_PROSPECTIVE:
                if task_suffix in MSK_CAIA_TASKS:
                    folds, ens = load_external_sparcp(
                        task_suffix, "MSK_CAIA_timeshift", metric_col,
                        data_dir=MSK_DIR,
                    )
                    mean, std = _fold_stats(folds)
                    data[task_suffix][ds_name] = (mean, std, ens if not np.isnan(mean) else np.nan)
                else:
                    data[task_suffix][ds_name] = (np.nan, np.nan, np.nan)

            elif ds_name in CAIA_SITE_SHEETS:
                folds, ens = load_caia_sparcp(task_suffix, ds_name, metric_col)
                mean, std  = _fold_stats(folds)
                data[task_suffix][ds_name] = (mean, std, ens if not np.isnan(mean) else np.nan)

            elif ds_name == "MIMICIV-hosp":
                if task_suffix in MIMIC_TASKS:
                    folds, ens = load_external_sparcp(task_suffix, "MIMICIV_hosp", metric_col)
                    mean, std  = _fold_stats(folds)
                    data[task_suffix][ds_name] = (mean, std, ens if not np.isnan(mean) else np.nan)
                else:
                    data[task_suffix][ds_name] = (np.nan, np.nan, np.nan)

            elif ds_name == "EHRSHOT":
                if task_suffix in EHRSHOT_TASKS:
                    folds, ens = load_external_sparcp(task_suffix, "stanfordEHRSHOT", metric_col)
                    mean, std  = _fold_stats(folds)
                    data[task_suffix][ds_name] = (mean, std, ens if not np.isnan(mean) else np.nan)
                else:
                    data[task_suffix][ds_name] = (np.nan, np.nan, np.nan)

            else:  # INSIGHT-A … INSIGHT-F
                site  = ds_name.split("-")[1]
                n_elig = _insight_n_eligible(site, task_suffix)
                if n_elig is not None and n_elig < MIN_N:
                    data[task_suffix][ds_name] = (np.nan, np.nan, np.nan)
                else:
                    site_root = os.path.join(INSIGHT_ROOT, site)
                    folds, ens = load_insight_sparcp(site_root, task_suffix, metric_col)
                    mean, std  = _fold_stats(folds)
                    data[task_suffix][ds_name] = (mean, std, ens if not np.isnan(mean) else np.nan)

        # MSK-CAIA ensemble line (vertical dotted line per subplot)
        if task_suffix in MSK_CAIA_TASKS:
            folds, ens = load_external_sparcp(
                task_suffix, "MSK_CAIA", metric_col, data_dir=MSK_DIR,
            )
            mean, _    = _fold_stats(folds)
            mskcaia_ens[task_suffix] = ens if not np.isnan(mean) else np.nan
        else:
            mskcaia_ens[task_suffix] = np.nan

    # Ensemble CIs, keyed like data; ci[task][ds] = (lo, hi) or None
    ci = {t: {ds: load_ensemble_ci(ds, t, mc, data[t][ds][2]) for ds in DATASET_ORDER}
          for t, _, mc in TASKS}
    return data, mskcaia_ens, ci


# =============================================================================
# Figure
# =============================================================================

def make_figure(all_data, mskcaia_ens, ci=None):
    n_ds   = len(DATASET_ORDER)
    ncols  = 3
    nrows  = 2

    # barh places item 0 at the bottom; reverse so MSK-CAIA prospective is on top
    y_labels = list(reversed(DATASET_ORDER))
    y_pos    = np.arange(n_ds)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 9.6))
    _left = 0.135 if SHARE_Y_LABELS else 0.18
    fig.subplots_adjust(
        left=_left, right=0.97,
        bottom=0.11, top=0.90,
        hspace=0.38, wspace=0.18,
    )

    for ti, (task_suffix, task_name, _) in enumerate(TASKS):
        row, col = divmod(ti, ncols)
        ax = axes[row][col]

        task_data = all_data[task_suffix]

        for yi, ds_name in enumerate(y_labels):
            mean_val, std_val, ens_val = task_data[ds_name]
            # Convert to percentage for display
            mean_pct = mean_val * 100 if not np.isnan(mean_val) else np.nan
            std_pct  = std_val  * 100 if not np.isnan(std_val)  else np.nan
            ens_pct  = ens_val  * 100 if not np.isnan(ens_val)  else np.nan
            color = COHORT_COLORS[ds_name]

            ci_val = (ci or {}).get(task_suffix, {}).get(ds_name)
            if CI_MODE == "ensemble":
                # bar at the ensemble value, asymmetric CI error bar (none if unavailable)
                mean_pct = ens_pct
                std_pct = (np.array([[max(ens_val - ci_val[0], 0) * 100],
                                     [max(ci_val[1] - ens_val, 0) * 100]])
                           if ci_val is not None else np.nan)

            if np.isnan(mean_pct):
                # Minimal stub indicating a placeholder / no data
                ax.barh(
                    yi, 1.0,
                    color=color, alpha=0.20, height=0.72,
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=1,
                )
            else:
                ax.barh(
                    yi, mean_pct,
                    xerr=(std_pct if not np.isscalar(std_pct) else
                          (std_pct if not np.isnan(std_pct) else 0)),
                    color=color,
                    alpha=0.82,
                    height=0.72,
                    edgecolor="black",
                    linewidth=BAR_LINEWIDTH,
                    capsize=3.0,
                    error_kw={
                        "linewidth": 1.0,
                        "capthick": 1.0,
                        "ecolor": "#333333",
                        "zorder": 4,
                    },
                    zorder=2,
                )

                # Red diamond for ensemble
                if CI_MODE == "folds" and not np.isnan(ens_pct):
                    ax.plot(
                        ens_pct, yi,
                        marker="D",
                        markersize=ENSEMBLE_SIZE,
                        color=ENSEMBLE_COLOR,
                        markeredgecolor=ENSEMBLE_EDGE,
                        markeredgewidth=ENSEMBLE_EDGEW,
                        zorder=10,
                    )

        # ── Dashed rule beneath MSK-CAIA prospective ──
        if SEPARATOR_BELOW in y_labels:
            ax.axhline(
                y_labels.index(SEPARATOR_BELOW) - 0.5,
                color="#555555",
                linewidth=0.9,
                linestyle=(0, (4, 3)),
                zorder=5,
            )

        # ── MSK-CAIA vertical dotted line ──
        msk_val = mskcaia_ens.get(task_suffix, np.nan)
        if not np.isnan(msk_val):
            ax.axvline(
                msk_val * 100,
                color=MSKCAIA_COLOR,
                linewidth=1.4,
                linestyle=(0, (3, 3)),
                zorder=5,
            )

        # ── Y axis ──
        ax.set_yticks(y_pos)
        if SHARE_Y_LABELS and col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(y_labels, fontsize=7.5)
            if _args.grouped:
                for _tl in ax.get_yticklabels():
                    if _tl.get_text() in PUBLIC_DATASETS:
                        _tl.set_color(PUBLIC_LABEL_COLOR)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_ylim(-0.55, n_ds - 0.45)

        # ── X axis ──
        ax.set_xlabel(f"{METRIC_LABEL} (%)", fontsize=9, labelpad=4)
        ax.set_xlim(45, 102)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.tick_params(axis="x", which="major", labelsize=7.5, length=3, width=0.6)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5, which="major")
        ax.set_axisbelow(True)

        # ── Title ──
        ax.set_title(task_name, fontsize=11, fontweight="bold", pad=8)

        # ── Spines ──
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_linewidth(0.7)
        ax.spines["bottom"].set_color("#333333")

    # ==========================================================================
    # Legend
    # ==========================================================================

    # 12 cohort patches + Ensemble diamond + MSK-CAIA line = 14
    # Layout: 7 cols × 2 rows (fills exactly, no spacer needed)
    legend_handles = [
        mpatches.Patch(
            facecolor=COHORT_COLORS[ds],
            alpha=0.82,
            edgecolor="black",
            linewidth=0.8,
            label=ds,
        )
        for ds in DATASET_ORDER          # 12 items
    ]
    if _args.grouped:
        legend_handles = [
            mpatches.Patch(facecolor=c, alpha=0.82, edgecolor="black", linewidth=0.8, label=g)
            for g, c in GROUP_COLORS.items() if g != "MSK"
        ]
    if CI_MODE == "folds":
        legend_handles.append(
            mlines.Line2D(
                [], [],
                marker="D", linestyle="None",
                color=ENSEMBLE_COLOR,
                markeredgecolor=ENSEMBLE_EDGE,
                markeredgewidth=ENSEMBLE_EDGEW,
                markersize=ENSEMBLE_SIZE,
                label="Ensemble",
            )
        )
    legend_handles.append(
        mlines.Line2D(
            [], [],
            color=MSKCAIA_COLOR, linewidth=1.4,
            linestyle=(0, (3, 3)),
            label="MSK-CAIA (retrospective)",
        )
    )
    if _args.grouped:
        legend_handles.append(mpatches.Patch(color="none", label="grey label = public dataset"))

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

    # ==========================================================================
    # Suptitle
    # ==========================================================================

    if CI_MODE == "ensemble":
        fig.text(0.5, 0.935,
                 "Bars: 10-fold ensemble, error bars: 95% bootstrap CI",
                 ha="center", fontsize=9, style="italic", color="#333333")
    fig.suptitle(
        f"SPARC-P {METRIC_LABEL} Across Clinical Tasks and External Cohorts",
        fontsize=12,
        fontweight="bold",
        y=0.97,
        ha="center",
    )

    # ==========================================================================
    # Save
    # ==========================================================================

    # Set SPARC_FIG_SUFFIX to write alongside an existing figure instead of
    # replacing it (e.g. SPARC_FIG_SUFFIX=_insight_zeroshot_fix).
    _suffix = os.environ.get("SPARC_FIG_SUFFIX", "") + ("_ensembleCI" if CI_MODE == "ensemble" else "") \
              + ("_overallauroc" if METRIC == "overall" else "") + ("_grouped" if _args.grouped else "")
    for ext in ("pdf", "png"):
        out = os.path.join(
            SAVE_DIR, f"sparcp_auc_by_task_cohort_with_prospective{_suffix}.{ext}")
        fig.savefig(
            out,
            bbox_inches="tight",
            facecolor="white",
            dpi=300 if ext == "png" else None,
        )
        print(f"Saved: {out}")

    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("Collecting data …")
    all_data, mskcaia_ens, ci = collect_all_data()

    # Print summary for quick verification
    for task_suffix, task_name, _ in TASKS:
        print(f"\n{task_name}:")
        for ds in DATASET_ORDER:
            mean, std, ens = all_data[task_suffix][ds]
            if np.isnan(mean):
                print(f"  {ds:<16s}  n/a")
            else:
                ens_str = f"  ens={ens:.4f}" if not np.isnan(ens) else ""
                print(f"  {ds:<16s}  mean={mean:.4f}  std={std:.4f}{ens_str}")
        msk = mskcaia_ens.get(task_suffix, np.nan)
        print(f"  {'MSK-CAIA':<16s}  {'n/a' if np.isnan(msk) else f'ens={msk:.4f}'}")

    print("\nGenerating figure …")
    make_figure(all_data, mskcaia_ens, ci)
    print("Done.")
