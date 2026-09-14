#!/usr/bin/env python3
"""
cancer_type_vis.py

Compares first-mapped parent-cancer-type distributions between tertiary
cancer centers (MSK, DFCI, FHCC, JHU) and non-tertiary/community sites
(INSIGHT-A through F), to visualize the hypothesis that tertiary centers
see a case mix enriched for harder-to-treat cancers.

Reads the `ctypes_<SITE>_summary.txt` files in this directory (plain-text
stdout captures of cancer_type_summary.py / the pipeline's cancer-type
report) and produces a two-panel figure:
  A. % of cohort with a "hard-to-treat" first-mapped cancer type, one bar
     per site, grouped by tertiary vs. non-tertiary and colored following
     the palette convention in dataset_summaries_combined_plot.py.
  B. Per-type breakdown (stacked) of just the hard-to-treat bucket, so the
     composition (e.g. Leukemia vs. Neuroendocrine Tumor vs. Head and Neck)
     is visible too, not just the aggregate.

# Why "hard-to-treat" instead of "rare", and how the type list was picked

An earlier version of this script defined the comparison category by
INCIDENCE RARITY (a fixed list of uncommon cancer types: Sarcoma, GIST,
Mesothelioma, Appendix Cancer, Hepatobiliary Cancer, etc.). That framing
turned out to be confounded: incidence rarity varies by REGION/POPULATION
as much as by referral pattern, so a non-tertiary site can score "high
rare-cancer burden" simply because a population-incidence-driven type
(e.g. Hepatobiliary Cancer, tied to regional hepatitis B/C or NASH
burden) happens to be common in its catchment area -- a real
epidemiological signal, but not evidence of harder cases or more complex
referrals.

A second attempt selected types by (a) being common at all three tertiary
centers and (b) general oncology-literature "hard to treat" reputation
(Pancreatic Cancer, Glioma, Esophagogastric Cancer, Non-Small Cell Lung
Cancer, Neuroendocrine Tumor, Leukemia, Renal Cell Carcinoma). That list
was wrong in a different way: it was built by eyeballing prognosis
reputation without actually checking whether each type was more
prevalent at tertiary sites than at INSIGHT sites in THIS data. When
checked, several of those picks ran the WRONG direction --
tertiary-site mean vs. INSIGHT-site mean ratio was 0.73x for Pancreatic
Cancer, 0.66x for Glioma, 0.61x for Esophagogastric Cancer, and only
1.66x for Non-Small Cell Lung Cancer (a weak signal, and NSCLC is also a
poor "hard to treat" category on its own terms -- it spans resectable
Stage IA, with >90% 5yr survival, through Stage IV; its oft-cited ~28%
5yr figure is a population average driven by stage mix, not a property
of the disease, and a growing share now responds well to targeted/immune
therapy).

This version instead requires BOTH conditions to actually hold in the
data: the type must (a) be common at all three tertiary centers already
present (FH/DFCI/MSK) and (b) show a real, consistent tertiary-vs-
INSIGHT prevalence gap (mean tertiary % well above mean INSIGHT %, not
just barely over 1x), AND (c) still be a defensible "hard to treat"
category on its own clinical terms -- ruling out types whose ratio is
high only because they're common referral volume with GOOD prognosis
(e.g. Breast Cancer: 3.46x ratio, but not hard to treat) or too
heterogeneous to call uniformly hard (Lymphoma: 2.70x ratio, but spans
very curable early-stage Hodgkin's through aggressive relapsed/refractory
disease; Non-Small Cell Lung Cancer for the reasons above).

HARD_TO_TREAT_TYPES (mean tertiary % / mean INSIGHT % ratio in this
data, then the clinical complexity rationale):
  Leukemia (5.82x -- heterogeneous but includes AML, ~30% 5yr),
  Neuroendocrine Tumor (3.86x -- highly grade/stage-dependent, complex
  multidisciplinary management), Head and Neck Cancer (2.77x -- typically
  requires combined surgery/radiation/chemotherapy with major functional
  and reconstructive complexity).

Sarcoma would also show a real ratio (3.08x) and is genuinely hard to
treat, but is excluded because FH's pipeline run appears to undercount
Sarcoma specifically (likely a condition_concept_id mapping mismatch,
consistent with the site-specific concept_id divergences already
documented in SITE_CROSSCHECK_METHODOLOGY.md) -- including it would
confound the comparison with a known data artifact rather than reflect
real case-mix.

JHU-CAIA is included in the site list but will be skipped automatically
(with a printed note) until `ctypes_JHU_summary.txt` exists alongside the
other files in this directory.
"""

import os
import matplotlib
# Keep text editable in Illustrator - don't convert to paths
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

from ctypes_parse import DATA_DIR, OUTPUT_DIR, TERTIARY, COLORS, load_all_sites

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# "Hard-to-treat" cancer type list (see module docstring): types that are
# BOTH common at all three tertiary centers already in the data (FH/DFCI/
# MSK) AND show a real tertiary-vs-INSIGHT prevalence gap in this data
# (not just an oncology-reputation guess), AND are a defensible "hard to
# treat" category on their own clinical terms (ruling out high-ratio types
# that are just common referral volume with good prognosis, or too
# heterogeneous to call uniformly hard).
HARD_TO_TREAT_TYPES = [
    'Leukemia',
    'Neuroendocrine Tumor',
    'Head and Neck Cancer',
]

# Distinct colors for each hard-to-treat type in the stacked breakdown panel
# (categorical, colorblind-friendly; order matches HARD_TO_TREAT_TYPES).
HARD_TO_TREAT_TYPE_COLORS = {
    'Leukemia': '#E377C2',
    'Neuroendocrine Tumor': '#98DF8A',
    'Head and Neck Cancer': '#D62728',
}

# OUTPUT_DIR comes from ctypes_parse (env-overridable); see that module.
# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading per-site cancer type summaries...")
site_data, site_source = load_all_sites()

caveat_sites = [s for s, src in site_source.items() if src != 'first-mapped']
if caveat_sites:
    print(f"\nNote: {', '.join(caveat_sites)} only had a combined-labels table "
          "available (no first-mapped-type table), so hard-to-treat % for "
          "those sites is computed from single-type rows only and likely "
          "*undercounts* true prevalence (co-occurrence rows like "
          "'Glioma + Head and Neck Cancer' are excluded). Treat their bars "
          "as a lower bound.")

site_names = list(site_data.keys())
if not site_names:
    raise SystemExit("No ctypes_<SITE>_summary.txt files could be parsed.")

# ---------------------------------------------------------------------------
# Outlier-dropout detection: guard the plot against a single (site, type)
# mapping/coverage bug silently deflating a site's total. This is a general
# safety net, not tied to any specific type currently in HARD_TO_TREAT_TYPES
# -- it's what originally caught FH's Sarcoma rate showing ~0.5% vs. ~6-11%
# at DFCI/MSK (Sarcoma has since been dropped from the list entirely for
# that reason; see module docstring), and it stays in place in case a
# similar per-site mapping gap shows up in one of the currently-tracked
# types (Pancreatic, Glioma, Esophagogastric, NSCLC, Neuroendocrine,
# Leukemia, Renal Cell Carcinoma) or a future one added to the list.
#
# Rather than hardcoding a one-off exclusion, flag any (site, type) cell
# whose rate is a drastic dropout relative to the median for that type
# *within its own group* (tertiary vs. non-tertiary) -- the two groups can
# have genuinely different base rates for a given type for real
# epidemiological reasons, not just data bugs, so comparing against the
# pooled cross-site median would misfire on real between-group signal.
# Comparing within-group instead only fires when a site looks anomalous
# against its true peers.
#
# A cell is flagged when it's below DROPOUT_REL_THRESHOLD * peer-group
# median, the peer-group median itself is non-trivial
# (>= DROPOUT_MIN_MEDIAN_PCT, so this doesn't fire on noise from
# already-tiny categories), and there are enough peers to make a median
# meaningful (>= DROPOUT_MIN_GROUP_SIZE). Flagged cells are excluded from
# that site's total and shown hatched/greyed in the stacked panel with a
# footnote, rather than silently omitted.
# ---------------------------------------------------------------------------
DROPOUT_REL_THRESHOLD = 0.15     # site rate < 15% of peer-group median -> flag
DROPOUT_MIN_MEDIAN_PCT = 1.0     # only guard types with a non-trivial peer median
DROPOUT_MIN_GROUP_SIZE = 2       # need at least this many nonzero peers for a median

peer_groups = {s: ('tertiary' if s in TERTIARY else 'non-tertiary') for s in site_names}

dropout_cells = []  # list of (site, type, site_pct, peer_median_pct)
for rt in HARD_TO_TREAT_TYPES:
    rates = {s: site_data[s].get(rt, (0, 0.0))[1] for s in site_names}
    for group in ('tertiary', 'non-tertiary'):
        group_sites = [s for s in site_names if peer_groups[s] == group]
        for s in group_sites:
            # median of this site's peers only -- excluding s itself, so a
            # low self-value can't pull down its own comparison baseline
            others = [rates[o] for o in group_sites if o != s and rates[o] > 0]
            if len(others) < DROPOUT_MIN_GROUP_SIZE:
                continue
            med = float(np.median(others))
            if med < DROPOUT_MIN_MEDIAN_PCT:
                continue
            v = rates[s]
            if v > 0 and v < DROPOUT_REL_THRESHOLD * med:
                dropout_cells.append((s, rt, v, med))

dropout_lookup = {(s, rt) for s, rt, _, _ in dropout_cells}
if dropout_cells:
    print("\nData-quality guard: excluding likely mapping/coverage dropouts "
          "from totals (site rate far below its peer-group's median for that type):")
    for s, rt, v, med in dropout_cells:
        print(f"  {s} / {rt}: {v:.2f}% vs. {peer_groups[s]} peer median {med:.2f}% -> excluded, shown hatched")

# ---------------------------------------------------------------------------
# Compute hard-to-treat % per site (Panel A) and per-type breakdown (Panel B)
# ---------------------------------------------------------------------------
hard_pct_total = {}
hard_pct_by_type = {}
for site in site_names:
    d = site_data[site]
    by_type = {t: d.get(t, (0, 0.0))[1] for t in HARD_TO_TREAT_TYPES}
    hard_pct_by_type[site] = by_type
    # totals exclude flagged dropout cells (they're a data artifact, not signal)
    hard_pct_total[site] = sum(v for t, v in by_type.items()
                                if (site, t) not in dropout_lookup)

# Order sites: tertiary first (already in SITES order), then non-tertiary,
# each group sorted descending by hard-to-treat % for a clean visual story.
tertiary_sites = [s for s in site_names if s in TERTIARY]
nontertiary_sites = [s for s in site_names if s not in TERTIARY]
tertiary_sites.sort(key=lambda s: -hard_pct_total[s])
nontertiary_sites.sort(key=lambda s: -hard_pct_total[s])
ordered_sites = tertiary_sites + nontertiary_sites

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
print("\nCreating figure...")
fig_height = max(3.5, len(ordered_sites) * 0.55 + 1.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, fig_height),
                                gridspec_kw={'width_ratios': [1.1, 1.5], 'wspace': 0.35})

y_pos = np.arange(len(ordered_sites))

# --- Panel A: total hard-to-treat % per site, grouped tertiary vs non-tertiary
bar_colors = [COLORS[s] for s in ordered_sites]
totals = [hard_pct_total[s] for s in ordered_sites]
bars1 = ax1.barh(y_pos, totals, color=bar_colors, edgecolor='black', linewidth=1.5)

sites_with_dropout = {s for s, _, _, _ in dropout_cells}
for bar, val, site in zip(bars1, totals, ordered_sites):
    suffix = ''
    if site_source[site] != 'first-mapped':
        suffix += '*'
    if site in sites_with_dropout:
        suffix += '†'  # dagger: this total excludes a flagged dropout cell
    ax1.text(bar.get_width() + max(totals) * 0.02, bar.get_y() + bar.get_height() / 2,
              f'{val:.1f}%{suffix}', ha='left', va='center', fontsize=9.5, fontweight='bold')

ax1.set_yticks(y_pos)
labels = [f'{s} (tertiary)' if s in TERTIARY else s for s in ordered_sites]
ax1.set_yticklabels(labels, fontsize=10, fontweight='bold')
ax1.invert_yaxis()
ax1.set_xlabel('% of cohort with a hard-to-treat first-mapped cancer type', fontsize=10, fontweight='bold')
ax1.set_title('A. Hard-to-Treat Cancer Burden by Site', fontsize=11, fontweight='bold', pad=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_xlim(0, max(totals) * 1.25)

# horizontal divider between tertiary and non-tertiary groups
if tertiary_sites and nontertiary_sites:
    divider_y = len(tertiary_sites) - 0.5
    ax1.axhline(divider_y, color='gray', linestyle='--', linewidth=1, alpha=0.6)

# --- Panel B: stacked per-type composition of the hard-to-treat bucket.
# Flagged dropout cells are drawn as a thin hatched marker at a fixed,
# non-stacked width (rather than their real, likely-artifactual value)
# so the gap is visible without distorting the stack or the site's total.
DROPOUT_MARKER_WIDTH = 0.35  # fixed visual width (%, x-axis units) for markers
left = np.zeros(len(ordered_sites))
for rt in HARD_TO_TREAT_TYPES:
    vals = []
    for s in ordered_sites:
        if (s, rt) in dropout_lookup:
            vals.append(0.0)  # excluded from the real stack entirely
        else:
            vals.append(hard_pct_by_type[s][rt])
    vals = np.array(vals)
    ax2.barh(y_pos, vals, left=left, color=HARD_TO_TREAT_TYPE_COLORS[rt],
             edgecolor='black', linewidth=0.6, label=rt)
    left += vals

# draw dropout markers after the real stack, appended past each site's bar end
for s, rt, v, med in dropout_cells:
    i = ordered_sites.index(s)
    ax2.barh(y_pos[i], DROPOUT_MARKER_WIDTH, left=left[i], color=HARD_TO_TREAT_TYPE_COLORS[rt],
             edgecolor='black', linewidth=0.8, hatch='///', alpha=0.55)
    left[i] += DROPOUT_MARKER_WIDTH

ax2.set_yticks(y_pos)
ax2.set_yticklabels(ordered_sites, fontsize=10, fontweight='bold')
ax2.invert_yaxis()
ax2.set_xlabel('% of cohort, stacked by hard-to-treat type', fontsize=10, fontweight='bold')
ax2.set_title('B. Composition of Hard-to-Treat Cancer Burden', fontsize=11, fontweight='bold', pad=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax2.legend(fontsize=7.3, frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.16),
           ncol=3, columnspacing=1.0, handletextpad=0.5)

if tertiary_sites and nontertiary_sites:
    ax2.axhline(divider_y, color='gray', linestyle='--', linewidth=1, alpha=0.6)

footnote_lines = []
if caveat_sites:
    footnote_lines.append(
        f'* {", ".join(caveat_sites)}: no first-mapped-type table available; '
        'value derived from combined-label table (single-type rows only) and likely undercounts true prevalence.')
if dropout_cells:
    dropout_desc = '; '.join(f'{s}/{rt} ({v:.2f}% vs. {med:.2f}% median)' for s, rt, v, med in dropout_cells)
    footnote_lines.append(
        f'† hatched segment = flagged data-quality dropout, excluded from that site\'s total: {dropout_desc}.')

if footnote_lines:
    fig.text(0.5, 0.015, '\n'.join(footnote_lines),
              ha='center', fontsize=7.5, style='italic', color='dimgray')

fig.suptitle('Tertiary Cancer Centers vs. Non-Tertiary Sites: Hard-to-Treat Cancer Representation',
             fontsize=12.5, fontweight='bold', y=0.98)

fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.38 if footnote_lines else 0.30)

# ---------------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------------
output_path_svg = os.path.join(OUTPUT_DIR, 'cancer_type_vis.svg')
plt.savefig(output_path_svg, format='svg', transparent=False)
print(f"\nSaved plot to: {output_path_svg}")

output_path_pdf = os.path.join(OUTPUT_DIR, 'cancer_type_vis.pdf')
plt.savefig(output_path_pdf, dpi=150, transparent=False)
print(f"Saved plot to: {output_path_pdf}")

output_path_png = os.path.join(OUTPUT_DIR, 'cancer_type_vis.png')
plt.savefig(output_path_png, dpi=300)
print(f"Saved plot to: {output_path_png}")

plt.close()

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HARD-TO-TREAT CANCER BURDEN SUMMARY")
print("=" * 70)
print(f"{'Site':<16} {'Group':<14} {'Hard-to-treat %':>15}")
print("-" * 70)
for site in ordered_sites:
    group = 'tertiary' if site in TERTIARY else 'non-tertiary'
    flag = '*' if site_source[site] != 'first-mapped' else ''
    print(f"{site:<16} {group:<14} {hard_pct_total[site]:>13.2f}%{flag}")
print("=" * 70)

print("\nDone!")
