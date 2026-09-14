#!/usr/bin/env python3
"""
cancer_type_volume_vis.py

Compares ABSOLUTE patient volume in hard-to-treat cancer categories between
tertiary cancer centers (MSK, DFCI, FHCC, JHU) and non-tertiary INSIGHT
sites -- a third, deliberately different framing from cancer_type_vis.py
(% of cohort) and cancer_type_diversity_vis.py (evenness of case mix).

Why volume, after the other two didn't separate the groups: both prior
metrics are normalized by cohort size, which answers "what fraction of
this site's practice is hard-to-treat cancer" or "how evenly spread is
this site's practice." Neither answers "how much experience treating
hard-to-treat cancers has this site actually accumulated" -- a tertiary
center with a much larger total cohort could have far more absolute
hard-to-treat-cancer patients even if its hard-to-treat-cancer percentage
or diversity looks similar to a much smaller community site. Absolute
volume is the framing that isolates this, at the cost of conflating it
with overall cohort size (a site with more patients overall will tend to
have more of everything, hard-to-treat or not) -- panel B below makes
that scale relationship explicit so it isn't hidden.

Uses the same hard-to-treat type list as cancer_type_vis.py (Leukemia,
Neuroendocrine Tumor, Head and Neck Cancer -- selected for being common
at all three tertiary centers with a real, data-observed tertiary-vs-
INSIGHT prevalence gap, not an oncology-reputation guess; see that
script's docstring for the full derivation).

Produces a two-panel figure:
  A. Absolute number of hard-to-treat-type patients per site (log-scale
     bars), grouped tertiary vs. non-tertiary, following the color
     convention in dataset_summaries_combined_plot.py.
  B. Hard-to-treat patient count vs. total cohort size (scatter), so
     volume differences driven purely by overall cohort size vs. by a
     genuinely higher hard-to-treat-cancer concentration are visually
     separable.
"""

import os
import matplotlib
# Keep text editable in Illustrator - don't convert to paths
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

from ctypes_parse import DATA_DIR, OUTPUT_DIR, TERTIARY, COLORS, load_all_sites

# OUTPUT_DIR comes from ctypes_parse (env-overridable); see that module.
# Same "hard-to-treat" list as cancer_type_vis.py: types common at all
# three tertiary centers (FH/DFCI/MSK), with a real tertiary-vs-INSIGHT
# prevalence gap actually observed in this data (not just an oncology-
# reputation guess), and a defensible hard-to-treat clinical rationale.
# See that script's docstring for the full derivation and for why earlier
# rarity-based and reputation-based lists (including Sarcoma, Pancreatic
# Cancer, Glioma, NSCLC) were dropped.
HARD_TO_TREAT_TYPES = [
    'Leukemia',
    'Neuroendocrine Tumor',
    'Head and Neck Cancer',
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading per-site cancer type summaries...")
site_data, site_source = load_all_sites()

caveat_sites = [s for s, src in site_source.items() if src != 'first-mapped']
if caveat_sites:
    print(f"\nNote: {', '.join(caveat_sites)} only had a combined-labels table "
          "available (no first-mapped-type table); hard-to-treat counts for "
          "those sites come from single-type rows only and likely undercount "
          "true volume (co-occurrence patients are dropped).")

site_names = list(site_data.keys())
if not site_names:
    raise SystemExit("No ctypes_<SITE>_summary.txt files could be parsed.")

# ---------------------------------------------------------------------------
# Compute absolute hard-to-treat patient counts and total cohort size per site
# ---------------------------------------------------------------------------
hard_count_total = {}
total_cohort = {}
for s in site_names:
    d = site_data[s]
    hard_count_total[s] = sum(d.get(t, (0, 0.0))[0] for t in HARD_TO_TREAT_TYPES)
    # Total cohort = sum of all counts in the (single-type, first-mapped)
    # table. This slightly undercounts true cohort size when there are
    # multi-type ties (a patient counted under >1 type), so it's a lower
    # bound -- consistent across sites, so it doesn't bias the comparison.
    total_cohort[s] = sum(v[0] for v in d.values())

tertiary_sites = [s for s in site_names if s in TERTIARY]
nontertiary_sites = [s for s in site_names if s not in TERTIARY]
tertiary_sites.sort(key=lambda s: -hard_count_total[s])
nontertiary_sites.sort(key=lambda s: -hard_count_total[s])
ordered_sites = tertiary_sites + nontertiary_sites

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
print("\nCreating figure...")
fig_height = max(3.5, len(ordered_sites) * 0.55 + 1.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, fig_height),
                                gridspec_kw={'width_ratios': [1.1, 1.3], 'wspace': 0.32})

y_pos = np.arange(len(ordered_sites))

# --- Panel A: absolute hard-to-treat patient counts (log scale)
bar_colors = [COLORS[s] for s in ordered_sites]
counts = [hard_count_total[s] for s in ordered_sites]
bars1 = ax1.barh(y_pos, counts, color=bar_colors, edgecolor='black', linewidth=1.5)

for bar, val in zip(bars1, counts):
    ax1.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height() / 2,
              f'{val:,}', ha='left', va='center', fontsize=9.5, fontweight='bold')

ax1.set_yticks(y_pos)
labels = [f'{s} (tertiary)' if s in TERTIARY else s for s in ordered_sites]
ax1.set_yticklabels(labels, fontsize=10, fontweight='bold')
ax1.invert_yaxis()
ax1.set_xlabel('# patients with a hard-to-treat first-mapped type (log scale)',
               fontsize=10, fontweight='bold')
ax1.set_title('A. Absolute Hard-to-Treat Cancer Volume', fontsize=11, fontweight='bold', pad=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5, which='major')
ax1.set_xscale('log')
ax1.tick_params(axis='x', which='minor', length=0)
ax1.set_xlim(min(counts) * 0.5, max(counts) * 4)

if tertiary_sites and nontertiary_sites:
    divider_y = len(tertiary_sites) - 0.5
    ax1.axhline(divider_y, color='gray', linestyle='--', linewidth=1, alpha=0.6)

# --- Panel B: hard-to-treat count vs. total cohort size (log-log scatter)
for s in ordered_sites:
    marker = 'o' if s in TERTIARY else '^'
    ax2.scatter(total_cohort[s], hard_count_total[s], color=COLORS[s],
                s=140, marker=marker, edgecolor='black', linewidth=1.2,
                label=s, zorder=3)
    ax2.annotate(s, (total_cohort[s], hard_count_total[s]),
                 textcoords='offset points', xytext=(6, 4), fontsize=7.5)

# reference line: constant hard-to-treat fraction (using the pooled mean rate) so
# points above the line have MORE hard-to-treat volume than expected for
# their cohort size, and points below have less
all_fracs = [hard_count_total[s] / total_cohort[s] for s in site_names]
mean_frac = float(np.mean(all_fracs))
xs = np.array([min(total_cohort.values()) * 0.7, max(total_cohort.values()) * 1.3])
ax2.plot(xs, xs * mean_frac, ':', color='gray', linewidth=1.3, zorder=1,
         label=f'mean hard-to-treat rate ({mean_frac*100:.1f}%)')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Total cohort size (log scale)', fontsize=10, fontweight='bold')
ax2.set_ylabel('# hard-to-treat patients (log scale)', fontsize=10, fontweight='bold')
ax2.set_title('B. Volume vs. Overall Cohort Size', fontsize=11, fontweight='bold', pad=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(alpha=0.3, linestyle='-', linewidth=0.5, which='major')
ax2.tick_params(axis='both', which='minor', length=0)
ax2.legend(fontsize=6.8, frameon=True, loc='upper left', ncol=1)

footnote_lines = []
if caveat_sites:
    footnote_lines.append(
        f'{", ".join(caveat_sites)}: counts from combined-label single-type rows only (see note above); likely undercounts.')
footnote_lines.append(
    'Panel B: circles = tertiary centers, triangles = non-tertiary INSIGHT sites. '
    'Points above the dotted line have more hard-to-treat volume than expected from cohort size alone.')
fig.text(0.5, 0.02, '\n'.join(footnote_lines), ha='center', fontsize=7.5, style='italic', color='dimgray')

fig.suptitle('Tertiary Cancer Centers vs. Non-Tertiary Sites: Absolute Hard-to-Treat Cancer Volume',
             fontsize=12.5, fontweight='bold', y=0.98)

fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.18)

# ---------------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------------
output_path_svg = os.path.join(OUTPUT_DIR, 'cancer_type_volume_vis.svg')
plt.savefig(output_path_svg, format='svg', transparent=False)
print(f"\nSaved plot to: {output_path_svg}")

output_path_pdf = os.path.join(OUTPUT_DIR, 'cancer_type_volume_vis.pdf')
plt.savefig(output_path_pdf, dpi=150, transparent=False)
print(f"Saved plot to: {output_path_pdf}")

output_path_png = os.path.join(OUTPUT_DIR, 'cancer_type_volume_vis.png')
plt.savefig(output_path_png, dpi=300)
print(f"Saved plot to: {output_path_png}")

plt.close()

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("ABSOLUTE RARE/COMPLEX CANCER VOLUME SUMMARY")
print("=" * 90)
print(f"{'Site':<16} {'Group':<14} {'Hard-to-treat n':>16} {'Total cohort':>14} {'Rate':>8}")
print("-" * 90)
for s in ordered_sites:
    group = 'tertiary' if s in TERTIARY else 'non-tertiary'
    rate = hard_count_total[s] / total_cohort[s] * 100
    print(f"{s:<16} {group:<14} {hard_count_total[s]:>16,} {total_cohort[s]:>14,} {rate:>7.2f}%")
print("=" * 90)

print("\nDone!")
