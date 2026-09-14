#!/usr/bin/env python3
"""
cancer_type_foldchange_vis.py

For every eligible parent cancer type EXCEPT "others" and "Unknown
Primary" (coding/catch-all buckets, not real diagnoses -- excluded so the
comparison stays about actual disease types), compares mean first-mapped
% at the three tertiary centers (FH/DFCI/MSK) against mean first-mapped %
across the six INSIGHT (non-tertiary) sites, and shows which types are
enriched in each direction by at least FC_THRESHOLD-fold.

This replaces picking a fixed "hard-to-treat" or "rare" type list by hand
(see cancer_type_vis.py's docstring for that history and why earlier lists
were confounded) with a direct, systematic view of the full 30-type
landscape: every type gets classified and shown, not just a pre-selected
subset, so the reader can judge for themselves which differences look like
referral-pattern / complexity signal vs. population-incidence / screening-
practice signal (e.g. Hepatobiliary Cancer and Prostate/Colorectal are
large INSIGHT-side effects that are more plausibly regional-incidence or
screening-driven than complexity-driven).

Method: for each type, ratio = mean(tertiary %) / mean(INSIGHT %). A type
is "tertiary-enriched" if ratio >= FC_THRESHOLD, "INSIGHT-enriched" if
ratio <= 1/FC_THRESHOLD, else "similar" (not shown in the fold-change
panels, but listed in the console summary). This uses GROUP MEANS, not
the stricter no-overlap test used in an earlier ad hoc analysis -- a
fold-change view is about magnitude of the average difference, and
showing the per-site range alongside each bar (as thin whisker marks)
keeps the reader able to see when a type's separation isn't clean even
though the means differ by a lot.

Produces a three-panel figure:
  A. Tertiary-enriched types (ratio >= FC_THRESHOLD), bars = mean tertiary
     %, sorted by descending fold-change, with mean INSIGHT % and each
     site's individual rate shown as markers for context.
  B. INSIGHT-enriched types (ratio <= 1/FC_THRESHOLD), mirrored.
  C. A related but distinct phenomenon, not folded into A/B: the % of
     each site's patients who had >1 parent cancer type tied for their
     EARLIEST diagnosis day (a same-day multi-primary/synchronous-cancer
     tie). This is what makes the first-mapped-type counts behind panels
     A/B NOT sum to each site's total patient count -- a tied patient is
     counted once under each tied type (see cancer_type_vis.py and
     add_cancer_type_counts.py's docstrings for the same convention). It
     turns out to be a strikingly clean, independent tertiary-vs-INSIGHT
     signal in its own right (every tertiary site's rate exceeds every
     INSIGHT site's rate in this data), so it's shown directly rather than
     buried as a caveat.
Colors follow the convention in dataset_summaries_combined_plot.py
(tertiary sites: teal/coral/mauve/lavender; INSIGHT sites: blue gradient).
"""

import os
import matplotlib
# Keep text editable in Illustrator - don't convert to paths
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

from ctypes_parse import DATA_DIR, OUTPUT_DIR, TERTIARY, COLORS, load_all_sites, load_all_sites_multi_type_stats

# OUTPUT_DIR comes from ctypes_parse (env-overridable); see that module.
# Fold-change threshold defining "enriched" in either direction. Tune this
# and re-run to see how the two panels change -- 1.5x is a permissive
# default; 2x or 3x show progressively starker, more robust differences
# (see the console summary's full ranked list either way, independent of
# this threshold, so nothing is hidden by the choice).
FC_THRESHOLD = 1.5

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading per-site cancer type summaries...")
site_data, site_source = load_all_sites()

caveat_sites = [s for s, src in site_source.items() if src != 'first-mapped']
if caveat_sites:
    print(f"\nNote: {', '.join(caveat_sites)} only had a combined-labels table "
          "available (no first-mapped-type table); their per-type % comes "
          "from single-type rows only and likely undercounts true prevalence.")

site_names = list(site_data.keys())
tertiary_sites = [s for s in site_names if s in TERTIARY]
nontertiary_sites = [s for s in site_names if s not in TERTIARY]
if not tertiary_sites or not nontertiary_sites:
    raise SystemExit("Need at least one tertiary and one non-tertiary site to compare.")

print("\nLoading per-site multi-type-tie stats (for panel C)...")
multi_type_stats = load_all_sites_multi_type_stats()  # {site: (total, multi, pct)}
multi_type_sites = [s for s in site_names if s in multi_type_stats]

# "others" and "Unknown Primary" are coding/catch-all buckets (not
# determinable primary site, or not covered by the parent-type lookup),
# not real cancer diagnoses -- they measure diagnostic/coding completeness
# rather than disease characteristics, so they're excluded from this
# type-vs-type comparison entirely (same reasoning as dropping Unknown
# Primary from the earlier hard-to-treat list in cancer_type_vis.py).
EXCLUDED_TYPES = {'others', 'Unknown Primary'}

all_types = sorted(
    set().union(*[set(site_data[s].keys()) for s in site_names]) - EXCLUDED_TYPES
)

# ---------------------------------------------------------------------------
# Compute per-type fold-change (mean tertiary % / mean INSIGHT %)
# ---------------------------------------------------------------------------
rows = []  # (type, mean_tert, mean_non, ratio, tert_rates_by_site, non_rates_by_site)
for t in all_types:
    tert_rates = {s: site_data[s].get(t, (0, 0.0))[1] for s in tertiary_sites}
    non_rates = {s: site_data[s].get(t, (0, 0.0))[1] for s in nontertiary_sites}
    mean_tert = float(np.mean(list(tert_rates.values())))
    mean_non = float(np.mean(list(non_rates.values())))
    if mean_tert == 0 and mean_non == 0:
        continue  # type absent everywhere -- nothing to show
    ratio = (mean_tert / mean_non) if mean_non > 0 else float('inf')
    rows.append((t, mean_tert, mean_non, ratio, tert_rates, non_rates))

# Infinite-ratio types (zero mean rate in EVERY INSIGHT site) aren't a
# meaningful fold-change -- a true zero-in-six-sites result for a real
# diagnosis is itself implausible and much more likely reflects a
# concept-code/vocabulary gap on the INSIGHT side (the type's codes may
# simply not exist in INSIGHT's source vocabulary) than genuine absence of
# disease. Rather than plot a meaningless infinite bar, these are flagged
# separately and excluded from the fold-change panels and ranking.
infinite_ratio = [r for r in rows if r[3] == float('inf')]
rows = [r for r in rows if r[3] != float('inf')]
if infinite_ratio:
    print(f"\nExcluded {len(infinite_ratio)} type(s) with an infinite tertiary/INSIGHT "
          "ratio (zero mean rate in every INSIGHT site -- likely a coding/vocabulary "
          "gap on the INSIGHT side, not a real fold-change):")
    for t, mean_tert, mean_non, _, _, _ in infinite_ratio:
        print(f"  {t}: mean tertiary {mean_tert:.2f}%, mean INSIGHT {mean_non:.2f}%")

tertiary_enriched = [r for r in rows if r[3] >= FC_THRESHOLD]
insight_enriched = [r for r in rows if r[3] <= 1.0 / FC_THRESHOLD]
similar = [r for r in rows if r not in tertiary_enriched and r not in insight_enriched]

tertiary_enriched.sort(key=lambda r: -r[3] if r[3] != float('inf') else -1e9)
insight_enriched.sort(key=lambda r: r[3])

print(f"\nAt FC_THRESHOLD={FC_THRESHOLD}x: "
      f"{len(tertiary_enriched)} tertiary-enriched, "
      f"{len(insight_enriched)} INSIGHT-enriched, "
      f"{len(similar)} similar/neither.")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
print("\nCreating figure...")
n_left = len(tertiary_enriched)
n_right = len(insight_enriched)
n_c = len(multi_type_sites)
fig_height = max(3.5, max(n_left, n_right, n_c) * 0.42 + 2.0)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, fig_height),
                                     gridspec_kw={'wspace': 0.5, 'width_ratios': [1, 1, 0.85]})

TERTIARY_BAR_COLOR = '#4A90A4'   # teal, matches MSK-CAIA in the shared palette
INSIGHT_BAR_COLOR = '#5B9BD5'    # blue, matches INSIGHT-A in the shared palette


def _plot_panel(ax, panel_rows, bar_color, title, xlabel, site_group_for_dots, dot_color_map):
    y_pos = np.arange(len(panel_rows))
    means = [r[1] if bar_color == TERTIARY_BAR_COLOR else r[2] for r in panel_rows]
    ax.barh(y_pos, means, color=bar_color, edgecolor='black', linewidth=1.2, zorder=2)

    # overlay each individual site's rate as a small dot, so the reader can
    # see whether the group mean reflects a consistent effect or is driven
    # by one outlier site
    for i, row in enumerate(panel_rows):
        rates_dict = row[4] if site_group_for_dots == 'tertiary' else row[5]
        for site, val in rates_dict.items():
            ax.scatter(val, i, color=dot_color_map[site], edgecolor='black',
                       linewidth=0.6, s=32, zorder=3)

    ax.set_yticks(y_pos)
    labels = [f'{r[0]}  ({r[3]:.2f}x)' if r[3] != float('inf') else f'{r[0]}  (∞)'
              for r in panel_rows]
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    if means:
        ax.set_xlim(0, max(means) * 1.3)


if tertiary_enriched:
    _plot_panel(ax1, tertiary_enriched, TERTIARY_BAR_COLOR,
                f'A. Tertiary-Enriched Types (≥{FC_THRESHOLD}x)',
                'Mean first-mapped % (bar = tertiary mean;\ndots = individual site rates)',
                'tertiary', COLORS)
else:
    ax1.text(0.5, 0.5, f'No types ≥{FC_THRESHOLD}x tertiary-enriched',
              ha='center', va='center', transform=ax1.transAxes, fontsize=10, style='italic')
    ax1.set_axis_off()

if insight_enriched:
    _plot_panel(ax2, insight_enriched, INSIGHT_BAR_COLOR,
                f'B. INSIGHT-Enriched Types (≤{1/FC_THRESHOLD:.2f}x)',
                'Mean first-mapped % (bar = INSIGHT mean;\ndots = individual site rates)',
                'non-tertiary', COLORS)
else:
    ax2.text(0.5, 0.5, f'No types ≤{1/FC_THRESHOLD:.2f}x INSIGHT-enriched',
              ha='center', va='center', transform=ax2.transAxes, fontsize=10, style='italic')
    ax2.set_axis_off()

# --- Panel C: % of patients with >1 type tied for their earliest
# diagnosis day, one bar per site, tertiary vs. non-tertiary grouped and
# each sorted descending (same convention as cancer_type_vis.py's panels).
if multi_type_sites:
    c_tertiary = [s for s in multi_type_sites if s in TERTIARY]
    c_nontertiary = [s for s in multi_type_sites if s not in TERTIARY]
    c_tertiary.sort(key=lambda s: -multi_type_stats[s][2])
    c_nontertiary.sort(key=lambda s: -multi_type_stats[s][2])
    c_ordered = c_tertiary + c_nontertiary

    y_pos_c = np.arange(len(c_ordered))
    pcts_c = [multi_type_stats[s][2] for s in c_ordered]
    bar_colors_c = [COLORS[s] for s in c_ordered]
    bars_c = ax3.barh(y_pos_c, pcts_c, color=bar_colors_c, edgecolor='black', linewidth=1.2, zorder=2)

    for bar, val in zip(bars_c, pcts_c):
        ax3.text(bar.get_width() + max(pcts_c) * 0.02, bar.get_y() + bar.get_height() / 2,
                  f'{val:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')

    ax3.set_yticks(y_pos_c)
    labels_c = [f'{s} (tertiary)' if s in TERTIARY else s for s in c_ordered]
    ax3.set_yticklabels(labels_c, fontsize=9.5)
    ax3.invert_yaxis()
    ax3.set_xlabel('% of patients with >1 type tied\nfor earliest diagnosis day', fontsize=10, fontweight='bold')
    ax3.set_title('C. Multi-Type First-Diagnosis Rate', fontsize=11, fontweight='bold', pad=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.set_xlim(0, max(pcts_c) * 1.3)

    if c_tertiary and c_nontertiary:
        divider_y_c = len(c_tertiary) - 0.5
        ax3.axhline(divider_y_c, color='gray', linestyle='--', linewidth=1, alpha=0.6)
else:
    ax3.text(0.5, 0.5, 'No multi-type-tie stats available',
              ha='center', va='center', transform=ax3.transAxes, fontsize=10, style='italic')
    ax3.set_axis_off()

# shared legend for the site-color dots
legend_sites = tertiary_sites + nontertiary_sites
legend_handles = [plt.Line2D([0], [0], marker='o', linestyle='none', markersize=6,
                              markerfacecolor=COLORS[s], markeredgecolor='black', label=s)
                   for s in legend_sites]
fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(legend_sites), 9),
           fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(f'Tertiary vs. INSIGHT: Cancer Type Fold-Change (threshold={FC_THRESHOLD}x) and Multi-Type First Diagnoses',
             fontsize=12.5, fontweight='bold', y=1.02)

footnote = (f'{len(similar)} type(s) fell between {1/FC_THRESHOLD:.2f}x and {FC_THRESHOLD}x '
            '(not shown; see console summary for the full ranked list).')
if infinite_ratio:
    footnote += (f' {len(infinite_ratio)} type(s) with an infinite ratio (zero in every '
                 'INSIGHT site) excluded as a likely coding gap, not a real fold-change '
                 '(see console summary).')
if caveat_sites:
    footnote += f' {", ".join(caveat_sites)}: rates from combined-label single-type rows only (see note above).'
fig.text(0.5, -0.10, footnote, ha='center', fontsize=8, style='italic', color='dimgray')

fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.18)

# ---------------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------------
output_path_svg = os.path.join(OUTPUT_DIR, 'cancer_type_foldchange_vis.svg')
plt.savefig(output_path_svg, format='svg', transparent=False, bbox_inches='tight')
print(f"\nSaved plot to: {output_path_svg}")

output_path_pdf = os.path.join(OUTPUT_DIR, 'cancer_type_foldchange_vis.pdf')
plt.savefig(output_path_pdf, dpi=150, transparent=False, bbox_inches='tight')
print(f"Saved plot to: {output_path_pdf}")

output_path_png = os.path.join(OUTPUT_DIR, 'cancer_type_foldchange_vis.png')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {output_path_png}")

plt.close()

# ---------------------------------------------------------------------------
# Print full ranked summary (independent of FC_THRESHOLD)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("FULL RANKED FOLD-CHANGE SUMMARY (all types, tertiary mean / INSIGHT mean)")
print("=" * 90)
all_rows_sorted = sorted(rows, key=lambda r: -r[3] if r[3] != float('inf') else -1e9)
print(f"{'Type':<28} {'mean tertiary':>13} {'mean INSIGHT':>13} {'ratio':>10}")
print("-" * 90)
for t, mean_tert, mean_non, ratio, _, _ in all_rows_sorted:
    ratio_str = f"{ratio:.2f}x" if ratio != float('inf') else "inf"
    print(f"{t:<28} {mean_tert:>12.2f}% {mean_non:>12.2f}% {ratio_str:>10}")
print("=" * 90)

if multi_type_sites:
    print("\n" + "=" * 90)
    print("MULTI-TYPE FIRST-DIAGNOSIS RATE (panel C)")
    print("=" * 90)
    print(f"{'Site':<16} {'Group':<14} {'Total patients':>15} {'Multi-type n':>13} {'%':>8}")
    print("-" * 90)
    c_all_sorted = sorted(multi_type_sites, key=lambda s: -multi_type_stats[s][2])
    for s in c_all_sorted:
        total, multi, pct = multi_type_stats[s]
        group = 'tertiary' if s in TERTIARY else 'non-tertiary'
        print(f"{s:<16} {group:<14} {total:>15,} {multi:>13,} {pct:>7.2f}%")
    print("=" * 90)

print("\nDone!")
