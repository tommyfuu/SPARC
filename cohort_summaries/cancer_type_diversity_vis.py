#!/usr/bin/env python3
"""
cancer_type_diversity_vis.py

Compares case-mix DIVERSITY (evenness across all 30 parent cancer types)
between tertiary cancer centers (MSK, DFCI, FHCC, JHU) and non-tertiary
INSIGHT sites, as an alternative to cancer_type_vis.py's rare/complex-%
framing.

Why a separate metric: "% of cohort with a rare/complex first-mapped
type" turned out to be confounded in two ways once inspected closely.
(1) It's sensitive to a single type's data-quality issues (FH's Sarcoma
mapping looks broken, dragging FH's whole rare/complex % down — see the
dropout guard in cancer_type_vis.py). (2) More fundamentally, it's a
proportion, so a tertiary center that ALSO sees huge absolute volumes of
common cancers (e.g. Breast >20% at MSK/DFCI) gets its rare-type
percentage diluted even if it draws a broad range of complex cases in
practice. Removing the confounded Sarcoma category from that comparison
also erased most of the tertiary-vs-non-tertiary gap it was meant to show.

Shannon diversity sidesteps both problems: it summarizes how EVEN the
full 30-type distribution is, independent of which specific type carries
the signal, and independent of any single type's absolute rate. A site
whose cohort is spread broadly across many cancer types (tertiary
referral pattern) should show higher diversity than one dominated by 1-2
common, screening-driven types (community/regional pattern) even if both
see, say, only 5% Sarcoma.

Metrics (computed over each site's full first-mapped-type distribution,
all 30 eligible types, from all_CAIA_cancer_types.txt):
  - Shannon entropy H = -sum(p_i * log(p_i))
  - Effective number of types = exp(H) ("Hill number" q=1) — interpretable
    as "the cohort behaves as if evenly spread across this many types"
  - Simpson diversity index (1 - sum(p_i^2)) as a secondary/robustness check

Produces a two-panel figure:
  A. Effective number of types per site (bar), grouped tertiary vs.
     non-tertiary, following the color convention in
     dataset_summaries_combined_plot.py.
  B. Full-distribution "evenness" curve per site: sorted type proportions
     (largest first) on a shared axis, so the shape of dominance vs.
     spread is visible directly, not just summarized into one number.
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
# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading per-site cancer type summaries...")
site_data, site_source = load_all_sites()

caveat_sites = [s for s, src in site_source.items() if src != 'first-mapped']
if caveat_sites:
    print(f"\nNote: {', '.join(caveat_sites)} only had a combined-labels table "
          "available (no first-mapped-type table), so their distribution is "
          "built from single-type rows only -- co-occurrence patients are "
          "dropped rather than mis-assigned, which slightly flattens/narrows "
          "the true distribution for those sites.")

site_names = list(site_data.keys())
if not site_names:
    raise SystemExit("No ctypes_<SITE>_summary.txt files could be parsed.")

# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------
def diversity_metrics(type_pct_dict):
    """
    type_pct_dict: {type_name: (count, pct_of_cohort)}. Percentages need not
    sum to exactly 100 (first-mapped tables can have ties counted under
    multiple types), so proportions are renormalized to sum to 1 before
    computing diversity -- diversity metrics describe the SHAPE of the
    distribution, which is what matters here, not its absolute scale.
    """
    pcts = np.array([v[1] for v in type_pct_dict.values()], dtype=float)
    pcts = pcts[pcts > 0]
    p = pcts / pcts.sum()
    shannon_h = float(-np.sum(p * np.log(p)))
    effective_n = float(np.exp(shannon_h))
    simpson = float(1.0 - np.sum(p ** 2))
    return {
        'shannon_h': shannon_h,
        'effective_n': effective_n,
        'simpson': simpson,
        'n_types_present': int(len(p)),
        'sorted_p': np.sort(p)[::-1],  # largest first, for panel B
    }

metrics = {s: diversity_metrics(site_data[s]) for s in site_names}

# Order sites: tertiary first, then non-tertiary, each sorted descending by
# effective number of types for a clean visual story.
tertiary_sites = [s for s in site_names if s in TERTIARY]
nontertiary_sites = [s for s in site_names if s not in TERTIARY]
tertiary_sites.sort(key=lambda s: -metrics[s]['effective_n'])
nontertiary_sites.sort(key=lambda s: -metrics[s]['effective_n'])
ordered_sites = tertiary_sites + nontertiary_sites

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
print("\nCreating figure...")
fig_height = max(3.5, len(ordered_sites) * 0.55 + 1.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, fig_height),
                                gridspec_kw={'width_ratios': [1.1, 1.4], 'wspace': 0.3})

y_pos = np.arange(len(ordered_sites))

# --- Panel A: effective number of types per site
bar_colors = [COLORS[s] for s in ordered_sites]
eff_n = [metrics[s]['effective_n'] for s in ordered_sites]
bars1 = ax1.barh(y_pos, eff_n, color=bar_colors, edgecolor='black', linewidth=1.5)

for bar, val in zip(bars1, eff_n):
    ax1.text(bar.get_width() + max(eff_n) * 0.02, bar.get_y() + bar.get_height() / 2,
              f'{val:.1f}', ha='left', va='center', fontsize=9.5, fontweight='bold')

ax1.set_yticks(y_pos)
labels = [f'{s} (tertiary)' if s in TERTIARY else s for s in ordered_sites]
ax1.set_yticklabels(labels, fontsize=10, fontweight='bold')
ax1.invert_yaxis()
ax1.set_xlabel('Effective number of cancer types\n(exp(Shannon entropy), out of 30 possible)',
               fontsize=10, fontweight='bold')
ax1.set_title('A. Case-Mix Diversity by Site', fontsize=11, fontweight='bold', pad=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_xlim(0, max(eff_n) * 1.25)

if tertiary_sites and nontertiary_sites:
    divider_y = len(tertiary_sites) - 0.5
    ax1.axhline(divider_y, color='gray', linestyle='--', linewidth=1, alpha=0.6)

# --- Panel B: sorted type-proportion curves (evenness shape), one line per site
for s in ordered_sites:
    sp = metrics[s]['sorted_p'] * 100  # back to % for readability
    rank = np.arange(1, len(sp) + 1)
    style = '-' if s in TERTIARY else '--'
    lw = 2.2 if s in TERTIARY else 1.6
    ax2.plot(rank, sp, style, color=COLORS[s], linewidth=lw, label=s,
              marker='o', markersize=3)

ax2.set_xlabel('Cancer type rank (1 = most common at that site)', fontsize=10, fontweight='bold')
ax2.set_ylabel('% of cohort (renormalized)', fontsize=10, fontweight='bold')
ax2.set_title('B. Distribution Shape: Dominance vs. Spread', fontsize=11, fontweight='bold', pad=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(alpha=0.3, linestyle='-', linewidth=0.5)
ax2.legend(fontsize=7.5, frameon=True, loc='upper right', ncol=2)
ax2.set_xlim(0.5, 15.5)  # top 15 ranks is where the shape differences live

footnote_lines = []
if caveat_sites:
    footnote_lines.append(
        f'{", ".join(caveat_sites)}: distribution built from combined-label single-type rows only (see note above).')
footnote_lines.append(
    'Solid lines/warm-purple bars = tertiary centers; dashed lines/blue bars = non-tertiary INSIGHT sites. '
    'A flatter, right-shifted curve in panel B = more even spread across many cancer types.')
fig.text(0.5, 0.02, '\n'.join(footnote_lines), ha='center', fontsize=7.5, style='italic', color='dimgray')

fig.suptitle('Tertiary Cancer Centers vs. Non-Tertiary Sites: Case-Mix Diversity',
             fontsize=12.5, fontweight='bold', y=0.98)

fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.16)

# ---------------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------------
output_path_svg = os.path.join(OUTPUT_DIR, 'cancer_type_diversity_vis.svg')
plt.savefig(output_path_svg, format='svg', transparent=False)
print(f"\nSaved plot to: {output_path_svg}")

output_path_pdf = os.path.join(OUTPUT_DIR, 'cancer_type_diversity_vis.pdf')
plt.savefig(output_path_pdf, dpi=150, transparent=False)
print(f"Saved plot to: {output_path_pdf}")

output_path_png = os.path.join(OUTPUT_DIR, 'cancer_type_diversity_vis.png')
plt.savefig(output_path_png, dpi=300)
print(f"Saved plot to: {output_path_png}")

plt.close()

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("CASE-MIX DIVERSITY SUMMARY")
print("=" * 90)
print(f"{'Site':<16} {'Group':<14} {'Shannon H':>10} {'Effective N':>12} {'Simpson':>9} {'#Types':>8}")
print("-" * 90)
for s in ordered_sites:
    group = 'tertiary' if s in TERTIARY else 'non-tertiary'
    m = metrics[s]
    print(f"{s:<16} {group:<14} {m['shannon_h']:>10.3f} {m['effective_n']:>12.2f} "
          f"{m['simpson']:>9.3f} {m['n_types_present']:>8}")
print("=" * 90)

print("\nDone!")
