#!/usr/bin/env python3
"""Cohort summary figure: (A) patients, (B) lab and vital measurements, (C) median visits per
patient with IQR, for the 13 cohorts (MSK-CAIA, MSK-CAIA prospective, DFCI/FHCC/JHU-CAIA,
INSIGHT A-F, MIMICIV-hosp, EHRSHOT).

--succinct drops DFCI-CAIA; --grouped colours by cohort type and greys the public-dataset labels.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
# editable text in Illustrator
matplotlib.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
matplotlib.rcParams['pdf.fonttype'] = 42  # Type 42 (TrueType) - stays as live, editable text in Illustrator
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import OrderedDict

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate dataset summary plots')
parser.add_argument('--succinct', action='store_true',
                    help='If set, exclude DFCI-CAIA from visualizations')
parser.add_argument('--grouped', action='store_true',
                    help='One colour per cohort type instead of per cohort, public-dataset '
                         'labels in grey; adds _grouped to the output filenames')
args = parser.parse_args()

# Configuration
BASE_DIR = os.environ.get('SPARC_DATASET_SUMMARIES_DIR',
    '/path/to/cohort_summaries/dataset_summaries')

# display name -> summary directory, in plot order
ALL_DATASETS = OrderedDict([
    ('MSK-CAIA', 'parsimonious_MSK_CAIA'),
    ('MSK-CAIA\nprospective', 'parsimonious_MSK_CAIA_timeshift'),
    ('DFCI-CAIA', 'parsimonious_DFCI_CAIA_all_060526'),
    ('FHCC-CAIA', 'parsimonious_FHCC_CAIA'),
    ('JHU-CAIA', 'parsimonious_JHU_CAIA'),
    ('INSIGHT-A', 'parsimonious_INSIGHT_A'),
    ('INSIGHT-B', 'parsimonious_INSIGHT_B'),
    ('INSIGHT-C', 'parsimonious_INSIGHT_C'),
    ('INSIGHT-D', 'parsimonious_INSIGHT_D'),
    ('INSIGHT-E', 'parsimonious_INSIGHT_E'),
    ('INSIGHT-F', 'parsimonious_INSIGHT_F'),
    ('MIMICIV-hosp', 'parsimonious_MIMICIV_hosp'),
    ('EHRSHOT', 'parsimonious_StanfordEHRSHOT'),
])

if args.succinct:
    DATASETS = OrderedDict([(k, v) for k, v in ALL_DATASETS.items() if k != 'DFCI-CAIA'])
    print("Running in succinct mode: DFCI-CAIA excluded from visualizations")
else:
    DATASETS = ALL_DATASETS

# File name patterns
DATASET_FILE_PATTERNS = {
    'DFCI-CAIA': 'dataset_summary_DFCI_CAIA_all_060526_suppressed.csv',
    'MSK-CAIA': 'dataset_summary_MSK_CAIA.csv',
    'MSK-CAIA\nprospective': 'dataset_summary_MSK_CAIA_timeshift.csv',
    'FHCC-CAIA': 'dataset_summary_FHCC_CAIA_all_082726.csv',
    'JHU-CAIA': 'dataset_summary_JHU_CAIA_042926.csv',
    'MIMICIV-hosp': 'dataset_summary_MIMICIV_hosp.csv',
    'EHRSHOT': 'dataset_summary_StanfordEHRSHOT.csv',
    'INSIGHT-A': 'dataset_summary_insight.csv',
    'INSIGHT-B': 'dataset_summary_insight.csv',
    'INSIGHT-C': 'dataset_summary_insight.csv',
    'INSIGHT-D': 'dataset_summary_insight.csv',
    'INSIGHT-E': 'dataset_summary_insight.csv',
    'INSIGHT-F': 'dataset_summary_insight.csv',
}

# per-cohort colours
COLORS = {
    'MSK-CAIA': '#4A90A4',       # teal
    'MSK-CAIA\nprospective': '#5FA8BD',  # lighter teal
    'DFCI-CAIA': '#E8866F',      # coral
    'FHCC-CAIA': '#B87FB8',      # mauve/purple-pink
    'JHU-CAIA': '#9B8BC4',       # lavender
    'MIMICIV-hosp': '#8172B2',   # purple
    'EHRSHOT': '#D4A76A',        # amber
    'INSIGHT-A': '#5B9BD5',      # blue
    'INSIGHT-B': '#70AADE',      # light blue
    'INSIGHT-C': '#85B9E7',      # lighter blue
    'INSIGHT-D': '#4682B4',      # steel blue
    'INSIGHT-E': '#3A6FA1',      # darker blue
    'INSIGHT-F': '#2C5C8E',      # darkest blue
}

# panel B: lighter = labs, darker = vitals
COLORS_LAB = {
    'MSK-CAIA': '#78B3C4',       # light teal
    'MSK-CAIA\nprospective': '#8EC5D6',  # lighter teal variant
    'DFCI-CAIA': '#F5A89A',      # light coral
    'FHCC-CAIA': '#D4A8D4',      # light mauve
    'JHU-CAIA': '#BEB0DC',       # light lavender
    'MIMICIV-hosp': '#A89CCE',   # light purple
    'EHRSHOT': '#E4C89A',        # light amber
    'INSIGHT-A': '#8DC3EA',      # light blue
    'INSIGHT-B': '#A1CEEE',      # lighter blue
    'INSIGHT-C': '#B5D9F2',      # lightest blue
    'INSIGHT-D': '#7AA8D0',      # light steel blue
    'INSIGHT-E': '#6895BD',      # lighter dark blue
    'INSIGHT-F': '#5682AA',      # light darkest blue
}

COLORS_VITAL = {
    'MSK-CAIA': '#2D6B7A',       # dark teal
    'MSK-CAIA\nprospective': '#407A8C',  # darker teal variant
    'DFCI-CAIA': '#C6654F',      # dark coral
    'FHCC-CAIA': '#8E5A8E',      # dark mauve
    'JHU-CAIA': '#6E5FA0',       # dark lavender
    'MIMICIV-hosp': '#5B4D8A',   # dark purple
    'EHRSHOT': '#B38A4A',        # dark amber
    'INSIGHT-A': '#3875A8',      # dark blue
    'INSIGHT-B': '#4682B4',      # darker blue
    'INSIGHT-C': '#5B9BD5',      # even darker blue
    'INSIGHT-D': '#2C5C8E',      # darkest steel blue
    'INSIGHT-E': '#1F4D7B',      # even darker blue
    'INSIGHT-F': '#123E68',      # darkest blue
}

# --grouped: muted colours per cohort type; panel B uses a lighter
# (labs) and darker (vitals) shade of the group colour
GROUP_COLORS = {
    'MSK': '#3E9A97',                                    # muted teal
    'MSK prospective': '#C27C5D',                        # muted terracotta
    'Tertiary cancer centers': '#D4B651',                # muted mustard yellow
    'Comprehensive hospitals (INSIGHT)': '#5B84B1',      # muted blue
    'Comprehensive hospitals (public)': '#A7BED8',       # lighter blue, same hue
}
COHORT_GROUP = {
    'MSK-CAIA': 'MSK',
    'MSK-CAIA\nprospective': 'MSK prospective',
    'DFCI-CAIA': 'Tertiary cancer centers',
    'FHCC-CAIA': 'Tertiary cancer centers',
    'JHU-CAIA': 'Tertiary cancer centers',
    **{f'INSIGHT-{c}': 'Comprehensive hospitals (INSIGHT)' for c in 'ABCDEF'},
    'MIMICIV-hosp': 'Comprehensive hospitals (public)',
    'EHRSHOT': 'Comprehensive hospitals (public)',
}
PUBLIC_DATASETS = {'MIMICIV-hosp', 'EHRSHOT'}
PUBLIC_LABEL_COLOR = '#8C8C8C'


def _shade(hex_color, f):
    # f > 0 mixes toward white, f < 0 toward black
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    t = (1.0, 1.0, 1.0) if f > 0 else (0.0, 0.0, 0.0)
    f = abs(f)
    return mc.to_hex(tuple(c + (tc - c) * f for c, tc in zip((r, g, b), t)))


if args.grouped:
    COLORS = {ds: GROUP_COLORS[g] for ds, g in COHORT_GROUP.items()}
    COLORS_LAB = {ds: _shade(c, 0.35) for ds, c in COLORS.items()}
    COLORS_VITAL = {ds: _shade(c, -0.25) for ds, c in COLORS.items()}

OUTPUT_DIR = os.environ.get('SPARC_COHORT_SUMMARIES_DIR',
    '/path/to/cohort_summaries')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading dataset summaries...")
data = {}
for ds_name, ds_dir in DATASETS.items():
    csv_path = os.path.join(BASE_DIR, ds_dir, DATASET_FILE_PATTERNS[ds_name])
    df = pd.read_csv(csv_path)
    data[ds_name] = dict(zip(df['metric'], df['value']))
    print(f"  Loaded {ds_name}: {int(data[ds_name]['n_patients']):,} patients")

# Figure: 1 x 3 panels
print("\nCreating composite figure...")

dataset_names = list(DATASETS.keys())
y_pos = np.arange(len(dataset_names))

TICK_FS = 10

# height scales with the number of cohorts
num_datasets = len(dataset_names)
fig_width = 12.5 * 1.3  # 16.25
fig_height = max(3.5, num_datasets * 0.5 + 1.5) * 0.9
fig = plt.figure(figsize=(fig_width, fig_height))
gs = GridSpec(1, 3, figure=fig, wspace=0.08, width_ratios=[1.5, 1.8, 1.2])

# Panel A: patient counts, linear scale (a log axis makes a 546-patient cohort
# look comparable to a 224,047-patient one)
ax1 = fig.add_subplot(gs[0, 0])
patient_counts = [int(data[ds]['n_patients']) for ds in dataset_names]
bars1 = ax1.barh(y_pos, patient_counts, color=[COLORS[ds] for ds in dataset_names],
                 edgecolor='black', linewidth=1.5)

# label offset is a fraction of the axis range so small bars stay readable
label_offset = max(patient_counts) * 0.012
for i, (bar, count) in enumerate(zip(bars1, patient_counts)):
    width = bar.get_width()
    ax1.text(width + label_offset, bar.get_y() + bar.get_height() / 2,
             f'{count:,}',
             ha='left', va='center', fontsize=10, fontweight='bold')

ax1.set_xlabel('Number of Patients', fontsize=10, fontweight='bold')
ax1.set_title('A. Patient Counts', fontsize=11, fontweight='bold', pad=8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(dataset_names, fontsize=TICK_FS, fontweight='bold')
if args.grouped:
    for _tl in ax1.get_yticklabels():
        if _tl.get_text() in PUBLIC_DATASETS:
            _tl.set_color(PUBLIC_LABEL_COLOR)
    # cohort-type legend in the empty lower right of panel A
    from matplotlib.patches import Rectangle
    ax1.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black', label=g)
                        for g, c in GROUP_COLORS.items() if any(COHORT_GROUP[d] == g for d in dataset_names)]
               + [Rectangle((0, 0), 1, 1, color='none', label='grey label = public dataset')],
               fontsize=8.5, frameon=True, loc='lower right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5, which='major')  # Only major grid
# headroom on the right for the value labels
ax1.set_xlim(0, max(patient_counts) * 1.18)
ax1.invert_yaxis()  # Highest on top

# Panel B: lab and vital measurements, log scale
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
lab_counts = [int(data[ds]['lab_measurements']) for ds in dataset_names]
vital_counts = [int(data[ds]['vitals_measurements']) for ds in dataset_names]

lab_counts_m = [count / 1e6 for count in lab_counts]
vital_counts_m = [count / 1e6 for count in vital_counts]

bar_height = 0.35
y_pos_b = np.arange(len(dataset_names))

lab_colors = [COLORS_LAB[ds] for ds in dataset_names]
vital_colors = [COLORS_VITAL[ds] for ds in dataset_names]

bars2a = ax2.barh(y_pos_b - bar_height/2, lab_counts_m, bar_height,
                  color=lab_colors, edgecolor='black', linewidth=1)
bars2b = ax2.barh(y_pos_b + bar_height/2, vital_counts_m, bar_height,
                  color=vital_colors, edgecolor='black', linewidth=1)

for i, (bar, count) in enumerate(zip(bars2a, lab_counts_m)):
    if count > 0:
        ax2.text(count * 1.18, bar.get_y() + bar.get_height() / 2,
                 f'{count:.1f}M',
                 ha='left', va='center', fontsize=9, fontweight='bold')

for i, (bar, count) in enumerate(zip(bars2b, vital_counts_m)):
    if count > 0:
        ax2.text(count * 1.18, bar.get_y() + bar.get_height() / 2,
                 f'{count:.1f}M',
                 ha='left', va='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Measurements (millions, log scale)', fontsize=10, fontweight='bold')
ax2.set_title('B. Lab and Vital Measurements', fontsize=11, fontweight='bold', pad=8)

from matplotlib.patches import Rectangle
legend_elements = [
    Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label='Lab measurements (lighter)'),
    Rectangle((0, 0), 1, 1, facecolor='darkgray', edgecolor='black', label='Vital measurements (darker)')
]
ax2.legend(handles=legend_elements, fontsize=8.5, frameon=True, loc='upper center',
           bbox_to_anchor=(0.5, -0.28), ncol=2)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5, which='major')  # Only major grid
ax2.set_xscale('log')
ax2.tick_params(axis='x', which='minor', length=0)  # Hide minor ticks
min_val = min([v for v in lab_counts_m + vital_counts_m if v > 0])
max_val = max(lab_counts_m + vital_counts_m)
# headroom for the value labels
ax2.set_xlim(min_val * 0.3, max_val * 7)
plt.setp(ax2.get_yticklabels(), visible=False)

# Panel C: median visits per patient with IQR
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
median_visits = [int(data[ds]['n_dates_per_pt_median']) for ds in dataset_names]
p25_visits = [int(data[ds]['n_dates_per_pt_p25']) for ds in dataset_names]
p75_visits = [int(data[ds]['n_dates_per_pt_p75']) for ds in dataset_names]

xerr_lower = [med - p25 for med, p25 in zip(median_visits, p25_visits)]
xerr_upper = [p75 - med for med, p75 in zip(median_visits, p75_visits)]

bars3 = ax3.barh(y_pos, median_visits, color=[COLORS[ds] for ds in dataset_names],
                 edgecolor='black', linewidth=1.5)
ax3.errorbar(median_visits, y_pos,
             xerr=[xerr_lower, xerr_upper],
             fmt='none', ecolor='black', elinewidth=1.5, capsize=4, capthick=1.5,
             label='IQR (25th\u201375th pct.)')

# value labels sit past the IQR whisker
for i, (bar, med) in enumerate(zip(bars3, median_visits)):
    label_x = p75_visits[i] + max(median_visits) * 0.04
    ax3.text(label_x, bar.get_y() + bar.get_height() / 2,
             f'{med}',
             ha='left', va='center', fontsize=10, fontweight='bold')

ax3.set_xlabel('Median Visits per Patient (with IQR)', fontsize=10, fontweight='bold')
ax3.set_title('C. Total Sampling Frequency', fontsize=11, fontweight='bold', pad=8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
# headroom for the widest whisker and label
ax3.set_xlim(0, max(p75_visits) * 1.25)
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.legend(fontsize=9, frameon=True, loc='lower right')

# Save figure
# fixed margins rather than bbox_inches='tight': Illustrator rejects PDFs whose page box
# comes from matplotlib's post-hoc crop ("transformation would make objects too large")
if num_datasets > 5:
    top_margin = 0.92
    bottom_margin = 0.15
else:
    top_margin = 0.88
    bottom_margin = 0.30
fig.subplots_adjust(left=0.13, right=0.98, top=top_margin, bottom=bottom_margin)

suffix = ('_succinct' if args.succinct else '') + ('_grouped' if args.grouped else '')

output_path_svg = os.path.join(OUTPUT_DIR, f'dataset_summaries_combined_plot{suffix}.svg')
plt.savefig(output_path_svg, format='svg', transparent=False)
print(f"\nSaved combined plot to: {output_path_svg}")

output_path = os.path.join(OUTPUT_DIR, f'dataset_summaries_combined_plot{suffix}.pdf')
plt.savefig(output_path, dpi=150, transparent=False)
print(f"Saved combined plot to: {output_path}")

output_path_png = os.path.join(OUTPUT_DIR, f'dataset_summaries_combined_plot{suffix}.png')
plt.savefig(output_path_png, dpi=300)
print(f"Saved combined plot to: {output_path_png}")

plt.close()

# Print summary statistics
print()
print("DATASET SUMMARY TABLE")
print()
print(f"{'Dataset':<20} {'Patients':>12} {'Lab Meas.':>14} {'Vital Meas.':>14} {'Med Visits':>12}")
print("-"*70)
for ds in dataset_names:
    print(f"{ds:<20} {int(data[ds]['n_patients']):>12,} "
          f"{int(data[ds]['lab_measurements']):>14,} "
          f"{int(data[ds]['vitals_measurements']):>14,} "
          f"{int(data[ds]['n_dates_per_pt_median']):>12}")
print()

print("\nDone!")