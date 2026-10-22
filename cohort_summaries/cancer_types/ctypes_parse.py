#!/usr/bin/env python3
"""Parse ctypes_<SITE>_summary.txt cancer-type reports.

Shared by cancer_type_vis.py and cancer_type_diversity_vis.py, along with the site list and palette.
"""

import os
import re
from collections import OrderedDict

# input and output dirs; override with environment variables:
#     export SPARC_CANCER_TYPES_DIR=/path/to/cohort_summaries/cancer_types
#     export SPARC_CANCER_TYPES_OUT=/path/to/write/figures
DEFAULT_CANCER_TYPES_DIR = '/path/to/cohort_summaries/cancer_types'

DATA_DIR = os.environ.get('SPARC_CANCER_TYPES_DIR', DEFAULT_CANCER_TYPES_DIR)
OUTPUT_DIR = os.environ.get('SPARC_CANCER_TYPES_OUT', DATA_DIR)

# reference list of eligible cancer types
CANCER_TYPES_LIST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'all_CAIA_cancer_types.txt'
)

# Site display name -> ctypes_<KEY>_summary.txt file stem, in plotting order
# (tertiary centers first, then non-tertiary INSIGHT cohorts).
SITES = OrderedDict([
    ('MSK-CAIA', 'MSK'),
    ('DFCI-CAIA', 'DFCI'),
    ('FHCC-CAIA', 'FH'),
    ('JHU-CAIA', 'JHU'),
    ('INSIGHT-A', 'INSIGHT-A'),
    ('INSIGHT-B', 'INSIGHT-B'),
    ('INSIGHT-C', 'INSIGHT-C'),
    ('INSIGHT-D', 'INSIGHT-D'),
    ('INSIGHT-E', 'INSIGHT-E'),
    ('INSIGHT-F', 'INSIGHT-F'),
])

TERTIARY = {'MSK-CAIA', 'DFCI-CAIA', 'FHCC-CAIA', 'JHU-CAIA'}

# colors as in dataset_summaries_combined_plot.py: warm hues for tertiary
# centers, blue gradient for INSIGHT cohorts
COLORS = {
    'MSK-CAIA': '#4A90A4',       # teal
    'DFCI-CAIA': '#E8866F',      # coral
    'FHCC-CAIA': '#B87FB8',      # mauve/purple-pink
    'JHU-CAIA': '#9B8BC4',       # lavender
    'INSIGHT-A': '#5B9BD5',      # blue
    'INSIGHT-B': '#70AADE',      # light blue
    'INSIGHT-C': '#85B9E7',      # lighter blue
    'INSIGHT-D': '#4682B4',      # steel blue
    'INSIGHT-E': '#3A6FA1',      # darker blue
    'INSIGHT-F': '#2C5C8E',      # darkest blue
}

# parsing
FIRST_MAPPED_RE = (
    r'CANCER TYPE SUMMARY \(first-mapped type\).*?\n'
    r'parent_cancer_type.*?\n-+\n(.*?)(?:\n=|\Z)'
)
COMBINED_RE = (
    r'CANCER TYPE SUMMARY \(combined labels.*?\n'
    r'(?:label.*?\n)?-+\n(.*?)(?:\n=|\Z)'
)
ROW_RE = re.compile(r'^(.*?)\s+([\d,]+)\s+([\d.]+)%$')


def _parse_table(compact_text, header_regex):
    m = re.search(header_regex, compact_text, re.S)
    if not m:
        return None
    out = {}
    for line in m.group(1).split('\n'):
        line = line.strip()
        if not line or line.startswith('='):
            break
        mm = ROW_RE.match(line)
        if mm:
            label, count, pct = mm.groups()
            out[label.strip()] = (int(count.replace(',', '')), float(pct))
    return out


def load_site_cancer_types(file_stem):
    """Return ({parent_cancer_type: (count, pct)}, source_label), or (None, None) if the file is missing.

    Uses the first-mapped table; otherwise falls back to single-type rows of the combined table (undercounts multi-type patients).
    """
    path = os.path.join(DATA_DIR, f'ctypes_{file_stem}_summary.txt')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        text = f.read()
    # Some files were pasted from a terminal with a blank line after every
    # content line: collapse those so both formats parse identically.
    compact = '\n'.join(l for l in text.split('\n') if l.strip() != '')

    d = _parse_table(compact, FIRST_MAPPED_RE)
    if d:
        return d, 'first-mapped'

    d = _parse_table(compact, COMBINED_RE)
    if d:
        d = {k: v for k, v in d.items() if '+' not in k}
        return d, 'combined (single-type rows only, undercounts)'

    return None, None


TOTAL_PATIENTS_RE = re.compile(r'Total patients with >=1 mapped cancer type:\s*([\d,]+)')
MULTI_TYPE_RE = re.compile(r'Patients with >1 type tied for first appearance:\s*([\d,]+)')


def load_site_multi_type_stats(file_stem):
    """Return (total_patients, multi_type_patients, pct), or None if missing.

    multi_type_patients: patients with >1 parent cancer type tied on the earliest diagnosis day.
    """
    path = os.path.join(DATA_DIR, f'ctypes_{file_stem}_summary.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    compact = '\n'.join(l for l in text.split('\n') if l.strip() != '')

    total_m = TOTAL_PATIENTS_RE.search(compact)
    multi_m = MULTI_TYPE_RE.search(compact)
    if not total_m or not multi_m:
        return None
    total = int(total_m.group(1).replace(',', ''))
    multi = int(multi_m.group(1).replace(',', ''))
    pct = (multi / total * 100) if total else 0.0
    return total, multi, pct


