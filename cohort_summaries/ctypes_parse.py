#!/usr/bin/env python3
"""
ctypes_parse.py

Shared parsing utilities for the `ctypes_<SITE>_summary.txt` plain-text
cancer-type report captures in this directory, used by both
cancer_type_vis.py (rare/complex type comparison) and
cancer_type_diversity_vis.py (case-mix diversity comparison).

Also defines the shared site list, tertiary/non-tertiary grouping, and
color palette so both scripts stay visually and structurally consistent.
"""

import os
import re
from collections import OrderedDict

# Where the ctypes_<SITE>_summary.txt inputs live, and where figures/CSVs are
# written. These are generated artefacts, so they sit in the results tree
# rather than next to the code. Override either with an environment variable:
#     export SPARC_CANCER_TYPES_DIR=/path/to/cohort_summaries/cancer_types
#     export SPARC_CANCER_TYPES_OUT=/path/to/write/figures
DEFAULT_CANCER_TYPES_DIR = (
    '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_fixed_models/results_vis'
    '/cohort_summaries/cancer_types'
)

DATA_DIR = os.environ.get('SPARC_CANCER_TYPES_DIR', DEFAULT_CANCER_TYPES_DIR)
OUTPUT_DIR = os.environ.get('SPARC_CANCER_TYPES_OUT', DATA_DIR)

# Reference list of eligible cancer types, kept with the code (it is an input,
# not an output).
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

# Colors - following the convention in dataset_summaries_combined_plot.py
# (consistent, colorblind-friendly palette; tertiary centers get warm/varied
# hues, INSIGHT cohorts get a blue gradient).
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

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
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
    """
    Returns (type_pct_dict, source_label) for a given ctypes_<stem>_summary.txt
    file, or (None, None) if the file doesn't exist.

    `type_pct_dict` maps parent_cancer_type -> (count, pct_of_cohort).

    Prefers the "first-mapped type" table (clean per-type counts, single-type
    rows so no co-occurrence double counting). Falls back to the "combined
    labels" table's single-type-only rows (excluding any "A + B" co-occurrence
    rows) when the first-mapped table isn't present in the file — this
    slightly *undercounts* multi-type patients but is the best available
    signal from a combined-only capture.
    """
    path = os.path.join(DATA_DIR, f'ctypes_{file_stem}_summary.txt')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        text = f.read()
    # Some files were pasted from a terminal with a blank line after every
    # content line; collapse those so both formats parse identically.
    compact = '\n'.join(l for l in text.split('\n') if l.strip() != '')

    d = _parse_table(compact, FIRST_MAPPED_RE)
    if d:
        return d, 'first-mapped'

    d = _parse_table(compact, COMBINED_RE)
    if d:
        d = {k: v for k, v in d.items() if '+' not in k}
        return d, 'combined (single-type rows only, undercounts)'

    return None, None


def load_all_sites(verbose=True):
    """
    Loads every site in SITES, returns (site_data, site_source) dicts keyed
    by display name, skipping (with a printed note if verbose) any site
    whose file doesn't exist yet (e.g. JHU-CAIA before its run completes).
    """
    site_data = OrderedDict()
    site_source = {}
    for display_name, stem in SITES.items():
        d, src = load_site_cancer_types(stem)
        if d is None:
            if verbose:
                print(f"  Skipping {display_name}: ctypes_{stem}_summary.txt not found")
            continue
        site_data[display_name] = d
        site_source[display_name] = src
        if verbose:
            flag = '' if src == 'first-mapped' else '  (** see caveat)'
            print(f"  Loaded {display_name}: {len(d)} types [{src}]{flag}")
    return site_data, site_source


TOTAL_PATIENTS_RE = re.compile(r'Total patients with >=1 mapped cancer type:\s*([\d,]+)')
MULTI_TYPE_RE = re.compile(r'Patients with >1 type tied for first appearance:\s*([\d,]+)')


def load_site_multi_type_stats(file_stem):
    """
    Returns (total_patients, multi_type_patients, pct) for a given
    ctypes_<stem>_summary.txt file, or None if the file doesn't exist or
    either figure can't be found in it.

    `multi_type_patients` is the count of patients whose earliest mapped
    diagnosis day had >1 parent cancer type tied for first appearance (so
    they're counted once under each tied type in the first-mapped-type
    table rather than being a single unambiguous first diagnosis).
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


def load_all_sites_multi_type_stats(verbose=True):
    """
    Loads multi-type-tie stats for every site in SITES, returns
    {display_name: (total, multi, pct)}, skipping any site whose file
    doesn't exist or doesn't contain both figures.
    """
    out = OrderedDict()
    for display_name, stem in SITES.items():
        stats = load_site_multi_type_stats(stem)
        if stats is None:
            if verbose:
                print(f"  Skipping {display_name} (multi-type stats): not found")
            continue
        out[display_name] = stats
    return out
