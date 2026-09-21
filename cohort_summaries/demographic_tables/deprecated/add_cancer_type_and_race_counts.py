#!/usr/bin/env python3
"""
add_cancer_type_and_race_counts.py

Adds per-cancer-type patient counts AND harmonized race/ethnicity counts to
the Table S1 dataset demographics CSV, in the same dataset row order as the
original file.

Race/ethnicity columns
----------------------
Sourced from `race_ethnicity/harmonized/<STEM>_{race,ethnicity}_harmonized.csv`
— the 6 harmonized race categories and 3 ethnicity categories produced by
`data/MIMICIV_preprocessing/11_create_harmonized_race_ethnicity.py`.

A "Race/ethnicity denominator" column is written FIRST, before the category
columns, and it is not decorative: for some datasets the race/ethnicity
extract covers a wider population than the modelling cohort in the row's
"# Patients" column. MSK, MSK (prospective), MIMICIV-hosp and INSIGHT-A..F
are cohort-restricted and agree with "# Patients"; JHU-CAIA and FHCC-CAIA
are still aggregated over their full site extract and run ~24% high. Making
the denominator explicit keeps that visible instead of implying every column
in a row shares one denominator.

Counts are copied verbatim from the harmonized CSVs, so a suppressed cell
stays as its 'n<=10' label rather than being silently coerced to 0. That
means a row's race columns will not always sum exactly to the denominator.

DFCI-CAIA has no race/ethnicity extract and gets blank cells, the same
convention used for missing cancer-type data below.

Cancer-type columns
-------------------

For each dataset row, adds a "Total patients with >=1 mapped cancer type"
column FIRST, then one column per named cancer type (29 of the 30
eligible parent cancer types from
src/sparc_preprocessing/data/all_CAIA_cancer_types.txt, EXCLUDING "Unknown
Primary" -- see below), followed by one final "Other/Unknown Primary"
column.

IMPORTANT -- the 29+1 type columns do NOT sum to the "Total patients"
column: a patient with >1 parent cancer type tied for their earliest
diagnosis day is counted once under EACH tied type (the same convention
used throughout cancer_type_vis.py and friends -- see those scripts'
docstrings). In this data that inflates the row sum by roughly 14-23%
over the true unique-patient total depending on site, so the "Total
patients" column is the actual per-site denominator; the type columns are
a per-diagnosis-type breakdown, not a partition of patients.

Column source: each dataset's `ctypes_<STEM>_summary.txt` first-mapped-type
table (patient counts, not %) -- the same table used by cancer_type_vis.py
and friends. Counts here can sum to more than a dataset's total mapped
patients if a patient had >1 type tie for first appearance (first-mapped
tables count such patients once under each tied type); see those scripts'
docstrings for the full explanation of that convention.

"Other/Unknown Primary" = the "Unknown Primary" named-type count PLUS any
"others" catch-all count already present in that site's first-mapped
table (a residual bucket some sites report for condition codes present in
the data but not covered by the named parent-type lookup). Both measure
diagnostic/coding completeness rather than a real disease category, which
is why they're combined into one residual column instead of listed as
named types (see cancer_type_foldchange_vis.py's docstring for the same
reasoning applied there).

A dataset row with no matching ctypes_<STEM>_summary.txt file (JHU-CAIA,
MSK (prospective) -- no separate cancer-type run captured yet) gets blank
cells for every added column rather than zeros, so "no data available" is
never confused with "zero patients of this type".
"""

import csv
import os
import re

# DATA_DIR (ctypes_<SITE>_summary.txt inputs and the supplementary CSVs) and
# CANCER_TYPES_LIST_PATH are resolved in ctypes_parse so every cancer-type
# script agrees on one location. Both honour SPARC_CANCER_TYPES_DIR /
# SPARC_CANCER_TYPES_OUT.
from ctypes_parse import DATA_DIR, OUTPUT_DIR, CANCER_TYPES_LIST_PATH

INPUT_CSV = os.path.join(
    DATA_DIR, 'SPARC_Supplementary_Tables - Table S1 Dataset Demographics.csv'
)
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    'SPARC_Supplementary_Tables - Table S1 Dataset Demographics'
    '_with_ctypes_and_race.csv',
)
TYPES_LIST_PATH = CANCER_TYPES_LIST_PATH

# Harmonized race/ethnicity CSVs live beside the other cohort_summaries
# outputs, not in the cancer-types folder. Override with
# SPARC_RACE_ETHNICITY_HARMONIZED_DIR.
RACE_ETHNICITY_DIR = os.environ.get(
    'SPARC_RACE_ETHNICITY_HARMONIZED_DIR',
    os.path.join(
        os.path.dirname(os.path.normpath(DATA_DIR)),
        'race_ethnicity', 'harmonized',
    ),
)

# The 6 / 3 harmonized categories, in the order they are written out.
RACE_CATEGORIES = [
    'White',
    'Black or African American',
    'Asian',
    'American Indian or Alaska Native',
    'Native Hawaiian or Other Pacific Islander',
    'Other/Unknown/Refused',
]
ETHNICITY_CATEGORIES = [
    'Hispanic or Latino',
    'Not Hispanic or Latino',
    'Unknown',
]

# Dataset name (as in the "Dataset" column) -> harmonized file stem.
# DFCI-CAIA is absent: no race/ethnicity extract exists for it.
DATASET_TO_RACE_STEM = {
    'MSK': 'MSK_CAIA',
    'MSK (prospective)': 'MSK_CAIA_timeshift',
    'FHCC-CAIA': 'FHCC_CAIA',
    'JHU-CAIA': 'JHU_CAIA',
    'INSIGHT-A': 'insight_A',
    'INSIGHT-B': 'insight_B',
    'INSIGHT-C': 'insight_C',
    'INSIGHT-D': 'insight_D',
    'INSIGHT-E': 'insight_E',
    'INSIGHT-F': 'insight_F',
    'MIMICIV-hosp': 'MIMICIV_hosp',
    'EHRSHOT': 'EHRSHOT',
}

# Dataset name (exactly as it appears in the "Dataset" column of the input
# CSV) -> ctypes_<STEM>_summary.txt file stem. Datasets with no entry here
# get blank cells for every added column.
DATASET_TO_CTYPES_STEM = {
    'MSK': 'MSK',
    'DFCI-CAIA': 'DFCI',
    'FHCC-CAIA': 'FH',
    'INSIGHT-A': 'INSIGHT-A',
    'INSIGHT-B': 'INSIGHT-B',
    'INSIGHT-C': 'INSIGHT-C',
    'INSIGHT-D': 'INSIGHT-D',
    'INSIGHT-E': 'INSIGHT-E',
    'INSIGHT-F': 'INSIGHT-F',
    'MIMICIV-hosp': 'MIMICIV-hosp',
    'EHRSHOT': 'EHRSHOT',
    # 'MSK (prospective)' and 'JHU-CAIA' intentionally omitted: no
    # ctypes_<STEM>_summary.txt exists for either yet.
}

FIRST_MAPPED_RE = (
    r'CANCER TYPE SUMMARY \(first-mapped type\).*?\n'
    r'parent_cancer_type.*?\n-+\n(.*?)(?:\n=|\Z)'
)
ROW_RE = re.compile(r'^(.*?)\s+([\d,]+)\s+([\d.]+)%$')


TOTAL_PATIENTS_RE = re.compile(r'Total patients with >=1 mapped cancer type:\s*([\d,]+)')

# Age *subgroup* columns ("Age <=4", "Age 5-9", "Age>=85") — matches a digit or
# comparator right after "Age", so it never catches "Age at First Lab (mean ±
# std)". Used to find where the age bins start so race/ethnicity can go before
# them.
AGE_BIN_RE = re.compile(r'^Age\s*(?:<=|>=|\d)')


def load_type_counts_and_total(stem):
    """Returns ({type_name: count}, total_mapped_patients) from
    ctypes_<stem>_summary.txt's first-mapped-type table, or (None, None) if
    the file doesn't exist or the total line can't be found."""
    path = os.path.join(DATA_DIR, f'ctypes_{stem}_summary.txt')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        text = f.read()
    compact = '\n'.join(l for l in text.split('\n') if l.strip() != '')

    total_m = TOTAL_PATIENTS_RE.search(compact)
    total = int(total_m.group(1).replace(',', '')) if total_m else None

    m = re.search(FIRST_MAPPED_RE, compact, re.S)
    if not m:
        return None, total
    out = {}
    for line in m.group(1).split('\n'):
        line = line.strip()
        if not line or line.startswith('='):
            break
        mm = ROW_RE.match(line)
        if mm:
            label, count, _pct = mm.groups()
            out[label.strip()] = int(count.replace(',', ''))
    return out, total


def _read_harmonized(path):
    """Returns {category: cell_as_written} from a harmonized CSV, or None.

    Cells are kept as strings so a suppressed 'n<=10' survives into the
    output instead of being coerced to 0.
    """
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                out[row[0].strip()] = row[1].strip()
    return out


def load_race_ethnicity(stem):
    """Returns (race_cells, ethnicity_cells, denominator) for a dataset.

    The denominator is the ethnicity total, which is the more reliable of the
    two: ethnicity has 3 categories and rarely suppresses a cell, whereas the
    race tables routinely do, which would make a race-derived total read low.
    Returns (None, None, None) when no extract exists for this dataset.
    """
    if not stem:
        return None, None, None
    race = _read_harmonized(
        os.path.join(RACE_ETHNICITY_DIR, f'{stem}_race_harmonized.csv')
    )
    eth = _read_harmonized(
        os.path.join(RACE_ETHNICITY_DIR, f'{stem}_ethnicity_harmonized.csv')
    )
    if race is None and eth is None:
        return None, None, None
    denom = None
    if eth:
        denom = sum(int(v) for v in eth.values() if v.isdigit())
    return race or {}, eth or {}, denom


def main():
    with open(TYPES_LIST_PATH) as f:
        all_30_types = [line.strip() for line in f if line.strip()]
    named_types = [t for t in all_30_types if t != 'Unknown Primary']
    assert len(named_types) == 29, f"expected 29 named types, got {len(named_types)}"

    with open(INPUT_CSV, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        input_rows = list(reader)

    race_header = (['Race/ethnicity denominator']
                   + [f'{c} (race)' for c in RACE_CATEGORIES]
                   + [f'{c} (ethnicity)' for c in ETHNICITY_CATEGORIES])
    n_race_cols = len(race_header)

    # Race/ethnicity is inserted directly after the summary demographics
    # (Dataset / # Patients / Male % / Age mean±std) and BEFORE the age
    # subgroup bins, so the demographic columns read together. Cancer-type
    # columns stay at the end.
    age_start = next(
        (i for i, c in enumerate(header) if AGE_BIN_RE.match(c.strip())),
        len(header),
    )

    new_header = (header[:age_start] + race_header + header[age_start:]
                  + ['Total patients with >=1 mapped cancer type']
                  + named_types + ['Other/Unknown Primary'])
    new_rows = []
    missing_datasets = []
    missing_race = []
    denominator_mismatches = []

    for row in input_rows:
        dataset_name = row[0]
        stem = DATASET_TO_CTYPES_STEM.get(dataset_name)
        counts, total = load_type_counts_and_total(stem) if stem else (None, None)

        if counts is None:
            missing_datasets.append(dataset_name)
            ctype_cells = [''] * (len(named_types) + 2)
        else:
            type_cells = [str(counts.get(t, 0)) for t in named_types]
            other_unknown = counts.get('Unknown Primary', 0) + counts.get('others', 0)
            total_cell = str(total) if total is not None else ''
            ctype_cells = [total_cell] + type_cells + [str(other_unknown)]

        # ---- harmonized race / ethnicity ----
        race, eth, denom = load_race_ethnicity(
            DATASET_TO_RACE_STEM.get(dataset_name)
        )
        if race is None:
            missing_race.append(dataset_name)
            race_cells = [''] * n_race_cols
        else:
            race_cells = [str(denom) if denom is not None else '']
            race_cells += [race.get(c, '0') for c in RACE_CATEGORIES]
            race_cells += [eth.get(c, '0') for c in ETHNICITY_CATEGORIES]
            # Flag rows whose race/ethnicity population is not the cohort in
            # the "# Patients" column — see this module's docstring.
            try:
                n_patients = int(str(row[1]).replace(',', ''))
                if denom is not None and abs(denom - n_patients) > 0.01 * n_patients:
                    denominator_mismatches.append(
                        (dataset_name, n_patients, denom)
                    )
            except (ValueError, IndexError):
                pass

        new_row = (row[:age_start] + race_cells + row[age_start:] + ctype_cells)
        new_rows.append(new_row)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Added 1 total-patients column + {len(named_types)} named-type columns "
          "+ 1 Other/Unknown Primary column.")
    print(f"Added {n_race_cols} race/ethnicity columns "
          f"({len(RACE_CATEGORIES)} race + {len(ETHNICITY_CATEGORIES)} ethnicity "
          "+ 1 denominator).")
    if missing_datasets:
        print(f"\nNo ctypes_<STEM>_summary.txt found for: {', '.join(missing_datasets)} "
              "-- left blank for all added columns.")
    if missing_race:
        print(f"\nNo harmonized race/ethnicity found for: {', '.join(missing_race)} "
              "-- left blank for all race/ethnicity columns.")
    if denominator_mismatches:
        print("\nWARNING: race/ethnicity population differs from '# Patients' "
              "by >1% for:")
        for name, n_patients, denom in denominator_mismatches:
            print(f"  {name}: # Patients {n_patients:,} vs race/ethnicity "
                  f"{denom:,} ({denom / n_patients:.3f}x)")
        print("  Those rows' race/ethnicity columns describe a wider population "
              "than the rest of the row.")


if __name__ == '__main__':
    main()
