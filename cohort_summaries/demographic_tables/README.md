# demographic_tables

The demographics-table pipeline, in run order. Each stage writes what the next
one reads, so they can be run independently once their inputs exist.

| # | script | reads | writes |
|---|---|---|---|
| 1 | `dem_table_1_dataset_summaries.py` | SPARC data volume (per dataset) | `cohort_summaries/dataset_summaries/<dataset>/dataset_summary_*.csv`, `age_distribution_*.csv`, PDFs |
| 2 | `dem_table_2_race_ethnicity_cohort.py` | patient-level `{id: (race, ethnicity)}` pickle + the modelling cohort | `cohort_summaries/race_ethnicity/<SITE>/<stem>_{race,ethnicity}.csv` |
| 3 | `dem_table_3_harmonize_race_ethnicity.py` | `race_ethnicity/<SITE>/*` (raw site labels) | `race_ethnicity/harmonized/<stem>_{race,ethnicity}_harmonized.csv` |
| 4 | `dem_table_4_build_table.py` | stages 1 and 3, plus `cancer_types/ctypes_<SITE>_summary.txt` | `demographics_table.csv` and `cancer_types/...Table S1..._with_ctypes_and_race.csv` |

Stage 4 replaces what used to be `demographics_table.py` + a manual
rename/reorder + `add_cancer_type_and_race_counts.py`. Both originals are in
`deprecated/` pending review; stage 4 reproduces their output exactly (the one
intended difference is `MSK (prospective)` Male% now reading `46.00` rather
than a hand-typed `46`).

## Typical run

```bash
# only stages 3 and 4 are needed to refresh the table after a race extract changes
python3 dem_table_3_harmonize_race_ethnicity.py
python3 dem_table_4_build_table.py            # prints a consistency report

# stage 2 only applies to a site that supplies a patient-level race pickle
python3 dem_table_2_race_ethnicity_cohort.py -ds MSK_CAIA -cp prior
```

Every stage takes explicit `-*` path flags; defaults point at
`lilac_transfer_091126/SPARC_fixed_models/results_vis/cohort_summaries`.

## Known denominator discrepancies

Stage 4 ends with a consistency report flagging any row where a denominator
exceeds that dataset's patient count. The root cause is always the same — an
aggregate computed over a wider population than the modelling cohort, which
drops patients whose labs are empty after `caia_portion` truncation.

* **race/ethnicity denominator** — fixed everywhere as of 2026-09-17. MSK and
  MSK (prospective) were fixed by stage 2, which needs a patient-level pickle.
  JHU-CAIA and FHCC-CAIA had no such pickle and instead got re-extracted site
  CSVs; running stage 3 propagated them (JHU `219,187 -> 176,294`, FHCC
  `179,756 -> 145,883`). EHRSHOT reads 541 against 546 patients, which is
  genuine missing data, not inflation.
* **cancer-type total** — still open for MSK (`+13,401`) and FHCC-CAIA
  (`+2,436`). `ctypes_<SITE>_summary.txt` reports `len(pickle)`, and the MSK
  cancer-type pickle holds 237,448 patients against a 224,047-patient cohort.
  The fix is to intersect the pickle with the cohort the way stage 2 does for
  race; the corrected figure will be *below* 224,047, since some cohort
  patients have no cancer-type record at all.

## Conventions worth knowing before editing stage 4

The two output tables differ deliberately, and these were the reason a manual
step sat between the old scripts:

* age-bin labels — counts table uses `Age <=4` / `Age 5-9` / `Age>=85` (ASCII,
  no space before `>=`); pct table uses `Age ≤4` / `Age 5–9` / `Age ≥85` (en dash)
* absent category or age bin — counts table writes `0`, pct table writes blank;
  blank means "no extract at all" in both
* row order — pct table alphabetical, counts table publication order
* suppressed cells stay verbatim (`n<=10`), so race columns need not sum to the
  denominator. Note stage 3 coerces `n<=10` to `0` when harmonizing, so
  suppression does not survive into the harmonized CSVs.
* the 29+1 cancer-type columns do not partition patients: a patient with
  several types tied for earliest diagnosis is counted under each, inflating
  the row sum ~14-23%. The `Total patients ...` column is the real denominator.

## Dependency kept outside this folder

`dem_table_4_build_table.py` imports `ctypes_parse` from the parent
`cohort_summaries/` directory rather than vendoring it, because it is shared
with the cancer-type visualisation scripts (`cancer_type_vis.py` and friends),
which exist in the lilac copy but have not been ported into this repo yet.
