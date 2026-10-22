"""
Split the combined INSIGHT preprocessing pickles by site_id (from demographics.csv) into <output_dir>/<site>/.
With --run-preprocessing, the full pipeline is run first. Output formats: see README.md.
Usage: python run_preprocessing_split.py [--source-dir <combined>] [--output-dir <out>] [--run-preprocessing]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the package root is importable when called as a plain script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from shared.utils import load_enrollment_dates, load_first_lab_dates, build_reference_dates


# ---------------------------------------------------------------------------
# Pickle filename → (pipeline tag, description)
# ---------------------------------------------------------------------------
PIPELINE_FILES: dict[str, str] = {
    "labs":        "ssid_to_date_to_labs_dict.pickle",
    "vitals":      "ssid_to_date_to_vitals_dict.pickle",
    "oral_meds":   "ssid_to_date_to_oral_meds_dict.pickle",
    "nonoral_meds":"ssid_to_date_to_nonoral_meds_dict.pickle",
    "admissions":  "ssid_to_hospital_adm_dsch_dict.pickle",
    "stage":       "ssid_to_date_to_STAGE_dict.pickle",
    "demographics":"ssid_to_demographics_dict.pickle",
}

ALL_PIPELINES = list(PIPELINE_FILES.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ssid_to_site(demographics_csv: str) -> dict[str, str]:
    """Return {ssid: site_id} from demographics.csv."""
    df = pd.read_csv(demographics_csv, dtype={"ssid": str, "site_id": str},
                     usecols=["ssid", "site_id"])
    df = df.dropna(subset=["site_id"])
    mapping = dict(zip(df["ssid"], df["site_id"]))
    print(f"  {len(mapping):,} ssid→site_id pairs loaded from demographics.csv")
    sites = sorted(set(mapping.values()))
    counts = {s: sum(1 for v in mapping.values() if v == s) for s in sites}
    print(f"  Sites: {counts}")
    return mapping


def split_and_save(
    combined: dict,
    filename: str,
    ssid_to_site: dict[str, str],
    output_dir: str,
    sites_filter: set[str] | None,
) -> None:
    """Split a combined ssid-keyed dict by site_id and save per-site pickles."""
    per_site: dict[str, dict] = {}
    skipped = 0
    for ssid, data in combined.items():
        ssid_str = str(ssid)
        site = ssid_to_site.get(ssid_str)
        if site is None:
            skipped += 1
            continue
        if sites_filter and site not in sites_filter:
            continue
        per_site.setdefault(site, {})[ssid_str] = data

    if skipped:
        print(f"    {skipped:,} ssids skipped (not in demographics.csv)")

    for site in sorted(per_site):
        site_data = per_site[site]
        site_dir = os.path.join(output_dir, site)
        os.makedirs(site_dir, exist_ok=True)
        out_path = os.path.join(site_dir, filename)
        with open(out_path, "wb") as fh:
            pickle.dump(site_data, fh, protocol=5)
        print(f"    [{site}] {len(site_data):,} patients → {out_path}")


def split_from_source(
    source_dir: str,
    output_dir: str,
    ssid_to_site: dict[str, str],
    pipelines: list[str],
    sites_filter: set[str] | None,
) -> None:
    """Load each combined pickle from source_dir and split it by site."""
    for pipeline in pipelines:
        filename = PIPELINE_FILES[pipeline]
        src_path = os.path.join(source_dir, filename)
        print(f"\n── {pipeline} ({filename}) ────────────────────────────────")
        if not os.path.exists(src_path):
            print(f"  MISSING: {src_path} — skipping.")
            continue
        print(f"  Loading: {src_path}")
        with open(src_path, "rb") as fh:
            combined = pickle.load(fh)
        print(f"  Loaded {len(combined):,} patients")
        split_and_save(combined, filename, ssid_to_site, output_dir, sites_filter)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (mirrors run_preprocessing.py)
# ---------------------------------------------------------------------------

def run_and_split_labs(
    data_dir: str, output_dir: str, reference_dates: dict,
    min_records: int, ssid_to_site: dict, sites_filter: set | None,
) -> None:
    from labs.build_dict import stream_build_lab_dict
    filename = PIPELINE_FILES["labs"]
    print("── Labs pipeline ─────────────────────────────────────────────────")
    csv_path = os.path.join(data_dir, config.LAB_FILE)
    labs_dict = stream_build_lab_dict(
        csv_path, config.LOINC_TO_CANONICAL, reference_dates,
        canonical_to_accepted_units=config.CANONICAL_TO_ACCEPTED_UNITS,
        sentinel_values=config.LAB_RESULT_SENTINEL_VALUES,
        min_records=min_records, chunksize=config.CHUNK_SIZE,
        canonical_unit_aliases=config.CANONICAL_UNIT_ALIASES,
    )
    if not labs_dict:
        print("  WARNING: No lab tests passed the cleaning filters.")
        return
    print(f"  Combined: {len(labs_dict):,} patients")
    split_and_save(labs_dict, filename, ssid_to_site, output_dir, sites_filter)


def run_and_split_vitals(
    data_dir: str, output_dir: str, reference_dates: dict,
    min_records: int, ssid_to_site: dict, sites_filter: set | None,
) -> None:
    from vitals.load import load_vitals
    from vitals.load_obs_clin import load_obs_clin_vitals
    from vitals.clean import clean_vitals
    from vitals.build_dict import build_vital_dict
    filename = PIPELINE_FILES["vitals"]
    print("── Vitals pipeline ───────────────────────────────────────────────")
    vital_path = os.path.join(data_dir, config.VITAL_FILE)
    obs_path   = os.path.join(data_dir, config.OBS_CLIN_FILE)
    df_vital = load_vitals(vital_path, config.VITAL_COL_TO_CANONICAL)
    df_obs   = load_obs_clin_vitals(
        obs_path, config.OBS_CLIN_LOINC_TO_VITAL,
        config.OBS_CLIN_TEMP_FAHRENHEIT_LOINC,
        obs_clin_accepted_units=config.OBS_CLIN_ACCEPTED_UNITS,
        temp_celsius_unit=config.OBS_CLIN_TEMP_CELSIUS_UNIT,
        temp_fahrenheit_unit=config.OBS_CLIN_TEMP_FAHRENHEIT_UNIT,
    )
    df_all = pd.concat([df_vital, df_obs], ignore_index=True)
    dfs_by_test = clean_vitals(df_all, min_records=min_records,
                               valid_ranges=config.VITAL_VALID_RANGES)
    if not dfs_by_test:
        print("  WARNING: No vital tests passed the cleaning filters.")
        return
    vitals_dict = build_vital_dict(dfs_by_test, config.ORDERED_VITALS, reference_dates)
    print(f"  Combined: {len(vitals_dict):,} patients")
    split_and_save(vitals_dict, filename, ssid_to_site, output_dir, sites_filter)


def run_and_split_meds(
    data_dir: str, output_dir: str, reference_dates: dict,
    rxnorm_cache_path: str, ssid_to_site: dict, sites_filter: set | None,
) -> None:
    from medications.rxnorm import resolve_rxnorm_names
    from medications.build_dict import stream_build_prescribing_dicts, substitute_drug_names
    from medications.medadmin import stream_build_medadmin_dict, merge_into_nonoral
    oral_filename    = PIPELINE_FILES["oral_meds"]
    nonoral_filename = PIPELINE_FILES["nonoral_meds"]
    print("── Medications pipeline ──────────────────────────────────────────")
    csv_path = os.path.join(data_dir, config.PRESCRIBING_FILE)
    oral_dict, nonoral_dict, unique_cuis = stream_build_prescribing_dicts(
        csv_path, reference_dates, config.ORAL_ROUTES, chunksize=config.CHUNK_SIZE,
    )
    cui_to_name = resolve_rxnorm_names(unique_cuis, rxnorm_cache_path)
    substitute_drug_names(oral_dict, cui_to_name)
    print(f"  Oral combined: {len(oral_dict):,} patients")
    split_and_save(oral_dict, oral_filename, ssid_to_site, output_dir, sites_filter)

    substitute_drug_names(nonoral_dict, cui_to_name)
    medadmin_path = os.path.join(data_dir, config.MED_ADMIN_FILE)
    if os.path.exists(medadmin_path):
        supplement, ma_cuis = stream_build_medadmin_dict(
            medadmin_path, reference_dates, config.ORAL_ROUTES, chunksize=config.CHUNK_SIZE,
        )
        if supplement:
            new_cuis = ma_cuis - set(cui_to_name.keys())
            if new_cuis:
                cui_to_name.update(resolve_rxnorm_names(new_cuis, rxnorm_cache_path))
            substitute_drug_names(supplement, cui_to_name)
            nonoral_dict, _, _ = merge_into_nonoral(nonoral_dict, supplement)
    print(f"  Non-oral combined: {len(nonoral_dict):,} patients")
    split_and_save(nonoral_dict, nonoral_filename, ssid_to_site, output_dir, sites_filter)


def run_and_split_admissions(
    data_dir: str, output_dir: str, reference_dates: dict,
    ssid_to_site: dict, sites_filter: set | None,
) -> None:
    from admissions.load import load_encounters
    from admissions.clean import clean_admissions
    from admissions.build_dict import build_admission_dict
    filename = PIPELINE_FILES["admissions"]
    print("── Admissions pipeline ───────────────────────────────────────────")
    csv_path = os.path.join(data_dir, config.ENCOUNTER_FILE)
    df = load_encounters(csv_path, inpatient_enc_types=config.INPATIENT_ENC_TYPES)
    df_clean = clean_admissions(df, config.INPATIENT_ENC_TYPES)
    adm_dict = build_admission_dict(df_clean, reference_dates)
    print(f"  Combined: {len(adm_dict):,} patients")
    split_and_save(adm_dict, filename, ssid_to_site, output_dir, sites_filter)


def run_and_split_stage(
    data_dir: str, output_dir: str, reference_dates: dict,
    ssid_to_site: dict, sites_filter: set | None,
) -> None:
    from stage.load import load_diagnoses_for_staging
    from stage.build_dict import build_stage_dict
    filename = PIPELINE_FILES["stage"]
    print("── Stage pipeline ────────────────────────────────────────────────")
    csv_path = os.path.join(data_dir, config.DIAGNOSIS_FILE)
    df = load_diagnoses_for_staging(csv_path, config.METASTASIS_ICD10_PREFIXES,
                                    chunksize=config.CHUNK_SIZE)
    if df.empty:
        print("  WARNING: No metastasis records found.")
        return
    stage_dict = build_stage_dict(df, reference_dates)
    print(f"  Combined: {len(stage_dict):,} patients")
    split_and_save(stage_dict, filename, ssid_to_site, output_dir, sites_filter)


def run_and_split_demographics(
    data_dir: str, output_dir: str, enrollment_dates: dict,
    ssid_to_site: dict, sites_filter: set | None,
) -> None:
    from demographics.load import load_demographics
    from demographics.build_dict import build_demographics_dict
    filename = PIPELINE_FILES["demographics"]
    print("── Demographics pipeline ─────────────────────────────────────────")
    csv_path = os.path.join(data_dir, config.DEMOGRAPHICS_FILE)
    df = load_demographics(csv_path)
    demo_dict = build_demographics_dict(
        df, enrollment_dates,
        config.SEX_CODE_TO_LABEL, config.RACE_CODE_TO_LABEL, config.HISPANIC_CODE_TO_LABEL,
    )
    print(f"  Combined: {len(demo_dict):,} patients")
    split_and_save(demo_dict, filename, ssid_to_site, output_dir, sites_filter)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Insight preprocessing output by site_id."
    )
    parser.add_argument(
        "--source-dir",
        default=os.path.join(config.OUTPUT_DIR, "_combined"),
        help="Directory with combined pickles (default: full_rerun/). "
             "Used in split-only mode (without --run-preprocessing).",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help="Root output directory; per-site subdirs are created here.",
    )
    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help=f"Raw Insight CSVs (default: {config.DATA_DIR}). "
             "Only needed with --run-preprocessing.",
    )
    parser.add_argument(
        "--demographics-csv",
        default=None,
        help="Path to demographics.csv (default: <data-dir>/demographics.csv).",
    )
    parser.add_argument(
        "--run-preprocessing",
        action="store_true",
        help="Run the full preprocessing pipeline before splitting.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=ALL_PIPELINES,
        default=ALL_PIPELINES,
        help="Which pipelines to process (default: all).",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Restrict output to these site IDs (default: all sites).",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=config.MIN_RECORD_THRESHOLD,
        help=f"Minimum records per test (labs/vitals, default: {config.MIN_RECORD_THRESHOLD}). "
             "Only used with --run-preprocessing.",
    )
    parser.add_argument(
        "--rxnorm-cache",
        default=None,
        help="RxNorm name cache JSON path. Only used with --run-preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    demographics_csv = args.demographics_csv or os.path.join(
        args.data_dir, config.DEMOGRAPHICS_FILE
    )
    print(f"Loading site map from: {demographics_csv}")
    ssid_to_site = load_ssid_to_site(demographics_csv)

    sites_filter: set[str] | None = set(args.sites) if args.sites else None
    if sites_filter:
        print(f"Restricting to sites: {sorted(sites_filter)}")

    print(f"Pipelines: {args.pipelines}")
    print(f"Output root: {args.output_dir}")

    # ------------------------------------------------------------------
    # Mode A: split existing combined pickles (default, fast)
    # ------------------------------------------------------------------
    if not args.run_preprocessing:
        print(f"\nSplit-only mode: reading combined pickles from {args.source_dir}")
        split_from_source(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            ssid_to_site=ssid_to_site,
            pipelines=args.pipelines,
            sites_filter=sites_filter,
        )
        print("\nDone.")
        return

    # ------------------------------------------------------------------
    # Mode B: run full preprocessing then split
    # ------------------------------------------------------------------
    print(f"\nFull-pipeline mode: processing from {args.data_dir}")

    enrollment_csv = os.path.join(args.data_dir, config.ENROLLMENT_FILE)
    print(f"Loading enrollment dates from: {enrollment_csv}")
    enrollment_dates = load_enrollment_dates(enrollment_csv)

    # Map from pipeline tag → (group, function)
    # We split the tags into canonical pipeline groups to match run_preprocessing.py
    pipeline_group_map: dict[str, str] = {
        "labs":        "labs",
        "vitals":      "vitals",
        "oral_meds":   "meds",
        "nonoral_meds":"meds",
        "admissions":  "admissions",
        "stage":       "stage",
        "demographics":"demographics",
    }
    groups_needed = set(pipeline_group_map[p] for p in args.pipelines)

    # Demographics is static and anchored on enrollment alone, so a
    # demographics-only run skips this full scan of the 50GB lab file.
    reference_dates: dict = {}
    if groups_needed - {"demographics"}:
        lab_csv = os.path.join(args.data_dir, config.LAB_FILE)
        print(f"Computing first-lab dates from: {lab_csv}")
        first_lab_dates = load_first_lab_dates(lab_csv, chunksize=config.CHUNK_SIZE)
        reference_dates = build_reference_dates(first_lab_dates, enrollment_dates)
        print(f"  {len(reference_dates):,} patients total")

    if "labs" in groups_needed:
        run_and_split_labs(
            args.data_dir, args.output_dir, reference_dates,
            args.min_records, ssid_to_site, sites_filter,
        )

    if "vitals" in groups_needed:
        run_and_split_vitals(
            args.data_dir, args.output_dir, reference_dates,
            args.min_records, ssid_to_site, sites_filter,
        )

    if "meds" in groups_needed:
        cache_path = args.rxnorm_cache or config.RXNORM_CACHE_FILE
        run_and_split_meds(
            args.data_dir, args.output_dir, reference_dates,
            cache_path, ssid_to_site, sites_filter,
        )

    if "admissions" in groups_needed:
        run_and_split_admissions(
            args.data_dir, args.output_dir, reference_dates,
            ssid_to_site, sites_filter,
        )

    if "stage" in groups_needed:
        run_and_split_stage(
            args.data_dir, args.output_dir, reference_dates,
            ssid_to_site, sites_filter,
        )

    if "demographics" in groups_needed:
        run_and_split_demographics(
            args.data_dir, args.output_dir, enrollment_dates,
            ssid_to_site, sites_filter,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
