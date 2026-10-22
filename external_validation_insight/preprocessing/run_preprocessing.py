"""
Run the INSIGHT preprocessing: PCORnet CDM CSVs to SPARC-compatible pickles (labs, vitals, oral and non-oral meds,
admissions, stage, demographics). Output formats: see README.md.
Usage: python run_preprocessing.py [--data-dir <in>] [--output-dir <out>] [--pipelines labs vitals ...]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter
from statistics import median

import pandas as pd

import config
from shared.utils import load_enrollment_dates, load_first_lab_dates, build_reference_dates
from labs.load import load_labs
from labs.clean import clean_labs
from labs.build_dict import build_lab_dict, stream_build_lab_dict
from vitals.load import load_vitals
from vitals.load_obs_clin import load_obs_clin_vitals
from vitals.clean import clean_vitals
from vitals.build_dict import build_vital_dict
from medications.load import load_prescriptions
from medications.rxnorm import resolve_rxnorm_names
from medications.clean import clean_medications
from medications.build_dict import (
    build_oral_dict, build_nonoral_dict,
    stream_build_prescribing_dicts, substitute_drug_names,
)
from medications.medadmin import (
    load_medadmin_nonoral, build_medadmin_dict, merge_into_nonoral,
    stream_build_medadmin_dict,
)
from admissions.load import load_encounters
from admissions.clean import clean_admissions
from admissions.build_dict import build_admission_dict
from stage.load import load_diagnoses_for_staging
from stage.build_dict import build_stage_dict
from demographics.load import load_demographics
from demographics.build_dict import build_demographics_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Insight PCORnet CDM data into SPARC-compatible pickles."
    )
    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help=f"Directory containing insight CSV files (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help=f"Directory for output pickle files (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=["labs", "vitals", "meds", "admissions", "stage", "demographics"],
        default=["labs", "vitals", "meds", "admissions", "stage", "demographics"],
        help="Which pipelines to run (default: all six)",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=config.MIN_RECORD_THRESHOLD,
        help=f"Minimum records per test to include (default: {config.MIN_RECORD_THRESHOLD})",
    )
    parser.add_argument(
        "--rxnorm-cache",
        default=None,
        help="Path to RxNorm name cache JSON (default: from config.py)",
    )
    return parser.parse_args()


def run_labs(data_dir: str, output_dir: str, enrollment_dates: dict, min_records: int) -> None:
    print("── Labs pipeline ─────────────────────────────────────────────────")
    csv_path = os.path.join(data_dir, config.LAB_FILE)
    print(f"  Streaming: {csv_path}")
    # Single chunked pass (load + clean + build combined) — never materializes
    # the full ~150M-row LOINC-matched set in memory. Peak RSS is bounded by
    # the output dict size, not the ~50GB/330M-row input file.
    labs_dict = stream_build_lab_dict(
        csv_path,
        config.LOINC_TO_CANONICAL,
        enrollment_dates,
        canonical_to_accepted_units=config.CANONICAL_TO_ACCEPTED_UNITS,
        sentinel_values=config.LAB_RESULT_SENTINEL_VALUES,
        min_records=min_records,
        chunksize=config.CHUNK_SIZE,
        canonical_unit_aliases=config.CANONICAL_UNIT_ALIASES,
    )
    if not labs_dict:
        print("  WARNING: No lab tests passed the cleaning filters. Skipping labs output.")
        return

    n_patients = len(labs_dict)
    n_dates = sum(len(v) for v in labs_dict.values())
    n_records = sum(len(vals) for dates in labs_dict.values() for vals in dates.values())
    print(f"  Result: {n_patients:,} patients, {n_dates:,} patient-date entries, "
          f"{n_records:,} test values total")

    out_path = os.path.join(output_dir, "ssid_to_date_to_labs_dict.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(labs_dict, f)
    print(f"  Saved: {out_path}")


def run_vitals(data_dir: str, output_dir: str, enrollment_dates: dict, min_records: int) -> None:
    print("── Vitals pipeline ───────────────────────────────────────────────")

    # ── Source 1: vital.csv (BMI, BP Systolic, BP Diastolic) ──────────────
    vital_path = os.path.join(data_dir, config.VITAL_FILE)
    print(f"  Loading: {vital_path}")
    df_vital = load_vitals(vital_path, config.VITAL_COL_TO_CANONICAL)
    print(f"  vital.csv: {len(df_vital):,} long-format rows, {df_vital['ssid'].nunique():,} patients")

    # ── Source 2: obs_clin.csv (O2, Heart Rate, Resp Rate, Temp C) ────────
    obs_path = os.path.join(data_dir, config.OBS_CLIN_FILE)
    print(f"  Loading: {obs_path}")
    df_obs = load_obs_clin_vitals(
        obs_path,
        config.OBS_CLIN_LOINC_TO_VITAL,
        config.OBS_CLIN_TEMP_FAHRENHEIT_LOINC,
        obs_clin_accepted_units=config.OBS_CLIN_ACCEPTED_UNITS,
        temp_celsius_unit=config.OBS_CLIN_TEMP_CELSIUS_UNIT,
        temp_fahrenheit_unit=config.OBS_CLIN_TEMP_FAHRENHEIT_UNIT,
    )
    print(f"  obs_clin.csv: {len(df_obs):,} long-format rows, {df_obs['ssid'].nunique():,} patients")

    # ── Combine and clean ──────────────────────────────────────────────────
    df_all = pd.concat([df_vital, df_obs], ignore_index=True)
    print(f"  Combined: {len(df_all):,} rows")

    print("  Cleaning ...")
    dfs_by_test = clean_vitals(
        df_all,
        min_records=min_records,
        valid_ranges=config.VITAL_VALID_RANGES,
    )
    if not dfs_by_test:
        print("  WARNING: No vital tests passed the cleaning filters. Skipping vitals output.")
        return

    n_tests = len(dfs_by_test)
    n_rows = sum(len(v) for v in dfs_by_test.values())
    print(f"  Kept {n_tests} canonical vitals, {n_rows:,} records total")
    print(f"  Vitals: {sorted(dfs_by_test.keys())}")

    print("  Building output dictionary ...")
    vitals_dict = build_vital_dict(dfs_by_test, config.ORDERED_VITALS, enrollment_dates)
    n_patients = len(vitals_dict)
    n_dates = sum(len(v) for v in vitals_dict.values())
    print(f"  Result: {n_patients:,} patients, {n_dates:,} patient-date entries")

    out_path = os.path.join(output_dir, "ssid_to_date_to_vitals_dict.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(vitals_dict, f)
    print(f"  Saved: {out_path}")


def run_meds(
    data_dir: str,
    output_dir: str,
    enrollment_dates: dict,
    rxnorm_cache_path: str,
) -> None:
    print("── Medications pipeline ──────────────────────────────────────────")
    # Streaming: load + clean + split + build in one chunked pass, never
    # materializing the full ~32M-row filtered prescribing.csv (or the
    # ~58M-row filtered med_admin.csv) in memory — the prior accumulate-then-
    # concat pipeline needed an 80GB job; this session's cgroup caps at 40GB.
    # CUI resolution needs the full unique-CUI set up front, so entries carry
    # the raw CUI in place of drug_name until substitute_drug_names() fixes
    # them up afterward (a cheap O(output size) pass, not a second CSV read).
    csv_path = os.path.join(data_dir, config.PRESCRIBING_FILE)
    print(f"  Streaming: {csv_path}")
    oral_dict, nonoral_dict, unique_cuis = stream_build_prescribing_dicts(
        csv_path, enrollment_dates, config.ORAL_ROUTES, chunksize=config.CHUNK_SIZE,
    )
    print(f"  Found {len(unique_cuis):,} unique RxNorm CUIs")
    cui_to_name = resolve_rxnorm_names(unique_cuis, rxnorm_cache_path)

    substitute_drug_names(oral_dict, cui_to_name)
    n_pts = len(oral_dict)
    n_dates = sum(len(v) for v in oral_dict.values())
    print(f"  Oral result: {n_pts:,} patients, {n_dates:,} start-date entries")
    out_path = os.path.join(output_dir, "ssid_to_date_to_oral_meds_dict.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(oral_dict, f)
    print(f"  Saved: {out_path}")

    substitute_drug_names(nonoral_dict, cui_to_name)
    n_pts = len(nonoral_dict)
    n_dates = sum(len(v) for v in nonoral_dict.values())
    print(f"  Non-oral (prescribing): {n_pts:,} patients, {n_dates:,} start-date entries")

    # Non-oral meds supplement: med_admin.csv (actual IV/SC administrations)
    medadmin_path = os.path.join(data_dir, config.MED_ADMIN_FILE)
    if os.path.exists(medadmin_path):
        print(f"  Streaming: {medadmin_path}")
        supplement, ma_cuis = stream_build_medadmin_dict(
            medadmin_path, enrollment_dates, config.ORAL_ROUTES, chunksize=config.CHUNK_SIZE,
        )
        n_ma_pts = len(supplement)
        n_ma_entries = sum(len(v) for dates in supplement.values() for v in dates.values())
        print(f"  med_admin.csv (non-oral/unknown routes): {n_ma_entries:,} entries, {n_ma_pts:,} patients")

        if supplement:
            # Resolve any CUIs in med_admin that prescribing didn't cover
            new_cuis = ma_cuis - set(cui_to_name.keys())
            if new_cuis:
                print(f"  Resolving {len(new_cuis):,} additional CUIs from med_admin.csv ...")
                extra = resolve_rxnorm_names(new_cuis, rxnorm_cache_path)
                cui_to_name.update(extra)

            substitute_drug_names(supplement, cui_to_name)
            print(f"  Supplement: {n_ma_pts:,} patients with parenteral admin records")

            nonoral_dict, new_pts, new_entries = merge_into_nonoral(nonoral_dict, supplement)
            print(f"  Merged: {new_pts:,} new patients, {new_entries:,} new entries")

        n_pts = len(nonoral_dict)
        n_dates = sum(len(v) for v in nonoral_dict.values())
        print(f"  Non-oral total (prescribing + medadmin): {n_pts:,} patients, {n_dates:,} start-date entries")
    else:
        print(f"  WARNING: {medadmin_path} not found — non-oral dict contains prescribing.csv only")

    out_path = os.path.join(output_dir, "ssid_to_date_to_nonoral_meds_dict.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(nonoral_dict, f, protocol=5)
    print(f"  Saved: {out_path}")


def run_admissions(data_dir: str, output_dir: str, enrollment_dates: dict) -> None:
    print("── Admissions pipeline ───────────────────────────────────────────")
    csv_path = os.path.join(data_dir, config.ENCOUNTER_FILE)
    print(f"  Loading: {csv_path}")
    df = load_encounters(csv_path, inpatient_enc_types=config.INPATIENT_ENC_TYPES)
    print(f"  Loaded {len(df):,} rows, {df['ssid'].nunique():,} patients")

    print(f"  Filtering to inpatient enc_types: {sorted(config.INPATIENT_ENC_TYPES)} ...")
    df_clean = clean_admissions(df, config.INPATIENT_ENC_TYPES)
    print(f"  Inpatient encounters: {len(df_clean):,} rows, {df_clean['ssid'].nunique():,} patients")

    print("  Building hospital admissions dictionary ...")
    adm_dict = build_admission_dict(df_clean, enrollment_dates)
    n_pts = len(adm_dict)
    n_stays = sum(len(v) for v in adm_dict.values())
    print(f"  Result: {n_pts:,} patients, {n_stays:,} total admission stays")

    out_path = os.path.join(output_dir, "ssid_to_hospital_adm_dsch_dict.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(adm_dict, f)
    print(f"  Saved: {out_path}")
    print("  Note: ICU admissions pickle not produced — no ICU enc_type in insight data.")


def run_stage(data_dir: str, output_dir: str, enrollment_dates: dict) -> None:
    print("── Stage pipeline ────────────────────────────────────────────────")
    csv_path = os.path.join(data_dir, config.DIAGNOSIS_FILE)
    print(f"  Loading: {csv_path}")
    print(f"  Filtering to metastasis prefixes: {config.METASTASIS_ICD10_PREFIXES}")
    df = load_diagnoses_for_staging(
        csv_path,
        config.METASTASIS_ICD10_PREFIXES,
        chunksize=config.CHUNK_SIZE,
    )
    print(f"  Loaded {len(df):,} metastasis diagnosis rows, {df['ssid'].nunique():,} patients")

    if df.empty:
        print("  WARNING: No metastasis records found. Skipping stage output.")
        return

    print("  Building stage dictionary ...")
    stage_dict = build_stage_dict(df, enrollment_dates)
    n_pts = len(stage_dict)
    n_dates = sum(len(v) for v in stage_dict.values())
    print(f"  Result: {n_pts:,} patients, {n_dates:,} patient-date entries")
    print("  Note: Only Stage 4 produced (inferred from ICD-10 C77-C79, C800 codes).")

    out_path = os.path.join(output_dir, "ssid_to_date_to_STAGE_dict.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(stage_dict, f)
    print(f"  Saved: {out_path}")


def run_demographics(data_dir: str, output_dir: str, enrollment_dates: dict) -> None:
    print("── Demographics pipeline ─────────────────────────────────────────")
    csv_path = os.path.join(data_dir, config.DEMOGRAPHICS_FILE)
    print(f"  Loading: {csv_path}")
    df = load_demographics(csv_path)
    print(f"  Loaded {len(df):,} patients")

    print("  Building demographics dictionary ...")
    demo_dict = build_demographics_dict(
        df,
        enrollment_dates,
        config.SEX_CODE_TO_LABEL,
        config.RACE_CODE_TO_LABEL,
        config.HISPANIC_CODE_TO_LABEL,
    )
    ages = [v["age"] for v in demo_dict.values() if v["age"] is not None]
    print(f"  Result: {len(demo_dict):,} patients, {len(ages):,} with a computable age")
    if ages:
        print(f"  age: median {median(ages):.1f}, range {min(ages):.1f}-{max(ages):.1f} years")
    for field in ("sex", "race", "ethnicity"):
        counts = Counter(v[field] for v in demo_dict.values())
        print(f"  {field}: {dict(counts.most_common())}")

    out_path = os.path.join(output_dir, "ssid_to_demographics_dict.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(demo_dict, f)
    print(f"  Saved: {out_path}")


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    enrollment_csv = os.path.join(args.data_dir, config.ENROLLMENT_FILE)
    print(f"Loading enrollment dates from: {enrollment_csv}")
    enrollment_dates = load_enrollment_dates(enrollment_csv)
    print(f"  {len(enrollment_dates):,} patients with enrollment dates")

    # Compute first-lab date per patient from lab_result_cm.csv.
    # This is the CAIA-compatible time-zero: all feature offsets (labs, vitals,
    # meds, admissions, stage) are measured from first lab date so they share a
    # common timeline. Patients without any lab fall back to enrollment date.
    # Demographics is static and anchored on enrollment alone, so a
    # demographics-only run skips this full scan of the 50GB lab file.
    timeseries_pipelines = [p for p in args.pipelines if p != "demographics"]
    reference_dates: dict = {}
    if timeseries_pipelines:
        lab_csv = os.path.join(args.data_dir, config.LAB_FILE)
        print(f"Computing first-lab dates from: {lab_csv}")
        first_lab_dates = load_first_lab_dates(lab_csv, chunksize=config.CHUNK_SIZE)
        print(f"  {len(first_lab_dates):,} patients with lab data")
        reference_dates = build_reference_dates(first_lab_dates, enrollment_dates)
        print(f"  {len(reference_dates):,} patients total (first-lab date preferred, enrollment fallback)")

    if "labs" in args.pipelines:
        run_labs(args.data_dir, args.output_dir, reference_dates, args.min_records)

    if "vitals" in args.pipelines:
        run_vitals(args.data_dir, args.output_dir, reference_dates, args.min_records)

    if "meds" in args.pipelines:
        cache_path = args.rxnorm_cache or config.RXNORM_CACHE_FILE
        run_meds(args.data_dir, args.output_dir, reference_dates, cache_path)

    if "admissions" in args.pipelines:
        run_admissions(args.data_dir, args.output_dir, reference_dates)

    if "stage" in args.pipelines:
        run_stage(args.data_dir, args.output_dir, reference_dates)

    if "demographics" in args.pipelines:
        run_demographics(args.data_dir, args.output_dir, enrollment_dates)

    print("Done.")


if __name__ == "__main__":
    # Add the repo root to sys.path so relative imports work when invoked
    # as a script (python run_preprocessing.py) without installing the package.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
