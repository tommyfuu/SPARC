#!/usr/bin/env python3
"""Check the raw PCORnet CSVs before the long run: every file the pipeline reads exists and has the columns it uses.
Prints headers and aggregate site counts only, no patient rows.
Usage: python preflight_check.py [--data-dir <INSIGHT_DATA_DIR>]
"""
import os, sys, csv, argparse
from collections import Counter
import config

REQUIRED = {
    config.ENROLLMENT_FILE: ["ssid", "enr_start_date"],
    config.DEMOGRAPHICS_FILE: ["ssid", "birth_date", "sex", "race", "hispanic", "site_id"],
    config.LAB_FILE: ["ssid", "lab_loinc", "result_date", "result_num", "result_unit"],
    config.VITAL_FILE: ["ssid", "measure_date"],
    config.OBS_CLIN_FILE: ["ssid", "obsclin_start_date", "obsclin_code", "obsclin_result_num", "obsclin_result_unit"],
    config.PRESCRIBING_FILE: ["ssid", "rxnorm_cui", "rx_start_date", "rx_end_date", "rx_route"],
    config.MED_ADMIN_FILE: ["ssid", "medadmin_code", "medadmin_route", "medadmin_start_date", "medadmin_stop_date"],
    config.ENCOUNTER_FILE: ["ssid", "enc_type", "admit_date", "discharge_date"],
    config.DIAGNOSIS_FILE: ["ssid", "admit_date", "dx"],
    "death.csv": ["ssid", "death_date"],   # read by the SPARC loader (mortality task)
}

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=config.DATA_DIR)
args = ap.parse_args()
print(f"data dir: {args.data_dir}\nRxNorm cache: {config.RXNORM_CACHE_FILE} "
      f"({'found' if os.path.exists(config.RXNORM_CACHE_FILE) else 'MISSING'})\n")

ok = True
for fname, cols in REQUIRED.items():
    p = os.path.join(args.data_dir, fname)
    if not os.path.exists(p):
        print(f"MISSING FILE  {fname}"); ok = False; continue
    with open(p, newline="") as fh:
        header = [h.strip().lower() for h in next(csv.reader(fh))]
    missing = [c for c in cols if c not in header]
    print(f"{'OK     ' if not missing else 'COLUMNS'}  {fname:<20} {os.path.getsize(p)/1e9:7.2f} GB"
          + (f"  missing: {missing}" if missing else ""))
    ok &= not missing

# per-site patient counts: the split step expects sites A-F
p = os.path.join(args.data_dir, config.DEMOGRAPHICS_FILE)
if os.path.exists(p):
    with open(p, newline="") as fh:
        sites = Counter(r.get("site_id", "") for r in csv.DictReader(fh))
    print(f"\nsite_id counts in {config.DEMOGRAPHICS_FILE}: {dict(sorted(sites.items()))}")
    if set(s for s in sites if s) - set("ABCDEF"):
        print("NOTE: sites other than A-F found; create_splits.py and make_views.py only handle A-F")
print("\nAll inputs look usable." if ok else "\nFix the problems above before running.")
sys.exit(0 if ok else 1)
