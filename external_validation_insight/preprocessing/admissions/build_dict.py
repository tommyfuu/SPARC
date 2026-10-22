"""Build ssid_to_hospital_adm_dsch_dict.pickle from inpatient encounters.

Output: {ssid: [(admit_offset, dsch_offset), ...]}, days since enr_start_date, sorted.
No ICU pickle: insight has no ICU-specific enc_type.
"""

from __future__ import annotations

import pandas as pd


def build_admission_dict(
    df: pd.DataFrame,
    enrollment_dates: dict[str, pd.Timestamp],
) -> dict[str, list[tuple[int, int]]]:
    """Build {ssid: [(admit_offset, dsch_offset), ...]} sorted by date.

    Patients without an enrollment date are skipped. A missing discharge date
    falls back to the admit offset.
    """
    enr_series = pd.Series(enrollment_dates)

    # Vectorized enrollment date lookup
    ref_dates = df["ssid"].map(enr_series)
    valid = ref_dates.notna()
    df2 = df[valid]
    ref2 = ref_dates[valid]

    if df2.empty:
        return {}

    # Vectorized offset computation (admit_date already Timestamp from clean_admissions)
    admit_offsets = (df2["admit_date"] - ref2).dt.days.astype(int)

    # Discharge offsets: NaT → fall back to admit offset
    dsch_raw = (df2["discharge_date"] - ref2).dt.days
    dsch_offsets = dsch_raw.where(df2["discharge_date"].notna(), other=admit_offsets).astype(int)

    out: dict[str, list[tuple[int, int]]] = {}
    for ssid, admit_off, dsch_off in zip(df2["ssid"], admit_offsets, dsch_offsets):
        out.setdefault(ssid, []).append((int(admit_off), int(dsch_off)))

    # Sort each patient's admissions chronologically.
    # The binary / binary_icu_pre tasks anchor on admission_dates[-1] (see
    # data_loaders_inf_risk_fixed.py:866), taking it to be the patient's LAST
    # admission: it sets the prediction cutoff (discharge) and bounds the
    # feature window. Appending in encounter-file row order made [-1] the
    # last row in the file rather than the latest date, so for any patient
    # whose admissions were not already in date order, both the label horizon
    # and the input window came from the wrong stay.
    for ssid in out:
        out[ssid].sort()

    return out
