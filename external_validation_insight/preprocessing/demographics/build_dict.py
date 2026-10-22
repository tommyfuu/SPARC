"""Build ssid_to_demographics_dict.pickle (static, no date_offset level).

Output: {ssid: {age, age_binned_5, sex, race, ethnicity}}.
Age is at the earliest enr_start_date, matching dataset_loading.py::_load_age.
"""

from __future__ import annotations

import pandas as pd

_UNKNOWN = "Unknown"
_AGE_BIN_YEARS = 5
_DAYS_PER_YEAR = 365.25


def _report_unmapped(codes: pd.Series, code_to_label: dict[str, str], field: str) -> None:
    """Warn about codes missing from the map (they decode to _UNKNOWN)."""
    unmapped = sorted(set(codes.unique()) - set(code_to_label))
    if unmapped:
        print(f"  WARNING: unmapped {field} codes decoded as '{_UNKNOWN}': {unmapped}")


def build_demographics_dict(
    df: pd.DataFrame,
    enrollment_dates: dict[str, pd.Timestamp],
    sex_code_to_label: dict[str, str],
    race_code_to_label: dict[str, str],
    hispanic_code_to_label: dict[str, str],
) -> dict[str, dict]:
    """Build {ssid: {age, age_binned_5, sex, race, ethnicity}}.

    age is None if birth date or enrollment date is missing, or enrollment
    precedes birth. Unmapped codes become "Unknown".
    """
    enr_series = pd.Series(enrollment_dates)

    ref_dates = df["ssid"].map(enr_series)
    ages = (ref_dates - df["birth_date"]).dt.days / _DAYS_PER_YEAR
    ages = ages.where(ages >= 0)

    _report_unmapped(df["sex"], sex_code_to_label, "sex")
    _report_unmapped(df["race"], race_code_to_label, "race")
    _report_unmapped(df["hispanic"], hispanic_code_to_label, "hispanic")

    out: dict[str, dict] = {}
    for ssid, age, sex, race, hispanic in zip(
        df["ssid"], ages, df["sex"], df["race"], df["hispanic"]
    ):
        known_age = pd.notna(age)
        out[ssid] = {
            "age": float(age) if known_age else None,
            "age_binned_5": int(age // _AGE_BIN_YEARS) if known_age else None,
            "sex": sex_code_to_label.get(sex, _UNKNOWN),
            "race": race_code_to_label.get(race, _UNKNOWN),
            "ethnicity": hispanic_code_to_label.get(hispanic, _UNKNOWN),
        }

    return out
