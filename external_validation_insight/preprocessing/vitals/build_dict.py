"""Build the vitals dict in SPARC format.

{ssid: {date_offset: [9 values in ORDERED_VITALS order]}}; missing values are the
string 'nan' (invasive BP is always missing in Insight). Positions 5-8 come from obs_clin.csv.
"""

from __future__ import annotations

import pandas as pd

_NAN = "nan"  # String sentinel used by existing SPARC pipeline for missing values


def build_vital_dict(
    dfs_by_test: dict[str, pd.DataFrame],
    ordered_vitals: list[str],
    enrollment_dates: dict[str, pd.Timestamp],
) -> dict[str, dict[int, list]]:
    """Build {ssid: {date_offset: [values in ordered_vitals order]}}.

    Patients without a reference date are skipped.
    """
    enr_series = pd.Series(enrollment_dates)

    # Build O(1) lookup per test: (ssid, date_offset) → float value
    # Uses vectorized offset computation instead of iterrows
    lookup: dict[str, dict[tuple[str, int], float]] = {}

    for canonical, df in dfs_by_test.items():
        ref_dates = df["ssid"].map(enr_series)
        valid = ref_dates.notna() & df["CLEANED_TEST_VALUE"].notna()
        df2 = df[valid]
        if df2.empty:
            lookup[canonical] = {}
            continue

        ref2 = ref_dates[valid]
        measure_dates = pd.to_datetime(df2["measure_date"], errors="coerce")
        valid2 = measure_dates.notna()
        df2 = df2[valid2]
        ref2 = ref2[valid2]
        measure_dates = measure_dates[valid2]

        offsets = (measure_dates - ref2).dt.days

        test_lookup: dict[tuple[str, int], float] = {}
        for ssid, offset, value in zip(df2["ssid"], offsets, df2["CLEANED_TEST_VALUE"]):
            test_lookup[(ssid, int(offset))] = float(value)
        lookup[canonical] = test_lookup

    # Collect all (ssid, offset) pairs that have at least one vital
    all_keys: set[tuple[str, int]] = set()
    for test_lookup in lookup.values():
        all_keys.update(test_lookup.keys())

    # Build output dict
    out: dict[str, dict[int, list]] = {}
    for ssid, offset in all_keys:
        vec = [
            lookup.get(vital, {}).get((ssid, offset), _NAN)
            for vital in ordered_vitals
        ]
        out.setdefault(ssid, {})[offset] = vec

    return out
