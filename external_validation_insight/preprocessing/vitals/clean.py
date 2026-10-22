"""Clean long-format vital sign measurements and group them by canonical test."""

from __future__ import annotations

import pandas as pd

from shared.utils import deduplicate


def clean_vitals(
    df: pd.DataFrame,
    min_records: int = 100,
    valid_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Clean vitals and return {canonical_test: DataFrame}.

    Keeps one row per (ssid, test, date), drops values outside valid_ranges
    (data-entry errors), and drops tests with fewer than min_records rows.
    """
    # Step 1: deduplicate
    df = deduplicate(
        df,
        group_cols=["ssid", "CANONICAL_TEST", "measure_date"],
        sort_col="measure_date",
    )

    # Step 2: rename value column
    df = df.rename(columns={"value": "CLEANED_TEST_VALUE"})

    # Step 3: range filtering
    if valid_ranges:
        keep = pd.Series(True, index=df.index)
        for canonical, (lo, hi) in valid_ranges.items():
            is_this_test = df["CANONICAL_TEST"] == canonical
            in_range = df["CLEANED_TEST_VALUE"].between(lo, hi)
            keep &= ~is_this_test | in_range
        dropped = (~keep).sum()
        if dropped > 0:
            print(f"  Range filter: dropped {dropped:,} rows outside plausible value ranges")
        df = df[keep]
        if df.empty:
            return {}

    # Step 4: filter by record count
    counts = df.groupby("CANONICAL_TEST")["ssid"].transform("count")
    df = df[counts >= min_records]

    result: dict[str, pd.DataFrame] = {}
    for canonical, group in df.groupby("CANONICAL_TEST"):
        result[canonical] = group[
            ["ssid", "measure_date", "CANONICAL_TEST", "CLEANED_TEST_VALUE"]
        ].reset_index(drop=True)

    return result
