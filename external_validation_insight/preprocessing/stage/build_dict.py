"""Build ssid_to_date_to_STAGE_dict.pickle: {ssid: {date_offset: 4.0}}.

Insight has no TNM/numeric stage, so only Stage 4 is produced, inferred from
metastasis ICD-10 codes (C77-C79, C80.0). Offset = days since enr_start_date.
"""

from __future__ import annotations

import pandas as pd

_STAGE_4 = 4.0


def build_stage_dict(
    df: pd.DataFrame,
    enrollment_dates: dict[str, pd.Timestamp],
) -> dict[str, dict[int, float]]:
    """Build {ssid: {date_offset: 4.0}} from metastatic diagnosis records.

    Skips patients without an enrollment date and rows with bad admit_date.
    """
    enr_series = pd.Series(enrollment_dates)

    # Vectorized enrollment date lookup
    ref_dates = df["ssid"].map(enr_series)
    valid = ref_dates.notna()
    df2 = df[valid].copy()
    ref2 = ref_dates[valid]

    if df2.empty:
        return {}

    # Vectorized date parsing and offset computation
    admit_dates = pd.to_datetime(df2["admit_date"], errors="coerce")
    valid2 = admit_dates.notna()
    df2 = df2[valid2]
    ref2 = ref2[valid2]
    admit_dates = admit_dates[valid2]

    offsets = (admit_dates - ref2).dt.days.astype(int)

    out: dict[str, dict[int, float]] = {}
    for ssid, offset in zip(df2["ssid"], offsets):
        out.setdefault(ssid, {})[int(offset)] = _STAGE_4

    return out
