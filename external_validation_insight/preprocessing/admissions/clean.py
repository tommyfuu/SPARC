"""Filter encounters to inpatient stays (IP, EI) and parse dates."""

from __future__ import annotations

import pandas as pd


def clean_admissions(
    df: pd.DataFrame,
    inpatient_enc_types: frozenset[str],
) -> pd.DataFrame:
    """Keep inpatient encounters and parse admit/discharge dates.

    Rows without an admit_date are dropped; discharge_date may be NaT.
    """
    df = df.copy()

    # Filter to inpatient
    df = df[df["enc_type"].isin(inpatient_enc_types)]

    # Parse dates
    df["admit_date"] = pd.to_datetime(df["admit_date"], errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")

    # Drop rows with no admit date
    df = df.dropna(subset=["admit_date"])

    return df.reset_index(drop=True)
