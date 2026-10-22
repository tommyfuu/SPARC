"""Load PCORnet CDM demographics.csv (small, read in one pass).

sex, race and hispanic stay as normalised codes; build_dict.py decodes them.
"""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {"ssid", "birth_date", "sex", "race", "hispanic"}
_READ_COLS = ["ssid", "birth_date", "sex", "race", "hispanic"]
_CODE_COLS = ["sex", "race", "hispanic"]


def load_demographics(csv_path: str) -> pd.DataFrame:
    """Load one row per patient from demographics.csv.

    Code columns are stripped and uppercased, bad birth dates become NaT, and
    duplicate ssids keep the first row. Raises ValueError on missing columns.
    """
    df = pd.read_csv(csv_path, dtype=str, usecols=lambda c: c in set(_READ_COLS))

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"demographics.csv is missing columns: {missing}")

    df["ssid"] = df["ssid"].astype(str).str.strip()
    for col in _CODE_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip().str.upper()
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

    df = df.drop_duplicates(subset=["ssid"], keep="first")
    return df[_READ_COLS].reset_index(drop=True)
