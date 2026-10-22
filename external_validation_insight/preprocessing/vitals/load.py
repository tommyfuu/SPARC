"""Load BMI and BP from PCORnet vital.csv (chunked) into long format.

Other vitals come from obs_clin.csv (see load_obs_clin.py).
"""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {"ssid", "measure_date"}


def load_vitals(
    csv_path: str,
    vital_col_map: dict[str, str],
    cohort_ssids: set[str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Melt vital.csv into rows of (ssid, measure_date, CANONICAL_TEST, value).

    Drops non-numeric values. Raises ValueError if columns are missing.
    """
    vital_cols = list(vital_col_map.keys())
    read_cols = ["ssid", "measure_date"] + vital_cols

    chunks = []
    reader = pd.read_csv(
        csv_path,
        dtype={"ssid": str},
        usecols=lambda c: c in set(read_cols),
        chunksize=chunksize,
    )

    header_checked = False
    for chunk in reader:
        if not header_checked:
            missing = (_REQUIRED_COLS | set(vital_cols)) - set(chunk.columns)
            if missing:
                raise ValueError(f"vital.csv is missing columns: {missing}")
            header_checked = True

        chunk["ssid"] = chunk["ssid"].astype(str).str.strip()

        if cohort_ssids is not None:
            chunk = chunk[chunk["ssid"].isin(cohort_ssids)]

        if chunk.empty:
            continue

        # Melt wide → long
        long = chunk.melt(
            id_vars=["ssid", "measure_date"],
            value_vars=vital_cols,
            var_name="vital_col",
            value_name="value",
        )
        long["CANONICAL_TEST"] = long["vital_col"].map(vital_col_map)
        long = long.drop(columns=["vital_col"])

        # Drop null values (already-numeric columns)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])

        if not long.empty:
            chunks.append(long[["ssid", "measure_date", "CANONICAL_TEST", "value"]])

    if not chunks:
        return pd.DataFrame(columns=["ssid", "measure_date", "CANONICAL_TEST", "value"])

    return pd.concat(chunks, ignore_index=True)
