"""Load encounters from PCORnet CDM encounter.csv (~2.3 GB).

Read in chunks with early enc_type filtering.
"""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {"ssid", "enc_type", "admit_date", "discharge_date"}
_READ_COLS = ["ssid", "enc_type", "admit_date", "discharge_date"]


def load_encounters(
    csv_path: str,
    inpatient_enc_types: frozenset[str] | None = None,
    cohort_ssids: set[str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load encounters, optionally filtered by enc_type and cohort ssids.

    Raises ValueError if required columns are missing.
    """
    chunks = []
    reader = pd.read_csv(
        csv_path,
        dtype={"ssid": str},
        usecols=lambda c: c in set(_READ_COLS),
        chunksize=chunksize,
    )

    header_checked = False
    for chunk in reader:
        if not header_checked:
            missing = _REQUIRED_COLS - set(chunk.columns)
            if missing:
                raise ValueError(f"encounter.csv is missing columns: {missing}")
            header_checked = True

        chunk["ssid"] = chunk["ssid"].astype(str).str.strip()
        chunk["enc_type"] = chunk["enc_type"].astype(str).str.strip().str.upper()

        # Early filter to inpatient enc_types
        if inpatient_enc_types is not None:
            chunk = chunk[chunk["enc_type"].isin(inpatient_enc_types)]

        if cohort_ssids is not None:
            chunk = chunk[chunk["ssid"].isin(cohort_ssids)]

        if not chunk.empty:
            chunks.append(chunk[_READ_COLS])

    if not chunks:
        return pd.DataFrame(columns=_READ_COLS)

    return pd.concat(chunks, ignore_index=True)
