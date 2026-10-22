"""Load lab results from PCORnet CDM lab_result_cm.csv (~50 GB).

Read in chunks with early LOINC filtering.
"""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {"ssid", "lab_loinc", "result_date", "result_num"}
_READ_COLS = ["ssid", "lab_loinc", "result_date", "result_num", "result_unit"]


def load_labs(
    csv_path: str,
    known_loincs: set[str] | None = None,
    cohort_ssids: set[str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load lab results, optionally filtered by LOINC and cohort ssids.

    Raises ValueError if required columns are missing.
    """
    chunks = []
    reader = pd.read_csv(
        csv_path,
        dtype={"ssid": str, "lab_loinc": str},
        usecols=lambda c: c in set(_READ_COLS),
        chunksize=chunksize,
    )

    header_checked = False
    for chunk in reader:
        if not header_checked:
            missing = _REQUIRED_COLS - set(chunk.columns)
            if missing:
                raise ValueError(f"lab_result_cm.csv is missing columns: {missing}")
            header_checked = True

        chunk["ssid"] = chunk["ssid"].astype(str).str.strip()
        chunk["lab_loinc"] = chunk["lab_loinc"].astype(str).str.strip()

        # Early LOINC filter — keeps only rows we can map to a canonical name
        if known_loincs is not None:
            chunk = chunk[chunk["lab_loinc"].isin(known_loincs)]

        if cohort_ssids is not None:
            chunk = chunk[chunk["ssid"].isin(cohort_ssids)]

        if not chunk.empty:
            chunks.append(chunk[_READ_COLS])

    if not chunks:
        return pd.DataFrame(columns=_READ_COLS)

    return pd.concat(chunks, ignore_index=True)
