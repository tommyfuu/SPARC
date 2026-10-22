"""Load metastasis diagnoses (C77-C79, C800) from PCORnet CDM diagnosis.csv (~11 GB).

Read in chunks with early ICD-10 prefix filtering. Used to infer Stage 4 only.
"""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {"ssid", "admit_date", "dx"}
_READ_COLS = ["ssid", "admit_date", "dx"]


def load_diagnoses_for_staging(
    csv_path: str,
    metastasis_prefixes: tuple[str, ...],
    cohort_ssids: set[str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load diagnosis rows whose dx starts with a metastasis prefix.

    Raises ValueError if required columns are missing.
    """
    chunks = []
    reader = pd.read_csv(
        csv_path,
        dtype={"ssid": str, "dx": str},
        usecols=lambda c: c in set(_READ_COLS),
        chunksize=chunksize,
    )

    header_checked = False
    for chunk in reader:
        if not header_checked:
            missing = _REQUIRED_COLS - set(chunk.columns)
            if missing:
                raise ValueError(f"diagnosis.csv is missing columns: {missing}")
            header_checked = True

        chunk["ssid"] = chunk["ssid"].astype(str).str.strip()
        chunk["dx"] = chunk["dx"].astype(str).str.strip()

        # Early filter: keep only metastasis ICD-10 codes
        meta_mask = chunk["dx"].str.startswith(metastasis_prefixes, na=False)
        chunk = chunk[meta_mask]

        if cohort_ssids is not None:
            chunk = chunk[chunk["ssid"].isin(cohort_ssids)]

        if not chunk.empty:
            chunks.append(chunk[_READ_COLS])

    if not chunks:
        return pd.DataFrame(columns=_READ_COLS)

    return pd.concat(chunks, ignore_index=True)
