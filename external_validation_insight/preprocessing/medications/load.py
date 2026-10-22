"""Load prescriptions from PCORnet CDM prescribing.csv (chunked; ~12 GB file)."""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {
    "ssid", "rxnorm_cui", "rx_start_date", "rx_end_date", "rx_route",
    "rx_dose_ordered", "rx_dose_ordered_unit", "rx_quantity",
    "rx_frequency", "rx_days_supply",
}
_READ_COLS = [
    "ssid", "rxnorm_cui", "rx_start_date", "rx_end_date", "rx_route",
    "rx_dose_ordered", "rx_dose_ordered_unit", "rx_quantity",
    "rx_frequency", "rx_days_supply",
]


def load_prescriptions(
    csv_path: str,
    cohort_ssids: set[str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load prescriptions, dropping rows without a CUI or start date.

    Optionally restricts to cohort_ssids. Raises ValueError if columns are missing.
    """
    chunks = []
    reader = pd.read_csv(
        csv_path,
        dtype={"ssid": str, "rxnorm_cui": str},
        usecols=lambda c: c in set(_READ_COLS),
        chunksize=chunksize,
    )

    header_checked = False
    for chunk in reader:
        if not header_checked:
            missing = _REQUIRED_COLS - set(chunk.columns)
            if missing:
                raise ValueError(f"prescribing.csv is missing columns: {missing}")
            header_checked = True

        chunk["ssid"] = chunk["ssid"].astype(str).str.strip()
        chunk["rxnorm_cui"] = chunk["rxnorm_cui"].astype(str).str.strip()
        chunk["rx_route"] = chunk["rx_route"].astype(str).str.strip()

        # Drop rows with no CUI or no start date
        chunk = chunk[
            chunk["rxnorm_cui"].notna()
            & ~chunk["rxnorm_cui"].isin({"nan", "", "None"})
        ]
        chunk = chunk[
            chunk["rx_start_date"].notna()
            & ~chunk["rx_start_date"].astype(str).isin({"nan", "", "None"})
        ]

        if cohort_ssids is not None:
            chunk = chunk[chunk["ssid"].isin(cohort_ssids)]

        if not chunk.empty:
            chunks.append(chunk[_READ_COLS])

    if not chunks:
        return pd.DataFrame(columns=_READ_COLS)

    return pd.concat(chunks, ignore_index=True)
