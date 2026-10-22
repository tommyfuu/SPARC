"""Load O2, heart rate, resp rate and temperature from PCORnet obs_clin.csv by LOINC code.

Complements vital.csv (BMI, BP); both long-format outputs are concatenated before
cleaning. Temperature in [degF] is converted to Celsius.
"""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {"ssid", "obsclin_start_date", "obsclin_code", "obsclin_result_num"}
_READ_COLS = ["ssid", "obsclin_start_date", "obsclin_code",
              "obsclin_result_num", "obsclin_result_unit"]


def load_obs_clin_vitals(
    csv_path: str,
    obs_clin_loinc_map: dict[str, str],
    temp_fahrenheit_loinc: str,
    obs_clin_accepted_units: dict[str, frozenset[str]] | None = None,
    temp_celsius_unit: str = "cel",
    temp_fahrenheit_unit: str = "[degf]",
    cohort_ssids: set[str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load vital-sign rows from obs_clin.csv in the same long format as load_vitals().

    Drops rows whose unit is not in obs_clin_accepted_units for that LOINC
    (if given) and converts Fahrenheit temperatures to Celsius.
    """
    vital_loincs = set(obs_clin_loinc_map.keys())

    chunks = []
    reader = pd.read_csv(
        csv_path,
        dtype={"ssid": str, "obsclin_code": str, "obsclin_result_unit": str},
        usecols=lambda c: c in set(_READ_COLS),
        chunksize=chunksize,
    )

    header_checked = False
    for chunk in reader:
        if not header_checked:
            missing = _REQUIRED_COLS - set(chunk.columns)
            if missing:
                raise ValueError(f"obs_clin.csv is missing columns: {missing}")
            header_checked = True

        chunk["ssid"] = chunk["ssid"].astype(str).str.strip()
        chunk["obsclin_code"] = chunk["obsclin_code"].astype(str).str.strip()

        # Early filter: only the vital-sign LOINC codes we need
        chunk = chunk[chunk["obsclin_code"].isin(vital_loincs)]
        if chunk.empty:
            continue

        if cohort_ssids is not None:
            chunk = chunk[chunk["ssid"].isin(cohort_ssids)]
            if chunk.empty:
                continue

        chunk["value"] = pd.to_numeric(chunk["obsclin_result_num"], errors="coerce")
        chunk = chunk.dropna(subset=["value"])
        if chunk.empty:
            continue

        # Unit filtering: drop rows with unaccepted units per LOINC
        if obs_clin_accepted_units is not None and "obsclin_result_unit" in chunk.columns:
            unit_lower = chunk["obsclin_result_unit"].astype(str).str.strip().str.lower()
            keep = pd.Series(True, index=chunk.index)
            for loinc, accepted in obs_clin_accepted_units.items():
                is_this = chunk["obsclin_code"] == loinc
                keep &= ~is_this | unit_lower.isin(accepted)
            chunk = chunk[keep]
            if chunk.empty:
                continue

        # Temperature conversion: [degF] → °C; 'Cel' rows kept as-is
        temp_mask = chunk["obsclin_code"] == temp_fahrenheit_loinc
        if temp_mask.any() and "obsclin_result_unit" in chunk.columns:
            unit_lower = chunk.loc[temp_mask, "obsclin_result_unit"].astype(str).str.strip().str.lower()
            f_mask = temp_mask & (chunk["obsclin_result_unit"].astype(str).str.strip().str.lower() == temp_fahrenheit_unit)
            chunk.loc[f_mask, "value"] = (chunk.loc[f_mask, "value"] - 32) * 5 / 9
            # Celsius rows already correct — no action needed

        chunk["CANONICAL_TEST"] = chunk["obsclin_code"].map(obs_clin_loinc_map)
        chunk = chunk.rename(columns={"obsclin_start_date": "measure_date"})

        if not chunk.empty:
            chunks.append(chunk[["ssid", "measure_date", "CANONICAL_TEST", "value"]])

    if not chunks:
        return pd.DataFrame(columns=["ssid", "measure_date", "CANONICAL_TEST", "value"])

    return pd.concat(chunks, ignore_index=True)
