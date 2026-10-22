"""Build oral and non-oral medication dicts in SPARC format.

Oral: {ssid: {start_offset: [[name, class, type, end_offset], ...]}}; non-oral adds
dose, unit, dispense_cnt, qty, freq, duration. class/type are '' and dispense_cnt
is NaN (not in PCORnet). start_offset = (rx_start_date - enrollment date).days.
"""

from __future__ import annotations

import pandas as pd

_NAN = float("nan")


def _prepare(
    df: pd.DataFrame,
    enr_series: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Keep rows with an enrollment date and valid start date; compute day offsets.

    end_offsets is NaN where the end date is missing or unparseable.
    """
    ref_dates = df["ssid"].map(enr_series)
    valid = ref_dates.notna()
    df2 = df[valid].copy()
    ref2 = ref_dates[valid]

    start_dates = pd.to_datetime(df2["rx_start_date"], errors="coerce")
    valid2 = start_dates.notna()
    df2 = df2[valid2].copy()
    ref2 = ref2[valid2]
    start_dates = start_dates[valid2]

    start_offsets = (start_dates - ref2).dt.days.astype(int)

    end_dates = pd.to_datetime(df2["rx_end_date"], errors="coerce")
    end_offsets_days = (end_dates - ref2).dt.days
    # NaT end dates → NaN float (matches existing pipeline)
    end_offsets = end_offsets_days.where(end_dates.notna(), other=_NAN)

    return df2, ref2, start_offsets, end_offsets


def build_oral_dict(
    oral_df: pd.DataFrame,
    enrollment_dates: dict[str, pd.Timestamp],
) -> dict[str, dict[int, list]]:
    """Build {ssid: {start_offset: [[name, class, type, end_offset], ...]}}."""
    enr_series = pd.Series(enrollment_dates)
    df2, _, start_offsets, end_offsets = _prepare(oral_df, enr_series)

    out: dict[str, dict[int, list]] = {}
    for ssid, start_off, end_off, drug_name in zip(
        df2["ssid"], start_offsets, end_offsets, df2["drug_name"]
    ):
        end_val = int(end_off) if pd.notna(end_off) else _NAN
        record = [drug_name, "", "", end_val]
        out.setdefault(ssid, {}).setdefault(int(start_off), []).append(record)

    return out


def build_nonoral_dict(
    nonoral_df: pd.DataFrame,
    enrollment_dates: dict[str, pd.Timestamp],
) -> dict[str, dict[int, list]]:
    """Build the non-oral dict (oral fields plus dose, unit, disp, qty, freq, dur)."""
    enr_series = pd.Series(enrollment_dates)
    df2, _, start_offsets, end_offsets = _prepare(nonoral_df, enr_series)

    # Vectorize scalar field extraction
    doses = pd.to_numeric(df2["rx_dose_ordered"], errors="coerce")
    dose_units = _vec_str_or_empty(df2["rx_dose_ordered_unit"])
    qtys = pd.to_numeric(df2["rx_quantity"], errors="coerce")
    freqs = _vec_str_or_empty(df2["rx_frequency"])
    durations = _vec_str_or_empty(df2["rx_days_supply"])

    out: dict[str, dict[int, list]] = {}
    for ssid, start_off, end_off, drug_name, dose, dose_unit, qty, freq, duration in zip(
        df2["ssid"], start_offsets, end_offsets, df2["drug_name"],
        doses, dose_units, qtys, freqs, durations,
    ):
        end_val = int(end_off) if pd.notna(end_off) else _NAN
        dose_val = float(dose) if pd.notna(dose) else _NAN
        qty_val = float(qty) if pd.notna(qty) else _NAN

        record = [
            drug_name, "", "", end_val,
            dose_val, dose_unit, _NAN, qty_val, freq, duration,
        ]
        out.setdefault(ssid, {}).setdefault(int(start_off), []).append(record)

    return out


def _vec_str_or_empty(series: pd.Series) -> pd.Series:
    """Convert to stripped str, with '' for NaN/'none'/'nan'."""
    result = series.astype(str).str.strip()
    bad = series.isna() | result.str.lower().isin({"nan", "none", ""})
    return result.where(~bad, other="")


_EXCLUDED_ROUTES = frozenset({"OT", "NI", "UN", "", "NAN", "NONE"})
_READ_COLS = [
    "ssid", "rxnorm_cui", "rx_start_date", "rx_end_date", "rx_route",
    "rx_dose_ordered", "rx_dose_ordered_unit", "rx_quantity",
    "rx_frequency", "rx_days_supply",
]


def substitute_drug_names(meds_dict: dict[str, dict[int, list]], cui_to_name: dict[str, str]) -> None:
    """Replace each entry's CUI (index 0) with its drug name, in place.

    Unmapped CUIs are kept as-is.
    """
    for date_dict in meds_dict.values():
        for entries in date_dict.values():
            for entry in entries:
                entry[0] = cui_to_name.get(entry[0], entry[0])


def stream_build_prescribing_dicts(
    csv_path: str,
    enrollment_dates: dict[str, pd.Timestamp],
    oral_routes: frozenset[str],
    chunksize: int = 500_000,
) -> tuple[dict[str, dict[int, list]], dict[str, dict[int, list]], set[str]]:
    """Build oral and non-oral dicts from prescribing.csv in one chunked pass.

    Avoids holding all rows in memory. Entries store the raw CUI as the name;
    call substitute_drug_names() once all CUIs are resolved.
    Returns (oral_dict, nonoral_dict, unique_cuis).
    """
    oral_routes_upper = frozenset(r.upper() for r in oral_routes)
    enr_series = pd.Series(enrollment_dates)

    oral_out: dict[str, dict[int, list]] = {}
    nonoral_out: dict[str, dict[int, list]] = {}
    unique_cuis: set[str] = set()

    reader = pd.read_csv(
        csv_path,
        dtype={"ssid": str, "rxnorm_cui": str},
        usecols=lambda c: c in set(_READ_COLS),
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk["ssid"] = chunk["ssid"].astype(str).str.strip()
        chunk["rxnorm_cui"] = chunk["rxnorm_cui"].astype(str).str.strip()
        chunk["rx_route"] = chunk["rx_route"].astype(str).str.strip()

        chunk = chunk[
            chunk["rxnorm_cui"].notna()
            & ~chunk["rxnorm_cui"].isin({"nan", "", "None"})
        ]
        chunk = chunk[
            chunk["rx_start_date"].notna()
            & ~chunk["rx_start_date"].astype(str).isin({"nan", "", "None"})
        ]
        if chunk.empty:
            continue

        route_upper = chunk["rx_route"].str.upper()
        chunk = chunk[~route_upper.isin(_EXCLUDED_ROUTES) | route_upper.isin(oral_routes_upper)]
        if chunk.empty:
            continue
        route_upper = route_upper.loc[chunk.index]

        ref_dates = chunk["ssid"].map(enr_series)
        valid = ref_dates.notna()
        chunk = chunk[valid]
        if chunk.empty:
            continue
        ref2 = ref_dates[valid]
        route_upper = route_upper[valid]

        start_dates = pd.to_datetime(chunk["rx_start_date"], errors="coerce")
        valid2 = start_dates.notna()
        chunk = chunk[valid2]
        ref2 = ref2[valid2]
        route_upper = route_upper[valid2]
        start_dates = start_dates[valid2]
        if chunk.empty:
            continue

        start_offsets = (start_dates - ref2).dt.days
        end_dates = pd.to_datetime(chunk["rx_end_date"], errors="coerce")
        end_offsets = (end_dates - ref2).dt.days

        doses = pd.to_numeric(chunk["rx_dose_ordered"], errors="coerce")
        dose_units = _vec_str_or_empty(chunk["rx_dose_ordered_unit"])
        qtys = pd.to_numeric(chunk["rx_quantity"], errors="coerce")
        freqs = _vec_str_or_empty(chunk["rx_frequency"])
        durations = _vec_str_or_empty(chunk["rx_days_supply"])

        for ssid, cui, is_oral, start_off, end_off, dose, dose_unit, qty, freq, duration in zip(
            chunk["ssid"], chunk["rxnorm_cui"], route_upper.isin(oral_routes_upper),
            start_offsets, end_offsets, doses, dose_units, qtys, freqs, durations,
        ):
            unique_cuis.add(cui)
            end_val = int(end_off) if pd.notna(end_off) else _NAN
            start_off_i = int(start_off)

            if is_oral:
                record = [cui, "", "", end_val]
                oral_out.setdefault(ssid, {}).setdefault(start_off_i, []).append(record)
            else:
                dose_val = float(dose) if pd.notna(dose) else _NAN
                qty_val = float(qty) if pd.notna(qty) else _NAN
                record = [cui, "", "", end_val, dose_val, dose_unit, _NAN, qty_val, freq, duration]
                nonoral_out.setdefault(ssid, {}).setdefault(start_off_i, []).append(record)

    return oral_out, nonoral_out, unique_cuis
