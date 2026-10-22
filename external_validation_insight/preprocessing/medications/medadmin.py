"""Build a non-oral medication supplement from med_admin.csv (actual administrations).

Merged into the non-oral dict. Entry: [drug_name, start_str, stop_str, route, dose,
dose_unit, nan, '', nan, 'medadmin']; only index 0 is used downstream.
"""

from __future__ import annotations

import math

import pandas as pd

_NAN = float("nan")

_NEEDED_COLS = frozenset({
    "ssid",
    "medadmin_code",
    "medadmin_route",
    "medadmin_start_date",
    "medadmin_stop_date",
    "medadmin_dose_admin",
    "medadmin_dose_admin_unit",
})


def load_medadmin_nonoral(
    med_admin_csv: str,
    oral_routes: frozenset[str],
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read med_admin.csv and return rows whose route is not explicitly oral.

    Blank/unknown routes are kept, since med_admin records actual administrations.
    """
    # Routes that definitively mean oral/enteral — exclude these
    _ORAL_UPPER = frozenset(r.upper() for r in oral_routes)

    kept: list[pd.DataFrame] = []
    chunks_read = rows_kept = 0

    for chunk in pd.read_csv(
        med_admin_csv,
        dtype=str,
        chunksize=chunksize,
        usecols=lambda c: c in _NEEDED_COLS,
    ):
        chunks_read += 1
        if chunks_read % 20 == 0:
            total_read = chunks_read * chunksize
            print(f"    ...{total_read // 1_000_000}M rows read, {rows_kept:,} kept so far")

        if "medadmin_route" not in chunk.columns:
            continue

        route_norm = (
            chunk["medadmin_route"]
            .fillna("")
            .str.strip()
            .str.strip('"')
            .str.upper()
        )
        # Keep anything that is NOT explicitly an oral route
        mask = ~route_norm.isin(_ORAL_UPPER)
        if mask.any():
            sub = chunk[mask].copy()
            sub["medadmin_route"] = route_norm[mask]
            kept.append(sub)
            rows_kept += mask.sum()

    if not kept:
        return pd.DataFrame()
    return pd.concat(kept, ignore_index=True)


def build_medadmin_dict(
    df: pd.DataFrame,
    enrollment_dates: dict,      # {ssid: pd.Timestamp}
    cui_to_name: dict[str, str], # {cui: drug_name_upper}
) -> dict[str, dict[int, list]]:
    """Build the supplement dict {ssid: {start_offset: [entry, ...]}} from med_admin rows.

    Drops unmapped CUIs; a missing/bad stop date falls back to the start date.
    """
    if df.empty:
        return {}

    df = df.copy()

    # ── CUI → drug name ───────────────────────────────────────────────────────
    if "medadmin_code" in df.columns:
        df["medadmin_code"] = df["medadmin_code"].str.strip().str.strip('"')
    else:
        return {}
    df["drug_name"] = df["medadmin_code"].map(cui_to_name)
    df = df[df["drug_name"].notna()]
    if df.empty:
        return {}

    # ── Filter to enrolled patients ────────────────────────────────────────────
    enr = pd.Series(enrollment_dates)
    ref = df["ssid"].map(enr)
    df = df[ref.notna()].copy()
    ref = ref[ref.notna()]
    if df.empty:
        return {}

    # ── Compute start offset ──────────────────────────────────────────────────
    start_col = "medadmin_start_date"
    start_strs = df[start_col].str.strip().str.strip('"') if start_col in df.columns else pd.Series("", index=df.index)
    start_dates = pd.to_datetime(start_strs, errors="coerce")
    valid = start_dates.notna()
    df = df[valid].copy()
    ref = ref[valid]
    start_strs = start_strs[valid]
    start_dates = start_dates[valid]
    start_offsets = (start_dates - ref).dt.days.astype(int)

    # ── Stop date string (fall back to start date if missing/unparseable) ─────
    stop_col = "medadmin_stop_date"
    if stop_col in df.columns:
        stop_strs_raw = df[stop_col].str.strip().str.strip('"')
        stop_dates = pd.to_datetime(stop_strs_raw, errors="coerce")
        stop_strs = stop_strs_raw.where(stop_dates.notna(), other=start_strs)
    else:
        stop_strs = start_strs

    # ── Dose fields ───────────────────────────────────────────────────────────
    dose_col = "medadmin_dose_admin"
    unit_col = "medadmin_dose_admin_unit"
    doses = pd.to_numeric(df[dose_col], errors="coerce") if dose_col in df.columns else pd.Series(_NAN, index=df.index)
    dose_units = df[unit_col].fillna("").str.strip() if unit_col in df.columns else pd.Series("", index=df.index)

    # ── Build output dict ─────────────────────────────────────────────────────
    out: dict[str, dict[int, list]] = {}
    for ssid, start_off, start_str, stop_str, route, drug_name, dose, dose_unit in zip(
        df["ssid"],
        start_offsets,
        start_strs,
        stop_strs,
        df.get("medadmin_route", pd.Series("", index=df.index)),
        df["drug_name"],
        doses,
        dose_units,
    ):
        dose_val = float(dose) if pd.notna(dose) else _NAN
        entry = [drug_name, start_str, stop_str, route,
                 dose_val, dose_unit, _NAN, "", _NAN, "medadmin"]
        out.setdefault(ssid, {}).setdefault(int(start_off), []).append(entry)

    return out


def stream_build_medadmin_dict(
    csv_path: str,
    enrollment_dates: dict,
    oral_routes: frozenset[str],
    chunksize: int = 500_000,
) -> tuple[dict[str, dict[int, list]], set[str]]:
    """Build the med_admin supplement dict in one chunked pass (low memory).

    Entries store the raw CUI as the name; call substitute_drug_names() later.
    Returns (supplement_dict, unique_cuis).
    """
    oral_upper = frozenset(r.upper() for r in oral_routes)
    enr_series = pd.Series(enrollment_dates)

    out: dict[str, dict[int, list]] = {}
    unique_cuis: set[str] = set()
    chunks_read = rows_kept = 0

    for chunk in pd.read_csv(
        csv_path,
        dtype=str,
        chunksize=chunksize,
        usecols=lambda c: c in _NEEDED_COLS,
    ):
        chunks_read += 1
        if chunks_read % 20 == 0:
            total_read = chunks_read * chunksize
            print(f"    ...{total_read // 1_000_000}M rows read, {rows_kept:,} kept so far")

        if "medadmin_route" not in chunk.columns or "medadmin_code" not in chunk.columns:
            continue

        route_norm = (
            chunk["medadmin_route"].fillna("").str.strip().str.strip('"').str.upper()
        )
        chunk = chunk[~route_norm.isin(oral_upper)].copy()
        if chunk.empty:
            continue
        chunk["medadmin_route"] = route_norm[chunk.index]

        chunk["medadmin_code"] = chunk["medadmin_code"].str.strip().str.strip('"')

        ref = chunk["ssid"].map(enr_series)
        chunk = chunk[ref.notna()]
        if chunk.empty:
            continue
        ref = ref[ref.notna()]

        start_col = "medadmin_start_date"
        start_strs = (
            chunk[start_col].str.strip().str.strip('"')
            if start_col in chunk.columns else pd.Series("", index=chunk.index)
        )
        start_dates = pd.to_datetime(start_strs, errors="coerce")
        valid = start_dates.notna()
        chunk = chunk[valid]
        if chunk.empty:
            continue
        ref = ref[valid]
        start_strs = start_strs[valid]
        start_dates = start_dates[valid]
        start_offsets = (start_dates - ref).dt.days.astype(int)

        stop_col = "medadmin_stop_date"
        if stop_col in chunk.columns:
            stop_strs_raw = chunk[stop_col].str.strip().str.strip('"')
            stop_dates = pd.to_datetime(stop_strs_raw, errors="coerce")
            stop_strs = stop_strs_raw.where(stop_dates.notna(), other=start_strs)
        else:
            stop_strs = start_strs

        dose_col = "medadmin_dose_admin"
        unit_col = "medadmin_dose_admin_unit"
        doses = pd.to_numeric(chunk[dose_col], errors="coerce") if dose_col in chunk.columns else pd.Series(_NAN, index=chunk.index)
        dose_units = chunk[unit_col].fillna("").str.strip() if unit_col in chunk.columns else pd.Series("", index=chunk.index)

        rows_kept += len(chunk)

        for ssid, start_off, start_str, stop_str, route, cui, dose, dose_unit in zip(
            chunk["ssid"], start_offsets, start_strs, stop_strs,
            chunk["medadmin_route"], chunk["medadmin_code"], doses, dose_units,
        ):
            unique_cuis.add(cui)
            dose_val = float(dose) if pd.notna(dose) else _NAN
            entry = [cui, start_str, stop_str, route, dose_val, dose_unit, _NAN, "", _NAN, "medadmin"]
            out.setdefault(ssid, {}).setdefault(int(start_off), []).append(entry)

    return out, unique_cuis


def merge_into_nonoral(
    nonoral_dict: dict[str, dict[int, list]],
    supplement: dict[str, dict[int, list]],
) -> tuple[dict[str, dict[int, list]], int, int]:
    """Merge supplement into nonoral_dict in place; return (dict, n_new_patients, n_new_entries)."""
    new_pts = new_entries = 0
    for ssid, date_dict in supplement.items():
        if ssid not in nonoral_dict:
            nonoral_dict[ssid] = {}
            new_pts += 1
        for offset, entries in date_dict.items():
            nonoral_dict[ssid].setdefault(offset, []).extend(entries)
            new_entries += len(entries)
    return nonoral_dict, new_pts, new_entries
