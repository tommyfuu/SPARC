"""Shared utilities for Insight preprocessing.

Reference dates, date offsets, value cleaning helpers, and streaming
combined-JSON output.
"""

from __future__ import annotations

import gc
import json
import locale

import pandas as pd


# ── Invalid value filter ──────────────────────────────────────────────────────
# Strings that represent missing/invalid measurements. Replicated from the
# existing SPARC_data_preprocessing pipeline for consistency.
INVALID_VALUES: frozenset[str] = frozenset(
    {
        "",
        " ",
        ".",
        "+",
        "+++",
        "++++",
        "nan",
        "nan\n",
        "N/A",
        "NA",
        "n/a",
        "NM",
        "NI",
        "OT",
        "UN",
        "not measured",
        "NOT MEASURED",
        "Not Measured",
        "not applicable",
        "Not Applicable",
        "NOT APPLICABLE",
        "unknown",
        "Unknown",
        "UNKNOWN",
        "clotted",
        "CLOTTED",
        "Clotted",
        "hemolyzed",
        "Hemolyzed",
        "HEMOLYZED",
        "lipemic",
        "Lipemic",
        "LIPEMIC",
        "icteric",
        "Icteric",
        "ICTERIC",
        "QNS",
        "qns",
        "CANCELED",
        "Canceled",
        "canceled",
        "Error-Disregard",
        "FLAG",
        "FOOTNOTE",
        "see note",
        "See Note",
        "SEE NOTE",
        "#FORCED",
        "not seen",
        "Not Seen",
        "NOT SEEN",
    }
)


# ── Reference dates ───────────────────────────────────────────────────────────

def load_enrollment_dates(enrollment_csv_path: str) -> dict[str, pd.Timestamp]:
    """Return {ssid: enr_start_date} using the earliest enrollment start per patient."""
    df = pd.read_csv(enrollment_csv_path, dtype={"ssid": str})
    df["enr_start_date"] = pd.to_datetime(df["enr_start_date"], errors="coerce")
    df = df.dropna(subset=["enr_start_date"])
    earliest = df.groupby("ssid")["enr_start_date"].min()
    return earliest.to_dict()


def load_first_lab_dates(
    lab_csv_path: str,
    chunksize: int = 500_000,
) -> dict[str, pd.Timestamp]:
    """Return {ssid: earliest result_date} from lab_result_cm.csv.

    This is the time-zero for all feature offsets (CAIA-compatible).
    """
    mins: dict[str, pd.Timestamp] = {}
    chunks_read = 0
    for chunk in pd.read_csv(
        lab_csv_path,
        dtype=str,
        chunksize=chunksize,
        usecols=["ssid", "result_date"],
    ):
        chunks_read += 1
        if chunks_read % 100 == 0:
            print(f"    ...{chunks_read * chunksize // 1_000_000}M rows scanned, "
                  f"{len(mins):,} patients with labs so far")
        chunk["result_date"] = pd.to_datetime(chunk["result_date"], errors="coerce")
        chunk = chunk.dropna(subset=["result_date"])
        for ssid, date in zip(chunk["ssid"], chunk["result_date"]):
            if ssid not in mins or date < mins[ssid]:
                mins[ssid] = date
    return mins


def build_reference_dates(
    first_lab_dates: dict[str, pd.Timestamp],
    enrollment_dates: dict[str, pd.Timestamp],
) -> dict[str, pd.Timestamp]:
    """Return reference dates: first lab date, else enrollment start date."""
    ref = dict(enrollment_dates)
    ref.update(first_lab_dates)  # first_lab_dates takes precedence
    return ref


# ── Date offset ───────────────────────────────────────────────────────────────

def compute_date_offset(date_val: str | pd.Timestamp, reference_date: pd.Timestamp) -> int:
    """Return days from reference_date to date_val (negative if before)."""
    ts = pd.Timestamp(date_val) if not isinstance(date_val, pd.Timestamp) else date_val
    return int((ts - reference_date).days)


# ── Value cleaning helpers ─────────────────────────────────────────────────────

def strip_and_filter(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Strip whitespace from value_col and drop rows whose value is in INVALID_VALUES."""
    df = df.copy()
    df[value_col] = df[value_col].astype(str).str.strip()
    return df[~df[value_col].isin(INVALID_VALUES)].reset_index(drop=True)


def deduplicate(
    df: pd.DataFrame,
    group_cols: list[str],
    sort_col: str,
    keep: str = "first",
) -> pd.DataFrame:
    """Keep one row per group_cols after sorting by sort_col (keep='first' or 'last')."""
    df = df.sort_values(sort_col)
    return df.drop_duplicates(subset=group_cols, keep=keep).reset_index(drop=True)


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convert a string series to float (NaN if unparseable).

    Tries locale-aware parsing first, then falls back to pd.to_numeric.
    """
    try:
        locale.setlocale(locale.LC_NUMERIC, "")
        converted = series.map(
            lambda x: locale.atof(str(x)) if pd.notna(x) and str(x).strip() not in INVALID_VALUES else float("nan")
        )
        if converted.notna().any():
            return converted
    except (ValueError, locale.Error):
        pass
    return pd.to_numeric(series, errors="coerce")


# ── Streaming combined-JSON assembly ──────────────────────────────────────────
# Ported from the CAIA/DFCI preprocessing pipeline (sparc_preprocessing_042726.py)
# so this repo's combined JSON output is structured the same way: one top-level
# dict, each key's value converted and written independently so the whole
# combined structure is never held in memory at once.

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return {"__set__": list(o)}
        return super().default(o)


def stringify_keys(obj):
    """Recursively cast dict keys to str, including nested ones."""
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [stringify_keys(x) for x in obj]
    return obj


def stream_dump_combined(json_path: str, sources: list[tuple[str, object]],
                          encoder_cls: type[json.JSONEncoder] | None = None) -> None:
    """Write (key, value) sources as one JSON dict, one key at a time, to limit memory.

    Each value is converted with stringify_keys and freed after writing.
    """
    encoder_cls = encoder_cls or CustomEncoder
    with open(json_path, "w") as f:
        f.write("{")
        first = True
        for key, source in sources:
            if not first:
                f.write(",")
            first = False
            converted = stringify_keys(source)
            f.write(json.dumps(key) + ":")
            json.dump(converted, f, cls=encoder_cls)
            del converted
            gc.collect()
            print(f"  ✓ Streamed {key}")
        f.write("}")
