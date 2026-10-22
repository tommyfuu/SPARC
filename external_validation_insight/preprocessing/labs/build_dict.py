"""Build the lab result dictionary for SPARC.

Output: {ssid: {date_offset: {canonical_test: value}}}, offset = days since enr_start_date.
"""

from __future__ import annotations

import pandas as pd

from shared.utils import compute_date_offset

_READ_COLS = ["ssid", "lab_loinc", "result_date", "result_num", "result_unit"]


def build_lab_dict(
    dfs_by_test: dict[str, pd.DataFrame],
    enrollment_dates: dict[str, pd.Timestamp],
) -> dict[str, dict[int, dict[str, float]]]:
    """Build {ssid: {date_offset: {test: value}}} from clean_labs() output.

    Skips patients without an enrollment date and NaN values.
    """
    enr_series = pd.Series(enrollment_dates)
    out: dict[str, dict[int, dict[str, float]]] = {}

    for canonical, df in dfs_by_test.items():
        # Vectorized enrollment date lookup
        ref_dates = df["ssid"].map(enr_series)
        valid = ref_dates.notna() & df["result_num"].notna()
        df2 = df[valid]
        if df2.empty:
            continue

        ref2 = ref_dates[valid]
        result_dates = pd.to_datetime(df2["result_date"], errors="coerce")
        valid2 = result_dates.notna()
        df2 = df2[valid2]
        ref2 = ref2[valid2]
        result_dates = result_dates[valid2]

        # Vectorized offset computation
        offsets = (result_dates - ref2).dt.days

        # Fast iteration over arrays (10-100x faster than iterrows)
        for ssid, offset, value in zip(df2["ssid"], offsets, df2["result_num"]):
            out.setdefault(ssid, {}).setdefault(int(offset), {})[canonical] = float(value)

    return out


def stream_build_lab_dict(
    csv_path: str,
    loinc_to_canonical: dict[str, str],
    enrollment_dates: dict[str, pd.Timestamp],
    canonical_to_accepted_units: dict[str, frozenset[str]] | None = None,
    sentinel_values: frozenset[float] | None = None,
    min_records: int = 100,
    chunksize: int = 500_000,
    canonical_unit_aliases: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Load, clean and build the lab dict in one chunked pass over the CSV.

    Memory-bounded alternative to load_labs/clean_labs/build_lab_dict (the
    ~50 GB file does not fit otherwise). Same filters and output structure;
    keeps the first value seen per (ssid, offset, test).
    """
    known_loincs = set(loinc_to_canonical)
    enr_series = pd.Series(enrollment_dates)

    out: dict[str, dict[int, dict[str, float]]] = {}
    counts: dict[str, int] = {}
    # Unit-filter accounting, accumulated across chunks and reported once at
    # the end. An aggregate per-chunk count previously hid the site F failure,
    # where ALT/AST/ALP each lost >90% of their rows to a malformed "uL" unit.
    unit_seen: dict[str, int] = {}
    unit_dropped: dict[str, int] = {}
    unit_offenders: dict[str, dict[str, int]] = {}
    unit_aliased: dict[str, int] = {}

    reader = pd.read_csv(
        csv_path,
        dtype={"ssid": str, "lab_loinc": str},
        usecols=lambda c: c in set(_READ_COLS),
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk["ssid"] = chunk["ssid"].astype(str).str.strip()
        chunk["lab_loinc"] = chunk["lab_loinc"].astype(str).str.strip()
        chunk = chunk[chunk["lab_loinc"].isin(known_loincs)]
        if chunk.empty:
            continue

        chunk["result_num"] = pd.to_numeric(chunk["result_num"], errors="coerce")
        chunk = chunk.dropna(subset=["result_num"])
        if sentinel_values:
            chunk = chunk[~chunk["result_num"].isin(sentinel_values)]
        if chunk.empty:
            continue

        chunk["CANONICAL_TEST"] = chunk["lab_loinc"].map(loinc_to_canonical)

        if canonical_to_accepted_units:
            unit_lower = chunk["result_unit"].astype(str).str.strip().str.lower()

            # Normalise known-malformed unit strings before comparison
            # (config.CANONICAL_UNIT_ALIASES).
            if canonical_unit_aliases:
                for canonical, aliases in canonical_unit_aliases.items():
                    is_this_test = chunk["CANONICAL_TEST"] == canonical
                    if not is_this_test.any():
                        continue
                    for raw, normalised in aliases.items():
                        target = is_this_test & (unit_lower == raw)
                        n_fixed = int(target.sum())
                        if n_fixed:
                            unit_lower = unit_lower.mask(target, normalised)
                            unit_aliased[canonical] = (
                                unit_aliased.get(canonical, 0) + n_fixed)

            keep = pd.Series(True, index=chunk.index)
            for canonical, accepted in canonical_to_accepted_units.items():
                is_this_test = chunk["CANONICAL_TEST"] == canonical
                n_seen = int(is_this_test.sum())
                if n_seen:
                    ok = unit_lower.isin(accepted)
                    bad = is_this_test & ~ok
                    unit_seen[canonical] = unit_seen.get(canonical, 0) + n_seen
                    n_bad = int(bad.sum())
                    if n_bad:
                        unit_dropped[canonical] = (
                            unit_dropped.get(canonical, 0) + n_bad)
                        off = unit_offenders.setdefault(canonical, {})
                        for u, n in unit_lower[bad].value_counts().items():
                            off[u] = off.get(u, 0) + int(n)
                    keep &= ~is_this_test | ok
            chunk = chunk[keep]
            if chunk.empty:
                continue

        ref_dates = chunk["ssid"].map(enr_series)
        valid = ref_dates.notna()
        chunk = chunk[valid]
        if chunk.empty:
            continue
        ref2 = ref_dates[valid]
        result_dates = pd.to_datetime(chunk["result_date"], errors="coerce")
        valid2 = result_dates.notna()
        chunk = chunk[valid2]
        ref2 = ref2[valid2]
        offsets = (result_dates[valid2] - ref2).dt.days

        for ssid, offset, canonical, value in zip(
            chunk["ssid"], offsets, chunk["CANONICAL_TEST"], chunk["result_num"]
        ):
            date_bucket = out.setdefault(ssid, {}).setdefault(int(offset), {})
            if canonical not in date_bucket:
                date_bucket[canonical] = float(value)
                counts[canonical] = counts.get(canonical, 0) + 1

    # Unit-filter report (accumulated over all chunks)
    if unit_aliased:
        for canonical in sorted(unit_aliased):
            print(f"  Unit alias: {canonical}: normalised "
                  f"{unit_aliased[canonical]:,} rows via CANONICAL_UNIT_ALIASES")
    if unit_dropped:
        total_dropped = sum(unit_dropped.values())
        print(f"  Unit filter: dropped {total_dropped:,} rows with unexpected units")
        for canonical in sorted(unit_dropped):
            n_drop = unit_dropped[canonical]
            n_seen = unit_seen.get(canonical, 0)
            if n_seen and n_drop / n_seen >= 0.01:
                top = sorted(unit_offenders.get(canonical, {}).items(),
                             key=lambda kv: -kv[1])[:3]
                print(f"    WARNING {canonical}: dropped {n_drop:,}/{n_seen:,} "
                      f"({100 * n_drop / n_seen:.1f}%) — offending units: "
                      f"{ {u: f'{n:,}' for u, n in top} }")

    # Step 6 equivalent: drop canonical tests with fewer than min_records
    # surviving entries anywhere in the whole file.
    drop_tests = {c for c, n in counts.items() if n < min_records}
    if drop_tests:
        print(f"  Dropping tests below min_records={min_records}: {sorted(drop_tests)}")
        for dates in out.values():
            for vals in dates.values():
                for t in drop_tests:
                    vals.pop(t, None)

    return out
