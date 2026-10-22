"""Clean lab results and map LOINC codes to canonical test names.

Output: {canonical_test: DataFrame(ssid, result_date, CANONICAL_TEST, result_num)}.
"""

from __future__ import annotations

import pandas as pd

from shared.utils import deduplicate


def clean_labs(
    df: pd.DataFrame,
    loinc_to_canonical: dict[str, str],
    canonical_to_accepted_units: dict[str, frozenset[str]] | None = None,
    min_records: int = 100,
    sentinel_values: frozenset[float] | None = None,
    canonical_unit_aliases: dict[str, dict[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Clean lab data and group it by canonical test name.

    Drops null and sentinel results, unmapped LOINCs, and unaccepted units
    (after alias normalisation). Keeps the last result per
    (ssid, lab_loinc, result_date) and drops tests below min_records.
    """
    # Step 1: drop null result_num
    df["result_num"] = pd.to_numeric(df["result_num"], errors="coerce")
    df = df.dropna(subset=["result_num"])

    # Step 1b: drop null/redacted-value sentinels (e.g. 9999999). Left in,
    # these dominate the per-test mean/std used for normalization even at
    # a fraction of a percent of rows, since a single such value is orders
    # of magnitude larger than any real result.
    if sentinel_values:
        dropped = df["result_num"].isin(sentinel_values).sum()
        if dropped > 0:
            print(f"  Sentinel filter: dropped {dropped:,} rows with result_num in {set(sentinel_values)}")
        df = df[~df["result_num"].isin(sentinel_values)]
        if df.empty:
            return {}

    # Step 2: keep only LOINC codes we can map
    df = df[df["lab_loinc"].isin(loinc_to_canonical)]
    if df.empty:
        return {}

    # Step 3: map to canonical name
    df["CANONICAL_TEST"] = df["lab_loinc"].map(loinc_to_canonical)

    # Step 4: unit filtering
    if canonical_to_accepted_units is not None and "result_unit" in df.columns:
        unit_lower = df["result_unit"].astype(str).str.strip().str.lower()

        # Step 4a: normalise known-malformed unit strings before comparison.
        # See config.CANONICAL_UNIT_ALIASES — e.g. site F writes the liver
        # enzymes as "uL" (a volume) where it means "U/L".
        if canonical_unit_aliases:
            for canonical, aliases in canonical_unit_aliases.items():
                is_this_test = df["CANONICAL_TEST"] == canonical
                if not is_this_test.any():
                    continue
                for raw, normalised in aliases.items():
                    target = is_this_test & (unit_lower == raw)
                    n_fixed = int(target.sum())
                    if n_fixed:
                        unit_lower = unit_lower.mask(target, normalised)
                        print(f"  Unit alias: {canonical}: normalised "
                              f"{n_fixed:,} rows {raw!r} -> {normalised!r}")

        keep = pd.Series(True, index=df.index)
        for canonical, accepted in canonical_to_accepted_units.items():
            is_this_test = df["CANONICAL_TEST"] == canonical
            unit_ok = unit_lower.isin(accepted)
            # Drop rows that are this test but have an unaccepted unit
            keep &= ~is_this_test | unit_ok

        dropped = (~keep).sum()
        if dropped > 0:
            print(f"  Unit filter: dropped {dropped:,} rows with unexpected units")
            # Per-test breakdown: an aggregate count hid the site F failure,
            # where one test losing >90% of its rows looked like ordinary noise
            # against the total. Report any test losing a material share.
            for canonical in sorted(canonical_to_accepted_units):
                is_this_test = df["CANONICAL_TEST"] == canonical
                n_test = int(is_this_test.sum())
                if n_test == 0:
                    continue
                n_drop = int((is_this_test & ~keep).sum())
                if n_drop and n_drop / n_test >= 0.01:
                    bad = (unit_lower[is_this_test & ~keep]
                           .value_counts().head(3).to_dict())
                    print(f"    WARNING {canonical}: dropped {n_drop:,}/{n_test:,} "
                          f"({100 * n_drop / n_test:.1f}%) — offending units: {bad}")
        df = df[keep]
        if df.empty:
            return {}

    # Step 5: deduplicate — keep last per (ssid, lab_loinc, result_date)
    df = deduplicate(df, group_cols=["ssid", "lab_loinc", "result_date"], sort_col="result_date")

    # Step 6: filter by record count per canonical test
    counts = df.groupby("CANONICAL_TEST")["ssid"].transform("count")
    df = df[counts >= min_records]

    result: dict[str, pd.DataFrame] = {}
    for canonical, group in df.groupby("CANONICAL_TEST"):
        result[canonical] = group[["ssid", "result_date", "CANONICAL_TEST", "result_num"]].reset_index(drop=True)

    return result
