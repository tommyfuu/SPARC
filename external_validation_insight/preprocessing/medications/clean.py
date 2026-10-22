"""Clean prescriptions and split them into oral and non-oral DataFrames."""

from __future__ import annotations

import pandas as pd


def clean_medications(
    df: pd.DataFrame,
    cui_to_name: dict[str, str],
    oral_routes: frozenset[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve drug names and split prescriptions into (oral_df, nonoral_df).

    Rows with blank or unknown routes (OT, NI, UN) are dropped.
    """
    # Resolve drug names
    df["drug_name"] = df["rxnorm_cui"].map(cui_to_name).fillna(df["rxnorm_cui"])

    # Drop rows with unknown/blank route
    valid_route = df["rx_route"].str.upper().isin(
        oral_routes | _nonoral_routes(df, oral_routes)
    )
    df = df[valid_route].copy()

    oral_mask = df["rx_route"].str.upper().isin(oral_routes)
    oral_df = df[oral_mask].reset_index(drop=True)
    nonoral_df = df[~oral_mask].reset_index(drop=True)

    output_cols = [
        "ssid", "drug_name", "rx_start_date", "rx_end_date",
        "rx_dose_ordered", "rx_dose_ordered_unit", "rx_quantity",
        "rx_frequency", "rx_days_supply",
    ]
    return oral_df[output_cols], nonoral_df[output_cols]


def _nonoral_routes(df: pd.DataFrame, oral_routes: frozenset[str]) -> set[str]:
    """Return all route values in df that are not oral and not blank/OT."""
    excluded = oral_routes | {"OT", "NI", "UN", "", "NAN", "NONE"}
    routes = {r.upper() for r in df["rx_route"].dropna().unique()}
    return routes - excluded
