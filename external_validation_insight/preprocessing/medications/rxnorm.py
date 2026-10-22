"""Resolve RxNorm CUIs to generic drug names via the NIH RxNorm REST API.

Results are cached in a local JSON file (RXNORM_CACHE_FILE) so each CUI is fetched once.
"""

from __future__ import annotations

import json
import os
import time

import requests


_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
_REQUEST_DELAY = 0.05  # seconds between API calls (be polite to NIH servers)
_CONNECT_TIMEOUT = 15  # seconds for the connectivity probe (NIH TLS handshake can be slow)


def _load_cache(cache_path: str) -> dict[str, str]:
    """Load existing cache from disk, or return empty dict."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, str], cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _fetch_ingredient_name(cui: str) -> str | None:
    """Fetch the ingredient (IN) name for a CUI, uppercased.

    Falls back to the CUI's own name, then None if both lookups fail.
    """
    try:
        # Step 1: get ingredient (IN) concept
        resp = requests.get(
            f"{_BASE_URL}/rxcui/{cui}/related.json",
            params={"tty": "IN"},
            timeout=10,
        )
        if resp.status_code == 200:
            groups = (
                resp.json()
                .get("relatedGroup", {})
                .get("conceptGroup", [])
            )
            for group in groups:
                if group.get("tty") == "IN":
                    members = group.get("conceptProperties", [])
                    if members:
                        return members[0]["name"].upper()
    except Exception:
        pass

    time.sleep(_REQUEST_DELAY)

    try:
        # Step 2: fall back to the CUI's own name
        resp2 = requests.get(
            f"{_BASE_URL}/rxcui/{cui}/properties.json",
            timeout=10,
        )
        if resp2.status_code == 200:
            name = resp2.json().get("properties", {}).get("name", "")
            if name:
                return name.upper()
    except Exception:
        pass

    return None


def _api_reachable() -> bool:
    """Return True if the RxNorm API is reachable."""
    try:
        requests.get(
            f"{_BASE_URL}/rxcui/607044/properties.json",
            timeout=_CONNECT_TIMEOUT,
        )
        return True
    except Exception:
        return False


def resolve_rxnorm_names(
    cuis: set[str],
    cache_path: str,
    verbose: bool = True,
) -> dict[str, str]:
    """Map RxNorm CUIs to uppercase generic drug names, using the cache first.

    Unresolved CUIs (or all uncached ones if the API is unreachable) map to the
    CUI string itself.
    """
    cache = _load_cache(cache_path)
    uncached = [c for c in cuis if c not in cache]

    if not uncached:
        return {c: cache[c] for c in cuis if c in cache}

    if verbose:
        print(f"  Resolving {len(uncached)} new RxNorm CUIs "
              f"({len(cuis) - len(uncached)} already cached) ...")

    # Single connectivity probe before attempting bulk lookups
    if not _api_reachable():
        print("  WARNING: RxNorm API unreachable (no outbound internet?). "
              "Falling back to RxNorm CUI strings as drug names.")
        for cui in uncached:
            cache[cui] = cui  # CUI string as fallback name
        _save_cache(cache, cache_path)
        return {c: cache[c] for c in cuis if c in cache}

    for i, cui in enumerate(uncached):
        name = _fetch_ingredient_name(cui)
        cache[cui] = name if name is not None else cui  # fallback: CUI itself
        time.sleep(_REQUEST_DELAY)

        if verbose and (i + 1) % 50 == 0:
            print(f"    ... resolved {i + 1}/{len(uncached)}")

    _save_cache(cache, cache_path)
    if verbose:
        print(f"  Cache updated: {cache_path}")

    return {c: cache[c] for c in cuis if c in cache}
