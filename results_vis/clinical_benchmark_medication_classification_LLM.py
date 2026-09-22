"""
Classify every unique medication in the MSK non-oral meds dict into fields
usable by Lyman / Dranitsaris / FENCE. Produces a reusable {med_name: {...}} dict.

Run once; the output pickle is then loaded by the benchmark scripts.
"""
# Put the repo root, data/ and models/ on sys.path, from any working directory.
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
import pickle
import json
import time
import boto3
from pydantic import BaseModel, Field
from typing import List, Optional
from llm_utils import call_llm, extract_json_robust

# ---------------------------------------------------------------------------
# 1. Load the meds dict and extract UNIQUE medication names
# ---------------------------------------------------------------------------
MEDS_PATH = (sparc_paths.DATA_ROOT + '/MSK/postbc_DMP_ID_to_DATES_to_nonoral_drug_cats_wdosage_deidentified_061925.pickle')
with open(MEDS_PATH, "rb") as f:
    meds = pickle.load(f)   # {DMP_ID: {date: [ [name, class, regimen, cat, dose, unit, ...], ... ]}}

DRUG_NAME_IDX = 0
unique_meds = set()
for dmp_id, date_to_entries in meds.items():
    for date, entries in date_to_entries.items():
        for e in entries:
            try:
                nm = str(e[DRUG_NAME_IDX]).strip()
                if nm:
                    unique_meds.add(nm.upper())
            except (IndexError, TypeError):
                continue
unique_meds = sorted(unique_meds)
print(f"{len(unique_meds)} unique medications to classify")

# ---------------------------------------------------------------------------
# 2. Classification schema (fields needed across all 3 benchmarks)
# ---------------------------------------------------------------------------
class MedClass(BaseModel):
    is_chemotherapy: bool = Field(default=False,
        description="True if this is a cytotoxic chemotherapy / antineoplastic agent")
    chemo_class: str = Field(default="none",
        description="one of: taxane, platinum, nonplatinum_alkylating, topoisomerase, "
                    "antimetabolite, vinca, other_chemo, none")
    is_immunosuppressant: bool = Field(default=False,
        description="True if a NON-chemo immunosuppressant (corticosteroid, calcineurin "
                    "inhibitor, antimetabolite immunosuppressant, mTOR inhibitor, etc.)")
    is_targeted_or_immunotherapy: bool = Field(default=False,
        description="True if a targeted therapy or immune checkpoint inhibitor")
    notes: str = Field(default="")

PROMPT_TEMPLATE = """You are a clinical pharmacology expert classifying an oncology medication.

Medication name: "{med}"

Classify it and respond with ONLY a JSON object (no prose) with these exact keys:
- "is_chemotherapy": true/false (cytotoxic chemotherapy / antineoplastic agent)
- "chemo_class": one of ["taxane","platinum","nonplatinum_alkylating","topoisomerase","antimetabolite","vinca","other_chemo","none"] (use "none" if not chemo)
- "is_immunosuppressant": true/false (NON-chemo immunosuppressant: corticosteroids like dexamethasone/prednisone, calcineurin inhibitors like tacrolimus/cyclosporine, azathioprine, mycophenolate, sirolimus). Do NOT mark cytotoxic chemo as immunosuppressant here.
- "is_targeted_or_immunotherapy": true/false (targeted therapy or immune checkpoint inhibitor)
- "notes": brief note (mechanism/drug class), max 10 words

Respond with only the JSON object."""

# ---------------------------------------------------------------------------
# 3. Classify each unique med (with simple caching + retry)
# ---------------------------------------------------------------------------
client = boto3.client("bedrock-runtime", region_name="us-east-1")  # adjust region
MODEL_ID = "qwen.qwen3-next-80b-a3b"   # was the dead Claude ID

med_classification = {}
for i, med in enumerate(unique_meds):
    prompt = PROMPT_TEMPLATE.format(med=med)
    try:
        parsed, usage = call_llm(client, prompt, model_id=MODEL_ID)
        # call_llm already returns parsed JSON; validate/fill defaults via schema
        result = extract_json_robust(json.dumps(parsed), schema=MedClass) if parsed else MedClass().model_dump()
    except Exception as e:
        print(f"  [{med}] error: {e} -> defaults")
        result = MedClass().model_dump()
    med_classification[med] = result
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(unique_meds)} classified")
    time.sleep(0.1)  # gentle rate limit

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
OUT_PATH = sparc_paths.DATA_ROOT + '/MSK/med_classification_llm.pickle'
with open(OUT_PATH, "wb") as f:
    pickle.dump(med_classification, f)
print(f"\nSaved {len(med_classification)} classifications to {OUT_PATH}")

# quick summary
n_chemo = sum(1 for v in med_classification.values() if v.get("is_chemotherapy"))
n_immuno = sum(1 for v in med_classification.values() if v.get("is_immunosuppressant"))
print(f"  chemo: {n_chemo}, immunosuppressant: {n_immuno}")
# show a sample
for med in unique_meds[:15]:
    print(f"  {med:30s} -> {med_classification[med]}")