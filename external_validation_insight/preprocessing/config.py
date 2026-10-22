"""
Configuration for the INSIGHT preprocessing pipeline: paths, file names, code mappings and constants.
Input: sparc_paths.INSIGHT_RAW_DIR (or env INSIGHT_DATA_DIR); output: sparc_paths.INSIGHT_DATA_DIR
(or env INSIGHT_OUTPUT_DIR).
"""

import os
import sys

# repo root on sys.path for sparc_paths
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import sparc_paths

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("INSIGHT_DATA_DIR", sparc_paths.INSIGHT_RAW_DIR)
OUTPUT_DIR = os.environ.get("INSIGHT_OUTPUT_DIR", sparc_paths.INSIGHT_DATA_DIR)

# ── Input file names (production data, no test_ prefix) ───────────────────────
LAB_FILE       = "lab_result_cm.csv"
VITAL_FILE     = "vital.csv"
OBS_CLIN_FILE  = "obs_clin.csv"
PRESCRIBING_FILE = "prescribing.csv"
MED_ADMIN_FILE   = "med_admin.csv"
ENCOUNTER_FILE = "encounter.csv"
DIAGNOSIS_FILE = "diagnosis.csv"
ENROLLMENT_FILE = "enrollment.csv"
# demographics.csv in the original extract, demographic.csv in later pulls
DEMOGRAPHICS_FILE = os.environ.get("INSIGHT_DEMOGRAPHICS_FILE",
                                   "demographics.csv" if os.path.exists(os.path.join(DATA_DIR, "demographics.csv"))
                                   else "demographic.csv")

# ── Performance ───────────────────────────────────────────────────────────────
# Rows per chunk when reading large CSV files (50GB labs, 15GB obs_clin, etc.)
CHUNK_SIZE = 500_000

# ── Cohort filter ─────────────────────────────────────────────────────────────
MIN_RECORD_THRESHOLD = 0  # Inference-only pipeline — keep all tests regardless of record count

# ── Vitals pipeline — vital.csv (wide-format BP / BMI) ───────────────────────
# ORDERED_VITALS matches the existing SPARC pipeline exactly (positional index
# in the final 9-element vector).
ORDERED_VITALS = [
    "BMI",
    "Blood Pressure BP Systolic",
    "Blood Pressure Diastolic",
    "Invasive Blood Pressure BP Systolic",   # not in insight data → always 'nan'
    "Invasive Blood Pressure Diastolic",     # not in insight data → always 'nan'
    "O2",
    "Pulse Heart Rate",
    "Resp Rate",
    "Temperature C",
]

# Maps vital.csv column names → canonical ORDERED_VITALS names.
# ht and wt are intentionally excluded (not in ORDERED_VITALS).
VITAL_COL_TO_CANONICAL = {
    "original_bmi": "BMI",
    "systolic":     "Blood Pressure BP Systolic",
    "diastolic":    "Blood Pressure Diastolic",
}

# ── Vitals pipeline — obs_clin.csv (LOINC-coded clinical observations) ────────
# Maps obs_clin LOINC code → canonical ORDERED_VITALS name.
# These supplement vital.csv: positions 5-8 (O2, HR, RespRate, TempC) are only
# available in obs_clin.csv.
OBS_CLIN_LOINC_TO_VITAL = {
    "20564-1": "O2",              # Oxygen saturation (%)
    "8867-4":  "Pulse Heart Rate",# Heart rate (bpm)
    "9279-1":  "Resp Rate",       # Respiratory rate (breaths/min)
    "8310-5":  "Temperature C",   # Body temperature — stored in °F, converted below
}

# LOINC code whose values are recorded in Fahrenheit and must be converted to
# Celsius (C = (F - 32) × 5/9) before storing.
OBS_CLIN_TEMP_FAHRENHEIT_LOINC = "8310-5"

# ── Labs pipeline ─────────────────────────────────────────────────────────────
# Maps LOINC code → canonical lab name.
# Canonical names must match LABS_COMBINATION_MAP keys in the existing
# SPARC_data_preprocessing pipeline so SPARC can consume the output unchanged.
# Equivalent LOINC codes confirmed against Query Specifications Form_CF_032525.xlsx.
LOINC_TO_CANONICAL = {
    # ── CBC counts ──────────────────────────────────────────────────────────
    "718-7":   "HGB",
    "4544-3":  "HCT",
    "789-8":   "RBC",
    "26453-1": "RBC",   # Erythrocytes [#/vol] in Blood — alt code per spec
    "790-6":   "RBC",   # Erythrocytes [#/vol] in Blood — alt code per spec
    "6690-2":  "WBC",
    "777-3":   "Platelets",
    "787-2":   "MCV",
    "785-6":   "MCH",
    "28539-5": "MCH",   # MCH [Entitic mass] — alt code per spec
    "786-4":   "MCHC",
    "788-0":   "RDW",
    # 21000-5 (RDW-SD, fL) intentionally excluded — different numeric scale from
    # RDW-CV (788-0, %). Mixing them corrupts the RDW feature.
    # ── WBC differential — absolute counts (10³/µL) ────────────────────────
    # Map only absolute-count LOINCs; percentage LOINCs (770-8, 736-9, etc.)
    # are intentionally excluded — SPARC expects absolute counts for these
    # canonical names.
    "751-8":   "Neut",
    "731-0":   "Lymph",
    "742-7":   "Mono",
    "704-7":   "Baso",
    "711-2":   "Eos",           # Eosinophils [#/vol] absolute
    "26444-0": "Eos",           # Eosinophils [#/vol] alternate LOINC
    # ── Metabolic / CMP ─────────────────────────────────────────────────────
    "1751-7":  "Albumin",
    "1742-6":  "ALT",
    "1920-8":  "AST",
    "6768-6":  "ALP",
    "1975-2":  "Bilirubin, Total",
    "1968-7":  "Conjugated Bili (mg)",
    "3094-0":  "BUN",
    "2160-0":  "Creatinine",
    "2345-7":  "Glucose",
    "14749-6": "Glucose",       # Glucose [mol/vol] alternate LOINC per spec
    "17861-6": "Calcium",
    "2075-0":  "Chloride",
    "2823-3":  "Potassium",
    "6298-4":  "Potassium",     # Potassium [mol/vol] alternate LOINC per spec
    "2951-2":  "Sodium",
    "2028-9":  "CO2",
    "14627-4": "CO2",           # Bicarbonate alternate LOINC per spec
    "2885-2":  "Protein, Total",
    # ── eGFR ────────────────────────────────────────────────────────────────
    "62238-1": "eGFR (CKD-EPI 2021)",
    "48642-3": "eGFR (CKD-EPI 2021)",  # eGFR non-African-American
    "50044-7": "eGFR (CKD-EPI 2021)",  # eGFR African-American
    # ── Tumor markers ────────────────────────────────────────────────────────
    "1834-1":  "Alphafetoprotein",
    "2857-1":  "PSA",
    "19199-9": "PSA",           # PSA free+bound alternate
    "2039-6":  "CEA",
    "85319-2": "Cancer Antigen 125",
    "10334-1": "Cancer Antigen 15-3",
    "17842-6": "Cancer Antigen 19-9",
    "3016-3":  "TSH",
    "2638-5":  "Beta-2 Microglobulin",
}

# ── Medications pipeline ──────────────────────────────────────────────────────
# PCORnet CDM rx_route values that indicate enteral (oral) administration.
# Everything else (INTRAVENOUS, SUBCUTANEOUS, TOPICAL, etc.) is non-oral.
# "OT" (Other/unknown) and blank routes are excluded from both dicts.
ORAL_ROUTES: frozenset[str] = frozenset(
    {
        "ORAL",
        "SUBLINGUAL",
        "BUCCAL",
        "TRANSMUCOSAL",
        "NASOGASTRIC",
        "GASTROSTOMY",
    }
)

# PCORnet CDM medadmin_route values treated as parenteral (non-oral).
# Used to filter med_admin.csv, which captures actual IV/SC/IM administrations
# (chemotherapy, biologics, supportive agents) not reliably in prescribing.csv.
NON_ORAL_ROUTES: frozenset[str] = frozenset(
    {
        "INTRAVENOUS",
        "SUBCUTANEOUS",
        "INTRAMUSCULAR",
        "INTRATHECAL",
        "INTRAPERITONEAL",
        "INTRAVESICAL",
        "INTRADERMAL",
        "INTRAARTERIAL",
    }
)

# Cache file for RxNorm CUI → drug name lookups.
# Persists across runs so the RxNorm API is only hit once per CUI.
RXNORM_CACHE_FILE = os.environ.get(
    "RXNORM_CACHE_FILE",
    # shipped with the code so drug names resolve without internet access
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_mapping", "rxnorm_cache.json"),
)

# ── Admissions pipeline ───────────────────────────────────────────────────────
# PCORnet CDM enc_type values that represent inpatient hospital stays.
# EI = Emergency inpatient (ED visit that converts to inpatient admission).
INPATIENT_ENC_TYPES: frozenset[str] = frozenset({"IP", "EI"})

# ── Unit validation ───────────────────────────────────────────────────────────
# Maps canonical lab test name → frozenset of accepted result_unit strings
# (lowercased for case-insensitive comparison).
# Tests not listed here have no unit restriction.
#
# Key choices:
#   RBC:       10*6/ul = Million/mcL; drop 10*3/ul rows (1000× too small)
#   HGB:       g/dl only; drop spurious % rows
#   WBC/Plt:   10*3/ul = K/mcL; drop kU/L, kU/mL
#   MCHC:      g/dl and % are numerically identical (~32–36)
#   Na/K/Cl/CO2: meq/l ≡ mmol/l for monovalent ions — both accepted
CANONICAL_TO_ACCEPTED_UNITS: dict[str, frozenset[str]] = {
    # ── CBC ─────────────────────────────────────────────────────────────────
    "RBC":        frozenset({"10*6/ul"}),             # Million/mcL = 10^6/μL
    "HGB":        frozenset({"g/dl"}),                # drop spurious '%' rows
    "HCT":        frozenset({"%"}),
    "MCV":        frozenset({"fl"}),
    "MCH":        frozenset({"pg"}),
    "MCHC":       frozenset({"g/dl", "%"}),           # numerically identical
    "RDW":        frozenset({"%"}),
    "WBC":        frozenset({"10*3/ul"}),             # K/mcL = 10^3/μL
    "Platelets":  frozenset({"10*3/ul"}),             # K/mcL = 10^3/μL
    # ── WBC differential — absolute counts (10³/µL) ─────────────────────────
    "Neut":       frozenset({"10*3/ul", "10*3/mm3"}),   # 1 µL = 1 mm³ — numerically identical
    "Lymph":      frozenset({"10*3/ul", "10*3/mm3"}),
    "Mono":       frozenset({"10*3/ul", "10*3/mm3"}),
    "Baso":       frozenset({"10*3/ul", "10*3/mm3"}),
    "Eos":        frozenset({"10*3/ul", "10*3/mm3"}),
    # ── Metabolic / CMP ─────────────────────────────────────────────────────
    "Glucose":    frozenset({"mg/dl"}),
    "Calcium":    frozenset({"mg/dl"}),
    "Sodium":     frozenset({"meq/l", "mmol/l"}),     # mEq/L ≡ mmol/L for Na⁺
    "Potassium":  frozenset({"meq/l", "mmol/l"}),     # mEq/L ≡ mmol/L for K⁺
    "CO2":        frozenset({"meq/l", "mmol/l"}),     # mEq/L ≡ mmol/L for HCO₃⁻
    "Creatinine": frozenset({"mg/dl"}),
    "Chloride":   frozenset({"meq/l", "mmol/l"}),     # mEq/L ≡ mmol/L for Cl⁻
    "BUN":        frozenset({"mg/dl"}),
    "Albumin":    frozenset({"g/dl"}),
    "Protein, Total": frozenset({"g/dl"}),
    "Bilirubin, Total": frozenset({"mg/dl"}),
    "Conjugated Bili (mg)": frozenset({"mg/dl"}),
    "ALT":        frozenset({"[u]/l", "[iu]/l", "u/l", "iu/l"}),
    "AST":        frozenset({"[u]/l", "[iu]/l", "u/l", "iu/l"}),
    "ALP":        frozenset({"[u]/l", "[iu]/l", "u/l", "iu/l"}),
    # ── Tumor markers ────────────────────────────────────────────────────────
    "TSH":        frozenset({"m[iu]/l", "u[iu]/ml"}),   # actual units in Insight data; drop m[iu]/ml (1000× too large) and ot
    "Beta-2 Microglobulin": frozenset({"mg/l", "ug/ml"}),
    "Cancer Antigen 125":   frozenset({"[u]/ml"}),
    "Cancer Antigen 15-3":  frozenset({"[u]/ml"}),
    "Cancer Antigen 19-9":  frozenset({"[u]/ml"}),
    "Alphafetoprotein":     frozenset({"ng/ml"}),
    "PSA":        frozenset({"ng/ml"}),
    "CEA":        frozenset({"ng/ml"}),
}

# ── Unit aliases: malformed unit strings normalised before unit filtering ────
# Site F reports the liver enzymes with result_unit "uL" on ~90% of its rows
# (956,420 ALT / 947,903 AST / 953,383 ALP), while its bilirubin uses a
# well-formed "mg/dL". "uL" is a microlitre — a volume — which is meaningless
# for an enzyme activity concentration, and the values themselves sit squarely
# in the U/L range (mean 51.5; e.g. 169, 7, 19, 43), indistinguishable from the
# same site's "[IU]/L" rows. It is a corrupted rendering of "U/L".
#
# Without this normalisation the unit filter drops those rows, leaving site F
# with 6.4% of its ALT, 6.5% of its AST and 6.5% of its ALP — which is why
# liver toxicity was unevaluable at site F (<11 eligible patients with liver labs).
#
# Keyed by canonical test so the alias applies only where the raw string is
# unambiguously wrong; the accepted-unit sets above stay strict.
# Keys and values are lowercase, matched after .strip().str.lower().
CANONICAL_UNIT_ALIASES: dict[str, dict[str, str]] = {
    # "uL" -> "U/L": enzyme activity, site F (~90% of its ALT/AST/ALP rows).
    "ALT": {"ul": "u/l"},
    "AST": {"ul": "u/l"},
    "ALP": {"ul": "u/l"},
    # "kU/L" -> "10*3/uL": cell counts, site F (~22% of its WBC/platelet rows).
    # kU/L is an enzyme-activity unit and meaningless for a cell count; the
    # values match the valid rows' distribution test-for-test in the same site
    # (WBC 8.66 vs 7.71, Platelets 246.2 vs 212.7, Baso 0.025 vs 0.023,
    # Lymph 1.52 vs 1.37, Mono 0.51 vs 0.58), i.e. a mangled "K/uL".
    "WBC": {"ku/l": "10*3/ul"},
    "Platelets": {"ku/l": "10*3/ul"},
    "Neut": {"ku/l": "10*3/ul"},
    "Lymph": {"ku/l": "10*3/ul"},
    "Mono": {"ku/l": "10*3/ul"},
    "Baso": {"ku/l": "10*3/ul"},
    "Eos": {"ku/l": "10*3/ul"},
    # "U/mL" -> "[U]/mL": pure UCUM-notation variant of the same unit. The
    # accepted sets list only the bracketed form, so 100% of Cancer Antigen
    # 19-9 (54,893 rows) and Cancer Antigen 15-3 (90,825 rows) were dropped
    # and neither marker reached the model at all.
    "Cancer Antigen 19-9": {"u/ml": "[u]/ml"},
    "Cancer Antigen 15-3": {"u/ml": "[u]/ml"},
    "Cancer Antigen 125":  {"u/ml": "[u]/ml"},
}

# ── Unit problems that need a VALUE conversion, not a rename ─────────────────
# These are NOT in CANONICAL_UNIT_ALIASES because aliasing them would silently
# admit values on the wrong scale. Each needs a multiplier applied to
# result_num, which the current pipeline has no mechanism for:
#
#   "{cells}/uL" on Baso/Eos/Lymph/Mono/Neut (~63,170 rows each, ~316k total)
#       absolute cells per microlitre = 1000x the canonical 10*3/uL value.
#       Aliasing would inflate every one of these by 1000x.
#
#   "ng/mL" on Beta-2 Microglobulin (335 rows; the test is 100% dropped today)
#       1000x smaller than the accepted ug/mL.
#
#   "[IU]/mL" on Cancer Antigen 19-9 (4,047) and 15-3 (8,916)
#       IU and U are assay-standardisation dependent for these markers and are
#       not interchangeable by notation alone; left dropped pending a decision.
#
# Also left dropped, deliberately: "OT" (PCORnet "Other", ~1.4M rows across
# tests) and blank units (~1.6M rows, 0.7-4% of most tests) carry no unit
# information at all, so no safe normalisation exists.
UNITS_NEEDING_CONVERSION_UNRESOLVED: dict[str, dict[str, float]] = {
    "Baso":  {"{cells}/ul": 0.001},
    "Eos":   {"{cells}/ul": 0.001},
    "Lymph": {"{cells}/ul": 0.001},
    "Mono":  {"{cells}/ul": 0.001},
    "Neut":  {"{cells}/ul": 0.001},
    "Beta-2 Microglobulin": {"ng/ml": 0.001},
}

# ── Null/redacted-value sentinels ─────────────────────────────────────────────
# lab_result_cm.csv encodes some suppressed/unavailable results as 9999999
# instead of leaving result_num blank. Confirmed present across every canonical
# lab test in the processed output (0.001%-3.5% of rows per test) — clearly a
# sentinel, not a real result (e.g. RBC/PSA/CEA of 9,999,999 is impossible).
# Left unfiltered, a tiny fraction of these values dominates the per-test
# mean/std used for z-score normalization (a single value this large swamps
# the variance of millions of real, small-magnitude results).
LAB_RESULT_SENTINEL_VALUES: frozenset[float] = frozenset({9999999.0})

# obs_clin.csv accepted units per vital LOINC code (lowercased).
# Temperature rows with 'cel' are already in °C (no conversion needed).
# Temperature rows with '[degf]' are in °F (convert with (F-32)*5/9).
# Rows with 'ni' or other unknown units are dropped.
OBS_CLIN_ACCEPTED_UNITS: dict[str, frozenset[str]] = {
    "20564-1": frozenset({"%"}),                      # O2 saturation
    "8867-4":  frozenset({"min", "{cpm}"}),           # Heart rate (per minute)
    "9279-1":  frozenset({"min", "{cpm}"}),           # Resp rate (per minute)
    "8310-5":  frozenset({"[degf]", "cel"}),          # Temperature — see conversion logic
}
OBS_CLIN_TEMP_CELSIUS_UNIT  = "cel"    # already °C — keep as-is
OBS_CLIN_TEMP_FAHRENHEIT_UNIT = "[degf]"  # must convert to °C

# ── Vitals plausible-value ranges ─────────────────────────────────────────────
# Loosely-bounded physiological ranges used to drop data-entry errors from
# vital.csv / obs_clin.csv (e.g. a small number of "original_bmi" rows come
# through in the tens of thousands — likely raw weight or a unit slip, not a
# real BMI). Rows outside range are dropped; canonical tests with no entry
# here are not range-filtered. Bounds are intentionally wide (data-error vs.
# clinical-outlier, not a clinical plausibility judgment).
VITAL_VALID_RANGES: dict[str, tuple[float, float]] = {
    "BMI":                              (10, 100),
    "Blood Pressure BP Systolic":       (40, 300),
    "Blood Pressure Diastolic":         (20, 200),
    "O2":                               (50, 100),
    "Pulse Heart Rate":                 (20, 300),
    "Resp Rate":                        (4, 80),
    "Temperature C":                    (25, 45),
}

# ── Stage pipeline ────────────────────────────────────────────────────────────
# ICD-10 code prefixes that indicate distant metastatic disease (Stage 4).
# C77: secondary/unspecified malignant neoplasm of lymph nodes
# C78: secondary malignant neoplasm of respiratory and digestive organs
# C79: secondary malignant neoplasm of other and unspecified sites
# C800: disseminated malignant neoplasm (explicitly metastatic)
# NOTE: Stages 1–3 cannot be reliably inferred from ICD-10 codes alone;
# only Stage 4 is produced by this pipeline.
METASTASIS_ICD10_PREFIXES: tuple[str, ...] = ("C77", "C78", "C79", "C800")

# ── Demographics pipeline ─────────────────────────────────────────────────────
# demographics.csv stores sex, race and hispanic as PCORnet CDM codes, not
# labels. The maps below decode them verbatim — "No information" (NI) and
# "Unknown" (UN) are distinct PCORnet codes and are kept distinct here; collapse
# them downstream if an analysis wants a single missing category.
#
# Codes actually present in the Insight extract (207,041 patients):
#   race      01 02 03 04 05 06 07 UN
#   hispanic  Y N NI UN
#   sex       M F NI UN
# Any code outside these maps is decoded as "Unknown" and reported at build time.

# MALE/FEMALE are spelled to match GENDERS in SPARC/data/MSK_featurename_list.py.
SEX_CODE_TO_LABEL: dict[str, str] = {
    "M":  "MALE",
    "F":  "FEMALE",
    "A":  "Ambiguous",
    "OT": "Other",
    "NI": "No information",
    "UN": "Unknown",
}

RACE_CODE_TO_LABEL: dict[str, str] = {
    "01": "American Indian or Alaska Native",
    "02": "Asian",
    "03": "Black or African American",
    "04": "Native Hawaiian or Other Pacific Islander",
    "05": "White",
    "06": "Multiple race",
    "07": "Refuse to answer",
    "OT": "Other",
    "NI": "No information",
    "UN": "Unknown",
}

# PCORnet models ethnicity as the single "hispanic" field.
HISPANIC_CODE_TO_LABEL: dict[str, str] = {
    "Y":  "Hispanic",
    "N":  "Not Hispanic",
    "R":  "Refuse to answer",
    "OT": "Other",
    "NI": "No information",
    "UN": "Unknown",
}
