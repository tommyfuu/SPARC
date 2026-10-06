"""A synthetic cohort in the bundle format, for demos and tests. Not real patients."""

import numpy as np
from bundle_io import (
    AGE,
    DEATH,
    FOLD,
    GENDER,
    ICU,
    INPATIENT,
    LABS,
    LAST_CONTACT,
    MEDS,
    N_FOLDS,
    REQUIRED_KEYS,
    VITALS_KEY,
)
from MSK_featurename_list import ALL_INCLUDED_TESTS, VITALS

# Typical values in the units the bundle expects (CBC differential as absolute counts, K/uL).
LAB_MEANS = {
    'RBC': 4.5, 'MCV': 88.0, 'MCH': 29.0, 'MCHC': 33.0, 'RDW': 14.0, 'HGB': 13.0, 'HCT': 39.0,
    'WBC': 7.0, 'Platelets': 220.0, 'Neut': 4.5, 'Mono': 0.5, 'Lymph': 2.0, 'Baso': 0.05,
    'Eos': 0.2, 'Glucose': 95.0, 'Calcium': 9.2, 'Sodium': 139.0, 'Potassium': 4.0, 'CO2': 24.0,
    'Creatinine': 1.0, 'Chloride': 104.0, 'BUN': 15.0, 'Albumin': 4.0, 'Protein, Total': 7.0,
    'ALP': 75.0, 'ALT': 25.0, 'AST': 25.0, 'Bilirubin, Total': 0.7,
}
VITAL_MEANS = {
    'BMI': 25.0, 'Blood Pressure BP Systolic': 120.0, 'Blood Pressure Diastolic': 78.0, 'O2': 97.0,
    'Pulse Heart Rate': 72.0, 'Resp Rate': 16.0, 'Temperature C': 36.8,
}


def make_synthetic_bundle(n_patients=200, seed=0):
    """Return a bundle with labs, vitals, one hospital stay, outcomes and CISPLATIN cycles.

    Each patient has a hidden risk factor that lowers blood counts, raises creatinine and
    makes death and neutropenia more likely, so the labels are learnable.
    """
    rng = np.random.default_rng(seed)
    bundle = {key: {} for key in REQUIRED_KEYS}
    folds = rng.permutation(n_patients) % N_FOLDS

    for i in range(n_patients):
        pid = 1000 + i
        risk = rng.normal()
        days = np.cumsum(rng.integers(5, 15, size=rng.integers(20, 35))) - 5
        days = [int(d) for d in days]

        on_chemo = rng.random() < 0.7
        infusions = [days[8] + 21 * k for k in range(4)] if on_chemo else []
        nadir_days = {d + 7 for d in infusions}
        days = sorted(set(days) | nadir_days)
        neutropenic = on_chemo and rng.random() < 1 / (1 + np.exp(-1.5 * (risk - 0.5)))

        labs, vitals = {}, {}
        for d in days:
            values = [None] * len(ALL_INCLUDED_TESTS)
            for name, mean in LAB_MEANS.items():
                shift = {'HGB': -0.08, 'Platelets': -0.1, 'Creatinine': 0.15, 'Albumin': -0.05}.get(name, 0.0)
                value = mean * (1 + shift * risk + 0.08 * rng.normal())
                values[ALL_INCLUDED_TESTS.index(name)] = round(value, 3)
            if d in nadir_days and neutropenic:
                values[ALL_INCLUDED_TESTS.index('Neut')] = round(rng.uniform(0.2, 0.9), 3)
            labs[d] = values
            if rng.random() < 0.6:
                row = [None] * len(VITALS)
                for name, mean in VITAL_MEANS.items():
                    row[VITALS.index(name)] = round(mean * (1 + 0.03 * rng.normal()), 2)
                vitals[d] = row

        admit, discharge = days[-4], days[-1]
        dies = rng.random() < 1 / (1 + np.exp(-1.5 * (risk - 0.3)))
        bundle[LABS][pid] = labs
        bundle[VITALS_KEY][pid] = vitals
        bundle[GENDER][pid] = [int(rng.random() < 0.5)]
        bundle[AGE][pid] = int(rng.integers(8, 17))
        bundle[INPATIENT][pid] = [[admit, discharge]]
        if rng.random() < 0.3:
            bundle[ICU][pid] = [[discharge - 2, discharge]]
        if dies:
            bundle[DEATH][pid] = discharge + int(rng.integers(5, 150))
            bundle[LAST_CONTACT][pid] = bundle[DEATH][pid]
        else:
            bundle[LAST_CONTACT][pid] = discharge + int(rng.integers(200, 500))
        bundle[FOLD][pid] = int(folds[i])
        if infusions:
            bundle[MEDS][pid] = {d: [['CISPLATIN', d]] for d in infusions}
    return bundle
