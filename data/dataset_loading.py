"""
dataset_loading.py

Standalone module for loading, normalizing, and splitting all SPARC datasets.
Import this wherever you need data (training, evaluation, inference).

Usage:
    from dataset_loading import load_dataset

    data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
        dataset='MSK',
        data_source_path='/data/morrisq/fuc/SPARC_data/',
        option='neutropenia_risk',
        modules='practical',
        valfold=0,
        mode='train',
        rseed=1024,
        competing_risk_death_check=False,
        include_gcsf=False,
        data_fraction_option='default',
    )
"""

import os
import sys
import re
import copy
import json
import math
import pickle
import random

import numpy as np
import pandas as pd

# Make sure MSK_featurename_list is importable when this module is used from other directories.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from MSK_featurename_list import (
    ALL_INCLUDED_TESTS, BASIC_TESTS, BASIC_TESTS_INDICES,
    EXTRA_TESTS, EXTRA_TESTS_INDICES,
    BASIC_VITALS_INDICES, EXTRA_VITALS_INDICES,
    CBC_DIFFERENTIAL_TESTS,
    MSK_numfeature_dict, MSK_firstprinciple_numfeature_dict,
    one_hot_encode_dict,
    SMOKE_list, timeshift_ctype_mapping_dict,
    WBC_INDEX, ANEMIA_INDEX, CREATININE_INDEX,
    AST_INDEX, ALT_INDEX, ALP_INDEX, BILIRUBIN_INDEX,
    Temperature_C_INDEX,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    dataset: str,
    data_source_path: str,
    option: str,
    modules: str,
    valfold: int,
    mode: str,
    rseed: int = 1024,
    competing_risk_death_check: bool = False,
    include_gcsf: bool = False,
    data_fraction_option: str = 'default',
    quick_test: bool = False,
    normalize: bool = True,
    first_principle: bool = False,
    caia_portion: str = 'prior',
    med_regimen_eval: bool = False,
    bundle_missingness_rate: float = 0.0,
    bundle_missingness_seed: int = 42,
):
    """
    Load, normalize, and split all data needed for a SPARC training/evaluation run.

    Parameters
    ----------
    dataset : str
        One of 'MSK', 'MSK_timeshift', 'stanfordEHRSHOT', 'MIMICIV_hosp', or any
        string containing 'CAIA'.
    data_source_path : str
        Base directory where all dataset folders live.
    option : str
        Task option, e.g. 'binary', 'pretrain', 'neutropenia_risk', 'AKI_risk', ...
    modules : str
        Feature module set: 'all', 'all_labs', 'mutation_clinical', 'practical', or 'parsimonious'.
    valfold : int
        0-indexed validation fold (0–9).  Fold 10 is always the held-out test set.
    mode : str
        'train', 'test', 'recalibrate', or 'test_recalibrate'.
    rseed : int
        Random seed used for data-fraction shuffling and stanfordEHRSHOT splits.
    competing_risk_death_check : bool
        Whether to load death-date data for competing risk checks.
    include_gcsf : bool
        Whether to load GCSF administration dates (MSK only).
    data_fraction_option : str
        'default' uses all training data.  Pass '1/5', '1/25', or '1/125' to
        down-sample the training split.
    quick_test : bool
        If True, use a tiny subset of the data for quick testing of the training loop.
        This overrides data_fraction_option.
    normalize : bool
        If True (default), normalize the data. Set to False to get raw lab values.
    first_principle : bool
        If True, use only task-specific lab values (e.g., only creatinine for AKI,
        only HGB for anemia) instead of all basic labs. Only applies to side-effect tasks.
    caia_portion: str
        'prior', 'post', or 'entire' to specify which portion of the MSK-CAIA dataset to load.
        Ignored for non-MSK-CAIA datasets.
    med_regimen_eval: bool
        If True, load the unfiltered medication data (all meds taken on same day) for regimen analysis.
        If False (default), load the standard filtered medication data (only high-risk meds).
        Only applies to CAIA datasets.

    Returns
    -------
    data_dict_lab_dict : dict
        Feature dictionaries keyed by module name.
    modular_featuredim_dict : dict
        Maps module name → feature dimension (int).
    split_dict : dict
        {'train': [...], 'val': [...], 'test': [...]} of patient IDs.
    component : str
        Descriptive string used by the data loader (e.g. 'ts_only', 'all').
    include_infusion_dates : bool
        Whether infusion-date columns are needed by the data loader.
    """
    assert (
        dataset in ['MSK', 'MSK_timeshift', 'stanfordEHRSHOT', 'MIMICIV_hosp']
        or 'CAIA' in dataset
    ), (
        "dataset must be one of ['MSK', 'MSK_timeshift', 'stanfordEHRSHOT', "
        "'MIMICIV_hosp'] or contain 'CAIA'"
    )

    data_modules_to_include, component = _resolve_modules(modules)

    data_dict_lab_dict = {}
    modular_featuredim_dict = {}

    # ------------------------------------------------------------------
    # 1. CAIA datasets: load from a single JSON bundle
    # ------------------------------------------------------------------
    caia_shared = {}   # populated if 'CAIA' in dataset; re-used across blocks below
    if 'CAIA' in dataset:
        caia_shared = _load_caia(dataset, data_source_path, caia_portion=caia_portion, med_regimen_eval=med_regimen_eval)

    # ------------------------------------------------------------------
    # 2. Demographic data (cancer type, death date, last-contact date)
    # ------------------------------------------------------------------
    DMP_ID_to_PRIMARY_CTYPE_dict = {}
    DMP_ID_to_DATES_to_TESTSVALUES_DICT = {}  # populated when labs are loaded

    if dataset in ['MSK', 'MSK_timeshift']:
        (
            data_dict_lab_dict,
            DMP_ID_to_PRIMARY_CTYPE_dict,
            DMP_ID_to_DATES_to_TESTSVALUES_DICT,
            PRIMARY_CTYPES,
            dePHIed_demographic_df,
        ) = _load_msk_demographics(
            dataset, data_source_path, option, competing_risk_death_check,
            include_gcsf, data_dict_lab_dict,
        )

    elif dataset == 'MIMICIV_hosp':
        data_dict_lab_dict, DMP_ID_to_PRIMARY_CTYPE_dict = _load_mimic_demographics(
            dataset, data_source_path, option, competing_risk_death_check, data_dict_lab_dict,
        )

    elif 'CAIA' in dataset:
        data_dict_lab_dict = _load_caia_demographics(
            option, competing_risk_death_check, data_dict_lab_dict, caia_shared,
        )

    elif dataset == 'stanfordEHRSHOT':
        data_dict_lab_dict = _load_stanford_demographics(
            dataset, data_source_path, option, competing_risk_death_check, data_dict_lab_dict,
        )

    # ------------------------------------------------------------------
    # 3. Hospital / ICU admission-discharge dates
    # ------------------------------------------------------------------
    data_dict_lab_dict = _load_admission_dates(
        dataset, data_source_path, option, data_dict_lab_dict, caia_shared,
    )

    # ------------------------------------------------------------------
    # 4. Medication data  (side-effect tasks)
    # ------------------------------------------------------------------
    se_task = bool(re.match(
        r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$',
        option,
    ))
    print(f"DEBUG in load_dataset: option='{option}', se_task={se_task}")
    if se_task:
        data_dict_lab_dict, DMP_ID_to_DATES_to_TESTSVALUES_DICT = _load_medications(
            dataset, data_source_path, data_dict_lab_dict, DMP_ID_to_DATES_to_TESTSVALUES_DICT, caia_shared,
        )

    # ------------------------------------------------------------------
    # 5. Lab values
    # ------------------------------------------------------------------
    if (
        'basic_lab' in data_modules_to_include
        or 'extra_lab' in data_modules_to_include
        or 'cbc_differential' in data_modules_to_include
        or se_task
    ):
        (
            data_dict_lab_dict,
            modular_featuredim_dict,
            DMP_ID_to_DATES_to_TESTSVALUES_DICT,
        ) = _load_labs(
            dataset, data_source_path, option, data_modules_to_include,
            data_dict_lab_dict, modular_featuredim_dict, caia_shared,
            DMP_ID_to_DATES_to_TESTSVALUES_DICT, first_principle,
        )

    # ------------------------------------------------------------------
    # 6. Vital signs
    # ------------------------------------------------------------------
    if 'basic_vitals' in data_modules_to_include or 'extra_vitals' in data_modules_to_include:
        data_dict_lab_dict, modular_featuredim_dict = _load_vitals(
            dataset, data_source_path, option, data_modules_to_include,
            data_dict_lab_dict, modular_featuredim_dict,
            DMP_ID_to_DATES_to_TESTSVALUES_DICT, caia_shared, first_principle,
        )

    # ------------------------------------------------------------------
    # 7. Static demographic features (gender, age)
    # ------------------------------------------------------------------
    if 'gender' in data_modules_to_include:
        data_dict_lab_dict, modular_featuredim_dict = _load_gender(
            dataset, data_source_path, data_dict_lab_dict, modular_featuredim_dict,
            dePHIed_demographic_df if dataset in ['MSK', 'MSK_timeshift'] else None,
            caia_shared,
        )
    if 'age' in data_modules_to_include:
        data_dict_lab_dict, modular_featuredim_dict = _load_age(
            dataset, data_source_path, data_dict_lab_dict, modular_featuredim_dict,
            dePHIed_demographic_df if dataset in ['MSK', 'MSK_timeshift'] else None,
            caia_shared,
        )

    # ------------------------------------------------------------------
    # 8. Genomic / clinical features
    # ------------------------------------------------------------------
    if 'mutation' in data_modules_to_include:
        data_dict_lab_dict, modular_featuredim_dict = _load_mutation(
            dataset, data_source_path, data_dict_lab_dict, modular_featuredim_dict,
        )
    if 'clinical' in data_modules_to_include:
        data_dict_lab_dict, modular_featuredim_dict = _load_clinical(
            dataset, data_source_path, data_dict_lab_dict, modular_featuredim_dict,
            DMP_ID_to_PRIMARY_CTYPE_dict,
            PRIMARY_CTYPES if dataset in ['MSK', 'MSK_timeshift'] else [],
            dePHIed_demographic_df if dataset in ['MSK', 'MSK_timeshift'] else None,
        )

    print("modular_featuredim_dict", modular_featuredim_dict)

    # ------------------------------------------------------------------
    # 9. Normalization (with caching)
    # ------------------------------------------------------------------
    if normalize:
        data_dict_lab_dict = _normalize(
            dataset, data_source_path, modules, data_dict_lab_dict, first_principle,
        )
        print("normalization done")
    else:
        print("skipping normalization (normalize=False)")

    # ------------------------------------------------------------------
    # 9b. Bundle missingness (resource-scarce simulation)
    # ------------------------------------------------------------------
    if bundle_missingness_rate > 0.0:
        data_dict_lab_dict = _apply_bundle_missingness(
            data_dict_lab_dict, bundle_missingness_rate, bundle_missingness_seed,
        )

    # ------------------------------------------------------------------
    # 10. Train / val / test splits
    # ------------------------------------------------------------------
    split_dict = _build_splits(
        dataset, data_source_path, option, mode, valfold, rseed,
        data_fraction_option, DMP_ID_to_DATES_to_TESTSVALUES_DICT,
        data_dict_lab_dict, quick_test, caia_shared,
    )

    include_infusion_dates = se_task
    print(f"DEBUG at return: se_task={se_task}, include_infusion_dates={include_infusion_dates}")
    return data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_first_principle_indices(option: str):
    """
    Map task option to specific lab/vital indices for first-principle models.

    Returns:
        tuple: (lab_indices, vital_indices, lab_featdim, vital_featdim)
    """
    if bool(re.match(r'^(febrile_neutropenia)(_next_risk|_risk)?$', option)):
        return (
            [WBC_INDEX],
            [Temperature_C_INDEX],
            MSK_firstprinciple_numfeature_dict['febrile_neutropenia_lab'],
            MSK_firstprinciple_numfeature_dict['febrile_neutropenia_vitals']
        )
    elif bool(re.match(r'^(anemia)(_next_risk|_risk)?$', option)):
        return (
            [ANEMIA_INDEX],
            [],
            MSK_firstprinciple_numfeature_dict['anemia_lab'],
            MSK_firstprinciple_numfeature_dict['anemia_vitals']
        )
    elif bool(re.match(r'^(neutropenia)(_next_risk|_risk)?$', option)):
        return (
            [WBC_INDEX],
            [],
            MSK_firstprinciple_numfeature_dict['neutropenia_lab'],
            MSK_firstprinciple_numfeature_dict['neutropenia_vitals']
        )
    elif bool(re.match(r'^(AKI)(_next_risk|_risk)?$', option)):
        return (
            [CREATININE_INDEX],
            [],
            MSK_firstprinciple_numfeature_dict['AKI_lab'],
            MSK_firstprinciple_numfeature_dict['AKI_vitals']
        )
    elif bool(re.match(r'^(Liver_toxicity)(_next_risk|_risk)?$', option)):
        return (
            [AST_INDEX, ALT_INDEX, ALP_INDEX, BILIRUBIN_INDEX],
            [],
            MSK_firstprinciple_numfeature_dict['Liver_toxicity_lab'],
            MSK_firstprinciple_numfeature_dict['Liver_toxicity_vitals']
        )
    else:
        # Not a first-principle task, return None
        return None, None, None, None


def _resolve_modules(modules):
    if modules == 'all':
        return ['gender', 'age', 'basic_lab', 'basic_vitals', 'extra_lab', 'extra_vitals', 'mutation', 'clinical'], 'all'
    elif modules == 'all_labs':
        return ['gender', 'age', 'basic_lab', 'basic_vitals', 'extra_lab', 'extra_vitals'], 'all_lab'
    elif modules == 'mutation_clinical':
        return ['gender', 'age', 'mutation', 'clinical'], 'mutation_clinical_only'
    elif modules == 'practical':
        return ['gender', 'age', 'basic_lab', 'basic_vitals'], 'ts_only'
    elif modules == 'parsimonious':
        return ['gender', 'age', 'cbc_differential', 'basic_lab', 'basic_vitals'], 'parsimonious'
    else:
        raise ValueError(f"Unknown modules value: {modules!r}. Expected one of: all, all_labs, mutation_clinical, practical, parsimonious")


def _load_caia(dataset, data_source_path, caia_portion='prior', med_regimen_eval=False):
    """
    Load the single CAIA JSON bundle and return a dict of de-identified sub-dicts.

    Args:
        dataset: Dataset name (e.g., 'MSK_CAIA')
        data_source_path: Base path to data
        caia_portion: 'prior', 'post', or 'entire' for temporal splitting
        med_regimen_eval: If True, load unfiltered meds (all meds on same day) for regimen analysis
    """
    def custom_decoder(d):
        if "__set__" in d:
            return set(d["__set__"])
        return d

    if dataset == 'MSK_CAIA':
        data_path = f'{data_source_path}/{dataset}/DMP_ID_to_dicts_final_till2021_031226.json'
        # data_path = f'{data_source_path}/{dataset}/DMP_ID_to_dicts_final_till2021_050726.json'
    elif dataset == 'MSK_CAIA_timeshift':
        data_path = f'{data_source_path}/{dataset}/DMP_ID_to_dicts_final_after2021_031226.json'
        # data_path = f'{data_source_path}/{dataset}/DMP_ID_to_dicts_final_after2021_050726.json'
    else:
        data_path = f'{data_source_path}/{dataset}/DMP_ID_to_dicts_final.json'
        # check if this exists, if not, raise error
        if not os.path.exists(data_path):
            raise ValueError(f"CAIA dataset JSON file not found at expected path: {data_path}")
    # else:
    #     raise ValueError(f"Unknown CAIA dataset: {dataset}. Expected 'MSK_CAIA' or 'MSK_CAIA_timeshift'")

    with open(data_path, 'r') as f:
        raw = json.load(f, object_hook=custom_decoder)

    shared = {
        'hospital':  {int(k): v for k, v in raw['DMP_ID_to_inpatient_dates_dict_delta'].items()},
        'icu':       {int(k): v for k, v in raw['DMP_ID_to_icu_dates_dict_delta'].items()},
        'gender_oh': {int(k): v for k, v in raw['DMP_ID_to_GENDER_dict_oh'].items()},
        'age':       {int(k): v for k, v in raw['DMP_ID_to_firstlabdateagebinned_dict'].items()},
        'death':     {int(k): v for k, v in raw['DMP_ID_to_death_days_dict'].items()},
        'lastcontact': {int(k): v for k, v in raw['DMP_ID_to_last_contact_date_dict_delta'].items()},
        'labs':      {int(k): {int(d): v for d, v in dates.items()} for k, dates in raw['DMP_ID_to_DATES_to_TESTSVALUES_DICT'].items()},
        'vitals':    {int(k): {int(d): v for d, v in dates.items()} for k, dates in raw['DMP_ID_to_DATES_to_VITALSVALUES_DICT'].items()},
        'fold_num':  {int(k): v for k, v in raw['DMP_ID_to_fold_num_dict'].items()},
    }

    # meds: outer key int, inner key int, value list of (str, int) tuples
    # Use unfiltered version for regimen analysis (all meds on same day), filtered otherwise (only high-risk meds)
    if med_regimen_eval:
        meds_key = 'DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_delta_unfiltered'
        if meds_key not in raw:
            print(f"WARNING: {meds_key} not found in data bundle. Falling back to filtered meds.")
            meds_key = 'DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_delta'
    else:
        meds_key = 'DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_delta'

    meds_raw = raw[meds_key]
    shared['meds'] = {
        int(k): {
            int(date): [(med[0], int(med[1])) for med in meds]
            for date, meds in dates.items()
        }
        for k, dates in meds_raw.items()
    }

    if dataset == 'MSK_CAIA':
        # if caia_portion is entire, keep it as is; 
        # if prior, for all patientss, remove any data after 2021 by matching to data after the cutoff_day as enlisted in DMP_ID_to_cutoff_day_dict
        # if post, for all patients, remove any data before 2021 by matching to data before or on the cutoff_day as enlisted in DMP_ID_to_cutoff_day_dict
        ## note that not for the same patient it might not be in all the dicts, e.g. some patients might not have ICU admissions but have hospital admissions, or some patients might not have labs but have meds, etc. so we need to loop through all dicts and check for the presence of the patient id before applying the cutoff
        ### calculate average number of lab/vital dates per patient overall, prior to cutoff, and post/on cutoff to confirm that the cutoff is doing something reasonable
        cutoff_dict_raw = raw.get('DMP_ID_to_cutoff_day_dict', {})
        num_lab_vital_dates_overall, num_lab_vital_dates_prior, num_lab_vital_dates_post = 0, 0, 0
        pids_w_cutoff = set()
        for pid, dates in shared['labs'].items():
            # Skip patients not in cutoff_dict (their first lab was after the cutoff date, but included in the dataset as their first exposure day is prior from other dfs)
            if str(pid) not in cutoff_dict_raw:
                continue
            pids_w_cutoff.add(pid)
            overall_lab_vital_dates = set(shared['labs'].get(pid, {}).keys()) | set(shared['vitals'].get(pid, {}).keys())
            cutoff_day = cutoff_dict_raw[str(pid)]
            prior_lab_vital_dates = {d for d in overall_lab_vital_dates if d <= cutoff_day}
            post_lab_vital_dates = {d for d in overall_lab_vital_dates if d > cutoff_day}
            num_lab_vital_dates_overall += len(overall_lab_vital_dates)
            num_lab_vital_dates_prior += len(prior_lab_vital_dates)
            num_lab_vital_dates_post += len(post_lab_vital_dates)
        num_patients_with_cutoff = len(pids_w_cutoff)
        if num_patients_with_cutoff > 0:
            print(f"Patients with cutoff data: {num_patients_with_cutoff} / {len(shared['labs'])}")
            print(f"Average number of lab/vital dates per patient overall: {num_lab_vital_dates_overall / num_patients_with_cutoff:.2f}")
            print(f"Average number of lab/vital dates per patient prior to cutoff: {num_lab_vital_dates_prior / num_patients_with_cutoff:.2f}")
            print(f"Average number of lab/vital dates per patient post/on cutoff: {num_lab_vital_dates_post / num_patients_with_cutoff:.2f}")
        else:
            print(f"Warning: No patients have cutoff_day data in this dataset")
        
        cutoff_dict = {int(k): v for k, v in raw['DMP_ID_to_cutoff_day_dict'].items()}
        if caia_portion == 'prior':
            print("Loading prior portion of CAIA dataset (data on or before cutoff day)")
            for key in ['hospital', 'icu', 'death', 'lastcontact', 'labs', 'vitals', 'meds']:
                for pid in pids_w_cutoff:
                    if pid not in shared[key]:
                        continue
                    data = shared[key][pid]
                    cutoff_day = cutoff_dict[pid]
                    if key in ['hospital', 'icu']:
                        shared[key][pid] = [(admission, discharge) for admission, discharge in data if discharge <= cutoff_day]
                    elif key in ['death', 'lastcontact']:
                        # Keep death/lastcontact dates even if after cutoff - these are outcomes, not features
                        pass
                    elif key in ['labs', 'vitals']:
                        shared[key][pid] = {d: v for d, v in data.items() if d <= cutoff_day}
                    elif key == 'meds':
                        shared[key][pid] = {d: meds for d, meds in data.items() if d <= cutoff_day}
        elif caia_portion == 'post':
            print("Loading post portion of CAIA dataset (data after cutoff day)")
            for key in ['hospital', 'icu', 'death', 'lastcontact', 'labs', 'vitals', 'meds']:
                for pid in pids_w_cutoff:
                    if pid not in shared[key]:
                        continue
                    data = shared[key][pid]
                    cutoff_day = cutoff_dict[pid]
                    if key in ['hospital', 'icu']:
                        shared[key][pid] = [(admission, discharge) for admission, discharge in data if admission > cutoff_day]
                    elif key in ['death', 'lastcontact']:
                        if data <= cutoff_day:
                            shared[key][pid] = np.nan
                    elif key in ['labs', 'vitals']:
                        shared[key][pid] = {d: v for d, v in data.items() if d > cutoff_day}
                    elif key == 'meds':
                        shared[key][pid] = {d: meds for d, meds in data.items() if d > cutoff_day}
        elif caia_portion == 'entire':
            print("Loading entire CAIA dataset (no cutoff applied)")
        else:
            raise ValueError(f"Unknown caia_portion value: {caia_portion}. Expected 'prior', 'post', or 'entire'")

        # Remove patients who have empty data after filtering
        if caia_portion in ['prior', 'post']:
            for key in ['labs', 'vitals', 'hospital', 'icu', 'meds']:
                original_count = len(shared[key])
                pids_to_remove = [pid for pid, data in shared[key].items() if not data]
                for pid in pids_to_remove:
                    del shared[key][pid]
                if pids_to_remove:
                    print(f"Removed {len(pids_to_remove)} patients with no {key} data after {caia_portion} filtering ({original_count} → {len(shared[key])})")

    # Restrict all dicts to patients present in the labs dict
    valid = set(shared['labs'].keys())
    for key in ['hospital', 'icu', 'meds', 'gender_oh', 'age', 'death', 'lastcontact', 'vitals', 'fold_num']:
        shared[key] = {k: v for k, v in shared[key].items() if k in valid}

    print("CAIA dataset loaded successfully")
    return shared


def _load_msk_demographics(dataset, data_source_path, option, competing_risk_death_check, include_gcsf, data_dict_lab_dict):
    df = pd.read_csv(f'{data_source_path}/MSK/dePHIed_demographic_df_MSK.csv', index_col=0)
    PRIMARY_CTYPES = df.value_counts('primary_cancer_type_th').index.to_list()
    DMP_ID_to_PRIMARY_CTYPE_dict = dict(zip(df['DMP_ID'], df['primary_cancer_type_th']))
    DMP_ID_to_DATES_to_TESTSVALUES_DICT = {}

    if dataset == 'MSK':
        data_dict_lab_dict['DMP_ID_to_PRIMARY_CTYPE_dict'] = DMP_ID_to_PRIMARY_CTYPE_dict
        if option in ['binary', 'binary_icu_pre'] or competing_risk_death_check:
            print("loading deathdate and lastcontactdate for MSK")
            data_dict_lab_dict['DMP_IDs_to_deathdate_dict'] = dict(
                zip(df.DMP_ID, df.deathdate_anchord_by_firstsequencingdate))
        data_dict_lab_dict['DMP_IDs_to_lastcontactdate_dict'] = dict(
            zip(df.DMP_ID, df.lastcontactdate_anchord_by_firstsequencingdate))

    if dataset == 'MSK_timeshift':
        with open(f'{data_source_path}/MSK/postbc_DMP_ID_to_DATES_to_TESTSVALUES_DICT_deidentified_MSK.pickle', 'rb') as f:
            og_dict = pickle.load(f)
        og_DMP_IDs = og_dict.keys()
        df_ts = pd.read_csv(f'{data_source_path}/{dataset}/dePHIed_demographic_df_{dataset}.csv', index_col=0)
        ctype_dict = dict(zip(df_ts['DMP_ID'], df_ts['primary_cancer_type']))
        ctype_dict = {k: v for k, v in ctype_dict.items() if k not in og_DMP_IDs}
        mapping = {k: v + [k] for k, v in timeshift_ctype_mapping_dict.items()}
        reverse_map = {v: k for k, vals in mapping.items() for v in vals}
        DMP_ID_to_PRIMARY_CTYPE_dict = {k: reverse_map.get(v, 'others') for k, v in ctype_dict.items()}
        data_dict_lab_dict['DMP_ID_to_PRIMARY_CTYPE_dict'] = DMP_ID_to_PRIMARY_CTYPE_dict
        if option in ['binary', 'binary_icu_pre'] or competing_risk_death_check:
            data_dict_lab_dict['DMP_IDs_to_deathdate_dict'] = dict(
                zip(df_ts.DMP_ID, df_ts.deathdate_anchord_by_firstsequencingdate))
        data_dict_lab_dict['DMP_IDs_to_lastcontactdate_dict'] = dict(
            zip(df_ts.DMP_ID, df_ts.lastcontactdate_anchord_by_firstsequencingdate))
        df = df_ts  # expose the timeshift df as the demographic df

    # if include_gcsf:
    with open(f'/data/morrisq/fuc/SPARC_data/DMP_ID_to_gcsf_startdates_dict.pickle', 'rb') as f:
        data_dict_lab_dict['gcsf_dict'] = pickle.load(f)

    return data_dict_lab_dict, DMP_ID_to_PRIMARY_CTYPE_dict, DMP_ID_to_DATES_to_TESTSVALUES_DICT, PRIMARY_CTYPES, df


def _load_mimic_demographics(dataset, data_source_path, option, competing_risk_death_check, data_dict_lab_dict):
    with open(f'{data_source_path}/{dataset}/MIMICIV_subjectid_to_mapped_cancer_type_dict_080625.pkl', 'rb') as f:
        DMP_ID_to_PRIMARY_CTYPE_dict = pickle.load(f)
    data_dict_lab_dict['DMP_ID_to_PRIMARY_CTYPE_dict'] = DMP_ID_to_PRIMARY_CTYPE_dict
    with open(f'{data_source_path}/{dataset}/MIMICIV_cancer_subjectid_to_lastcontactdate_dict_112325.pkl', 'rb') as f:
        lastcontact = pickle.load(f)
    if option in ['binary', 'binary_icu_pre'] or competing_risk_death_check:
        with open(f'{data_source_path}/{dataset}/MIMICIV_subjectid_to_admission_discharge_dict_112325.pkl', 'rb') as f:
            data_dict_lab_dict['deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict'] = pickle.load(f)
        with open(f'{data_source_path}/{dataset}/MIMIDIV_subjectid_to_deathdate_dict_112325.pkl', 'rb') as f:
            death = pickle.load(f)
        data_dict_lab_dict['DMP_IDs_to_deathdate_dict'] = death
        for pid, lcd in lastcontact.items():
            if pid not in death or pd.isna(death[pid]):
                data_dict_lab_dict['DMP_IDs_to_deathdate_dict'][pid] = np.nan
    data_dict_lab_dict['DMP_IDs_to_lastcontactdate_dict'] = lastcontact
    return data_dict_lab_dict, DMP_ID_to_PRIMARY_CTYPE_dict


def _load_caia_demographics(option, competing_risk_death_check, data_dict_lab_dict, caia_shared):
    if option in ['binary', 'binary_icu_pre'] or competing_risk_death_check:
        data_dict_lab_dict['DMP_IDs_to_deathdate_dict'] = caia_shared['death']
        for pid, lcd in caia_shared['lastcontact'].items():
            if pid not in caia_shared['death'] or pd.isna(caia_shared['death'][pid]):
                data_dict_lab_dict['DMP_IDs_to_deathdate_dict'][pid] = np.nan
    data_dict_lab_dict['DMP_IDs_to_lastcontactdate_dict'] = caia_shared['lastcontact']
    return data_dict_lab_dict


def _load_stanford_demographics(dataset, data_source_path, option, competing_risk_death_check, data_dict_lab_dict):
    with open(f'{data_source_path}/{dataset}/stanford_ehrshots_patient_id_to_death_date_dict_delta.pickle', 'rb') as f:
        death_dict = pickle.load(f)
    with open(f'{data_source_path}/{dataset}/stanford_ehrshots_patient_id_to_last_contact_date_dict_delta.pickle', 'rb') as f:
        lastcontact_dict = pickle.load(f)

    if option in ['binary', 'binary_icu_pre'] or competing_risk_death_check:
        # Convert None to np.nan for death dates
        death = {pid: (date if date is not None else np.nan) for pid, date in death_dict.items()}
        data_dict_lab_dict['DMP_IDs_to_deathdate_dict'] = death
    data_dict_lab_dict['DMP_IDs_to_lastcontactdate_dict'] = lastcontact_dict
    return data_dict_lab_dict


def _load_admission_dates(dataset, data_source_path, option, data_dict_lab_dict, caia_shared):
    if option in ['pretrain', 'binary']:
        if dataset in ['MSK', 'MSK_timeshift']:
            path = f'{data_source_path}/{dataset}/deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict_{dataset}.pickle'
            with open(path, 'rb') as f:
                data_dict_lab_dict['deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict'] = pickle.load(f)
        elif dataset == 'stanfordEHRSHOT':
            # Load separate admission and discharge lists, then zip them into tuples
            with open(f'{data_source_path}/{dataset}/stanford_ehrshots_patient_id_to_ip_admission_dates_dict_delta.pickle', 'rb') as f:
                ip_adm = pickle.load(f)
            with open(f'{data_source_path}/{dataset}/stanford_ehrshots_patient_id_to_ip_discharge_dates_dict_delta.pickle', 'rb') as f:
                ip_dis = pickle.load(f)
            # Zip admission and discharge dates into tuples, handling None values
            data_dict_lab_dict['deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict'] = {
                pid: list(zip(ip_adm[pid] or [], ip_dis[pid] or [])) for pid in ip_adm
            }
        elif dataset == 'MIMICIV_hosp':
            path = f'{data_source_path}/{dataset}/MIMICIV_subjectid_to_admission_discharge_dict_112325.pkl'
            with open(path, 'rb') as f:
                data_dict_lab_dict['deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict'] = pickle.load(f)
        elif 'CAIA' in dataset:
            data_dict_lab_dict['deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict'] = caia_shared['hospital']

        if option == 'pretrain' and dataset == 'MIMICIV_hosp':
            data_dict_lab_dict['deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict'] = {}

    if option in ['pretrain', 'binary_icu_pre']:
        if dataset == 'stanfordEHRSHOT' and option == 'binary_icu_pre':
            raise ValueError(
                "ICU data is not valid for stanfordEHRSHOT dataset. "
                "The ICU data is derived from ER inpatient records, not actual ICU admissions. "
                "Please use a different task or dataset."
            )

        if dataset in ['MSK', 'MSK_timeshift']:
            path = f'{data_source_path}/{dataset}/deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict_{dataset}.pickle'
            with open(path, 'rb') as f:
                data_dict_lab_dict['deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict'] = pickle.load(f)
        elif 'CAIA' in dataset:
            data_dict_lab_dict['deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict'] = caia_shared['icu']
        elif dataset == 'stanfordEHRSHOT':
            # ICU data is not valid for Stanford (derived from ER inpatient, not actual ICU)
            # Set to empty dict for pretrain task
            print("WARNING: ICU data is not available for stanfordEHRSHOT (ER data is not equivalent to ICU)")
            data_dict_lab_dict['deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict'] = {}
        else:
            data_dict_lab_dict['deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict'] = {}

    return data_dict_lab_dict


def _load_medications(dataset, data_source_path, data_dict_lab_dict, DMP_ID_to_DATES_to_TESTSVALUES_DICT, caia_shared):
    if dataset == 'MSK':
        with open(f'{data_source_path}/{dataset}/postbc_DMP_ID_to_DATES_to_nonoral_drug_cats_wdosage_deidentified_061925.pickle', 'rb') as f:
            meds = pickle.load(f)
    elif dataset == 'MSK_timeshift':
        with open(f'{data_source_path}/{dataset}/postbc_DMP_ID_to_DATES_to_nonoral_drug_cats_wdosage_deidentified_timeshift_060825.pickle', 'rb') as f:
            meds = pickle.load(f)
    elif 'CAIA' in dataset:
        # CAIA medication data already loaded in caia_shared['meds']
        meds = caia_shared['meds']
    elif dataset == 'stanfordEHRSHOT':
        # raise ValueError("stanfordEHRSHOT dataset does not have non-oral meds data")
        with open(f'{data_source_path}/{dataset}/stanford_ehrshots_patient_id_to_drug_exposure_dates_dict_delta.pickle', 'rb') as f:
            drug_exp_raw = pickle.load(f)
        # Convert from drug-centric {pid: {drug: [dates]}} to date-centric {pid: {date: [(drug, dosage)]}}
        meds = {}
        for pid, drug_dict in drug_exp_raw.items():
            meds[pid] = {}
            if drug_dict:
                for drug_name, dates_list in drug_dict.items():
                    for date in dates_list:
                        if date not in meds[pid]:
                            meds[pid][date] = []
                        # Stanford data doesn't have dosage, use None as placeholder
                        meds[pid][date].append((drug_name, None))
    elif dataset == 'MIMICIV_hosp':
        with open(f'{data_source_path}/{dataset}/MIMICIV_subjectid_to_date_to_all_medications_wdosage_dict.pkl', 'rb') as f:
            meds = pickle.load(f)
    else:
        raise ValueError(f"Medication loading not implemented for dataset: {dataset}")
    data_dict_lab_dict['DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT'] = meds
    return data_dict_lab_dict, DMP_ID_to_DATES_to_TESTSVALUES_DICT


def _load_labs(dataset, data_source_path, option, data_modules_to_include,
               data_dict_lab_dict, modular_featuredim_dict, caia_shared,
               DMP_ID_to_DATES_to_TESTSVALUES_DICT, first_principle=False):
    print("LABS")
    se_task = bool(re.match(
        r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option))

    # Get first-principle indices if requested
    fp_lab_indices, fp_vital_indices, fp_lab_dim, fp_vital_dim = None, None, None, None
    if first_principle and se_task:
        fp_lab_indices, fp_vital_indices, fp_lab_dim, fp_vital_dim = _get_first_principle_indices(option)

    if dataset in ['MSK', 'MSK_timeshift'] or 'CAIA' in dataset:
        if dataset in ['MSK', 'MSK_timeshift']:
            with open(f'{data_source_path}/{dataset}/postbc_DMP_ID_to_DATES_to_TESTSVALUES_DICT_deidentified_{dataset}.pickle', 'rb') as f:
                DMP_ID_to_DATES_to_TESTSVALUES_DICT = pickle.load(f)
        elif 'CAIA' in dataset:
            DMP_ID_to_DATES_to_TESTSVALUES_DICT = caia_shared['labs']

        if 'basic_lab' in data_modules_to_include:
            if first_principle and fp_lab_indices is not None:
                print("FIRST-PRINCIPLE LABS")
                basic = {k: {d: [v[i] for i in fp_lab_indices] for d, v in dates.items()}
                         for k, dates in DMP_ID_to_DATES_to_TESTSVALUES_DICT.items()}
                data_dict_lab_dict['basic_lab'] = basic
                modular_featuredim_dict['basic_lab'] = fp_lab_dim
            else:
                print("BASIC LABS")
                basic = {k: {d: [v[i] for i in BASIC_TESTS_INDICES] for d, v in dates.items()}
                         for k, dates in DMP_ID_to_DATES_to_TESTSVALUES_DICT.items()}
                data_dict_lab_dict['basic_lab'] = basic
                modular_featuredim_dict['basic_lab'] = MSK_numfeature_dict['basic_lab']

        if 'cbc_differential' in data_modules_to_include:
            print("CBC DIFFERENTIAL")
            cbc_diff_indices = [ALL_INCLUDED_TESTS.index(test) for test in CBC_DIFFERENTIAL_TESTS]
            cbc_diff = {k: {d: [v[i] for i in cbc_diff_indices] for d, v in dates.items()}
                        for k, dates in DMP_ID_to_DATES_to_TESTSVALUES_DICT.items()}
            data_dict_lab_dict['cbc_differential'] = cbc_diff
            modular_featuredim_dict['cbc_differential'] = len(CBC_DIFFERENTIAL_TESTS)

        if 'extra_lab' in data_modules_to_include or bool(re.match(r'^(neutropenia)(_next_risk|_risk)?$', option)):
            print("EXTRA LABS")
            extra = {k: {d: [v[i] for i in EXTRA_TESTS_INDICES] for d, v in dates.items()}
                     for k, dates in DMP_ID_to_DATES_to_TESTSVALUES_DICT.items()}
            if 'extra_lab' in data_modules_to_include:
                data_dict_lab_dict['extra_lab'] = extra
                modular_featuredim_dict['extra_lab'] = MSK_numfeature_dict['extra_lab']

        if se_task:
            data_dict_lab_dict['lab_def_dict'] = DMP_ID_to_DATES_to_TESTSVALUES_DICT

    elif dataset == 'stanfordEHRSHOT':
        with open(f'{data_source_path}/{dataset}/stanford_ehrshots_mapped_cancer_patient_id_to_date_to_all_included_lab_values_list_dict.pickle', 'rb') as f:
            all_labs = pickle.load(f)
        # Convert string values to numeric for side-effect tasks
        def convert_to_numeric(v):
            if isinstance(v, str):
                if v in ['NAN', 'NA', 'TNP', 'Negative', '']:
                    return np.nan
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return np.nan
            return v

        for pid in all_labs:
            for date in all_labs[pid]:
                all_labs[pid][date] = [convert_to_numeric(v) for v in all_labs[pid][date]]
        # Store full lab dict for side-effect tasks
        DMP_ID_to_DATES_to_TESTSVALUES_DICT = all_labs
        if 'basic_lab' in data_modules_to_include:
            if first_principle and fp_lab_indices is not None:
                print("FIRST-PRINCIPLE LABS")
                basic = {k: {d: [v[i] for i in fp_lab_indices] for d, v in dates.items()}
                         for k, dates in all_labs.items()}
                data_dict_lab_dict['basic_lab'] = basic
                modular_featuredim_dict['basic_lab'] = fp_lab_dim
            else:
                print("BASIC LABS")
                basic = {k: {d: [v[i] for i in BASIC_TESTS_INDICES] for d, v in dates.items()}
                         for k, dates in all_labs.items()}
                data_dict_lab_dict['basic_lab'] = basic
                modular_featuredim_dict['basic_lab'] = MSK_numfeature_dict['basic_lab']
        if 'cbc_differential' in data_modules_to_include:
            print("CBC DIFFERENTIAL")
            cbc_diff_indices = [ALL_INCLUDED_TESTS.index(test) for test in CBC_DIFFERENTIAL_TESTS]
            cbc_diff = {k: {d: [v[i] for i in cbc_diff_indices] for d, v in dates.items()}
                        for k, dates in all_labs.items()}

            # Convert percentage to absolute counts for Stanford EHRSHOT
            # (Stanford stores percentages, model expects absolute k/mcL)
            print("  Converting CBC differential from percentage to absolute counts...")
            wbc_index_in_all = ALL_INCLUDED_TESTS.index('WBC')

            for patient_id in cbc_diff:
                for date in cbc_diff[patient_id]:
                    # Get WBC count for this date
                    if patient_id in all_labs and date in all_labs[patient_id]:
                        wbc_count = all_labs[patient_id][date][wbc_index_in_all]

                        # Convert each differential percentage to absolute
                        if not np.isnan(wbc_count) and wbc_count is not None:
                            for i in range(len(cbc_diff[patient_id][date])):
                                diff_pct = cbc_diff[patient_id][date][i]
                                if not np.isnan(diff_pct) and diff_pct is not None:
                                    # Formula: absolute (k/mcL) = (percentage / 100) * WBC (k/mcL)
                                    cbc_diff[patient_id][date][i] = (diff_pct / 100.0) * wbc_count

            data_dict_lab_dict['cbc_differential'] = cbc_diff
            modular_featuredim_dict['cbc_differential'] = len(CBC_DIFFERENTIAL_TESTS)
        if 'extra_lab' in data_modules_to_include:
            print("EXTRA LABS")
            extra = {k: {d: [v[i] for i in EXTRA_TESTS_INDICES] for d, v in dates.items()}
                     for k, dates in all_labs.items()}
            data_dict_lab_dict['extra_lab'] = extra
            modular_featuredim_dict['extra_lab'] = MSK_numfeature_dict['extra_lab']
        if se_task:
            data_dict_lab_dict['lab_def_dict'] = DMP_ID_to_DATES_to_TESTSVALUES_DICT

    elif dataset == 'MIMICIV_hosp':
        with open(f'{data_source_path}/{dataset}/MIMICIV_subjectid_to_date_to_all_included_lab_values_list_dict.pkl', 'rb') as f:
            all_labs = pickle.load(f)
        # Convert string values to numeric for side-effect tasks
        def convert_to_numeric(v):
            if isinstance(v, str):
                if v in ['NAN', 'NA', 'TNP', 'Negative', '']:
                    return np.nan
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return np.nan
            return v

        for pid in all_labs:
            for date in all_labs[pid]:
                all_labs[pid][date] = [convert_to_numeric(v) for v in all_labs[pid][date]]

        # Store full lab dict for side-effect tasks (keep original unfiltered data)
        DMP_ID_to_DATES_to_TESTSVALUES_DICT = all_labs

        if 'basic_lab' in data_modules_to_include:
            if first_principle and fp_lab_indices is not None:
                print("FIRST-PRINCIPLE LABS")
                basic = {k: {d: [v[i] for i in fp_lab_indices] for d, v in dates.items()}
                         for k, dates in all_labs.items()}
                data_dict_lab_dict['basic_lab'] = basic
                modular_featuredim_dict['basic_lab'] = fp_lab_dim
            else:
                print("BASIC LABS")
                basic = {k: {d: [v[i] for i in BASIC_TESTS_INDICES] for d, v in dates.items()}
                         for k, dates in all_labs.items()}
                data_dict_lab_dict['basic_lab'] = basic
                modular_featuredim_dict['basic_lab'] = MSK_numfeature_dict['basic_lab']

        if 'cbc_differential' in data_modules_to_include:
            print("CBC DIFFERENTIAL")
            cbc_diff_indices = [ALL_INCLUDED_TESTS.index(test) for test in CBC_DIFFERENTIAL_TESTS]
            cbc_diff = {k: {d: [v[i] for i in cbc_diff_indices] for d, v in dates.items()}
                        for k, dates in all_labs.items()}

            # Convert percentage to absolute counts for MIMIC-IV
            # (MIMIC stores percentages, model expects absolute k/mcL)
            print("  Converting CBC differential from percentage to absolute counts...")
            wbc_index_in_all = ALL_INCLUDED_TESTS.index('WBC')

            for patient_id in cbc_diff:
                for date in cbc_diff[patient_id]:
                    # Get WBC count for this date
                    if patient_id in all_labs and date in all_labs[patient_id]:
                        wbc_count = all_labs[patient_id][date][wbc_index_in_all]

                        # Convert each differential percentage to absolute
                        if not np.isnan(wbc_count) and wbc_count is not None:
                            for i in range(len(cbc_diff[patient_id][date])):
                                diff_pct = cbc_diff[patient_id][date][i]
                                if not np.isnan(diff_pct) and diff_pct is not None:
                                    # Formula: absolute (k/mcL) = (percentage / 100) * WBC (k/mcL)
                                    cbc_diff[patient_id][date][i] = (diff_pct / 100.0) * wbc_count

            data_dict_lab_dict['cbc_differential'] = cbc_diff
            modular_featuredim_dict['cbc_differential'] = len(CBC_DIFFERENTIAL_TESTS)

        if 'extra_lab' in data_modules_to_include:
            print("EXTRA LABS")
            extra = {k: {d: [v[i] for i in EXTRA_TESTS_INDICES] for d, v in dates.items()}
                     for k, dates in all_labs.items()}
            data_dict_lab_dict['extra_lab'] = extra
            modular_featuredim_dict['extra_lab'] = MSK_numfeature_dict['extra_lab']

        if se_task:
            data_dict_lab_dict['lab_def_dict'] = DMP_ID_to_DATES_to_TESTSVALUES_DICT

    return data_dict_lab_dict, modular_featuredim_dict, DMP_ID_to_DATES_to_TESTSVALUES_DICT


def _load_vitals(dataset, data_source_path, option, data_modules_to_include,
                 data_dict_lab_dict, modular_featuredim_dict,
                 DMP_ID_to_DATES_to_TESTSVALUES_DICT, caia_shared, first_principle=False):
    print("VITALS")
    se_task = bool(re.match(
        r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option))

    # Get first-principle indices if requested
    fp_lab_indices, fp_vital_indices, fp_lab_dim, fp_vital_dim = None, None, None, None
    if first_principle and se_task:
        fp_lab_indices, fp_vital_indices, fp_lab_dim, fp_vital_dim = _get_first_principle_indices(option)

    if dataset in ['MSK', 'MSK_timeshift']:
        with open(f'{data_source_path}/{dataset}/postbc_DMP_ID_to_DATE_to_VITALTESTS_dict_051425redo.pickle', 'rb') as f:
            vitals = pickle.load(f)
    elif 'CAIA' in dataset:
        vitals = caia_shared['vitals']
        # add patients present in labs but missing from vitals
        for pid in DMP_ID_to_DATES_to_TESTSVALUES_DICT:
            if pid not in vitals:
                vitals[pid] = {}
    elif dataset == 'stanfordEHRSHOT':
        with open(f'{data_source_path}/{dataset}/stanford_ehrshots_mapped_cancer_patient_id_to_date_to_vitals_values_list_dict.pickle', 'rb') as f:
            vitals = pickle.load(f)
        # Convert string values to numeric for side-effect tasks
        def convert_to_numeric(v):
            if isinstance(v, str):
                if v in ['NAN', 'NA', 'TNP', 'Negative', '']:
                    return np.nan
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return np.nan
            return v

        for pid in vitals:
            for date in vitals[pid]:
                vitals[pid][date] = [convert_to_numeric(v) for v in vitals[pid][date]]
    elif dataset == 'MIMICIV_hosp':
        with open(f'{data_source_path}/{dataset}/MIMICIV_subjectid_to_date_to_vitals_values_list_dict.pkl', 'rb') as f:
            vitals = pickle.load(f)
        # Convert string values to numeric for side-effect tasks
        def convert_to_numeric(v):
            if isinstance(v, str):
                if v in ['NAN', 'NA', 'TNP', 'Negative', '']:
                    return np.nan
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return np.nan
            return v

        for pid in vitals:
            for date in vitals[pid]:
                vitals[pid][date] = [convert_to_numeric(v) for v in vitals[pid][date]]
    else:
        vitals = {}

    if 'basic_vitals' in data_modules_to_include:
        if first_principle and fp_vital_indices is not None and len(fp_vital_indices) > 0:
            print("FIRST-PRINCIPLE VITALS")
            data_dict_lab_dict['basic_vitals'] = {
                k: {d: [v[i] for i in fp_vital_indices] for d, v in dates.items()}
                for k, dates in vitals.items()
            }
            modular_featuredim_dict['basic_vitals'] = fp_vital_dim
        elif first_principle and fp_vital_indices is not None and len(fp_vital_indices) == 0:
            # First-principle task but no vitals needed (e.g., anemia, neutropenia)
            print("FIRST-PRINCIPLE: No vitals needed for this task")
            data_dict_lab_dict['basic_vitals'] = {k: {} for k in vitals.keys()}
            modular_featuredim_dict['basic_vitals'] = 0
        else:
            print("BASIC VITALS")
            data_dict_lab_dict['basic_vitals'] = {
                k: {d: [v[i] for i in BASIC_VITALS_INDICES] for d, v in dates.items()}
                for k, dates in vitals.items()
            }
            modular_featuredim_dict['basic_vitals'] = MSK_numfeature_dict['basic_vitals']

    if 'extra_vitals' in data_modules_to_include:
        print("EXTRA VITALS")
        data_dict_lab_dict['extra_vitals'] = {
            k: {d: [v[i] for i in EXTRA_VITALS_INDICES] for d, v in dates.items()}
            for k, dates in vitals.items()
        }
        modular_featuredim_dict['extra_vitals'] = MSK_numfeature_dict['extra_vitals']

    if se_task:
        data_dict_lab_dict['vitals_def_dict'] = vitals

    return data_dict_lab_dict, modular_featuredim_dict


def _load_gender(dataset, data_source_path, data_dict_lab_dict, modular_featuredim_dict,
                 dePHIed_demographic_df, caia_shared):
    print("GENDERS")
    GENDERS = ['MALE', 'FEMALE']
    if dataset in ['MSK', 'MSK_timeshift']:
        gender_raw = dict(zip(dePHIed_demographic_df['DMP_ID'], dePHIed_demographic_df['GENDER']))
        gender_oh, _ = one_hot_encode_dict(gender_raw, feature_reduced='MALE', unique_vals=GENDERS)
        data_dict_lab_dict['gender'] = gender_oh
    elif 'CAIA' in dataset:
        data_dict_lab_dict['gender'] = caia_shared['gender_oh']
    elif dataset == 'stanfordEHRSHOT':
        with open(f'{data_source_path}/{dataset}/stanford_ehrshots_patient_id_to_gender_dict_oh.pickle', 'rb') as f:
            gender_data = pickle.load(f)
            # Handle both tuple (new format) and dict (old format)
            data_dict_lab_dict['gender'] = gender_data[0] if isinstance(gender_data, tuple) else gender_data
    elif dataset == 'MIMICIV_hosp':
        with open(f'{data_source_path}/{dataset}/MIMICIV_cancer_subjectid_to_gender_dict_080625.pkl', 'rb') as f:
            gender_raw = pickle.load(f)
        gender_oh, _ = one_hot_encode_dict(gender_raw, feature_reduced='MALE', unique_vals=GENDERS)
        data_dict_lab_dict['gender'] = gender_oh
    modular_featuredim_dict['gender'] = MSK_numfeature_dict['gender']
    return data_dict_lab_dict, modular_featuredim_dict


def _load_age(dataset, data_source_path, data_dict_lab_dict, modular_featuredim_dict,
              dePHIed_demographic_df, caia_shared):
    print("AGE")
    if dataset in ['MSK', 'MSK_timeshift']:
        age = dict(zip(dePHIed_demographic_df['DMP_ID'], dePHIed_demographic_df['age_at_first_seq_binned_5']))
        data_dict_lab_dict['age'] = {k: [v] for k, v in age.items()}
    elif 'CAIA' in dataset:
        print("AGE CAIA")
        data_dict_lab_dict['age'] = {k: [v] for k, v in caia_shared['age'].items()}
    elif dataset == 'stanfordEHRSHOT':
        with open(f'{data_source_path}/{dataset}/stanford_ehrshots_age_at_first_lab_test_dict_binned5.pickle', 'rb') as f:
            age = pickle.load(f)
        data_dict_lab_dict['age'] = {k: [v] for k, v in age.items()}
    elif dataset == 'MIMICIV_hosp':
        with open(f'{data_source_path}/{dataset}/MIMICIV_cancer_subjectid_to_age_dict_binned5_080625.pkl', 'rb') as f:
            age = pickle.load(f)
        data_dict_lab_dict['age'] = {k: [v] for k, v in age.items()}
    modular_featuredim_dict['age'] = MSK_numfeature_dict['age']
    return data_dict_lab_dict, modular_featuredim_dict


def _load_mutation(dataset, data_source_path, data_dict_lab_dict, modular_featuredim_dict):
    print("MUTATION")
    if dataset == 'MSK':
        # df = pd.read_csv('/data/morrisq/fuc/impact_genomic_ft/outputs_050725/feature_table_full_first_341_harmonized.csv')
        df = pd.read_csv('/data/morrisq/fuc/impact_genomic_ft/output_050725/impact_genomic_ft_341.csv')
    elif dataset == 'MSK_timeshift': # this actually does NOT exusr
        # df = pd.read_csv('/data/morrisq/fuc/impact_genomic_ft/outputs_050725_timeshift/timeshift_feature_table_full_first_341_harmonized.csv')
        raise ValueError(f"Mutation features not available for dataset: {dataset}")
    else:
        raise ValueError(f"Mutation features not available for dataset: {dataset}")
    cols_to_drop = [                                                                                                                                     
        'Unnamed: 0',                                                                                                                                    
        'SAMPLE_ID',                                                                                                                                     
        'MONTH_ADDED',                                                                                                                                   
        'WEEK_ADDED',                                                                                                                                    
        'GENE_PANEL',                                                                                                                                    
        'CANCER_TYPE',                                                                                                                                   
        'CANCER_TYPE_DETAILED',                                                                                                                          
        'SAMPLE_TYPE',                                                                                                                                   
        'PRIMARY_SITE',                                                                                                                                  
        'METASTATIC_SITE',                                                                                                                               
        'ANCESTRY_LABEL',                                                                                                                                
        'YOST_INDEX_IMPUTED_MEDIAN',                                                                                                                     
        'CURRENT_AGE_DEID',                                                                                                                              
        'SEX',                                                                                                                                           
        'Classification_Category',
        'MSI_TYPE'                                                                                                                       
    ]     
    # df = df.drop(columns=['YOST_INDEX_IMPUTED_MEDIAN', 'CURRENT_AGE_DEID', 'SEX'])
    df = df.drop(columns=cols_to_drop)

    # Fill NaNs with 0 for genomic features (MSI_SCORE is the only remaining column with NaNs)
    # NaN in MSI_SCORE means test not performed, treat as 0 (not detected) for ML purposes
    df = df.fillna(0)

    feat_cols = [c for c in df.columns if c != 'PATIENT_ID']
    data_dict_lab_dict['mutation'] = dict(zip(df['PATIENT_ID'], df[feat_cols].values))
    modular_featuredim_dict['mutation'] = MSK_numfeature_dict['mutation']
    return data_dict_lab_dict, modular_featuredim_dict                                                                              
                                                                  
def _load_clinical(dataset, data_source_path, data_dict_lab_dict, modular_featuredim_dict,
                   DMP_ID_to_PRIMARY_CTYPE_dict, PRIMARY_CTYPES, dePHIed_demographic_df):
    print("CLINICAL FEATURES")
    if dataset == 'MSK':
        # df = pd.read_csv('/data/morrisq/fuc/impact_genomic_ft/outputs_050725/feature_table_full_first_341_harmonized.csv')
        df = pd.read_csv('/data/morrisq/fuc/impact_genomic_ft/output_050725/impact_genomic_ft_341.csv')
    elif dataset == 'MSK_timeshift': # this actually does NOT exusr
        # df = pd.read_csv('/data/morrisq/fuc/impact_genomic_ft/outputs_050725_timeshift/timeshift_feature_table_full_first_341_harmonized.csv')
        raise ValueError(f"Mutation features not available for dataset: {dataset}")
    else:
        raise ValueError(f"Mutation features not available for dataset: {dataset}")
    clinical = dict(zip(df['PATIENT_ID'], df[['YOST_INDEX_IMPUTED_MEDIAN']].values))
    df_ecog = pd.read_csv(f'{data_source_path}/{dataset}/df_ecog.csv', index_col=0)
    df_smoke = pd.read_csv(f'{data_source_path}/{dataset}/df_smoke.csv', index_col=0)
    ecog = dict(zip(df_ecog['DMP_ID'], df_ecog['ECOG_KPS']))
    smoke = dict(zip(df_smoke['DMP_ID'], df_smoke['smoking_predictions_3_classes']))
    smoke_oh, _ = one_hot_encode_dict(smoke, feature_reduced='Unknown', unique_vals=SMOKE_list)
    ctype_oh, oh_ctype_l = one_hot_encode_dict(
        DMP_ID_to_PRIMARY_CTYPE_dict, feature_reduced='others', unique_vals=PRIMARY_CTYPES)
    for pid in clinical:
        clinical[pid] = list(clinical[pid])
        clinical[pid].append(ecog.get(pid, 0))
        clinical[pid].extend(smoke_oh.get(pid, [0, 0]))
        clinical[pid].extend(ctype_oh.get(pid, [0] * len(oh_ctype_l)))
    data_dict_lab_dict['clinical'] = clinical
    modular_featuredim_dict['clinical'] = MSK_numfeature_dict['clinical']
    return data_dict_lab_dict, modular_featuredim_dict


def _apply_bundle_missingness(data_dict_lab_dict, rate, seed):
    """
    Simulate resource-scarce clinical settings by dropping entire lab/vital bundles.

    For each encounter (patient × date) and each bundle, the ENTIRE bundle is
    absent (all features set to np.nan) with probability `rate`, independently
    per encounter. This reflects the clinical reality that panels are ordered — or
    not — as a unit: if a CBC wasn't ordered or the sample was lost, none of the
    CBC values are available.

    Missing features are set to np.nan, matching the pipeline's existing
    convention for unmeasured values. The data loader uses nanmean aggregation,
    so NaN entries are correctly excluded rather than treated as mean values.

    Bundles within basic_lab (23 features = BASIC_TESTS = RBC_CBC_TESTS + BMP_TESTS + LFT_TESTS):
        CBC:    indices  0-8  (RBC, MCV, MCH, MCHC, RDW, HGB, HCT, WBC, Platelets)  — 9 features
        BMP:    indices  9-16 (Glucose, Calcium, Sodium, Potassium, CO2, Creatinine, Chloride, BUN) — 8 features
        LFT:    indices 17-22 (Albumin, Protein Total, ALP, ALT, AST, Bilirubin Total) — 6 features

    Bundles within basic_vitals (7 features = BASIC_VITALS):
        Vitals: indices  0-6  (BMI, BP Sys, BP Dia, O2, Pulse, Resp, Temp) — 7 features

    Parameters
    ----------
    data_dict_lab_dict : dict
        Nested dict {module: {pid: {date: [feature_values]}}}.
        Modified in-place.
    rate : float
        Probability that an entire bundle is absent at a given encounter (0–1).
    seed : int
        Global seed for reproducible per-encounter bundle-absence draws.

    Returns
    -------
    data_dict_lab_dict : dict
        The same dict, with dropped bundle features set to np.nan per encounter.
    """
    from MSK_featurename_list import BASIC_TESTS, BASIC_VITALS

    # Bundle definitions: {module: {bundle_name: [feature_indices_within_module]}}
    bundles = {
        'basic_lab': {
            'CBC':    list(range(0, 9)),    # 9 features
            'BMP':    list(range(9, 17)),   # 8 features
            'LFT':    list(range(17, 23)),  # 6 features
        },
        'basic_vitals': {
            'Vitals': list(range(0, 7)),    # 7 features
        },
    }

    # Stable integer IDs for modules so numpy seeds are reproducible without
    # relying on hash() (which varies with PYTHONHASHSEED).
    MODULE_IDS = {'basic_lab': 0, 'basic_vitals': 1}

    for module, bundle_dict in bundles.items():
        if module not in data_dict_lab_dict or not data_dict_lab_dict[module]:
            continue
        feature_dict = data_dict_lab_dict[module]

        # Determine actual feature count from data
        any_pid = next(iter(feature_dict))
        if not feature_dict[any_pid]:
            continue
        any_date = next(iter(feature_dict[any_pid]))
        n_features = len(feature_dict[any_pid][any_date])

        # Clip bundle indices to actual feature count
        clipped_bundles = {
            bname: [i for i in bidx if i < n_features]
            for bname, bidx in bundle_dict.items()
        }

        print(f"[bundle_missingness] {module}: whole-bundle dropout, rate={rate}, "
              f"bundles: { {b: len(v) for b, v in clipped_bundles.items()} }")

        # Sort encounters for stable ordering regardless of dict insertion order.
        # Stability ensures the same masks are produced across runs on the same dataset.
        encounters = sorted(
            (pid, d) for pid in feature_dict for d in feature_dict[pid]
        )
        n_enc = len(encounters)
        if n_enc == 0:
            continue

        # Stack all encounter feature vectors into a 2D float array (n_enc, n_features).
        # After _normalize(), all values are already float or np.nan.
        data_array = np.array(
            [feature_dict[pid][d] for pid, d in encounters], dtype=float
        )

        # # [OPTION A - partial feature masking within each bundle, commented out]
        # # Vectorized per-bundle masking: one numpy RNG call generates all n_enc masks.
        # for b_idx, (bundle_name, valid) in enumerate(clipped_bundles.items()):
        #     n_mask = n_mask_per_bundle[bundle_name]
        #     if n_mask == 0:
        #         continue
        #     valid_arr = np.array(valid)
        #     # Seed is (global_seed, module_id, bundle_position) — fully deterministic,
        #     # no Python hash() or md5 needed.
        #     b_rng = np.random.default_rng([seed, MODULE_IDS[module], b_idx])
        #     # Shape: (n_enc, len(valid)) — each row is an independent permutation.
        #     all_perms = b_rng.permuted(np.tile(valid_arr, (n_enc, 1)), axis=1)
        #     # Fancy-index NaN assignment: rows = encounter indices, cols = masked features.
        #     data_array[np.arange(n_enc)[:, None], all_perms[:, :n_mask]] = np.nan

        # [OPTION B - whole-bundle Bernoulli dropout]
        # For each bundle, the ENTIRE panel is absent with probability `rate`, independently
        # per encounter. Clinically: a panel either was ordered and resulted, or wasn't — not
        # partially missing.
        for b_idx, (bundle_name, valid) in enumerate(clipped_bundles.items()):
            if not valid:
                continue
            valid_arr = np.array(valid)
            # Seed is (global_seed, module_id, bundle_position) — fully deterministic.
            b_rng = np.random.default_rng([seed, MODULE_IDS[module], b_idx])
            # Bernoulli draw: True = entire bundle absent at this encounter.
            absent = b_rng.random(n_enc) < rate  # shape (n_enc,)
            absent_rows = np.where(absent)[0]
            if absent_rows.size > 0:
                data_array[np.ix_(absent_rows, valid_arr)] = np.nan

        # Write results back; tolist() converts numpy float64 → Python float (NaN preserved).
        for enc_idx, (pid, d) in enumerate(encounters):
            feature_dict[pid][d] = data_array[enc_idx].tolist()

    return data_dict_lab_dict


def _normalize(dataset, data_source_path, modules, data_dict_lab_dict, first_principle=False):
    """
    Normalize numeric time-series features in-place using per-feature z-scores.
    Persists the normalized data dict and normalization factors to disk so
    subsequent runs skip recomputation.

    For first-principle runs, caching is disabled since different tasks use different features.
    """
    # Skip caching for first-principle runs (different tasks = different features)
    if first_principle:
        use_cache = False
        normalized_data_path = None
        norm_factor_path = None
    else:
        use_cache = True
        if dataset in ['MSK_CAIA', 'MSK_CAIA_timeshift']:
            if 'timeshift' not in dataset:
                # normalized_data_path = f'{data_source_path}/{dataset}/normalized_data_dict_lab_dict_{modules}_till2021_031226.pickle'
                normalized_data_path = f'{data_source_path}/{dataset}/normalized_data_dict_{modules}_till2021_031226.pickle'
                norm_factor_path = f'{data_source_path}/{dataset}/normalization_factor_dict_{modules}_till2021_031226.pickle'
            else:
                normalized_data_path = f'{data_source_path}/{dataset}/normalized_data_dict_{modules}_after2021_031226.pickle'
                norm_factor_path = f'{data_source_path}/{dataset}/normalization_factor_dict_{modules}_after2021_031226.pickle'
        else:
            normalized_data_path = f'{data_source_path}/{dataset}/normalized_data_dict_lab_dict_{dataset}_{modules}.pickle'
            norm_factor_path = f'{data_source_path}/{dataset}/normalization_factor_dict_{dataset}_{modules}.pickle'
        # normalized_data_path = f'{data_source_path}/{dataset}/normalized_data_dict_lab_dict_{dataset}_{modules}.pickle'
        # norm_factor_path = f'{data_source_path}/{dataset}/normalization_factor_dict_{dataset}_{modules}.pickle'


    print("use_cache", use_cache, "normalized_data_path", normalized_data_path, "norm_factor_path", norm_factor_path,
          os.path.exists(normalized_data_path) if normalized_data_path is not None else None,
          os.path.exists(norm_factor_path) if norm_factor_path is not None else None)

    # Define clinical ranges for filtering outliers during NORMALIZATION CALCULATION ONLY (MIMIC-IV)
    # These ranges are only applied when computing mean/std, NOT to the actual data
    mimic_filtering_ranges = None
    if dataset == 'MIMICIV_hosp':
        print("  Using filtered values for MIMIC-IV normalization calculation (data unchanged)")
        mimic_filtering_ranges = {
            'AST': (0, 1000),           # U/L - std is 2.71x MSK
            'CO2': (5, 60),             # mEq/L - std is 1.94x MSK
            'ALT': (0, 1000),           # U/L - std is 1.88x MSK
            'BUN': (2, 200),            # mg/dL - std is 1.73x MSK
            'Creatinine': (0.2, 15),    # mg/dL - std is 1.58x MSK
        }
        # Map test names to indices in BASIC_TESTS
        from MSK_featurename_list import BASIC_TESTS
        mimic_filtering_indices = {
            ('basic_lab', BASIC_TESTS.index(test_name)): (min_val, max_val)
            for test_name, (min_val, max_val) in mimic_filtering_ranges.items()
        }
    else:
        mimic_filtering_indices = {}

    if not use_cache or not os.path.exists(normalized_data_path):
        # Load pre-computed factors if available (different file from the full data).
        if use_cache and norm_factor_path and os.path.exists(norm_factor_path):
            with open(norm_factor_path, 'rb') as f:
                normalization_factor_dict = pickle.load(f)
        else:
            normalization_factor_dict = {}

        for k, feature_dict in data_dict_lab_dict.items():
            if k not in ['basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals', 'extra_vitals']:
                continue
            any_pid = next(iter(feature_dict))
            if not feature_dict[any_pid]:
                print(f"Skipping normalization for {k}: no data")
                continue
            any_date = next(iter(feature_dict[any_pid]))
            num_tests = len(feature_dict[any_pid][any_date])

            for test_idx in range(num_tests):
                if (k, test_idx) not in normalization_factor_dict:
                    vals = []
                    vals_capped = 0
                    for pid in feature_dict:
                        for d in feature_dict[pid]:
                            v = feature_dict[pid][d][test_idx]
                            try:
                                val = float(v)
                                # Apply filtering for MIMIC-IV normalization calculation only
                                if (k, test_idx) in mimic_filtering_indices and not np.isnan(val):
                                    min_val, max_val = mimic_filtering_indices[(k, test_idx)]
                                    original_val = val
                                    val = np.clip(val, min_val, max_val)
                                    if val != original_val:
                                        vals_capped += 1
                                vals.append(val)
                            except (ValueError, TypeError):
                                continue

                    # Log filtering statistics for MIMIC
                    if (k, test_idx) in mimic_filtering_indices and len(vals) > 0:
                        print(f"    {k}[{test_idx}]: capped {vals_capped}/{len(vals)} values ({100*vals_capped/len(vals):.2f}%) during normalization calculation")

                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                else:
                    mean, std = normalization_factor_dict[(k, test_idx)]
                print("test_idx", test_idx, "mean", mean, "std", std)
                normalization_factor_dict[(k, test_idx)] = (mean, std)

                for pid in feature_dict:
                    for d in feature_dict[pid]:
                        v = feature_dict[pid][d][test_idx]
                        try:
                            if v in ['NAN', 'NA', 'TNP', 'Negative']:
                                v = np.nan
                            fval = float(v)
                        except (ValueError, TypeError):
                            fval = np.nan
                        feature_dict[pid][d][test_idx] = (fval - mean) / std if std > 0 else 0.0

        # Save cache only if not first-principle
        if use_cache:
            if not os.path.exists(norm_factor_path):
                with open(norm_factor_path, 'wb') as f:
                    pickle.dump(normalization_factor_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            if not os.path.exists(normalized_data_path):
                with open(normalized_data_path, 'wb') as f:
                    pickle.dump(data_dict_lab_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load from cache (only for non-first-principle runs)
        print("loading existing normalized data_dict_lab_dict")
        with open(normalized_data_path, 'rb') as f:
            loaded = pickle.load(f)
        for k in loaded:
            if k in ['basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals', 'extra_vitals']:
                data_dict_lab_dict[k] = loaded[k]

    return data_dict_lab_dict


def _build_splits(dataset, data_source_path, option, mode, valfold, rseed,
                  data_fraction_option, DMP_ID_to_DATES_to_TESTSVALUES_DICT,
                  data_dict_lab_dict, quick_test, caia_shared):
    """Return {'train': [...], 'val': [...], 'test': [...]} for the requested dataset/fold/mode."""

    if dataset == 'MSK':
        with open(f'{data_source_path}/{dataset}/fold_dict_11.pickle', 'rb') as f:
            fold_dict = pickle.load(f)
        train_ids = [pid for i in range(10) if i != valfold for pid in fold_dict[i]]
        val_ids = fold_dict[valfold]
        test_ids = fold_dict[10]
        # Filter to patients with lab data
        train_ids = [p for p in train_ids if p in DMP_ID_to_DATES_to_TESTSVALUES_DICT]
        val_ids   = [p for p in val_ids   if p in DMP_ID_to_DATES_to_TESTSVALUES_DICT]
        test_ids  = [p for p in test_ids  if p in DMP_ID_to_DATES_to_TESTSVALUES_DICT]
        print("train, val, test split done, sizes are", len(train_ids), len(val_ids), len(test_ids))

        if data_fraction_option != 'default':
            fraction = int(data_fraction_option.split('/')[1])
            random.seed(rseed + 1)
            shuffled = copy.deepcopy(train_ids)
            random.shuffle(shuffled)
            train_ids = shuffled[:len(train_ids) // fraction]
            print("using data fraction", data_fraction_option, "→ train size", len(train_ids))

        split_dict = {'train': train_ids, 'val': val_ids, 'test': test_ids}
        if mode == 'recalibrate':
            split_dict = {'train': [], 'val': [], 'test': val_ids}
        elif mode in ['test', 'test_recalibrate']:
            split_dict = {'train': [], 'val': [], 'test': test_ids}

    elif dataset == 'MSK_timeshift':
        if mode == 'train':
            with open(f'{data_source_path}/{dataset}/fold_dict_1v1v1_{dataset}.pickle', 'rb') as f:
                fold_dict = pickle.load(f)
            split_dict = {'train': fold_dict['train'], 'val': fold_dict['val'], 'test': fold_dict['test']}
        else:
            fold_dict = {10: data_dict_lab_dict['age'].keys()}
            # get all DMP_ID in the MSK dataset and exclude from the fold_dict
            data_path = f'{data_source_path}/MSK/postbc_DMP_ID_to_DATES_to_TESTSVALUES_DICT_deidentified_MSK.pickle'
            with open(data_path, 'rb') as handle:
                og_set = pickle.load(handle)
            fold_dict[10] = [DMP_ID for DMP_ID in fold_dict[10] if DMP_ID not in og_set.keys()]
            split_dict = {'train': [], 'val': [], 'test': fold_dict[10]}
        print("train, val, test split done, sizes are",
            len(split_dict['train']), len(split_dict['val']), len(split_dict['test']))

    elif 'CAIA' in dataset:
        # if mode == 'test':
        #     all_ids = list(set(data_dict_lab_dict['age'].keys()))
        #     split_dict = {'train': [], 'val': [], 'test': all_ids}
        # else:

        # Build fold_dict (fold_num -> list of patient IDs) from fold_num dict (patient ID -> fold_num)
        fold_dict = {}
        for pid, fold_num in caia_shared['fold_num'].items():
            if fold_num not in fold_dict:
                fold_dict[fold_num] = []
            fold_dict[fold_num].append(pid)

        # Folds 0-9 are for cross-validation, fold 10 is held-out test set
        train_ids = [pid for i in range(11) if i != valfold and i != 10 for pid in fold_dict.get(i, [])]
        val_ids   = fold_dict.get(valfold, [])
        test_ids  = fold_dict.get(10, [])

        # Filter to patients with lab data
        train_ids = [p for p in train_ids if p in DMP_ID_to_DATES_to_TESTSVALUES_DICT]
        val_ids   = [p for p in val_ids   if p in DMP_ID_to_DATES_to_TESTSVALUES_DICT]
        test_ids  = [p for p in test_ids  if p in DMP_ID_to_DATES_to_TESTSVALUES_DICT]
        if mode == 'recalibrate':
            test_ids = val_ids
            val_ids = []
        elif mode in ['test', 'test_recalibrate']:
            train_ids = []
            val_ids = []
        elif mode == 'test_all':
            test_ids = train_ids + val_ids + test_ids
        print("train, val, test split done, sizes are", len(train_ids), len(val_ids), len(test_ids))

        if data_fraction_option != 'default':
            fraction = int(data_fraction_option.split('/')[1])
            random.seed(rseed + 1)
            shuffled = copy.deepcopy(train_ids)
            random.shuffle(shuffled)
            train_ids = shuffled[:len(train_ids) // fraction]
            print("using data fraction", data_fraction_option, "→ train size", len(train_ids))

        split_dict = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    elif dataset == 'stanfordEHRSHOT':
        with open(f'{data_source_path}/{dataset}/stanford_ehrshots_mapped_cancer_patient_id_to_date_to_vitals_values_list_dict.pickle', 'rb') as f:
            tmp = pickle.load(f)
        all_ids = list(tmp.keys())
        random.seed(rseed + 1)
        random.shuffle(all_ids)
        n = len(all_ids)
        split_dict = {
            'train': all_ids[:int(n * 0.33)],
            'val':   all_ids[int(n * 0.3):int(n * 0.66)],
            'test':  all_ids[int(n * 0.66):],
        }
        print("train, val, test split done, sizes are",
              len(split_dict['train']), len(split_dict['val']), len(split_dict['test']))

    elif dataset == 'MIMICIV_hosp':
        with open(f'{data_source_path}/{dataset}/split_dict_1v1v1_112425.pickle', 'rb') as f:
            raw = pickle.load(f)
        split_dict = {'train': raw['train'], 'val': raw['val'], 'test': raw['test']}

    else:
        raise ValueError(f"Unknown dataset for split building: {dataset!r}")
    
    if quick_test:
        for split in split_dict:
            split_dict[split] = split_dict[split][:512*4]
        print("QUICK TEST MODE: using only 4 batches (512*4 samples) for each split, new sizes are",
              len(split_dict['train']), len(split_dict['val']), len(split_dict['test']))

    return split_dict


