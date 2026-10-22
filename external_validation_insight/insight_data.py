"""INSIGHT data loading for the 'insight' (one site) and 'insight_pooled' (sites A-F) datasets of
dataset_loading.load_dataset. data_source_path is a processed site folder or the root holding A-F.
Without split files every patient is in test (full-site zero-shot evaluation).
"""
import os
import sys
import gc
import json
import pickle
import random

import numpy as np
import pandas as pd

# repo root (sparc_paths) and data/ (MSK_featurename_list, dataset_loading) importable from anywhere
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, 'data')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
from MSK_featurename_list import (ALL_INCLUDED_TESTS, BASIC_TESTS, CBC_DIFFERENTIAL_TESTS, BASIC_VITALS_INDICES,
                                  EXTRA_VITALS_INDICES, MSK_numfeature_dict)

INSIGHT_DATASETS = ('insight', 'insight_pooled')
SITES = ['A', 'B', 'C', 'D', 'E', 'F']
FEATURE_MODULES = ['basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals', 'extra_vitals']
Z_CLIP = 100.0   # normalized values are clipped to +/-100, as in the runs behind the published INSIGHT results


def _raw_csv(name):
    """Raw CDM table; the demographics table is demographics.csv in the original extract, demographic.csv in later pulls."""
    names = ['demographics.csv', 'demographic.csv'] if name == 'demographics' else [f'{name}.csv']
    for n in names:
        p = os.path.join(sparc_paths.INSIGHT_RAW_DIR, n)
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f'{names[0]} not found in {sparc_paths.INSIGHT_RAW_DIR} (set SPARC_INSIGHT_RAW_DIR)')


def _site_dirs(dataset, data_source_path):
    return [data_source_path] if dataset == 'insight' else [os.path.join(data_source_path, s) for s in SITES]


def _merged_pickle(dataset, data_source_path, fname):
    merged = {}
    for d in _site_dirs(dataset, data_source_path):
        with open(os.path.join(d, fname), 'rb') as f:
            merged.update(pickle.load(f))
    return merged


def _demographics(option, competing_risk_death_check, data_dict_lab_dict):
    """Death and last-contact days, both relative to the earliest enrollment date."""
    enr = _raw_csv('enrollment')
    enr['ssid'] = enr['ssid'].astype(str)
    enr['enr_start_date'] = pd.to_datetime(enr['enr_start_date'])
    enr['enr_end_date'] = pd.to_datetime(enr['enr_end_date'])
    enr_start = enr.groupby('ssid')['enr_start_date'].min()
    enr_end = enr.groupby('ssid')['enr_end_date'].max()

    death_df = _raw_csv('death')
    death_df['ssid'] = death_df['ssid'].astype(str)
    death_df['death_date'] = pd.to_datetime(death_df['death_date'])
    death_map = dict(zip(death_df['ssid'], death_df['death_date']))

    if option in ['binary', 'binary_icu_pre'] or competing_risk_death_check:
        data_dict_lab_dict['DMP_IDs_to_deathdate_dict'] = {
            ssid: ((death_map[ssid] - enr_start[ssid]).days
                   if ssid in death_map and not pd.isna(death_map[ssid]) else np.nan)
            for ssid in enr_start.index}
    data_dict_lab_dict['DMP_IDs_to_lastcontactdate_dict'] = {
        ssid: int((enr_end[ssid] - enr_start[ssid]).days) if ssid in enr_end.index else 0
        for ssid in enr_start.index}
    return data_dict_lab_dict


def _admissions(dataset, data_source_path, option, data_dict_lab_dict):
    if option in ['pretrain', 'binary']:
        data_dict_lab_dict['deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict'] = _merged_pickle(
            dataset, data_source_path, 'ssid_to_hospital_adm_dsch_dict.pickle')
    if option in ['pretrain', 'binary_icu_pre']:
        # PCORnet CDM has no ICU stays
        data_dict_lab_dict['deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict'] = {}
    return data_dict_lab_dict


def _labs(dataset, data_source_path, data_modules_to_include, se_task, data_dict_lab_dict, modular_featuredim_dict):
    need_basic = 'basic_lab' in data_modules_to_include
    need_cbc = 'cbc_differential' in data_modules_to_include
    if need_basic:
        data_dict_lab_dict['basic_lab'] = {}
    if need_cbc:
        data_dict_lab_dict['cbc_differential'] = {}
    if se_task:
        data_dict_lab_dict['lab_def_dict'] = {}
    for d in _site_dirs(dataset, data_source_path):
        with open(os.path.join(d, 'ssid_to_date_to_labs_dict.pickle'), 'rb') as f:
            raw_labs = pickle.load(f)   # {ssid: {day: {test name: value}}}
        for pid, dates in raw_labs.items():
            if need_basic:
                data_dict_lab_dict['basic_lab'][pid] = {
                    day: np.array([v.get(t, np.nan) for t in BASIC_TESTS], dtype=np.float64) for day, v in dates.items()}
            if need_cbc:
                data_dict_lab_dict['cbc_differential'][pid] = {
                    day: np.array([v.get(t, np.nan) for t in CBC_DIFFERENTIAL_TESTS], dtype=np.float64)
                    for day, v in dates.items()}
            if se_task:
                data_dict_lab_dict['lab_def_dict'][pid] = {
                    day: np.array([v.get(t, np.nan) for t in ALL_INCLUDED_TESTS], dtype=np.float64)
                    for day, v in dates.items()}
        del raw_labs
        gc.collect()
    if need_basic:
        modular_featuredim_dict['basic_lab'] = MSK_numfeature_dict['basic_lab']
    if need_cbc:
        modular_featuredim_dict['cbc_differential'] = len(CBC_DIFFERENTIAL_TESTS)
    return data_dict_lab_dict, modular_featuredim_dict


def _vitals(dataset, data_source_path, option, data_modules_to_include, se_task, first_principle,
            data_dict_lab_dict, modular_featuredim_dict):
    from dataset_loading import _get_first_principle_indices
    vitals = {}
    for d in _site_dirs(dataset, data_source_path):
        with open(os.path.join(d, 'ssid_to_date_to_vitals_dict.pickle'), 'rb') as f:
            raw = pickle.load(f)   # {ssid: {day: 9 values in VITALS order, 'nan' when missing}}
        for pid, dates in raw.items():
            vitals[pid] = {day: [np.nan if str(v) == 'nan' else float(v) for v in row] for day, row in dates.items()}
        del raw
        gc.collect()

    fp_vital_indices = fp_vital_dim = None
    if first_principle and se_task:
        _, fp_vital_indices, _, fp_vital_dim = _get_first_principle_indices(option)
    if 'basic_vitals' in data_modules_to_include:
        if first_principle and fp_vital_indices is not None and len(fp_vital_indices) > 0:
            data_dict_lab_dict['basic_vitals'] = {
                k: {day: [v[i] for i in fp_vital_indices] for day, v in dates.items()} for k, dates in vitals.items()}
            modular_featuredim_dict['basic_vitals'] = fp_vital_dim
        elif first_principle and fp_vital_indices is not None:
            data_dict_lab_dict['basic_vitals'] = {k: {} for k in vitals}
            modular_featuredim_dict['basic_vitals'] = 0
        else:
            data_dict_lab_dict['basic_vitals'] = {
                k: {day: [v[i] for i in BASIC_VITALS_INDICES] for day, v in dates.items()} for k, dates in vitals.items()}
            modular_featuredim_dict['basic_vitals'] = MSK_numfeature_dict['basic_vitals']
    if 'extra_vitals' in data_modules_to_include:
        data_dict_lab_dict['extra_vitals'] = {
            k: {day: [v[i] for i in EXTRA_VITALS_INDICES] for day, v in dates.items()} for k, dates in vitals.items()}
        modular_featuredim_dict['extra_vitals'] = MSK_numfeature_dict['extra_vitals']
    if se_task:
        data_dict_lab_dict['vitals_def_dict'] = vitals
    return data_dict_lab_dict, modular_featuredim_dict


def _gender_age(data_modules_to_include, data_dict_lab_dict, modular_featuredim_dict):
    demo = _raw_csv('demographics')
    demo['ssid'] = demo['ssid'].astype(str)
    if 'gender' in data_modules_to_include:
        data_dict_lab_dict['gender'] = {r['ssid']: [1.0 if r['sex'] == 'F' else 0.0] for _, r in demo.iterrows()}
        modular_featuredim_dict['gender'] = MSK_numfeature_dict['gender']
    if 'age' in data_modules_to_include:
        # 5-year bin of the age at the earliest enrollment
        demo['birth_date'] = pd.to_datetime(demo['birth_date'])
        enr = _raw_csv('enrollment')
        enr['ssid'] = enr['ssid'].astype(str)
        enr['enr_start_date'] = pd.to_datetime(enr['enr_start_date'])
        enr_date = enr.groupby('ssid')['enr_start_date'].min()
        data_dict_lab_dict['age'] = {
            ssid: [int(((enr_date[ssid] - birth).days / 365.25) // 5)]
            for ssid, birth in demo.set_index('ssid')['birth_date'].items() if ssid in enr_date.index}
        modular_featuredim_dict['age'] = MSK_numfeature_dict['age']
    return data_dict_lab_dict, modular_featuredim_dict


def _clip(data_dict_lab_dict):
    for k in FEATURE_MODULES:
        for dates in data_dict_lab_dict.get(k, {}).values():
            for day, arr in dates.items():
                a = np.asarray(arr, dtype=np.float64)
                if a.size:
                    dates[day] = np.clip(a, -Z_CLIP, Z_CLIP)   # NaN stays NaN
    return data_dict_lab_dict


def _splits(dataset, data_source_path, mode, rseed, data_fraction_option, data_dict_lab_dict, quick_test):
    lab_ids = set(data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('cbc_differential', {})).keys())

    def read(base, site):
        out = []
        for part in ('train', 'val', 'test'):
            with open(os.path.join(base, part, f'{site}.json')) as f:
                out.append([p for p in json.load(f) if p in lab_ids])
        return out

    if dataset == 'insight':
        site = os.path.basename(os.path.normpath(data_source_path))
        base = os.path.join(os.path.dirname(os.path.normpath(data_source_path)), 'split')
        if os.path.exists(os.path.join(base, 'train', f'{site}.json')):
            train, val, test = read(base, site)
            split_dict = ({'train': train, 'val': val, 'test': test} if mode == 'train'
                          else {'train': [], 'val': [], 'test': test})
            print(f'insight split ({site}): train={len(split_dict["train"])}, val={len(split_dict["val"])}, '
                  f'test={len(test)}')
        else:
            # sorted: a set's order changes between processes, which would change the bootstrap CIs between runs
            split_dict = {'train': [], 'val': [], 'test': sorted(lab_ids)}
            print(f'insight split ({site}): no split files, all {len(lab_ids)} patients in test')
    else:
        base = os.path.join(data_source_path, 'split')
        train, val, test = [], [], []
        for site in SITES:
            tr, va, te = read(base, site)
            train += tr; val += va; test += te
        split_dict = {'train': train, 'val': val, 'test': test}
        print(f'insight_pooled split: train={len(train)}, val={len(val)}, test={len(test)}')

    # training-set subsampling for the scaling study: '1/k' keeps 1/k of the training patients, 'p=<float>' that
    # proportion; val and test are unchanged, and the seed (rseed + 1) is the same for every fold
    if data_fraction_option != 'default' and split_dict['train']:
        ids = sorted(split_dict['train'])
        p = (1.0 / int(data_fraction_option.split('/')[1]) if data_fraction_option.startswith('1/')
             else float(data_fraction_option.split('=')[1]))
        random.seed(rseed + 1)
        random.shuffle(ids)
        split_dict['train'] = ids[:max(1, int(round(len(ids) * p)))]
        print(f'data fraction {data_fraction_option} (p={p:.4f}): {len(split_dict["train"])} of {len(ids)} '
              f'training patients')
    if quick_test:
        split_dict = {k: v[:512 * 4] for k, v in split_dict.items()}
    return split_dict


def load_insight(dataset, data_source_path, option, modules, mode, rseed=1024, competing_risk_death_check=False,
                 data_fraction_option='default', quick_test=False, normalize=True, first_principle=False):
    """Load an INSIGHT dataset; returns the same tuple as dataset_loading.load_dataset."""
    import re
    from dataset_loading import _resolve_modules, _normalize
    assert dataset in INSIGHT_DATASETS, dataset
    data_modules_to_include, component = _resolve_modules(modules)
    data_dict_lab_dict, modular_featuredim_dict = {}, {}
    se_task = bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option))

    data_dict_lab_dict = _demographics(option, competing_risk_death_check, data_dict_lab_dict)
    data_dict_lab_dict = _admissions(dataset, data_source_path, option, data_dict_lab_dict)
    if se_task:
        data_dict_lab_dict['DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT'] = _merged_pickle(
            dataset, data_source_path, 'ssid_to_date_to_nonoral_meds_dict.pickle')
    if {'basic_lab', 'extra_lab', 'cbc_differential'} & set(data_modules_to_include) or se_task:
        data_dict_lab_dict, modular_featuredim_dict = _labs(
            dataset, data_source_path, data_modules_to_include, se_task, data_dict_lab_dict, modular_featuredim_dict)
    if {'basic_vitals', 'extra_vitals'} & set(data_modules_to_include):
        data_dict_lab_dict, modular_featuredim_dict = _vitals(
            dataset, data_source_path, option, data_modules_to_include, se_task, first_principle,
            data_dict_lab_dict, modular_featuredim_dict)
    if {'gender', 'age'} & set(data_modules_to_include):
        data_dict_lab_dict, modular_featuredim_dict = _gender_age(
            data_modules_to_include, data_dict_lab_dict, modular_featuredim_dict)
    print('modular_featuredim_dict', modular_featuredim_dict)

    if normalize:
        # cache: <data_source_path>/<dataset>/normalized_data_dict_lab_dict_<dataset>_<modules>.pickle
        os.makedirs(os.path.join(data_source_path, dataset), exist_ok=True)
        data_dict_lab_dict = _clip(_normalize(dataset, data_source_path, modules, data_dict_lab_dict, first_principle))

    split_dict = _splits(dataset, data_source_path, mode, rseed, data_fraction_option, data_dict_lab_dict, quick_test)
    return data_dict_lab_dict, modular_featuredim_dict, split_dict, component, se_task
