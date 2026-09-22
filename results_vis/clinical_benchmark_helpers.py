#!/usr/bin/env python3
"""
Helper functions for evaluating clinical benchmark scores (Motwani, FENCE, Lyman).
Imported by cohort_summaries/cancer_types/check_cancer_types_insight.py. The
eval_8 script it was originally written for is in results_vis/deprecated/.
"""

import os
import pickle
import sys
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

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
from MSK_featurename_list import ALL_INCLUDED_TESTS

# ============================================================================
# SHARED UTILITIES
# ============================================================================

def extract_lab_value(lab_array, index):
    """Extract and validate a lab value from array."""
    if index >= len(lab_array):
        return None

    value = lab_array[index]

    if isinstance(value, str) or (isinstance(value, float) and np.isnan(value)):
        return None

    try:
        float_value = float(value)
        if np.isnan(float_value):
            return None
        return float_value
    except (ValueError, TypeError):
        return None


# ============================================================================
# AKI MOTWANI SCORE
# ============================================================================

AGE_COEFFICIENTS_MOTWANI = {
    '<=60': 0,
    '61-70': 1.5,
    '71-90': 2.5
}

HYPERTENSION_COEFFICIENTS = {
    'no': 0,
    'yes': 2
}

CISPLATIN_DOSE_COEFFICIENTS = {
    '<=100': 0,
    '101-150': 1,
    '>150': 3
}

ALBUMIN_COEFFICIENTS_MOTWANI = {
    '>3.5': 0,
    '2.0-3.5': 2,
    '<2.0': 2
}


def calculate_motwani_score(
    age: float,
    has_hypertension: bool,
    cisplatin_dose_mg: Optional[float],
    albumin: Optional[float]
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate Motwani Cisplatin AKI Score.

    Returns:
    --------
    total_score : float
        Total Motwani score
    score_components : Dict[str, float]
        Individual score components
    """
    score_components = {}

    # Age
    if age <= 60:
        age_score = AGE_COEFFICIENTS_MOTWANI['<=60']
    elif age <= 70:
        age_score = AGE_COEFFICIENTS_MOTWANI['61-70']
    else:
        age_score = AGE_COEFFICIENTS_MOTWANI['71-90']
    score_components['age'] = age_score

    # Hypertension
    htn_score = HYPERTENSION_COEFFICIENTS['yes'] if has_hypertension else HYPERTENSION_COEFFICIENTS['no']
    score_components['hypertension'] = htn_score

    # Cisplatin dose
    if cisplatin_dose_mg is None or cisplatin_dose_mg == 0:
        cisplatin_score = 0
    elif cisplatin_dose_mg <= 100:
        cisplatin_score = CISPLATIN_DOSE_COEFFICIENTS['<=100']
    elif cisplatin_dose_mg <= 150:
        cisplatin_score = CISPLATIN_DOSE_COEFFICIENTS['101-150']
    else:
        cisplatin_score = CISPLATIN_DOSE_COEFFICIENTS['>150']
    score_components['cisplatin_dose'] = cisplatin_score

    # Albumin
    if albumin is None or np.isnan(albumin):
        albumin_score = 0  # Missing = assume normal
    elif albumin > 3.5:
        albumin_score = ALBUMIN_COEFFICIENTS_MOTWANI['>3.5']
    else:
        albumin_score = ALBUMIN_COEFFICIENTS_MOTWANI['2.0-3.5']
    score_components['albumin'] = albumin_score

    # Total score
    total_score = sum(score_components.values())

    return total_score, score_components


def get_cisplatin_dose(medications: List, medication_name: str = 'CISPLATIN') -> float:
    """
    Extract cisplatin dose from medication list.

    Returns:
    --------
    dose : float
        Total cisplatin dose in mg, or 0 if not found
    """
    total_dose = 0.0

    for med in medications:
        if isinstance(med, (tuple, list)) and len(med) >= 2:
            med_name, dose = med[0], med[1]
            if med_name.upper() == medication_name.upper():
                try:
                    total_dose += float(dose)
                except (ValueError, TypeError):
                    pass

    return total_dose


def evaluate_aki_motwani_score(
    data_module,
    data_dict_lab_dict: Dict,
    data_source_path: str,
    dataset: str,
    device: torch.device,
    rseed: int = 1024,
    strict_mode: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate AKI Motwani score on test set.

    Parameters:
    -----------
    strict_mode : bool
        If True, only evaluate patients on first infusion (infusion #0)
        with cisplatin-containing therapies for fair comparison.

    Returns:
    --------
    all_motwani_scores : torch.Tensor
        Motwani scores for all samples
    all_y_events : torch.Tensor
        Event indicators (1=event, 0=censored)
    all_time_to_events : torch.Tensor
        Time to event
    all_num_infusions : torch.Tensor
        Infusion numbers
    """
    import pytorch_lightning as pl
    import pickle

    # Get data dictionaries
    age_dict = data_dict_lab_dict.get('age', {})
    lab_dict = data_dict_lab_dict.get('lab_def_dict', {})
    DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT = data_dict_lab_dict.get('DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT', {})

    # Load hypertension dictionary
    htn_dict_path = f'{data_source_path}/{dataset}/DMP_ID_to_hypertension_dates_delta_{dataset}.pickle'
    if os.path.exists(htn_dict_path):
        with open(htn_dict_path, 'rb') as f:
            hypertension_dict = pickle.load(f)
    else:
        hypertension_dict = {}

    # Get lab indices
    albumin_idx = ALL_INCLUDED_TESTS.index('Albumin')

    # Collect scores
    all_motwani_scores = []
    all_y_events = []
    all_time_to_events = []
    all_num_infusions = []

    debug_count = 0
    max_debug = 10

    # Strict mode tracking
    total_samples_seen = 0
    samples_filtered_out = 0

    pl.seed_everything(rseed)

    for batch in data_module.test_dataloader():
        # Unpack batch
        x_data, y_data, batch_info = batch

        # Extract info
        dmp_ids_raw, cancer_types_raw, cutoff_dates_raw = batch_info

        # Convert to lists
        if torch.is_tensor(dmp_ids_raw):
            batch_dmp_ids = dmp_ids_raw.tolist() if dmp_ids_raw.dim() > 0 else [dmp_ids_raw.item()]
        else:
            batch_dmp_ids = list(dmp_ids_raw) if isinstance(dmp_ids_raw, (list, tuple)) else [dmp_ids_raw]

        if torch.is_tensor(cutoff_dates_raw):
            batch_cutoff_dates = cutoff_dates_raw.tolist() if cutoff_dates_raw.dim() > 0 else [cutoff_dates_raw.item()]
        else:
            batch_cutoff_dates = list(cutoff_dates_raw) if isinstance(cutoff_dates_raw, (list, tuple)) else [cutoff_dates_raw]

        # Extract y_data
        fail_indicators = torch.as_tensor(y_data[0], device=device)
        time_to_event = torch.as_tensor(y_data[1], device=device)

        # Extract num_infusion from x_data[2] (meds_returned=False in eval_8)
        # Structure when meds_returned=False: (X_ts, X_dates, num_infusions, DMP_ID)
        #                                       [0]    [1]       [2]           [3]
        num_infusion = x_data[2] if len(x_data) > 2 else torch.zeros(len(batch_dmp_ids))

        # Ensure at least 1D
        if fail_indicators.dim() == 0:
            fail_indicators = fail_indicators.unsqueeze(0)
        if time_to_event.dim() == 0:
            time_to_event = time_to_event.unsqueeze(0)

        # Calculate Motwani scores
        motwani_scores_batch = []
        valid_indices = []  # Track which indices pass strict mode filter

        for idx, dmp_id in enumerate(batch_dmp_ids):
            try:
                if torch.is_tensor(dmp_id):
                    dmp_id = dmp_id.item()
                elif isinstance(dmp_id, (list, tuple)):
                    dmp_id = dmp_id[0]

                # Get infusion number for strict mode filtering
                inf_num = num_infusion[idx].item() if torch.is_tensor(num_infusion[idx]) else num_infusion[idx]

                # Get medications for strict mode filtering
                cutoff_date = batch_cutoff_dates[idx]
                patient_meds = []
                if dmp_id in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT:
                    if cutoff_date in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[dmp_id]:
                        patient_meds = DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[dmp_id][cutoff_date]

                # Check if cisplatin is present
                has_cisplatin = any('CISPLATIN' in (med[0].upper() if isinstance(med, (tuple, list)) else str(med).upper()) for med in patient_meds)

                # Track total samples
                total_samples_seen += 1

                # Apply strict mode filter: only first infusion (0-indexed) with cisplatin
                if strict_mode:
                    if inf_num != 0 or not has_cisplatin:
                        # Skip this sample
                        samples_filtered_out += 1
                        continue

                # Get age
                age_bin = age_dict.get(dmp_id, [10])[0]
                cutoff_date = batch_cutoff_dates[idx]
                age_at_sequencing = age_bin * 5
                age = age_at_sequencing + (cutoff_date / 365.25)

                # Get albumin
                albumin = None
                if dmp_id in lab_dict:
                    patient_labs = lab_dict[dmp_id]
                    if cutoff_date in patient_labs:
                        lab_array = patient_labs[cutoff_date]
                        albumin = extract_lab_value(lab_array, albumin_idx)

                # Check hypertension
                has_hypertension = False
                if dmp_id in hypertension_dict:
                    htn_dates = hypertension_dict[dmp_id]
                    if isinstance(htn_dates, dict):
                        htn_dates = list(htn_dates.keys())
                    has_hypertension = any(d <= cutoff_date for d in htn_dates)

                # Get cisplatin dose
                patient_meds = []
                if dmp_id in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT:
                    if cutoff_date in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[dmp_id]:
                        patient_meds = DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[dmp_id][cutoff_date]
                cisplatin_dose = get_cisplatin_dose(patient_meds, 'CISPLATIN')

                # Calculate Motwani score
                motwani_score, motwani_components = calculate_motwani_score(
                    age=age,
                    has_hypertension=has_hypertension,
                    cisplatin_dose_mg=cisplatin_dose,
                    albumin=albumin
                )

                # Debug first few patients
                if debug_count < max_debug:
                    print(f"\n{'='*70}")
                    print(f"MOTWANI DEBUG Patient {debug_count + 1}")
                    print(f"{'='*70}")
                    print(f"DMP_ID: {dmp_id}")
                    print(f"Demographics: Age={age:.1f} years")
                    print(f"\nClinical Factors:")
                    print(f"  Hypertension: {'Yes' if has_hypertension else 'No'}")
                    print(f"  Cisplatin dose: {cisplatin_dose:.1f} mg" if cisplatin_dose else "  Cisplatin dose: 0 mg (or not found)")
                    print(f"\nLab Values:")
                    print(f"  Albumin: {albumin:.2f} g/dL" if albumin is not None else "  Albumin: MISSING")

                    print(f"\nMotwani Score Components:")
                    for key, value in motwani_components.items():
                        print(f"  {key}: {value:.1f}")

                    print(f"\nTotal Score: {motwani_score:.1f}")
                    print(f"Ground Truth: Event={fail_indicators[idx].item()}, Time={time_to_event[idx].item()}")
                    print(f"{'='*70}")
                    debug_count += 1

                motwani_scores_batch.append(motwani_score)
                valid_indices.append(idx)  # Track this as a valid sample

            except Exception as e:
                print(f"Error calculating Motwani for patient {dmp_id}: {e}")
                import traceback
                traceback.print_exc()
                motwani_scores_batch.append(0)
                valid_indices.append(idx)  # Still include failed samples to maintain alignment

        # Store only valid samples (if strict mode is used)
        if len(valid_indices) > 0:
            valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
            all_y_events.append(fail_indicators[valid_indices_tensor].detach().cpu())
            all_time_to_events.append(time_to_event[valid_indices_tensor].detach().cpu())
            all_num_infusions.append(num_infusion[valid_indices_tensor].detach().cpu() if torch.is_tensor(num_infusion) else torch.tensor([num_infusion[i] for i in valid_indices]))
            all_motwani_scores.extend(motwani_scores_batch)

    # Concatenate
    all_y_events = torch.cat(all_y_events, dim=0)
    all_time_to_events = torch.cat(all_time_to_events, dim=0)
    all_num_infusions = torch.cat(all_num_infusions, dim=0)
    all_motwani_scores = torch.tensor(all_motwani_scores, dtype=torch.float32)

    # Summary information
    print(f"\n{'='*70}")
    print("MOTWANI SCORE SUMMARY")
    print(f"{'='*70}")
    if strict_mode:
        print("STRICT MODE: Only first infusion + cisplatin-containing therapies")
        print(f"  Total samples seen: {total_samples_seen}")
        print(f"  Samples filtered out: {samples_filtered_out} ({100*samples_filtered_out/total_samples_seen if total_samples_seen > 0 else 0:.1f}%)")
        print(f"  Samples included: {len(all_motwani_scores)} ({100*len(all_motwani_scores)/total_samples_seen if total_samples_seen > 0 else 0:.1f}%)")
    print(f"Total samples: {len(all_motwani_scores)}")
    print(f"Score range: [{all_motwani_scores.min():.1f}, {all_motwani_scores.max():.1f}]")
    print(f"Score mean ± std: {all_motwani_scores.mean():.1f} ± {all_motwani_scores.std():.1f}")
    print(f"\nScore Conversion: RAW POINT SCORES (NO NORMALIZATION)")
    print(f"  - Motwani scores are integer points (typically 0-11)")
    print(f"  - Score = Age points + Hypertension points + Cisplatin dose points + Albumin points")
    print(f"  - Age: ≤60 (0), 61-70 (+1.5), 71-90 (+2.5)")
    print(f"  - Hypertension: No (0), Yes (+2)")
    print(f"  - Cisplatin: ≤100mg (0), 101-150mg (+1), >150mg (+3)")
    print(f"  - Albumin: >3.5 g/dL (0), 2.0-3.5 g/dL (+2)")
    print(f"  - Used directly as risk scores (higher = higher risk)")
    print(f"  - C-index uses ranking, so raw scores should work")
    print(f"  - For calibrated probabilities, would need logistic calibration")
    print(f"\nEvent distribution:")
    print(f"  Total events: {all_y_events.sum()}/{len(all_y_events)} ({100*all_y_events.float().mean():.1f}%)")
    print(f"{'='*70}\n")

    return all_motwani_scores, all_y_events, all_time_to_events, all_num_infusions


# ============================================================================
# FEBRILE NEUTROPENIA FENCE SCORE
# ============================================================================

SEX_COEFFICIENTS = {
    'MALE': 0,
    'FEMALE': 8
}

AGE_COEFFICIENTS_FENCE = {
    '<65': 0,
    '65-74': 9,
    '>=75': 4
}

CANCER_TYPE_COEFFICIENTS = {
    'Esophagogastric Cancer': 0,
    'Glioma': -22,
    'Head and Neck Cancer': 52,
    'Breast Cancer': 32,
    'Mesothelioma': 28,
    'Non-Small Cell Lung Cancer': 47,
    'Small Cell Lung Cancer': 75,
    'Colorectal Cancer': -23,
    'Ovarian Cancer': 28,
    'Endometrial Cancer': -33,
    'Cervical Cancer': -33,
    'Bladder Cancer': 29,
    'Prostate Cancer': 6,
    'Germ Cell Tumor': 85,
    'Pancreatic Cancer': 55,
    'Mature B-Cell Neoplasms': 7,
    'Lymphoma': 7,  # Maps to Mature B-Cell Neoplasms
    'Other': 10,
    'others': 10
}

DISEASE_STAGE_COEFFICIENTS = {
    'Adjuvant': 0,
    'Neoadjuvant': -11,
    'Advanced': 7
}

ALBUMIN_COEFFICIENTS_FENCE = {
    '<normal': 10,
    'normal': 0,
    '>normal': -7,
    'missing': -3
}

BILIRUBIN_COEFFICIENTS = {
    '<5': 1,
    '5-25': 0,
    '>25': 17,
    'missing': -29
}

EGFR_COEFFICIENTS = {
    '<=60': 13,
    '60-90': 4,
    '>=90': 0,
    'missing': 19
}

CRP_COEFFICIENTS = {
    '<10': 0,
    '>=10': 2,
    'missing': -9
}

INFECTION_COEFFICIENTS = {
    'no': 0,
    'yes': 10
}

NUM_DRUGS_COEFFICIENTS = {
    1: 0,
    2: 10,
    3: 47,
    4: 56
}

CHEMO_CATEGORY_COEFFICIENTS = {
    'Platinums': -27,
    'Non-platinum alkylating agents': -2,
    'Taxanes': 31,
    'Topoisomerase inhibitors': -5,
    'Antimetabolites': -16,
    'Vinca alkaloids': -4,
    'Other chemotherapy': -50
}


def calculate_fence_score(
    sex: str,
    age: float,
    cancer_type: str,
    albumin: Optional[float],
    bilirubin: Optional[float],
    creatinine: Optional[float],
    medications: List[Tuple[str, int]],
    infection_before_chemo: bool = False,
    disease_stage: str = 'Advanced',
    classified_meds_dict: Dict = None
) -> Tuple[int, Dict[str, int]]:
    """Calculate FENCE score for a patient-infusion."""
    score_components = {}

    # Sex
    sex_score = SEX_COEFFICIENTS.get(sex.upper(), 0)
    score_components['sex'] = sex_score

    # Age
    if age < 65:
        age_score = AGE_COEFFICIENTS_FENCE['<65']
    elif age < 75:
        age_score = AGE_COEFFICIENTS_FENCE['65-74']
    else:
        age_score = AGE_COEFFICIENTS_FENCE['>=75']
    score_components['age'] = age_score

    # Cancer type
    cancer_score = CANCER_TYPE_COEFFICIENTS.get(cancer_type, CANCER_TYPE_COEFFICIENTS['Other'])
    score_components['cancer_type'] = cancer_score

    # Disease stage
    disease_score = DISEASE_STAGE_COEFFICIENTS.get(disease_stage, DISEASE_STAGE_COEFFICIENTS['Advanced'])
    score_components['disease_stage'] = disease_score

    # Albumin
    if albumin is None or np.isnan(albumin):
        albumin_score = ALBUMIN_COEFFICIENTS_FENCE['missing']
    elif albumin < 3.5:
        albumin_score = ALBUMIN_COEFFICIENTS_FENCE['<normal']
    elif albumin <= 5.0:
        albumin_score = ALBUMIN_COEFFICIENTS_FENCE['normal']
    else:
        albumin_score = ALBUMIN_COEFFICIENTS_FENCE['>normal']
    score_components['albumin'] = albumin_score

    # Bilirubin (convert mg/dL to µmol/L)
    if bilirubin is None or np.isnan(bilirubin):
        bilirubin_score = BILIRUBIN_COEFFICIENTS['missing']
    else:
        bilirubin_umol = bilirubin * 17.1
        if bilirubin_umol < 5:
            bilirubin_score = BILIRUBIN_COEFFICIENTS['<5']
        elif bilirubin_umol <= 25:
            bilirubin_score = BILIRUBIN_COEFFICIENTS['5-25']
        else:
            bilirubin_score = BILIRUBIN_COEFFICIENTS['>25']
    score_components['bilirubin'] = bilirubin_score

    # eGFR calculation
    if creatinine is None or np.isnan(creatinine):
        egfr_score = EGFR_COEFFICIENTS['missing']
    else:
        try:
            age_calc = max(age, 1)
            egfr = 175 * (creatinine ** -1.154) * (age_calc ** -0.203)
            if sex.upper() == 'FEMALE':
                egfr *= 0.742

            if egfr <= 60:
                egfr_score = EGFR_COEFFICIENTS['<=60']
            elif egfr <= 90:
                egfr_score = EGFR_COEFFICIENTS['60-90']
            else:
                egfr_score = EGFR_COEFFICIENTS['>=90']
        except (ValueError, TypeError):
            egfr_score = EGFR_COEFFICIENTS['missing']
    score_components['egfr'] = egfr_score

    # CRP - not available, assume missing
    crp_score = CRP_COEFFICIENTS['missing']
    score_components['crp'] = crp_score

    # Infection before chemotherapy
    infection_score = INFECTION_COEFFICIENTS['yes'] if infection_before_chemo else INFECTION_COEFFICIENTS['no']
    score_components['infection'] = infection_score

    # Medications
    unique_drugs = set([med[0] if isinstance(med, (tuple, list)) else med for med in medications])

    if classified_meds_dict is not None:
        chemo_drugs = set([
            drug for drug in unique_drugs
            if drug in classified_meds_dict and classified_meds_dict[drug].get('is_chemo', False)
        ])
        num_drugs = len(chemo_drugs)
    else:
        chemo_drugs = unique_drugs
        num_drugs = len(unique_drugs)

    # Number of drugs
    if num_drugs >= 4:
        num_drugs_score = NUM_DRUGS_COEFFICIENTS[4]
    elif num_drugs in NUM_DRUGS_COEFFICIENTS:
        num_drugs_score = NUM_DRUGS_COEFFICIENTS[num_drugs]
    else:
        num_drugs_score = NUM_DRUGS_COEFFICIENTS[1]
    score_components['num_drugs'] = num_drugs_score

    # Drug categories
    chemo_categories = classify_medications(chemo_drugs, classified_meds_dict)

    for category, present in chemo_categories.items():
        if present:
            score_components[f'chemo_{category}'] = CHEMO_CATEGORY_COEFFICIENTS[category]
        else:
            score_components[f'chemo_{category}'] = 0

    # Total score
    total_score = sum(score_components.values())

    return total_score, score_components


def classify_medications(medications: set, classified_meds_dict: Dict = None) -> Dict[str, bool]:
    """Classify medications into FENCE drug categories."""
    categories = {
        'Platinums': False,
        'Non-platinum alkylating agents': False,
        'Taxanes': False,
        'Topoisomerase inhibitors': False,
        'Antimetabolites': False,
        'Vinca alkaloids': False,
        'Other chemotherapy': False
    }

    if classified_meds_dict is None:
        return categories

    has_chemo = False
    for med_name in medications:
        if med_name not in classified_meds_dict:
            continue

        med_class = classified_meds_dict[med_name]
        if not med_class.get('is_chemo', False):
            continue

        has_chemo = True

        if med_class.get('is_platinum', False):
            categories['Platinums'] = True
        if med_class.get('is_alkylating_agent', False) and not med_class.get('is_platinum', False):
            categories['Non-platinum alkylating agents'] = True
        if med_class.get('is_taxane', False):
            categories['Taxanes'] = True
        if med_class.get('is_topoisomerase_inhibitor', False):
            categories['Topoisomerase inhibitors'] = True
        if med_class.get('is_vinca_alkaloid', False):
            categories['Vinca alkaloids'] = True
        if med_class.get('is_antimetabolite', False):
            categories['Antimetabolites'] = True

    if has_chemo and not any([v for k, v in categories.items() if k != 'Other chemotherapy']):
        categories['Other chemotherapy'] = True

    return categories


def calculate_csr_fence_score(
    first_inf_fence_score: float,
    medications: List[Tuple[str, str]],
    cycle_number: int,
    has_gcsf: bool,
    classified_meds_dict: Dict = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate CSR-FENCE score for cycles 2-6 (infusions 1-5).

    CSR-FENCE = FENCE group + Platinums + Taxanes + Radiotherapy + Cycle number +
                FN/neutropenia in previous cycles + G-CSF prophylaxis
    """
    score_components = {}

    # FENCE group (based on first infusion FENCE score)
    # Low risk (<25), Intermediate (25-48), High (49-62), Very high (>=63)
    if first_inf_fence_score < 25:
        fence_group_score = 0
    elif first_inf_fence_score < 49:
        fence_group_score = 4
    elif first_inf_fence_score < 63:
        fence_group_score = 4
    else:
        fence_group_score = 5
    score_components['fence_group'] = fence_group_score

    # Extract medication names and check categories
    unique_meds = set([med[0] if isinstance(med, (tuple, list)) else med for med in medications])

    # Check for platinums and taxanes using classified_meds_dict
    has_platinum = False
    has_taxane = False

    if classified_meds_dict is not None:
        for med_name in unique_meds:
            if med_name in classified_meds_dict:
                med_class = classified_meds_dict[med_name]
                if med_class.get('is_platinum', False):
                    has_platinum = True
                if med_class.get('is_taxane', False):
                    has_taxane = True

    # Platinums
    platinum_score = 1 if has_platinum else 0
    score_components['platinum'] = platinum_score

    # Taxanes
    taxane_score = 1 if has_taxane else 0
    score_components['taxane'] = taxane_score

    # Radiotherapy (not available in our data)
    radiotherapy_score = 0
    score_components['radiotherapy'] = radiotherapy_score

    # Cycle number (2-6)
    cycle_scores = {2: 0, 3: -1, 4: -1, 5: -2, 6: -2}
    cycle_score = cycle_scores.get(cycle_number, 0)
    score_components['cycle_number'] = cycle_score

    # FN or neutropenia in previous cycles
    # By our definition, we exclude these patients, so this is always 0
    prev_fn_score = 0
    score_components['prev_fn_neutropenia'] = prev_fn_score

    # G-CSF prophylaxis
    gcsf_score = -1 if has_gcsf else 0
    score_components['gcsf'] = gcsf_score

    # Total score
    total_score = sum(score_components.values())

    return total_score, score_components


def extract_medication_names(medications):
    """Extract medication names from medication list."""
    if not medications:
        return set()

    unique_names = set()
    for med in medications:
        if isinstance(med, (tuple, list)) and len(med) >= 1:
            med_name = med[0]
            unique_names.add(med_name)
        elif isinstance(med, str):
            unique_names.add(med)

    return unique_names


def evaluate_fn_fence_score(
    data_module,
    data_dict_lab_dict: Dict,
    data_source_path: str,
    dataset: str,
    device: torch.device,
    rseed: int = 1024,
    strict_mode: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate Febrile Neutropenia FENCE score on test set.

    Parameters:
    -----------
    strict_mode : bool
        If True, only evaluate patients on first infusion (infusion #0),
        discounting CSR-FENCE score for fair comparison.

    Returns:
    --------
    all_fence_scores : torch.Tensor
        FENCE scores for all samples
    all_y_events : torch.Tensor
        Event indicators
    all_time_to_events : torch.Tensor
        Time to event
    all_num_infusions : torch.Tensor
        Infusion numbers
    """
    import json
    import pytorch_lightning as pl
    import pickle

    # Load classified medications dictionary
    classified_meds_path = f'{data_source_path}/{dataset}/classified_medications.json'
    if os.path.exists(classified_meds_path):
        with open(classified_meds_path, 'r') as f:
            classified_meds_dict = json.load(f)
        print(f"Loaded {len(classified_meds_dict)} medication classifications")
    else:
        print(f"Warning: {classified_meds_path} not found, FENCE scores may be inaccurate")
        classified_meds_dict = {}

    # Load infection dictionary
    infection_dict_path = f'{data_source_path}/{dataset}/DMP_ID_to_infection_dates_delta_{dataset}.pickle'
    if os.path.exists(infection_dict_path):
        with open(infection_dict_path, 'rb') as f:
            infection_dict = pickle.load(f)
        print(f"Loaded infection data for {len(infection_dict)} patients")
    else:
        print(f"Warning: Infection dictionary not found at {infection_dict_path}")
        infection_dict = {}

    # Load stage dictionary
    stage_dict_path = f'{data_source_path}/{dataset}/DMP_ID_to_DATE_to_STAGE_DICT_{dataset}_021026.pickle'
    if os.path.exists(stage_dict_path):
        with open(stage_dict_path, 'rb') as f:
            stage_dict = pickle.load(f)
        print(f"Loaded stage dictionary with {len(stage_dict)} patients")
    else:
        print(f"Warning: Stage dictionary not found at {stage_dict_path}")
        stage_dict = {}

    # Get data dictionaries
    gender_dict = data_dict_lab_dict.get('gender', {})
    age_dict = data_dict_lab_dict.get('age', {})
    lab_dict = data_dict_lab_dict.get('lab_def_dict', {})
    cancer_type_dict = data_dict_lab_dict.get('cancer_type', {})
    DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT = data_dict_lab_dict.get('DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT', {})
    gcsf_dict = data_dict_lab_dict.get('gcsf_dict', {})
    print(f"G-CSF dict size: {len(gcsf_dict)}")

    # Load cancer type dictionary if not present
    if len(cancer_type_dict) == 0:
        print("\nCancer type dict empty for FENCE, attempting to load manually...")
        if 'CAIA' in dataset:
            # For CAIA, load from date-specific cancer types pickle and use earliest per patient
            cancer_type_pickle_path = f'{data_source_path}{dataset}/DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_{dataset}_till2021_050726.pickle'
            if os.path.exists(cancer_type_pickle_path):
                with open(cancer_type_pickle_path, 'rb') as f:
                    dmp_to_date_to_cancer = pickle.load(f)

                # Extract earliest cancer type for each patient
                cancer_type_dict = {}
                for dmp_id, date_dict in dmp_to_date_to_cancer.items():
                    if date_dict:
                        # Get the earliest date (minimum key)
                        earliest_date = min(date_dict.keys())
                        cancer_type_value = date_dict[earliest_date]

                        # Handle list of cancer types - select the one with highest coefficient (worst prognosis)
                        if isinstance(cancer_type_value, list) and cancer_type_value:
                            # Find cancer type with maximum coefficient for FENCE
                            max_coef = -1000
                            selected_cancer = 'Other'
                            for ct in cancer_type_value:
                                coef = CANCER_TYPE_COEFFICIENTS.get(ct, CANCER_TYPE_COEFFICIENTS['Other'])
                                if coef > max_coef:
                                    max_coef = coef
                                    selected_cancer = ct
                            cancer_type_dict[dmp_id] = selected_cancer
                        elif isinstance(cancer_type_value, list):
                            cancer_type_dict[dmp_id] = 'Other'
                        else:
                            cancer_type_dict[dmp_id] = cancer_type_value

                print(f"Loaded cancer type dict for FENCE with {len(cancer_type_dict)} patients")
            else:
                print(f"Warning: Cancer type pickle not found at {cancer_type_pickle_path}")
                cancer_type_dict = {}
        else:
            print(f"Warning: Cannot load cancer type dict for dataset {dataset}")
            cancer_type_dict = {}

    # Get lab indices
    albumin_idx = ALL_INCLUDED_TESTS.index('Albumin')
    bilirubin_idx = ALL_INCLUDED_TESTS.index('Bilirubin, Total')
    creatinine_idx = ALL_INCLUDED_TESTS.index('Creatinine')

    # Collect scores
    all_fence_scores = []
    all_y_events = []
    all_time_to_events = []
    all_num_infusions = []

    # Track first infusion FENCE scores for CSR-FENCE calculation
    patient_first_inf_fence_scores = {}

    # Debug counters
    debug_count = 0
    max_debug = 5

    # Strict mode tracking
    total_samples_seen = 0
    samples_filtered_out = 0
    all_infusion_nums_raw = []  # Track ALL infusion numbers before filtering

    pl.seed_everything(rseed)

    for batch in data_module.test_dataloader():
        # Unpack batch
        x_data, y_data, batch_info = batch

        # Extract info
        dmp_ids_raw, cancer_types_raw, cutoff_dates_raw = batch_info

        # Convert to lists
        if torch.is_tensor(dmp_ids_raw):
            batch_dmp_ids = dmp_ids_raw.tolist() if dmp_ids_raw.dim() > 0 else [dmp_ids_raw.item()]
        else:
            batch_dmp_ids = list(dmp_ids_raw) if isinstance(dmp_ids_raw, (list, tuple)) else [dmp_ids_raw]

        if torch.is_tensor(cutoff_dates_raw):
            batch_cutoff_dates = cutoff_dates_raw.tolist() if cutoff_dates_raw.dim() > 0 else [cutoff_dates_raw.item()]
        else:
            batch_cutoff_dates = list(cutoff_dates_raw) if isinstance(cutoff_dates_raw, (list, tuple)) else [cutoff_dates_raw]

        # Extract y_data
        fail_indicators = torch.as_tensor(y_data[0], device=device)
        time_to_event = torch.as_tensor(y_data[1], device=device)

        # Debug: check what x_data contains (only first batch)
        if debug_count == 0:
            print(f"\nDEBUG: x_data structure in FENCE evaluation:")
            print(f"  len(x_data) = {len(x_data)}")
            for i, x in enumerate(x_data):
                if torch.is_tensor(x):
                    print(f"  x_data[{i}]: tensor shape {x.shape}, dtype {x.dtype}")
                else:
                    print(f"  x_data[{i}]: {type(x)}")
            print()

        # Extract num_infusion from x_data[2] (meds_returned=False in eval_8)
        # Structure when meds_returned=False: (X_ts, X_dates, num_infusions, DMP_ID)
        #                                       [0]    [1]       [2]           [3]
        num_infusion = x_data[2] if len(x_data) > 2 else torch.zeros(len(batch_dmp_ids))

        if debug_count == 0:
            print(f"DEBUG: Extracted num_infusion from x_data[2]")
            if torch.is_tensor(num_infusion):
                print(f"  First 10 infusion numbers: {num_infusion[:10].cpu().tolist()}")
            else:
                print(f"  num_infusion value: {num_infusion}")

        # Ensure at least 1D
        if fail_indicators.dim() == 0:
            fail_indicators = fail_indicators.unsqueeze(0)
        if time_to_event.dim() == 0:
            time_to_event = time_to_event.unsqueeze(0)
        if not torch.is_tensor(num_infusion):
            num_infusion = torch.tensor([num_infusion])

        # Track ALL raw infusion numbers before filtering
        if torch.is_tensor(num_infusion):
            all_infusion_nums_raw.extend(num_infusion.cpu().tolist())
        else:
            all_infusion_nums_raw.append(num_infusion)

        # Calculate FENCE scores (use FENCE for first infusion, CSR-FENCE for later)
        fence_scores_batch = []
        valid_indices = []  # Track which indices pass strict mode filter

        for idx, dmp_id in enumerate(batch_dmp_ids):
            try:
                if torch.is_tensor(dmp_id):
                    dmp_id = dmp_id.item()
                elif isinstance(dmp_id, (list, tuple)):
                    dmp_id = dmp_id[0]

                cutoff_date = batch_cutoff_dates[idx]

                # Get infusion number
                inf_num = num_infusion[idx].item() if torch.is_tensor(num_infusion[idx]) else num_infusion[idx]

                # Track total samples
                total_samples_seen += 1

                # Apply strict mode filter: only first infusion (0-indexed)
                if strict_mode and inf_num != 0:
                    # Skip this sample
                    samples_filtered_out += 1
                    continue

                # Get sex (gender_dict stores as int: 0=MALE, 1=FEMALE per GENDERS list)
                gender_int = gender_dict.get(dmp_id, [0])[0]  # Default to MALE (0)
                sex = 'MALE' if gender_int == 0 else 'FEMALE'

                # Get age
                age_bin = age_dict.get(dmp_id, [10])[0]
                age_at_sequencing = age_bin * 5
                age = age_at_sequencing + (cutoff_date / 365.25)

                # Get cancer type
                cancer_type_raw = cancer_type_dict.get(dmp_id, 'Other')
                # Handle if it's a list (from old data loader) or string (from manual loading)
                if isinstance(cancer_type_raw, list):
                    cancer_type = cancer_type_raw[0] if cancer_type_raw else 'Other'
                else:
                    cancer_type = cancer_type_raw

                # Get disease stage
                disease_stage = 'Advanced'  # Default
                if dmp_id in stage_dict:
                    patient_stages = stage_dict[dmp_id]
                    stage_values = [v for d, v in patient_stages.items() if d <= cutoff_date]
                    if stage_values:
                        max_stage = max(stage_values)
                        if max_stage <= 1:
                            disease_stage = 'Adjuvant'
                        elif max_stage == 2:
                            disease_stage = 'Neoadjuvant'
                        else:
                            disease_stage = 'Advanced'

                # Get lab values
                albumin = None
                bilirubin = None
                creatinine = None
                if dmp_id in lab_dict:
                    patient_labs = lab_dict[dmp_id]
                    if cutoff_date in patient_labs:
                        lab_array = patient_labs[cutoff_date]
                        albumin = extract_lab_value(lab_array, albumin_idx)
                        bilirubin = extract_lab_value(lab_array, bilirubin_idx)
                        creatinine = extract_lab_value(lab_array, creatinine_idx)

                # Get medications
                patient_meds = []
                if dmp_id in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT:
                    if cutoff_date in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[dmp_id]:
                        patient_meds = DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[dmp_id][cutoff_date]

                # Check for infection within 30 days before infusion
                infection_before_chemo = False
                if dmp_id in infection_dict:
                    patient_infections = infection_dict[dmp_id]
                    if isinstance(patient_infections, dict):
                        infection_dates = list(patient_infections.keys())
                    else:
                        infection_dates = patient_infections
                    infections_in_window = [d for d in infection_dates if cutoff_date - 30 <= d < cutoff_date]
                    infection_before_chemo = len(infections_in_window) > 0

                # Check if patient received G-CSF within window after cutoff (used for all cycles)
                gcsf_lim = 2
                has_gcsf = False
                if dmp_id in gcsf_dict:
                    gcsf_dates = gcsf_dict[dmp_id]
                    # G-CSF given if any date falls in [cutoff_date, cutoff_date + gcsf_lim]
                    has_gcsf = any(cutoff_date <= d <= cutoff_date + gcsf_lim for d in gcsf_dates)

                # Calculate FENCE score for first infusion
                fence_score, _ = calculate_fence_score(
                    sex=sex,
                    age=age,
                    cancer_type=cancer_type,
                    albumin=albumin,
                    bilirubin=bilirubin,
                    creatinine=creatinine,
                    medications=list(patient_meds),
                    infection_before_chemo=infection_before_chemo,
                    disease_stage=disease_stage,
                    classified_meds_dict=classified_meds_dict
                )

                # Track first infusion FENCE scores for CSR-FENCE
                if inf_num == 0:
                    patient_first_inf_fence_scores[dmp_id] = fence_score

                # For cycles 2-6 (infusions 1-5), use CSR-FENCE instead
                final_score = fence_score  # Default to FENCE
                if inf_num >= 1 and inf_num <= 5:
                    # Get first infusion FENCE score for this patient
                    first_inf_fence = patient_first_inf_fence_scores.get(dmp_id, 50)  # Default to 50 if not found

                    # Calculate CSR-FENCE (has_gcsf already defined above)
                    cycle_number = int(inf_num) + 1  # Convert 0-indexed to 1-indexed cycle number
                    csr_fence_score, _ = calculate_csr_fence_score(
                        first_inf_fence_score=first_inf_fence,
                        medications=list(patient_meds),
                        cycle_number=cycle_number,
                        has_gcsf=has_gcsf,
                        classified_meds_dict=classified_meds_dict
                    )
                    final_score = csr_fence_score

                # Debug first few patients
                if debug_count < max_debug:
                    print(f"\n{'='*70}")
                    print(f"FENCE DEBUG Patient {debug_count + 1}")
                    print(f"{'='*70}")
                    print(f"DMP_ID: {dmp_id}, Infusion #: {inf_num}, Cycle: {int(inf_num) + 1}")
                    print(f"Demographics: Sex={sex}, Age={age:.1f}, Cancer={cancer_type}")
                    print(f"\nLab Values:")
                    print(f"  Albumin: {albumin:.2f} g/dL" if albumin is not None else "  Albumin: MISSING")
                    print(f"  Bilirubin: {bilirubin:.2f} mg/dL" if bilirubin is not None else "  Bilirubin: MISSING")
                    print(f"  Creatinine: {creatinine:.2f} mg/dL" if creatinine is not None else "  Creatinine: MISSING")
                    print(f"\nMedications: {len(patient_meds)} drugs")
                    if patient_meds:
                        for med in list(patient_meds)[:5]:  # Show first 5
                            print(f"  - {med}")
                    print(f"\nG-CSF: {'Yes' if has_gcsf else 'No'}")
                    print(f"Infection before chemo: {infection_before_chemo}")
                    print(f"Disease stage: {disease_stage}")

                    # Show FENCE components if available
                    if inf_num == 0:
                        print(f"\nFENCE Score Components:")
                        print(f"  Total FENCE score: {fence_score}")
                        print(f"  Used as final score: {final_score}")
                    elif inf_num >= 1 and inf_num <= 5:
                        print(f"\nCSR-FENCE Score (Cycle {int(inf_num) + 1}):")
                        print(f"  First infusion FENCE: {first_inf_fence}")
                        print(f"  Total CSR-FENCE score: {csr_fence_score}")
                        print(f"  Used as final score: {final_score}")
                    else:
                        print(f"\nFENCE Score (Infusion #{inf_num}, beyond CSR-FENCE range):")
                        print(f"  Using basic FENCE score: {fence_score}")
                        print(f"  Used as final score: {final_score}")

                    print(f"\nFinal Score Used: {final_score}")
                    print(f"Ground Truth: Event={fail_indicators[idx].item()}, Time={time_to_event[idx].item()}")
                    print(f"{'='*70}")
                    debug_count += 1

                fence_scores_batch.append(final_score)
                valid_indices.append(idx)  # Track this as a valid sample

            except Exception as e:
                import traceback
                print(f"Error calculating FENCE for patient {dmp_id}: {e}")
                traceback.print_exc()
                fence_scores_batch.append(0)
                valid_indices.append(idx)  # Still include failed samples to maintain alignment

        # Store only valid samples (if strict mode is used)
        if len(valid_indices) > 0:
            valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
            all_y_events.append(fail_indicators[valid_indices_tensor].detach().cpu())
            all_time_to_events.append(time_to_event[valid_indices_tensor].detach().cpu())
            all_num_infusions.append(num_infusion[valid_indices_tensor].detach().cpu() if torch.is_tensor(num_infusion) else torch.tensor([num_infusion[i] for i in valid_indices]))
            all_fence_scores.extend(fence_scores_batch)

    # Concatenate
    all_y_events = torch.cat(all_y_events, dim=0)
    all_time_to_events = torch.cat(all_time_to_events, dim=0)
    all_num_infusions = torch.cat(all_num_infusions, dim=0)
    all_fence_scores = torch.tensor(all_fence_scores, dtype=torch.float32)

    # Summary information
    print(f"\n{'='*70}")
    print("FENCE SCORE SUMMARY")
    print(f"{'='*70}")

    # Show raw infusion distribution BEFORE filtering
    import torch as torch_temp
    if len(all_infusion_nums_raw) > 0:
        raw_unique, raw_counts = torch_temp.unique(torch_temp.tensor(all_infusion_nums_raw), return_counts=True)
        print(f"RAW infusion distribution (BEFORE any filtering):")
        for inf, count in zip(raw_unique, raw_counts):
            print(f"  Infusion #{int(inf.item())}: {int(count.item())} samples ({100*count.item()/len(all_infusion_nums_raw):.1f}%)")
        print()

    if strict_mode:
        print("STRICT MODE: Only first infusion (cycle 1, discounting CSR-FENCE)")
        print(f"  Total samples seen: {total_samples_seen}")
        print(f"  Samples filtered out: {samples_filtered_out} ({100*samples_filtered_out/total_samples_seen if total_samples_seen > 0 else 0:.1f}%)")
        print(f"  Samples included: {len(all_fence_scores)} ({100*len(all_fence_scores)/total_samples_seen if total_samples_seen > 0 else 0:.1f}%)")
    print(f"Total samples: {len(all_fence_scores)}")
    print(f"Infusion number distribution (AFTER filtering):")
    unique_infusions, counts = torch.unique(all_num_infusions, return_counts=True)
    for inf, count in zip(unique_infusions, counts):
        print(f"  Infusion #{int(inf.item())}: {int(count.item())} samples ({100*count.item()/len(all_num_infusions):.1f}%)")
    print(f"Score range: [{all_fence_scores.min():.1f}, {all_fence_scores.max():.1f}]")
    print(f"Score mean ± std: {all_fence_scores.mean():.1f} ± {all_fence_scores.std():.1f}")
    print(f"\nScore Conversion: RAW INTEGER SCORES (NO NORMALIZATION)")
    print(f"  - FENCE scores are integer points (typically 0-100+)")
    print(f"  - Used directly as risk scores (higher = higher risk)")
    print(f"  - C-index uses ranking, so raw scores should work")
    print(f"  - For calibrated probabilities, would need logistic calibration")
    print(f"\nEvent distribution:")
    print(f"  Total events: {all_y_events.sum()}/{len(all_y_events)} ({100*all_y_events.float().mean():.1f}%)")
    print(f"{'='*70}\n")

    return all_fence_scores, all_y_events, all_time_to_events, all_num_infusions


# ============================================================================
# NEUTROPENIA LYMAN 2011 SCORE
# ============================================================================

AGE_COEFFICIENT_LYMAN = 0.02
PRIOR_CHEMO_COEFFICIENT = 0.5
AST_ELEVATED_COEFFICIENT = 0.3
ALP_ELEVATED_COEFFICIENT = 0.3
BILIRUBIN_ELEVATED_COEFFICIENT = 0.4
EGFR_REDUCED_COEFFICIENT = 0.5
WBC_LOW_COEFFICIENT = 0.6
IMMUNOSUPPRESSIVE_COEFFICIENT = 0.4
GCSF_COEFFICIENT = -0.5

CANCER_TYPE_COEFFICIENTS_LYMAN = {
    'Breast Cancer': 0.0,
    'Colorectal Cancer': 0.2,
    'Non-Small Cell Lung Cancer': 0.3,
    'Small Cell Lung Cancer': 0.4,
    'Mature B-Cell Neoplasms': 0.4,
    'Lymphoma': 0.4,  # Maps to Mature B-Cell Neoplasms
    'Ovarian Cancer': 0.3,
    'Other': 0.1,
    'others': 0.1
}

# Lab thresholds
AST_ULN = 40
ALP_ULN = 120
BILIRUBIN_ULN = 1.2
EGFR_THRESHOLD = 60
WBC_LLN = 4.0


def calculate_lyman_score(
    age: float,
    cancer_type: str,
    prior_chemo: bool,
    ast: Optional[float],
    alp: Optional[float],
    bilirubin: Optional[float],
    creatinine: Optional[float],
    sex: str,
    wbc: Optional[float],
    has_immunosuppressive: bool,
    has_gcsf: bool
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate Lyman 2011 score for chemotherapy-induced neutropenia.

    Returns logit score (to be converted to probability via sigmoid).
    """
    score_components = {}

    # Intercept (baseline log-odds)
    intercept = -2.0
    score_components['intercept'] = intercept

    # Age (continuous)
    age_score = age * AGE_COEFFICIENT_LYMAN
    score_components['age'] = age_score

    # Prior chemotherapy
    prior_chemo_score = PRIOR_CHEMO_COEFFICIENT if prior_chemo else 0.0
    score_components['prior_chemo'] = prior_chemo_score

    # AST elevated
    ast_score = 0.0
    if ast is not None and not np.isnan(ast):
        if ast > AST_ULN:
            ast_score = AST_ELEVATED_COEFFICIENT
    score_components['ast_elevated'] = ast_score

    # ALP elevated
    alp_score = 0.0
    if alp is not None and not np.isnan(alp):
        if alp > ALP_ULN:
            alp_score = ALP_ELEVATED_COEFFICIENT
    score_components['alp_elevated'] = alp_score

    # Bilirubin elevated
    bilirubin_score = 0.0
    if bilirubin is not None and not np.isnan(bilirubin):
        if bilirubin > BILIRUBIN_ULN:
            bilirubin_score = BILIRUBIN_ELEVATED_COEFFICIENT
    score_components['bilirubin_elevated'] = bilirubin_score

    # eGFR reduced (< 60 mL/min/1.73m²)
    egfr_score = 0.0
    if creatinine is not None and not np.isnan(creatinine):
        try:
            # Calculate eGFR using MDRD equation
            age_calc = max(age, 1)
            egfr = 175 * (creatinine ** -1.154) * (age_calc ** -0.203)
            if sex.upper() == 'FEMALE':
                egfr *= 0.742

            if egfr < EGFR_THRESHOLD:
                egfr_score = EGFR_REDUCED_COEFFICIENT
        except (ValueError, TypeError):
            egfr_score = 0.0
    score_components['egfr_reduced'] = egfr_score

    # WBC low
    wbc_score = 0.0
    if wbc is not None and not np.isnan(wbc):
        if wbc < WBC_LLN:
            wbc_score = WBC_LOW_COEFFICIENT
    score_components['wbc_low'] = wbc_score

    # Cancer type
    cancer_score = CANCER_TYPE_COEFFICIENTS_LYMAN.get(cancer_type, CANCER_TYPE_COEFFICIENTS_LYMAN['Other'])
    score_components['cancer_type'] = cancer_score

    # Concurrent immunosuppressive therapy
    immunosuppressive_score = IMMUNOSUPPRESSIVE_COEFFICIENT if has_immunosuppressive else 0.0
    score_components['immunosuppressive'] = immunosuppressive_score

    # G-CSF prophylaxis (protective)
    gcsf_score = GCSF_COEFFICIENT if has_gcsf else 0.0
    score_components['gcsf'] = gcsf_score

    # Total logit score
    total_score = sum(score_components.values())

    return total_score, score_components


def check_prior_chemotherapy(dmp_id, cutoff_date, med_dict):
    """
    Check if patient received chemotherapy before this infusion.

    Returns True if any chemotherapy was given before cutoff_date.
    """
    if dmp_id not in med_dict:
        return False

    patient_meds = med_dict[dmp_id]

    # Check all dates before cutoff
    prior_dates = [d for d in patient_meds.keys() if d < cutoff_date]

    return len(prior_dates) > 0


def check_immunosuppressive_meds(medications, classified_meds_dict):
    """
    Check if patient is receiving concurrent immunosuppressive therapy.

    According to Lyman 2011, this includes corticosteroids and other immunosuppressants.
    We use the is_immunosuppressive flag from the classified medications.
    """
    if not medications or not classified_meds_dict:
        return False

    for med in medications:
        med_name = med[0] if isinstance(med, (tuple, list)) else med

        if med_name in classified_meds_dict:
            if classified_meds_dict[med_name].get('is_immunosuppressive', False):
                # Exclude chemotherapy - we want concurrent non-chemo immunosuppressives
                if not classified_meds_dict[med_name].get('is_chemo', False):
                    return True

    return False


def evaluate_neutropenia_lyman_score(
    data_module,
    data_dict_lab_dict: Dict,
    data_source_path: str,
    dataset: str,
    device: torch.device,
    rseed: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate Neutropenia Lyman 2011 score on test set.

    Returns:
    --------
    all_lyman_scores : torch.Tensor
        Lyman scores for all samples
    all_y_events : torch.Tensor
        Event indicators
    all_time_to_events : torch.Tensor
        Time to event
    all_num_infusions : torch.Tensor
        Infusion numbers
    """
    import pytorch_lightning as pl
    import json
    import pickle

    pl.seed_everything(rseed)

    # Load classified medications dictionary (for immunosuppressive check)
    classified_meds_path = f'{data_source_path}/{dataset}/classified_medications_with_immunosuppressive.json'
    if not os.path.exists(classified_meds_path):
        # Try alternate path
        classified_meds_path = sparc_paths.REPO_ROOT + '/clinical_baselines/FN_FENCE_score_070126/med_classification/classified_medications_with_immunosuppressive_MSK_CAIA.json'

    if os.path.exists(classified_meds_path):
        with open(classified_meds_path, 'r') as f:
            classified_meds_list = json.load(f)
            # Convert list to dict
            classified_meds_dict = {med['name']: med for med in classified_meds_list}
        print(f"Loaded {len(classified_meds_dict)} medication classifications for Lyman")
    else:
        print(f"Warning: {classified_meds_path} not found, Lyman scores may be inaccurate")
        classified_meds_dict = {}

    # Extract dictionaries
    gender_dict = data_dict_lab_dict.get('gender', {})
    age_dict = data_dict_lab_dict.get('age', {})
    cancer_type_dict = data_dict_lab_dict.get('DMP_ID_to_PRIMARY_CTYPE_dict', {})
    lab_dict = data_dict_lab_dict.get('lab_def_dict', {})
    gcsf_dict = data_dict_lab_dict.get('gcsf_dict', {})
    DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT = data_dict_lab_dict.get('DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT', {})

    print(f"\n=== Data Dictionary Sizes ===")
    print(f"Gender dict size: {len(gender_dict)}")
    print(f"Age dict size: {len(age_dict)}")
    print(f"Cancer type dict size: {len(cancer_type_dict)}")
    print(f"Lab dict size: {len(lab_dict)}")
    print(f"G-CSF dict size: {len(gcsf_dict)}")
    print(f"Medications dict size: {len(DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT)}")
    print(f"{'='*30}\n")

    # Load G-CSF dictionary if not present
    if len(gcsf_dict) == 0:
        print("G-CSF dict empty, attempting to load manually...")
        gcsf_path = sparc_paths.DATA_ROOT + '/DMP_ID_to_gcsf_startdates_dict.pickle'
        if os.path.exists(gcsf_path):
            with open(gcsf_path, 'rb') as f:
                gcsf_dict = pickle.load(f)
            print(f"Loaded G-CSF dict with {len(gcsf_dict)} patients")
        else:
            print(f"Warning: G-CSF dict not found at {gcsf_path}")
            gcsf_dict = {}

    # Load cancer type dictionary if not present
    if len(cancer_type_dict) == 0:
        print("\nCancer type dict empty, attempting to load manually...")
        if 'CAIA' in dataset:
            # For CAIA, load from date-specific cancer types pickle and use earliest per patient
            cancer_type_pickle_path = f'{data_source_path}{dataset}/DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_{dataset}_till2021_050726.pickle'
            if os.path.exists(cancer_type_pickle_path):
                with open(cancer_type_pickle_path, 'rb') as f:
                    dmp_to_date_to_cancer = pickle.load(f)

                # Extract earliest cancer type for each patient
                cancer_type_dict = {}
                for dmp_id, date_dict in dmp_to_date_to_cancer.items():
                    if date_dict:
                        # Get the earliest date (minimum key)
                        earliest_date = min(date_dict.keys())
                        cancer_type_value = date_dict[earliest_date]

                        # Handle list of cancer types - select the one with highest coefficient (worst prognosis)
                        if isinstance(cancer_type_value, list) and cancer_type_value:
                            # Find cancer type with maximum coefficient
                            max_coef = -1
                            selected_cancer = 'Other'
                            for ct in cancer_type_value:
                                coef = CANCER_TYPE_COEFFICIENTS_LYMAN.get(ct, CANCER_TYPE_COEFFICIENTS_LYMAN['Other'])
                                if coef > max_coef:
                                    max_coef = coef
                                    selected_cancer = ct
                            cancer_type_dict[dmp_id] = selected_cancer
                        elif isinstance(cancer_type_value, list):
                            cancer_type_dict[dmp_id] = 'Other'
                        else:
                            cancer_type_dict[dmp_id] = cancer_type_value

                print(f"Loaded cancer type dict with {len(cancer_type_dict)} patients")
            else:
                print(f"Warning: Cancer type pickle not found at {cancer_type_pickle_path}")
                cancer_type_dict = {}
        else:
            print(f"Warning: Cannot load cancer type dict for dataset {dataset}")
            cancer_type_dict = {}

    # Get lab indices
    creatinine_idx = ALL_INCLUDED_TESTS.index('Creatinine')
    wbc_idx = ALL_INCLUDED_TESTS.index('WBC')
    ast_idx = ALL_INCLUDED_TESTS.index('AST')
    alp_idx = ALL_INCLUDED_TESTS.index('ALP')
    bilirubin_idx = ALL_INCLUDED_TESTS.index('Bilirubin, Total')

    all_y_events = []
    all_time_to_events = []
    all_num_infusions = []
    all_lyman_scores = []

    debug_count = 0
    max_debug = 10

    for batch in data_module.test_dataloader():
        x_data, y_data, batch_info = batch

        fail_indicators = torch.as_tensor(y_data[0], device=device)
        time_to_event = torch.as_tensor(y_data[1], device=device)

        # Extract num_infusion from x_data[2] (meds_returned=False in eval_8)
        # Structure when meds_returned=False: (X_ts, X_dates, num_infusions, DMP_ID)
        #                                       [0]    [1]       [2]           [3]
        num_infusion = x_data[2] if len(x_data) > 2 else torch.zeros(len(fail_indicators))

        if fail_indicators.dim() == 0:
            fail_indicators = fail_indicators.unsqueeze(0)
        if time_to_event.dim() == 0:
            time_to_event = time_to_event.unsqueeze(0)

        # Parse batch info
        dmp_ids_raw, cancer_types_raw, cutoff_dates_raw = batch_info

        if torch.is_tensor(dmp_ids_raw):
            batch_dmp_ids = dmp_ids_raw.tolist() if dmp_ids_raw.dim() > 0 else [dmp_ids_raw.item()]
        else:
            batch_dmp_ids = list(dmp_ids_raw) if isinstance(dmp_ids_raw, (list, tuple)) else [dmp_ids_raw]

        if torch.is_tensor(cancer_types_raw):
            batch_cancer_types = [cancer_types_raw.item()] if cancer_types_raw.dim() == 0 else cancer_types_raw.tolist()
        else:
            batch_cancer_types = list(cancer_types_raw) if isinstance(cancer_types_raw, (list, tuple)) else [cancer_types_raw]

        if torch.is_tensor(cutoff_dates_raw):
            batch_cutoff_dates = cutoff_dates_raw.tolist() if cutoff_dates_raw.dim() > 0 else [cutoff_dates_raw.item()]
        else:
            batch_cutoff_dates = list(cutoff_dates_raw) if isinstance(cutoff_dates_raw, (list, tuple)) else [cutoff_dates_raw]

        # Calculate Lyman scores
        lyman_scores_batch = []

        for idx, dmp_id in enumerate(batch_dmp_ids):
            try:
                if torch.is_tensor(dmp_id):
                    dmp_id = dmp_id.item()
                elif isinstance(dmp_id, (list, tuple)):
                    dmp_id = dmp_id[0]

                cutoff_date = batch_cutoff_dates[idx]

                # Get demographics (gender_dict: 0=MALE, 1=FEMALE per GENDERS list)
                sex = 'MALE' if gender_dict.get(dmp_id, [0])[0] == 0 else 'FEMALE'
                age_bin = age_dict.get(dmp_id, [10])[0]
                age_at_sequencing = age_bin * 5
                age = age_at_sequencing + (cutoff_date / 365.25)

                # Get cancer type - first try batch_info, then fall back to cancer_type_dict
                cancer_type = batch_cancer_types[idx] if batch_cancer_types[idx] else None
                if not cancer_type or cancer_type == 'Unknown':
                    # Fallback: look up directly in cancer_type_dict
                    if dmp_id in cancer_type_dict:
                        cancer_type = cancer_type_dict[dmp_id]
                    else:
                        cancer_type = 'Other'

                # Check prior chemotherapy
                prior_chemo = check_prior_chemotherapy(dmp_id, cutoff_date, DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT)

                # Get labs
                ast = None
                alp = None
                bilirubin = None
                creatinine = None
                wbc = None

                patient_labs = lab_dict.get(dmp_id, {})
                if patient_labs:
                    available_lab_dates = [d for d in patient_labs.keys() if d <= cutoff_date]
                    if available_lab_dates:
                        most_recent_date = max(available_lab_dates)
                        recent_labs = patient_labs[most_recent_date]

                        ast = extract_lab_value(recent_labs, ast_idx)
                        alp = extract_lab_value(recent_labs, alp_idx)
                        bilirubin = extract_lab_value(recent_labs, bilirubin_idx)
                        creatinine = extract_lab_value(recent_labs, creatinine_idx)
                        wbc = extract_lab_value(recent_labs, wbc_idx)

                # Get medications
                patient_meds = []
                if dmp_id in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT:
                    if cutoff_date in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[dmp_id]:
                        patient_meds = DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[dmp_id][cutoff_date]

                # Check for immunosuppressive meds
                has_immunosuppressive = check_immunosuppressive_meds(patient_meds, classified_meds_dict)

                # Check for G-CSF
                gcsf_lim = 2
                has_gcsf = False
                if dmp_id in gcsf_dict:
                    gcsf_dates = gcsf_dict[dmp_id]
                    has_gcsf = any(cutoff_date <= d <= cutoff_date + gcsf_lim for d in gcsf_dates)

                # Calculate Lyman score
                lyman_logit, lyman_components = calculate_lyman_score(
                    age=age,
                    cancer_type=cancer_type,
                    prior_chemo=prior_chemo,
                    ast=ast,
                    alp=alp,
                    bilirubin=bilirubin,
                    creatinine=creatinine,
                    sex=sex,
                    wbc=wbc,
                    has_immunosuppressive=has_immunosuppressive,
                    has_gcsf=has_gcsf
                )

                # Convert logit to probability using sigmoid
                lyman_prob = 1.0 / (1.0 + np.exp(-lyman_logit))

                # Debug first few patients
                if debug_count < max_debug:
                    print(f"\n{'='*70}")
                    print(f"LYMAN DEBUG Patient {debug_count + 1}")
                    print(f"{'='*70}")
                    print(f"DMP_ID: {dmp_id}")
                    print(f"Demographics: Sex={sex}, Age={age:.1f}, Cancer={cancer_type}")
                    print(f"\nLab Values:")
                    print(f"  AST: {ast:.1f} IU/L (ULN=40)" if ast is not None else "  AST: MISSING")
                    print(f"  ALP: {alp:.1f} IU/L (ULN=120)" if alp is not None else "  ALP: MISSING")
                    print(f"  Bilirubin: {bilirubin:.2f} mg/dL (ULN=1.2)" if bilirubin is not None else "  Bilirubin: MISSING")
                    print(f"  Creatinine: {creatinine:.2f} mg/dL" if creatinine is not None else "  Creatinine: MISSING")
                    print(f"  WBC: {wbc:.1f} K/µL (LLN=4.0)" if wbc is not None else "  WBC: MISSING")

                    # Calculate and show eGFR if creatinine available
                    if creatinine is not None:
                        try:
                            age_calc = max(age, 1)
                            egfr = 175 * (creatinine ** -1.154) * (age_calc ** -0.203)
                            if sex.upper() == 'FEMALE':
                                egfr *= 0.742
                            print(f"  eGFR: {egfr:.1f} mL/min/1.73m² (threshold=60)")
                        except:
                            print(f"  eGFR: Could not calculate")

                    print(f"\nMedications: {len(patient_meds)} drugs")
                    if patient_meds:
                        for med in list(patient_meds)[:5]:  # Show first 5
                            print(f"  - {med}")

                    print(f"\nClinical Factors:")
                    print(f"  Prior chemotherapy: {prior_chemo}")
                    print(f"  Immunosuppressive meds: {has_immunosuppressive}")
                    print(f"  G-CSF prophylaxis: {has_gcsf}")

                    print(f"\nLyman Score Components:")
                    for key, value in lyman_components.items():
                        print(f"  {key}: {value:.3f}")

                    print(f"\nTotal Logit: {lyman_logit:.3f}")
                    print(f"Probability (sigmoid): {lyman_prob:.4f}")
                    print(f"Ground Truth: Event={fail_indicators[idx].item()}, Time={time_to_event[idx].item()}")
                    print(f"{'='*70}")
                    debug_count += 1

                lyman_scores_batch.append(lyman_prob)

            except Exception as e:
                print(f"Error calculating Lyman for patient {dmp_id}: {e}")
                import traceback
                traceback.print_exc()
                lyman_scores_batch.append(0.5)

        all_y_events.append(fail_indicators.detach().cpu())
        all_time_to_events.append(time_to_event.detach().cpu())
        all_num_infusions.append(num_infusion.detach().cpu() if torch.is_tensor(num_infusion) else torch.tensor([num_infusion]))
        all_lyman_scores.extend(lyman_scores_batch)

    all_y_events = torch.cat(all_y_events, dim=0)
    all_time_to_events = torch.cat(all_time_to_events, dim=0)
    all_num_infusions = torch.cat(all_num_infusions, dim=0)
    all_lyman_scores = torch.tensor(all_lyman_scores, dtype=torch.float32)

    # Summary information
    print(f"\n{'='*70}")
    print("LYMAN SCORE SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_lyman_scores)}")
    print(f"Score range: [{all_lyman_scores.min():.4f}, {all_lyman_scores.max():.4f}]")
    print(f"Score mean ± std: {all_lyman_scores.mean():.4f} ± {all_lyman_scores.std():.4f}")
    print(f"\nScore Conversion: LOGIT → SIGMOID → PROBABILITY")
    print(f"  - Raw logit computed from weighted features")
    print(f"  - Converted to probability: P = 1 / (1 + exp(-logit))")
    print(f"  - Logit = intercept + Σ(feature_coefficient × feature_value)")
    print(f"  - Baseline intercept = -2.0")
    print(f"  - Probabilities should be in [0, 1] range")
    print(f"\nEvent distribution:")
    print(f"  Total events: {all_y_events.sum()}/{len(all_y_events)} ({100*all_y_events.float().mean():.1f}%)")
    print(f"{'='*70}\n")

    return all_lyman_scores, all_y_events, all_time_to_events, all_num_infusions
