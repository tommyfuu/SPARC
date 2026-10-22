import torch
# put the repo root, data/ and models/ on sys.path
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import copy
import pickle
from torch.utils.data import DataLoader
import random
import copy
# from MSK_featurename_list import *
from MSK_featurename_list import *
import time
import re
import math
from itertools import combinations
# from data_loader_side_effect_infusions_utils_risk import *
from data_loader_side_effect_infusions_utils_risk_fixed import *


# labels (y)
def y_tuple_calc(cutoffdate, deathdate, lastcontactdate):
    if not np.isnan(deathdate):
        # the tuple would mean that the patient dies in the next x days
        return (True, int(deathdate-cutoffdate))
    else:
        # the tuple would mean that the patient is censored in the next x days
        return (False, int(lastcontactdate-cutoffdate))

def fetch_mutation_clinical_data(DMP_ID, data_dict_lab_dict, MSK_numfeature_dict):
    '''fetches the mutation and clinical data for a given DMP_ID'''
    # print("DMP_ID", DMP_ID)
    mutation_clinical_out = []
    if 'mutation' in data_dict_lab_dict:
        if DMP_ID in data_dict_lab_dict['mutation']:
            mutation_data = data_dict_lab_dict['mutation'][DMP_ID]
        else:
            # if mutation data not available, fill with zeros
            mutation_data = [0]*MSK_numfeature_dict['mutation']
        mutation_clinical_out.extend(copy.deepcopy(mutation_data))
    if 'clinical' in data_dict_lab_dict:
        if DMP_ID in data_dict_lab_dict['clinical']:
            clinical_data = data_dict_lab_dict['clinical'][DMP_ID]
        else:
            # if clinical data not available, fill with zeros
            clinical_data = [0]*MSK_numfeature_dict['clinical']
        mutation_clinical_out.extend(copy.deepcopy(clinical_data))
        
    
    # if 'gender' in data_dict_lab_dict:
    #     clinical_data = data_dict_lab_dict['gender'][DMP_ID]
    #     mutation_clinical_out.extend(copy.deepcopy(clinical_data))
    # if 'age' in data_dict_lab_dict:
    #     clinical_data = [data_dict_lab_dict['age'][DMP_ID][0]/20] # 20 bins
    #     mutation_clinical_out.extend(copy.deepcopy(clinical_data))
    return mutation_clinical_out

def fetch_data_given_dates(dates, data_dict_lab_dict, DMP_ID, MSK_numfeature_dict, cutoffdate=None, max_time_len=15, task='pretrain', advanced_model='default', lb=None, dataset='MSK', med='CISPLATIN', DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT=None):
    '''fetches the data given the dates'''
    if len(dates) == 0:
        raise Exception("No dates found")
    
    # Pre-compute feature categories once
    all_features = set(data_dict_lab_dict.keys())
    static_features = [el for el in ['gender', 'age'] if el in all_features]
    # mutation_clinical_features = [el for el in ['mutation', 'clinical'] if el in all_features]
    mutation_clinical_features = [el for el in ['mutation', 'clinical'] if el in all_features]
    time_series_features = [el for el in ['basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals', 'extra_vitals'] if el in all_features]
    
    # Get static features once
    static_out = []
    for module in static_features:
        static_out.extend(data_dict_lab_dict[module][DMP_ID])
    
    # Get mutation and clinical features once
    if mutation_clinical_features:
        mutation_clinical_out = fetch_mutation_clinical_data(DMP_ID, data_dict_lab_dict, MSK_numfeature_dict)
        static_out.extend(mutation_clinical_out)
    
    # Pre-compute patient data for time series modules
    patient_module_data = {}
    for module in time_series_features:
        # patient_module_data[module] = data_dict_lab_dict[module][DMP_ID]
        # TODO: added 121925 -> if DMP_ID not in data_dict_lab_dict[module], set it to an empty dict
        if DMP_ID not in data_dict_lab_dict[module]:
            data_dict_lab_dict[module][DMP_ID] = {}
        else:
            patient_module_data[module] = data_dict_lab_dict[module][DMP_ID]
    
    if advanced_model == 'seft':
        return _process_seft_model(dates, patient_module_data, MSK_numfeature_dict, 
                                 cutoffdate, max_time_len, static_out)
    elif advanced_model == 'duett':
        return _process_duett_model(dates, patient_module_data, MSK_numfeature_dict, 
                                  max_time_len, static_out, cutoffdate, task, dataset)
    else:  # default, raindrop
        return _process_default_raindrop_model(dates, patient_module_data, MSK_numfeature_dict, 
                                             max_time_len, static_out, cutoffdate, task, 
                                             dataset, advanced_model)

def _process_seft_model(dates, patient_module_data, MSK_numfeature_dict, cutoffdate, max_time_len, static_out):
    """Optimized SEFT model processing"""
    dates = sorted(dates, reverse=True)
    pad_len = max_time_len
    
    times_out = []
    values_out = []
    measurements_out = []
    lengths_out = 0
    
    # Pre-compute cumulative feature lengths
    cumulative_lengths = {}
    existing_len = 0
    for module in ['basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals', 'extra_vitals']:
        if module in patient_module_data:
            cumulative_lengths[module] = existing_len
            existing_len += MSK_numfeature_dict[module]
    
    for date in dates:
        if pad_len <= 0:
            break
            
        for module, module_data in patient_module_data.items():
            if pad_len <= 0:
                break
                
            if date in module_data:
                current_l = module_data[date]
                module_start_idx = cumulative_lengths[module]
                
                for i, val in enumerate(current_l):
                    if pad_len <= 0:
                        break
                    if (val not in ['NA', 'nan', np.nan] and 
                        not (isinstance(val, float) and math.isnan(val))):
                        values_out.append(float(val))
                        measurements_out.append(i + module_start_idx)
                        times_out.append(date)
                        lengths_out += 1
                        pad_len -= 1
    
    # Pad if needed
    if pad_len > 0:
        values_out.extend([0] * pad_len)
        measurements_out.extend([0] * pad_len)
        times_out.extend([0] * pad_len)
    
    # Adjust by cutoffdate
    if cutoffdate is not None:
        times_out = [date - cutoffdate for date in times_out]
    
    # Convert to tensors
    return ((torch.tensor(static_out, dtype=torch.float32),
             torch.tensor(values_out, dtype=torch.float32),
             torch.tensor(measurements_out, dtype=torch.float32),
             lengths_out, pad_len),
            torch.tensor(times_out, dtype=torch.float32))

def _process_duett_model(dates, patient_module_data, MSK_numfeature_dict, max_time_len, 
                        static_out, cutoffdate, task, dataset):
    """Optimized DUETT model processing"""
    min_date, max_date = min(dates), max(dates)
    date_range = max_date - min_date
    
    # Pre-compute bin boundaries
    bin_ends = [min_date + date_range * i / max_time_len for i in range(max_time_len + 1)]
    
    current_patient_current_period_ts_data = []
    current_patient_current_period_dates = []
    
    for time_bin in range(max_time_len):
        bin_start, bin_end = bin_ends[time_bin], bin_ends[time_bin + 1]
        current_date_ts_data = []
        
        for module in patient_module_data:
            module_data = patient_module_data[module]
            
            # Filter dates in current bin
            bin_dates = [date for date in module_data.keys() 
                        if bin_start <= date < bin_end]
            
            if not bin_dates:
                # No data in this bin
                feature_count = MSK_numfeature_dict[module]
                nan_data = np.full(feature_count, np.nan)
                count_data = np.zeros(feature_count)
                combined_data = np.concatenate([nan_data, count_data])
            else:
                # Collect and process data for this bin
                bin_data = []
                for date in bin_dates:
                    date_data = module_data[date]
                    # Replace NA values with np.nan
                    cleaned_data = [np.nan if (x is None or x in ['NA', 'nan', 'NAN']) else float(x)
                                  for x in date_data]
                    bin_data.append(cleaned_data)
                
                bin_array = np.array(bin_data)
                
                # Calculate counts and means
                count_data = np.sum(~np.isnan(bin_array), axis=0)
                mean_data = np.nanmean(bin_array, axis=0)
                
                combined_data = np.concatenate([mean_data, count_data])
            
            current_date_ts_data.extend(combined_data)
        
        current_patient_current_period_ts_data.append(current_date_ts_data)
        current_patient_current_period_dates.append(bin_start)
    
    return _finalize_output(current_patient_current_period_ts_data, current_patient_current_period_dates,
                           static_out, cutoffdate, max_time_len, task, dataset, 'duett')

def _process_default_raindrop_model(dates, patient_module_data, MSK_numfeature_dict, max_time_len, 
                                   static_out, cutoffdate, task, dataset, advanced_model):
    """Optimized default/raindrop model processing"""
    current_patient_current_period_ts_data = []
    current_patient_current_period_dates = []
    
    # Pre-compute feature counts for efficiency
    feature_counts = {module: MSK_numfeature_dict[module] 
                     for module in patient_module_data}
    
    for date in dates:
        current_date_ts_data = []
        # print("date being processed", date, "patient_module_data keys", patient_module_data.keys())
        for module in patient_module_data:
            module_data = patient_module_data[module]
            
            if date in module_data:
                date_data = module_data[date]
                # Vectorized NA replacement
                cleaned_data = [np.nan if (x is None or x in ['NA', 'nan', 'NAN']) else x
                              for x in date_data]
            else:
                # Use pre-computed feature count
                cleaned_data = [np.nan] * feature_counts[module]
            # print("module", module, "feature_count", feature_counts[module], "cleaned_data", cleaned_data)
            
            current_date_ts_data.extend(cleaned_data)
        # print("current_date_ts_data", current_date_ts_data)
        current_patient_current_period_ts_data.append(current_date_ts_data)
        current_patient_current_period_dates.append(date)
    
    seq_len = len(current_patient_current_period_dates) if advanced_model == 'raindrop' else None
    
    return _finalize_output(current_patient_current_period_ts_data, current_patient_current_period_dates,
                           static_out, cutoffdate, max_time_len, task, dataset, advanced_model, seq_len)

def _finalize_output(ts_data, dates, static_out, cutoffdate, max_time_len, task, dataset, 
                    advanced_model, seq_len=None):
    """Common finalization logic for default/duett/raindrop models"""
    # Apply cutoff date adjustment
    if cutoffdate is not None:
        dates = dates_to_relative_dates(dates, cutoffdate)
    
    # Adjust max_time_len for pretrain task
    if task == 'pretrain':
        max_time_len += 1
    
    # Convert strings to floats for Stanford dataset
    if dataset == 'stanfordEHRSHOT':
        ts_data = [[float(x) if x not in ['NA', 'nan', 'NAN'] else np.nan 
                   for x in row] for row in ts_data]
    
    # Padding
    if len(dates) < max_time_len:
        pad_len = max_time_len - len(dates)
        if ts_data:
            total_feature_len = len(ts_data[0])
            ts_data.extend([[np.nan] * total_feature_len] * pad_len)
        dates.extend([np.nan] * pad_len)
    
    # Convert to tensors
    # print("ts_data after processing", len(ts_data), ts_data)
    ts_tensor = torch.tensor(ts_data, dtype=torch.float32)
    dates_tensor = torch.tensor(dates, dtype=torch.float32)
    
    # Handle static features
    if advanced_model in ['default', 'duett'] and static_out:
        static_tensor = torch.tensor(static_out, dtype=torch.float32)
        repeated_static = static_tensor.unsqueeze(-1).repeat(1, max_time_len).permute(1, 0)
        ts_tensor = torch.cat([repeated_static, ts_tensor], dim=1)
        return ts_tensor, dates_tensor
    elif advanced_model == 'raindrop':
        return (ts_tensor, static_out, seq_len), dates_tensor
    else:
        return ts_tensor, dates_tensor
        
def select_random_window(dates, k=15):
    """Random contiguous window of a sorted date list, length between 2 and min(len(dates), k) + 1."""
    if len(dates) < 2:
        raise ValueError("The list must have at least 2 dates to select a window.")
    
    max_window_length = min(len(dates), k)
    
    # up to max_window_length+1, so with k=15 the first 15 dates can predict the 16th
    window_length = np.random.randint(2, max_window_length + 2)
    
    max_start_index = max(len(dates) - window_length, 0)
    start_index = np.random.randint(0, max_start_index + 1)
    return dates[start_index : start_index + window_length]

def sample_from_geometric_likelihood(k, size=1):
    if k<1:
        raise ValueError("k must be greater than or equal to 1.")
    # truncate the infinite series at n = 1000
    max_n = 1000
    n_values = np.arange(max_n + 1)
    
    # unnormalized p(n) = 1/k^n
    unnormalized_probs = (1 / k) ** n_values
    
    probabilities = unnormalized_probs / unnormalized_probs.sum()
    
    samples = np.random.choice(n_values, size=size, p=probabilities)

    samples = samples.tolist()
    
    return samples

def dates_to_relative_dates(dates, cutoffdate):
    return [date-cutoffdate for date in dates]

def check_if_fits_criteria(criteria_medication_name, current_date_medications, exclusion_meds=None, dataset=None, regimen_analysis=False):
    """Return (fits, fits_meds): which sublists of criteria_medication_name are all present.
    CAIA regimen analyses use case-insensitive substring matching plus exclusion_meds;
    everything else uses exact matching.
    """
    fits = False
    fits_meds = []

    use_substring_matching = (dataset and 'CAIA' in dataset and regimen_analysis)

    if use_substring_matching:
        current_meds_upper = [med.upper() for med in current_date_medications]

        # any exclusion med rejects the date
        if exclusion_meds:
            for exclusion_med in exclusion_meds:
                exclusion_med_upper = exclusion_med.upper()
                if any(exclusion_med_upper in current_med_upper for current_med_upper in current_meds_upper):
                    return False, []

        for sub_med_names in criteria_medication_name:
            all_present = True
            for criteria_med in sub_med_names:
                criteria_med_upper = criteria_med.upper()
                if not any(criteria_med_upper in current_med_upper for current_med_upper in current_meds_upper):
                    all_present = False
                    break

            if all_present:
                fits = True
                fits_meds.append(sub_med_names)
    else:
        # exact matching
        for sub_med_names in criteria_medication_name:
            med_included = False not in [m in current_date_medications for m in sub_med_names]
            if med_included:
                fits = True
                fits_meds.append(sub_med_names)

    return fits, fits_meds

def find_medication_start_end_dates(postbc_DMP_ID_to_DATES_to_nonoral_drug_cats_deidentified, criteria_medication_name, as_feature = False, include_dosage = False, include_infusion_dates = True, exclusion_meds=None, dataset=None, regimen_analysis=False):
    """criteria_medication_name can be list of list of meds (each sublist can have one single med (standalone) or multiple meds (drug cocktail, only combination will trigger inclusion))"""

    patient_medication_dates = {}
    if as_feature:
        patient_medication_dates_to_med_names = {}

    if include_infusion_dates:
        dmp_id_to_included_infusion_dates = {}

    for dmp_id, date_to_medications in postbc_DMP_ID_to_DATES_to_nonoral_drug_cats_deidentified.items():
        first_start = None
        last_end = None

        # if as_feature:
        #     patient_medication_dates_to_med_names[dmp_id] = set()
        if include_dosage:
            medication_date_dosage_dict = {}
        for start_date, medications in date_to_medications.items():
            included_meds = []
            current_medications = set([medication[0] for medication in medications])
            med_included = False

            med_included, fits_meds = check_if_fits_criteria(criteria_medication_name, current_medications, exclusion_meds=exclusion_meds, dataset=dataset, regimen_analysis=regimen_analysis)

            if med_included:
                if as_feature:
                    # Store medications for each (patient, date) pair
                    if (dmp_id, start_date) not in patient_medication_dates_to_med_names:
                        patient_medication_dates_to_med_names[(dmp_id, start_date)] = set()
                    for med_list in fits_meds:
                        patient_medication_dates_to_med_names[(dmp_id, start_date)].add(tuple(med_list))
                if include_dosage:
                    if start_date not in medication_date_dosage_dict:
                        medication_date_dosage_dict[start_date] = {}
                    # for med in sub_med_names:
                    #     medication_date_dosage_dict[start_date][med] = []
                    #     for medication in medications:
                    #         if medication[0] == med:
                    #             medication_date_dosage_dict[start_date][med].append((medication[4], medication[5]))
                    # included_meds.append(common_name)
                    # break
                    # TODO: check if it works - we might be removing include dosage altogether
                    for fits_med_sublist in fits_meds:  # Use fits_meds from check_if_fits_criteria 
                        for med in fits_med_sublist:
                            medication_date_dosage_dict[start_date][med] = []
                            for medication in medications:
                                if medication[0] == med:
                                    medication_date_dosage_dict[start_date][med].append((medication[4], medication[5]))
            # start_date is used as both start and end of each non-oral medication
            if med_included:
                if first_start is None or start_date < first_start:
                    first_start = start_date

                if last_end is None or start_date > last_end:
                    last_end = start_date
                if include_infusion_dates:
                    if dmp_id not in dmp_id_to_included_infusion_dates:
                        dmp_id_to_included_infusion_dates[dmp_id] = []
                    # dmp_id_to_included_infusion_dates[dmp_id][DMP_ID].append((start_date, included_meds))
                    dmp_id_to_included_infusion_dates[dmp_id].append(start_date)
        # sort the infusion dates
        if include_infusion_dates and dmp_id in dmp_id_to_included_infusion_dates:
            dmp_id_to_included_infusion_dates[dmp_id] = sorted(dmp_id_to_included_infusion_dates[dmp_id])
            

        if first_start is not None:  # Only add if the medication was found for the patient
            patient_medication_dates[dmp_id] = (first_start, last_end)
            if include_dosage:
                patient_medication_dates[dmp_id] = (first_start, last_end, medication_date_dosage_dict)
    print("patient_medication_dates", list(patient_medication_dates.items())[:5])
    print("dmp_id_to_included_infusion_dates", {k:v for k,v in list(dmp_id_to_included_infusion_dates.items())[:5]})

    if as_feature:
        if not include_infusion_dates:
            return patient_medication_dates, patient_medication_dates_to_med_names
        else:
            return patient_medication_dates, patient_medication_dates_to_med_names, dmp_id_to_included_infusion_dates
    else:
        if not include_infusion_dates:
            return patient_medication_dates
        else:
            print("correct case")
            return patient_medication_dates, dmp_id_to_included_infusion_dates

def last_days_dates(dates, last_days, max_time_len, advanced_model='default'):
    if last_days: # in this case, grab the last max_time_len days
        # reverse sort so position 0 is always the last date before the cutoff
        dates = sorted(dates, reverse=True)
        if advanced_model not in ['duett']:
            dates = dates[:max_time_len]
    else: # in this case, first draw max_time_len dates from the randomly ordered dates and sort them
        np.random.shuffle(dates)
        if advanced_model not in ['duett']:
            dates = dates[:max_time_len]
        # dates = sorted(dates)
        dates = sorted(dates, reverse=True)
    return dates

import random
import math
from collections import defaultdict

def build_balanced_batches(
    batch_size,
    num_infusions,
    ys,
    DMP_IDs,
    other_fields,
    force_add_positives=False  
):
    # Preparation
    infusion_to_pos = defaultdict(list)
    infusion_to_neg = defaultdict(list)
    infusion_types = sorted(set(num_infusions))
    for idx, (inf, y) in enumerate(zip(num_infusions, ys)):
        if y == 1:
            infusion_to_pos[inf].append(idx)
        else:
            infusion_to_neg[inf].append(idx)
    total_samples = len(num_infusions)
    num_batches = math.ceil(total_samples / batch_size)
    # How many samples of each infusion in each batch? Proportional!
    infusion_counts = {inf: len(infusion_to_pos[inf]) + len(infusion_to_neg[inf]) for inf in infusion_types}
    infusion_props = {inf: infusion_counts[inf] / total_samples for inf in infusion_types}
    n_per_batch = {}
    assigned = 0
    for inf in infusion_types:
        n = max(1, int(round(batch_size * infusion_props[inf])))
        assigned += n
        n_per_batch[inf] = n
    while assigned < batch_size:
        largest_inf = max(infusion_types, key=lambda inf: infusion_props[inf])
        n_per_batch[largest_inf] += 1
        assigned += 1
    while assigned > batch_size:
        smallest_inf = min(
            infusion_types, 
            key=lambda inf: n_per_batch[inf] if n_per_batch[inf]>1 else float('inf')
        )
        if n_per_batch[smallest_inf] > 1:
            n_per_batch[smallest_inf] -= 1
            assigned -= 1
        else:
            break

    # Helper: Evenly distribute positives across batches
    def distribute_positives(num_batches, pos, force_add_positives=False):
        pos = pos.copy()
        random.shuffle(pos)
        n_pos = len(pos)
        pos_for_batches = [[] for _ in range(num_batches)]
        if n_pos == 0:
            return pos_for_batches

        # CASE A: Sufficient positives
        if n_pos >= num_batches:
            for i in range(num_batches):
                pos_for_batches[i].append(pos[i])
            # If more positives than batches, assign extras round-robin
            for i in range(num_batches, n_pos):
                pos_for_batches[i % num_batches].append(pos[i])
            return pos_for_batches

        # CASE B: Not enough positives
        if force_add_positives:
            # Place unique positives first, then randomly assign positives to fill rest
            assigned_indices = set()
            random_pos_indices = list(range(n_pos))
            random.shuffle(random_pos_indices)
            for i in range(n_pos):
                pos_for_batches[i].append(pos[random_pos_indices[i]])
                assigned_indices.add(random_pos_indices[i])
            for i in range(n_pos, num_batches):
                # Pick random positive index (with replacement)
                pos_for_batches[i].append(pos[random.choice(range(n_pos))])
            return pos_for_batches
        else:
            # Just assign available positives to first batches, none to overs
            for i in range(n_pos):
                pos_for_batches[i].append(pos[i])
            # Other batches remain without positives
            return pos_for_batches

    infusion_batches = {inf: [] for inf in infusion_types}
    for inf in infusion_types:
        pos = infusion_to_pos[inf]
        neg = infusion_to_neg[inf]
        n_batch = n_per_batch[inf]
        neg_indices = neg.copy()
        random.shuffle(neg_indices)
        curr_neg_start = 0
        pos_for_batches = distribute_positives(
            num_batches,
            pos,
            force_add_positives=force_add_positives
        )
        for b in range(num_batches):
            batch = []
            # Add ALL assigned positives for this batch (could be 0, 1, or many)
            batch += pos_for_batches[b]
            # Fill remainder with negatives
            needed = n_batch - len(batch)
            batch_neg = []
            while needed > 0 and curr_neg_start < len(neg_indices):
                batch_neg.append(neg_indices[curr_neg_start])
                curr_neg_start += 1
                needed -= 1
            # If run out of negatives, resample from all available negatives
            if needed > 0 and neg:
                batch_neg.extend(random.choices(neg, k=needed))
            batch.extend(batch_neg)
            random.shuffle(batch)
            infusion_batches[inf].append(batch)
        random.shuffle(neg_indices)

    # Merge batches from all infusion types
    final_batches = []
    for b in range(num_batches):
        batch = []
        for inf in infusion_types:
            batch += infusion_batches[inf][b]
        random.shuffle(batch)
        final_batches.append(batch)
    # Flatten list of all indices for reordering
    flat_indices = [idx for batch in final_batches for idx in batch]

    def reorder_list(old_list): return [old_list[idx] for idx in flat_indices]
    # print("num_infusions before reordering", len(num_infusions), num_infusions)
    # print("flat_indices", len(flat_indices), flat_indices)
    # # print(set(flat_indices))
    new_num_infusions = reorder_list(num_infusions)
    new_ys = reorder_list(ys)
    new_DMP_IDs = reorder_list(DMP_IDs)
    for name, field in other_fields.items():
        other_fields[name] = reorder_list(field)
    return new_num_infusions, new_ys, new_DMP_IDs, other_fields, final_batches


class FlexibleFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict_lab_dict, dataset_DMP_IDs, MSK_numfeature_dict, dataset = 'MSK', advanced_model = 'default', med_as_feature = False,
                    task = 'binary', med = 'CISPLATIN', binary_days = 180, med_l = [],
                    hyperparameters = {'batch_size': 512, 'max_time_len': 15}, rseed = 1024,
                    data_processing_parameters = {'adjust_from_cutoff': True, 'last_days': True, 'pretrain_idx': True, 'component': 'ts_only'},
                    shared_data_elements = {}, primary_ctype_returned = False, include_dosage = False, control_lack_confounds = False, side_effect_censoring = False,
                    include_infusion_dates = False, premed_only = False, force_add_positives = False, val_test_mode = False, competing_risk_death_check = False, included_infusion = 'random',
                    include_gcsf = False, gcsf_lim = None, max_windows_per_patient = 2, meds_returned = False,
                    test_shuffle=True,
                 **kwargs):
        '''data_dict_lab_dict maps module name -> per-patient data; task is e.g. 'pretrain', 'binary' or a side effect.
        adjust_from_cutoff makes dates relative to the cutoff (for pretrain, a geometric draw past the last date);
        the model's time_calibration overrides this. component: 'ts_only', 'mutation_clinical_only' or 'all'.
        '''
        self.data_dict_lab_dict = data_dict_lab_dict
        print("keys in data_dict_lab_dict", data_dict_lab_dict.keys())
        print("FIXEDDDDD")
        print("FIXEDDDDD")
        print("FIXEDDDDD")
        print("val_test_mode", val_test_mode)
        self.dataset_DMP_IDs = dataset_DMP_IDs
        self.MSK_numfeature_dict = MSK_numfeature_dict
        self.dataset = dataset
        if advanced_model == 'rnn_deephit':
            advanced_model = 'default'  # treat as default for data processing
        self.advanced_model = advanced_model
        self.task = task
        self.med = med
        self.binary_days = binary_days
        self.primary_ctype_returned = primary_ctype_returned 
        self.med_as_feature = med_as_feature
        self.include_dosage = include_dosage
        self.control_lack_confounds = control_lack_confounds
        self.premed_only = premed_only
        if med_as_feature and med_l == []:
            raise ValueError("med_l must be provided if med_as_feature is True")
        self.med_l = med_l
        self.side_effect_censoring = side_effect_censoring
        self.competing_risk_death_check = competing_risk_death_check
        self.included_infusion = included_infusion
        self.include_infusion_dates = include_infusion_dates
        self.include_gcsf = include_gcsf
        self.meds_returned = meds_returned
        self.test_shuffle = test_shuffle
        # Extract anc flag for CAIA datasets (absolute neutrophil count vs neutrophil percent)
        self.anc = kwargs.get('anc', False)
        print(f"DEBUG FlexibleFeatureDataset: anc flag set to {self.anc} (True=absolute neutrophil count, False=neutrophil percent)")

        # Store medication dictionary for meds_returned
        if self.meds_returned and 'DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT' in data_dict_lab_dict:
            self.DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT = data_dict_lab_dict['DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT']

        # included_infusion can be 'first' , 


        # hyperparameters
        self.batch_size, self.max_time_len = hyperparameters['batch_size'], hyperparameters['max_time_len']
        self.max_windows_per_patient = max_windows_per_patient
        self.val_test_mode = val_test_mode
        self.adjust_from_cutoff, self.last_days, self.component, self.pretrain_idx = data_processing_parameters['adjust_from_cutoff'], data_processing_parameters['last_days'], data_processing_parameters['component'], data_processing_parameters['pretrain_idx']
        # shared data elements
        for attr in shared_data_elements:
            print("setting", attr)
            setattr(self, attr, shared_data_elements[attr])

        if self.advanced_model == 'seft':
            from sklearn.preprocessing import LabelBinarizer
            all_indexes = list(range(sum([v for k,v in self.MSK_numfeature_dict.items() if k in self.data_dict_lab_dict.keys() and k in ['basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals', 'extra_vitals']])))
            lb = LabelBinarizer()
            lb.fit(all_indexes)
            self.lb = lb
        else:
            self.lb = None
            
        # # use the IMPACT subset of the DMP_IDs if 'mutation' is one of the keys
        # if 'mutation' in data_dict_lab_dict:
        #     self.dataset_DMP_IDs = [DMP_ID for DMP_ID in self.dataset_DMP_IDs if DMP_ID in data_dict_lab_dict['mutation']]
        
        # get dates (and ys if task is not pretrain)
        real_DMP_IDs = []
        num_removed_dates0 = 0
        num_removed_notenoughprior = 0
        num_removed_notenoughfuture = 0
        num_removed_baseline = 0
        num_removed_censor = 0
        self.num_infusions = []
        if self.include_gcsf:
            self.gcsf_groundtruth = []
            self.gcsf_lim = min(30, gcsf_lim) if gcsf_lim is not None else 30 # default behavior is 30 days post med prescription

        all_alldates = []
        all_alltestdates = []
        infusion_to_death_num = {}
        self.dates = []
        self.window_metadata = []  # Track (DMP_ID, original_patient_idx, window_idx) for debugging
        
        print("task", task)
        print("task", task)
        print("task", task)
        print("task", task)
        print("task", task)
        if task not in ['pretrain', 'binary', 'binary_icu_pre', 'binary_risk', 'cancer_typing', 'progression_risk', 'progression'] and not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', task)): 
            # print("task", task)
            raise ValueError('task must be either pretrain or binary or binary_icu_pre or binary_risk or anemia or neutropenia or AKI or Liver_toxicity or cancer_typing or progression_risk or progression')
        elif task in ['binary', 'binary_icu_pre', 'binary_risk', 'cancer_typing', 'progression_risk', 'progression'] or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next)?(_next_risk|_risk)?$', task)): 
            if task in ['binary_icu_pre']:
                admission_discharge_dates_tuple_dict = self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict
                print("total number of patients with ICU admission", len(admission_discharge_dates_tuple_dict))
            elif task in ['binary', 'binary_risk']:
                admission_discharge_dates_tuple_dict = self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict
                print("total number of patients with hospital admission", len(admission_discharge_dates_tuple_dict))

            if bool(re.match(r'^(febrile_neutropenia|AKI)(_next_risk|_risk)?$', task)):
                from itertools import combinations, permutations, product
                threshold_fn_days = 2
                import warnings

            # go through all DMP_IDs and get the y_tuples
            self.ys = []
            self.cutoffdates = []
            if self.meds_returned:
                self.meds_at_cutoff = []  # Store medications at each cutoff date

            if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', task)):
                print("len(patient_w_med_first_date_dict) prior to processing med", len(self.dataset_DMP_IDs))
                self.dataset_DMP_IDs = [DMP_ID for DMP_ID in self.dataset_DMP_IDs if DMP_ID in self.MED_DMP_ID_TO_START_END_DATE_DICT]
                print("len(patient_w_med_first_date_dict) after processingmed ", len(self.dataset_DMP_IDs))
                # self.DMP_ID_TO_INCLUDED_MED_DATES = self.DMP_ID_TO_INCLUDED_MED_DATES
            # binary, binary_icu_pre, binary_risk and progression tasks are not filtered by medication
            print("len(self.dataset_DMP_IDs)", len(self.dataset_DMP_IDs))
            
            for DMP_ID in self.dataset_DMP_IDs:
                # acquire all available dates
                all_dates = set()
                all_vital_dates = []
                if 'lab' in self.temporal_modalities and DMP_ID in self.data_dict_lab_dict['basic_lab']:
                    all_test_dates = set(self.data_dict_lab_dict['basic_lab'][DMP_ID].keys())
                if 'vitals' in self.temporal_modalities and DMP_ID in self.data_dict_lab_dict['basic_vitals']:
                    all_vital_dates = set(self.data_dict_lab_dict['basic_vitals'][DMP_ID].keys())
                # if 'lab' in self.temporal_modalities and 'vitals' in self.temporal_modalities and DMP_ID in self.data_dict_lab_dict['lab_def_dict']:
                if 'lab' in self.temporal_modalities and 'vitals' in self.temporal_modalities:
                    all_dates = all_dates.union(all_test_dates).union(all_vital_dates)
                elif 'lab' in self.temporal_modalities:
                    all_dates = all_dates.union(all_test_dates)
                elif 'vitals' in self.temporal_modalities:
                    all_dates = all_dates.union(all_vital_dates)
                all_dates = sorted(list(all_dates))
                all_alldates.append(all_dates)
                all_alltestdates.append(all_test_dates)

                # task-specific dates and cutoff processing
                if task in ['binary', 'binary_risk', 'binary_icu_pre']:
                    if DMP_ID in admission_discharge_dates_tuple_dict and len(admission_discharge_dates_tuple_dict[DMP_ID]) >= 1:
                        hospital_tuple = admission_discharge_dates_tuple_dict[DMP_ID][-1]
                        if task in ['binary', 'binary_risk']:
                            dates = [date for date in all_dates if date >= hospital_tuple[0] and date <= hospital_tuple[1]]
                            cutoffdate = hospital_tuple[1]
                        elif task == 'binary_icu_pre':
                            # strict < so only data before the ICU stay is included
                            dates = [date for date in all_dates if date < hospital_tuple[0]]
                            cutoffdate = hospital_tuple[0]
                    else:
                        # if the patient does not have any hospital admission, then use all dates
                        dates = []
                        cutoffdate = None
                        
                elif bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', task)): # side effect prediction
                    DMP_ID_to_DATES_to_TESTSVALUES_DICT = self.data_dict_lab_dict['lab_def_dict']
                    future_dates = [date for date in all_test_dates if date > self.MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][0] and date <= self.MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][1]]
                    dates = [date for date in all_dates if date <= self.MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][0]] # this is used for fetching actual feature values, can be as far back as needed

                if len(dates) == 0:
                    num_removed_dates0 += 1

                # y (prediction gt) and cutoff processing
                # print(len(dates),"NUMBER OF DATES ====")
                if len(dates) > 0:

                    if (task in ['binary',  'binary_icu_pre']):
                        # if DMP_ID not in self.DMP_IDs_to_deathdate_dict or pd.isna(self.DMP_IDs_to_deathdate_dict[DMP_ID]):
                        #     continue # if no death date, skip
                        # print(self.DMP_IDs_to_deathdate_dict)
                        # print("DMP_ID", DMP_ID, "cutoffdate")
                        # print("deathdate", self.DMP_IDs_to_deathdate_dict[DMP_ID])
                        # print("lastcontactdate", self.DMP_IDs_to_lastcontactdate_dict[DMP_ID])

                        if 'stanfordEHRSHOT' in self.dataset:
                            if DMP_ID not in self.DMP_IDs_to_lastcontactdate_dict:
                                continue # skip patients with no contact/outcome date info
                            current_y_tuple = y_tuple_calc(cutoffdate, self.DMP_IDs_to_deathdate_dict.get(DMP_ID, np.nan), self.DMP_IDs_to_lastcontactdate_dict[DMP_ID])
                        else:
                            current_y_tuple = y_tuple_calc(cutoffdate, self.DMP_IDs_to_deathdate_dict[DMP_ID], self.DMP_IDs_to_lastcontactdate_dict[DMP_ID])
                        # if the prediction window>=event window (so censoring doesn't matter); or that the event is not censored
                        # if (not (self.binary_days < current_y_tuple[1] and not current_y_tuple[0])):
                        if (not (current_y_tuple[1] < self.binary_days and not current_y_tuple[0])):
                            # print(DMP_ID, "ADDED HERE")
                            self.cutoffdates.append(cutoffdate)
                            real_DMP_IDs.append(DMP_ID)
                            self.dates.append(last_days_dates(dates, self.last_days, self.max_time_len, advanced_model=self.advanced_model))
                            if not current_y_tuple[0]: # the patient is alive
                                self.ys.append(0)
                            elif self.binary_days < current_y_tuple[1]: # the patient dies in the next x days, but decision boundary is earlier than x, so patient dies outside boundary
                                self.ys.append(0)
                            else: # the patient dies in the next x days, and decision boundary is later than x, so patient does die within boundary
                                self.ys.append(1)
                            self.num_infusions.append(0) # for consistency with side effect prediction, where one patient can have multiple infusions
                        else:
                            # print(DMP_ID,"REMOVED HERE")
                            num_removed_censor += 1
                            # if num_removed_censor % 50 == 0:
                            #     print(DMP_ID,"REMOVED HERE", num_removed_censor, "due to censoring with current_y_tuple", current_y_tuple)

                    elif bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', task)):
                        # get the input_data_dict from patient_specific_side_effect_exclusion_event_adjudication
                        input_data_dict = patient_specific_side_effect_exclusion_event_adjudication(DMP_ID, all_test_dates, all_dates, self.data_dict_lab_dict, self.MED_DMP_ID_TO_START_END_DATE_DICT, self.DMP_IDs_to_lastcontactdate_dict, self.DMP_ID_TO_INCLUDED_MED_DATES, task, all_vital_dates, include_gcsf=self.include_gcsf, anc=self.anc)
                        
                        # if competing risk of death check is on, then check how many die within each infusion window
                        if self.competing_risk_death_check:
                            current_death_date = self.DMP_IDs_to_deathdate_dict.get(DMP_ID, None)
                        # go through the input_data_dict and select those that pass the exclusion criteria
                        previous_included_dates = []
                        future_input_end_date = None
                        for idx, (removal_dict, data_input_dates, input_end_date, y, entire_y, current_num_durations_to_event_or_cutoff) in reversed(list(input_data_dict.items())):
                        # for idx, (removal_dict, data_input_dates, input_end_date, y, entire_y, current_num_durations_to_event_or_cutoff, current_death_date) in reversed_input_data_dict:
                            if not True in removal_dict.values():
                                current_included_dates = sorted(data_input_dates)[-self.max_time_len:] # get the last max_time_len dates
                                if (val_test_mode) or (len(set(previous_included_dates).intersection(set(current_included_dates))) < 0.5*len(current_included_dates)):
                                    if self.include_gcsf:
                                        y, earliest_neut_or_fn_date = y
                                        # print("for DMP_ID", DMP_ID, "cutoff", input_end_date, "y (FN)", y, "earliest_neut_or_fn_date", earliest_neut_or_fn_date)
                                    if 'risk' in task:
                                        if 'next_risk' in task:
                                            self.ys.append((y, 1))
                                        else:
                                            self.ys.append((entire_y, current_num_durations_to_event_or_cutoff)) # y unused here
                                            # # DEBUG: Print first 50 samples
                                            # if len(self.ys) <= 50:
                                            #     print(f"RISK: DMP_ID={DMP_ID}, idx={idx}, y={y}, entire_y={entire_y}, time_to_event={current_num_durations_to_event_or_cutoff}")
                                    else:
                                        self.ys.append(y)
                                        # # DEBUG: Print first 50 samples
                                        # if len(self.ys) <= 50:
                                        #     print(f"NON-RISK: DMP_ID={DMP_ID}, idx={idx}, y={y}")
                                    # gcsf label: given on/after input_end_date, and if y == 1 (FN), before earliest_neut_or_fn_date
                                    if self.include_gcsf:
                                        gcsf_given = False
                                        if DMP_ID in self.gcsf_dict:
                                            for gcsf_date in self.gcsf_dict[DMP_ID]:
                                                if gcsf_date >= input_end_date and gcsf_date <= min(input_end_date + self.gcsf_lim, future_input_end_date if future_input_end_date is not None else input_end_date + 30):
                                                    if y == 1:
                                                        if gcsf_date < earliest_neut_or_fn_date:
                                                            # print("positive GCSF given for DMP_ID", DMP_ID, "on date", gcsf_date, "input_end_date", input_end_date, "earliest_neut_or_fn_date", earliest_neut_or_fn_date)
                                                            gcsf_given = True
                                                            break
                                                    else:
                                                        # print("negative GCSF given for DMP_ID", DMP_ID, "on date", gcsf_date, "input_end_date", input_end_date)
                                                        gcsf_given = True
                                                        break
                                        future_input_end_date = input_end_date
                                        self.gcsf_groundtruth.append(gcsf_given)


                                    real_DMP_IDs.append(DMP_ID)
                                    self.cutoffdates.append(input_end_date)
                                    self.dates.append(last_days_dates(data_input_dates, self.last_days, self.max_time_len, advanced_model=self.advanced_model))
                                    self.num_infusions.append(idx)

                                    # Store medications at cutoff date if meds_returned=True
                                    # Medication format: list of tuples (med_name, dosage, ...)
                                    # Full tuple structure preserved for downstream use (e.g., dosage-based baselines)
                                    if self.meds_returned:
                                        if DMP_ID in self.DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT and input_end_date in self.DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[DMP_ID]:
                                            self.meds_at_cutoff.append(self.DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT[DMP_ID][input_end_date])
                                        else:
                                            self.meds_at_cutoff.append([])
                                    previous_included_dates = current_included_dates
                                    # check if death happens between the cutoffdate and the end of infusion window
                                    # only for a non-nan death date and the last infusion (idx==0)
                                    if self.competing_risk_death_check and pd.notna(current_death_date) and idx == 0:
                                        print("DMP_ID", DMP_ID, "input_end_date", input_end_date, "current_death_date", current_death_date)
                                        if current_death_date > input_end_date and current_death_date <= input_end_date + 30:
                                            if idx not in infusion_to_death_num:
                                                infusion_to_death_num[idx] = 0
                                            infusion_to_death_num[idx] += 1
                            
                        
            self.dataset_DMP_IDs = real_DMP_IDs
            print("after cleaning, len(self.dataset_DMP_IDs)", len(self.dataset_DMP_IDs))
            print("______")

            # make sure the remainder of length divided by batch size is not 1; if so, remove the last sample
            if len(self.ys)%self.batch_size == 1:
                self.dates = self.dates[:-1]
                self.cutoffdates = self.cutoffdates[:-1]
                self.ys = self.ys[:-1]
                self.dataset_DMP_IDs = self.dataset_DMP_IDs[:-1]
                self.num_infusions = self.num_infusions[:-1]
        else:
            self.cutoffdates = []
            # pretrain
            # Sample enough cutoffs for augmented data (10x should be plenty for max_windows_per_patient)
            distances_to_cutoffs = sample_from_geometric_likelihood(2, size = len(self.dataset_DMP_IDs) * 10)
            cutoff_idx = 0  # Initialize counter for indexing into cutoffs
            # make the admission_discharge_dates_tuple_dict
            for i, DMP_ID in enumerate(self.dataset_DMP_IDs):
                # acquire all available dates
                dates = set()
                if 'lab' in self.temporal_modalities and DMP_ID in self.data_dict_lab_dict['basic_lab']:
                    dates = dates.union(set(self.data_dict_lab_dict['basic_lab'][DMP_ID].keys()))
                if 'vitals' in self.temporal_modalities and DMP_ID in self.data_dict_lab_dict['basic_vitals']:
                    dates = dates.union(set(self.data_dict_lab_dict['basic_vitals'][DMP_ID].keys()))
                dates = sorted(list(dates))

                # for folks who have hospitalization dates, we will only use the dates before the last discharge, either last ICU or hospital (the earlier one), to prevent data leak
                if DMP_ID in self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict and DMP_ID in self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict:
                    max_date = min(self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict[DMP_ID][-1][1], self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict[DMP_ID][-1][1])
                elif DMP_ID in self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict:
                    max_date = self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict[DMP_ID][-1][1]
                elif DMP_ID in self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict:
                    max_date = self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict[DMP_ID][-1][1]

                if DMP_ID in self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict or DMP_ID in self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict:
                    dates = [date for date in dates if date <= max_date]

                # randomly select dates for pretraining (len>=2; <=15)
                if self.last_days: # in this case, grab the last max_time_len days
                    # sort the dates
                    dates = sorted(dates, reverse=True)
                    if len(dates) >= 2:
                        # MIN_TRAJECTORY_FOR_AUGMENTATION = 2 * self.max_time_len - 2
                        
                        # Only augment training data with sufficient trajectory length
                        # if not self.val_test_mode and len(dates) >= MIN_TRAJECTORY_FOR_AUGMENTATION:
                        if not self.val_test_mode:
                            # up to max_windows_per_patient random windows; each later window starts after the previous one ends
                            
                            # last discharge date caps the windows (prevents leakage)
                            max_date_constraint = float('inf')
                            if DMP_ID in self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict:
                                max_date_constraint = self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict[DMP_ID][-1][1]
                            elif DMP_ID in self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict:
                                max_date_constraint = self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict[DMP_ID][-1][1]
                            
                            window_idx = 0
                            available_dates = dates.copy()
                            
                            while window_idx < self.max_windows_per_patient:
                                # min_required_dates = self.max_time_len // 2  # COMMENTED: too aggressive for large max_time_len
                                min_required_dates = 10  # Fixed threshold to avoid over-filtering with large max_time_len
                                if len(available_dates) < min_required_dates:
                                    break
                                
                                window = select_random_window(available_dates, k=min(self.max_time_len, len(available_dates)))
                                
                                if max(window) > max_date_constraint:
                                    break
                                
                                sampled_distance = distances_to_cutoffs[cutoff_idx]
                                cutoff_idx += 1
                                cutoffdate = max(window) + sampled_distance
                                
                                if max_date_constraint != float('inf'):
                                    cutoffdate = min(cutoffdate, max_date_constraint)
                                
                                self.dates.append(window)
                                self.cutoffdates.append(cutoffdate)
                                real_DMP_IDs.append(DMP_ID)
                                self.window_metadata.append((DMP_ID, i, window_idx))
                                
                                last_date_in_window = max(window)
                                available_dates = [d for d in available_dates if d > last_date_in_window]
                                # print("DMP_ID", DMP_ID, "window_idx", window_idx, "window length", len(window), "remaining available_dates", len(available_dates))
                                window_idx += 1
                        else:
                            # val/test: a single random window
                            dates = select_random_window(dates, k=self.max_time_len)
                            cutoffdate = max(dates) + distances_to_cutoffs[cutoff_idx]
                            cutoff_idx += 1

                            if DMP_ID in self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict:
                                max_date_constraint = self.deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict[DMP_ID][-1][1]
                                cutoffdate = min(cutoffdate, max_date_constraint)
                            elif DMP_ID in self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict:
                                max_date_constraint = self.deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict[DMP_ID][-1][1]
                                cutoffdate = min(cutoffdate, max_date_constraint)

                            self.dates.append(dates)
                            self.cutoffdates.append(cutoffdate)
                            real_DMP_IDs.append(DMP_ID)
                            self.window_metadata.append((DMP_ID, i, 0))  # Single window always has idx 0
                else:
                    if len(dates) >= 2:
                        dates = np.random.choice(dates, size = min(np.random.randint(2, self.max_time_len+1), len(dates)), replace=False).tolist()
                        dates = sorted(dates, reverse=True)
                        cutoffdate = max(dates) + distances_to_cutoffs[cutoff_idx]
                        cutoff_idx += 1
                        self.dates.append(dates)
                        self.cutoffdates.append(cutoffdate)
                        real_DMP_IDs.append(DMP_ID)
                        self.window_metadata.append((DMP_ID, i, 0))  # Single window for non-last_days mode
                    
            self.dataset_DMP_IDs = real_DMP_IDs

            # make sure the remainder of length divided by batch size is not 1; if so, remove the last sample
            if len(self.dataset_DMP_IDs)%self.batch_size == 1:
                self.dataset_DMP_IDs = self.dataset_DMP_IDs[:-1]


        if self.data_dict_lab_dict != {}:
            if self.advanced_model in ['default', 'duett', 'raindrop']:
                self.X_ts = []
                self.X_dates = []
                if task == 'pretrain' and self.primary_ctype_returned:
                    self.ctype_l = [] # for contrastive learning objective
            else:
                self.static_l = []
                self.times_l = []
                self.values_l = []
                self.measurements_l = []
                self.lengths_l = []
                self.pad_lens = []
            # check if the task as a string contain 'anemia' or 'neutropenia' or 'AKI'
            if task not in ['pretrain', 'binary', 'binary_icu_pre', 'binary_risk', 'cancer_typing', 'progression_risk', 'progression'] and not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', task)):
                raise ValueError('task must be either pretrain or binary or anemia or binary_icu_pre or binary_risk or neutropenia or AKI')
            else:
                if self.component in ['ts_only', 'all', 'parsimonious']:
                    cleaned_DMP_IDs = []
                    for i, DMP_ID in enumerate(self.dataset_DMP_IDs):
                        dates = self.dates[i]
                        if self.adjust_from_cutoff:
                            current_ts_data, current_dates = fetch_data_given_dates(dates, self.data_dict_lab_dict, DMP_ID, self.MSK_numfeature_dict, self.cutoffdates[i], max_time_len=self.max_time_len, task=task, advanced_model=advanced_model, lb=self.lb, dataset=self.dataset)
                        else:
                            current_ts_data, current_dates = fetch_data_given_dates(dates, self.data_dict_lab_dict, DMP_ID, self.MSK_numfeature_dict, max_time_len=self.max_time_len, task=task, advanced_model=advanced_model, lb=self.lb, dataset=self.dataset)

                        # # Print last timepoint feature values for each patient
                        # if self.advanced_model in ['default', 'duett', 'raindrop']:
                        #     # For default/duett/raindrop, current_ts_data is a tensor [max_time_len, num_features]
                        #     # Index 0 is the most recent timepoint (dates are reverse sorted)
                        #     last_timepoint_features = current_ts_data[0] if isinstance(current_ts_data, torch.Tensor) else current_ts_data[0][0]
                        #     num_valid = (~torch.isnan(last_timepoint_features)).sum().item() if isinstance(last_timepoint_features, torch.Tensor) else 0
                        #     print(f"Last timepoint features: DMP_ID={DMP_ID}, infusion_idx={self.num_infusions[i] if i < len(self.num_infusions) else 'N/A'}, "
                        #           f"num_valid_features={num_valid}/{len(last_timepoint_features) if isinstance(last_timepoint_features, torch.Tensor) else 'N/A'}, "
                        #           f"cutoff_date={self.cutoffdates[i]}, last_date={dates[0] if dates else 'N/A'}")

                        if self.advanced_model in ['default', 'duett', 'raindrop']:
                            # For pretrain, skip if first timepoint (target) has all NaN
                            if task == 'pretrain' and torch.isnan(current_ts_data[0]).all():
                                continue  # Skip this patient

                            cleaned_DMP_IDs.append(DMP_ID)
                            self.X_ts.append(current_ts_data)
                            self.X_dates.append(current_dates)
                            if task == 'pretrain' and self.primary_ctype_returned:
                                self.ctype_l.append(PRIMARY_CTYPES.index(self.DMP_ID_to_PRIMARY_CTYPE_dict[DMP_ID]))
                        else:
                            static_out, values_out, measurements_out, lengths_out, pad_len = current_ts_data
                            times_out = current_dates
                            if lengths_out > 0:
                                cleaned_DMP_IDs.append(DMP_ID)
                                self.static_l.append(static_out)
                                self.times_l.append(times_out)
                                self.values_l.append(values_out)
                                self.measurements_l.append(measurements_out)
                                self.lengths_l.append(lengths_out)
                                self.pad_lens.append(pad_len)
                    if self.advanced_model == 'seft':
                        self.dataset_DMP_IDs = cleaned_DMP_IDs
                        if len(self.dataset_DMP_IDs) == 0:
                            print("No data to return")
                    elif self.advanced_model in ['default', 'duett'] and task == 'pretrain':
                        # For pretrain, use cleaned list (filtered out all-NaN targets)
                        orig_count = len(self.dataset_DMP_IDs)
                        self.dataset_DMP_IDs = cleaned_DMP_IDs
                        filtered_count = orig_count - len(cleaned_DMP_IDs)
                        if filtered_count > 0:
                            print(f"Filtered out {filtered_count} patients with all-NaN target features (kept {len(cleaned_DMP_IDs)}/{orig_count})")

                    if self.advanced_model in ['default', 'duett']:
                        print("len(self.dataset_DMP_IDs)", len(self.dataset_DMP_IDs),len(self.X_dates), np.mean(self.X_dates))
                    elif self.advanced_model == 'seft':
                        print("len(self.dataset_DMP_IDs)", len(self.dataset_DMP_IDs))
                    if task =='pretrain':
                        print("data loading complete, len(self.dataset_DMP_IDs)", len(self.dataset_DMP_IDs), len(self.X_dates), np.mean(self.X_dates))
                        # Debug: check if first patient has valid data at index 0
                        if len(self.X_ts) > 0:
                            first_ts = self.X_ts[0]
                            print(f"DEBUG pretrain: First patient ts shape: {first_ts.shape}")
                            print(f"DEBUG pretrain: First timepoint (index 0) all NaN: {torch.isnan(first_ts[0]).all().item()}")
                            print(f"DEBUG pretrain: First timepoint (index 0) any valid: {(~torch.isnan(first_ts[0])).any().item()}")
                            print(f"DEBUG pretrain: Number of valid values at index 0: {(~torch.isnan(first_ts[0])).sum().item()}")
                            # Check a few more patients
                            all_nan_count = sum(torch.isnan(ts[0]).all().item() for ts in self.X_ts[:min(100, len(self.X_ts))])
                            print(f"DEBUG pretrain: Out of first 100 patients, {all_nan_count} have all-NaN at index 0")
                    else:
                        print("num_removed, num_removed_dates0, num_removed_notenoughprior, num_removed_notenoughfuture, num_removed_baseline, num_removed_censor", num_removed_dates0, num_removed_notenoughprior, num_removed_notenoughfuture, num_removed_baseline, num_removed_censor)
                        if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', task)):
                            # for each number of infusions, print the number of samples and the number of positive samples
                            all_num_infusions_ordered = sorted(list(set(self.num_infusions)))
                            num_infusion_to_indices_dict = {num_infusion: [i for i, x in enumerate(self.num_infusions) if x == num_infusion] for num_infusion in all_num_infusions_ordered}
                            for num_infusion in all_num_infusions_ordered:
                                current_num_infusion_idxs = [i for i, x in enumerate(self.num_infusions) if x == num_infusion]
                                current_ys = [self.ys[i] for i in current_num_infusion_idxs]
                                # print("num_infusion, num_samples, num_pos_ys", num_infusion, len(current_ys), sum(current_ys))
                                # remove num_infusions where there is only 1 sample as AUROC is not defined
                                # also remove num_infusions where it's all of one class
                                if len(current_num_infusion_idxs) == 1:
                                    idx_to_remove = current_num_infusion_idxs
                                    for idx in sorted(idx_to_remove, reverse=True):
                                        self.num_infusions.pop(idx)
                                        self.ys.pop(idx)
                                        self.cutoffdates.pop(idx)
                                        self.dates.pop(idx)
                                        if self.advanced_model in ['default', 'duett', 'raindrop']:
                                            self.X_ts.pop(idx)
                                            self.X_dates.pop(idx)
                                            if task == 'pretrain' and self.primary_ctype_returned:
                                                self.ctype_l.pop(idx)
                                        else:
                                            self.static_l.pop(idx)
                                            self.times_l.pop(idx)
                                            self.values_l.pop(idx)
                                            self.measurements_l.pop(idx)
                                            self.lengths_l.pop(idx)
                                            self.pad_lens.pop(idx)
                                        self.dataset_DMP_IDs.pop(idx)
                                        if self.include_gcsf:
                                            self.gcsf_groundtruth.pop(idx)
                                else:
                                    current_num_infusion_idxs = [i for i, x in enumerate(self.num_infusions) if x == num_infusion]
                                    current_ys = [self.ys[i] for i in current_num_infusion_idxs]
                                    if 'risk' not in task:
                                        print("num_infusion, num_samples, num_pos_ys", num_infusion, len(current_ys), sum(current_ys))
                                    elif 'risk' in task:
                                        print("num_infusion, num_samples, num_pos_ys", num_infusion, len(current_ys), sum([y for y, _ in current_ys]))
                                    num_infusion_to_indices_dict[num_infusion] = current_num_infusion_idxs
                            # also identify the patients with over 100 infusions
                            DMP_IDs_w_over100infusions = set([self.dataset_DMP_IDs[i] for i, x in enumerate(self.num_infusions) if x >= 100])
                            print("sanity checking DMP_IDs_w_over100infusions", DMP_IDs_w_over100infusions)
                        if 'risk' not in task:
                            if self.advanced_model in ['default', 'duett']:
                                print("data loading complete, len(self.dataset_DMP_IDs), len(set(dataset_DMP_IDs)), self.num_pos_ys", len(self.dataset_DMP_IDs), len(set(self.dataset_DMP_IDs)), len(self.X_dates), np.mean(self.X_dates), len(self.ys), sum(self.ys))
                            elif self.advanced_model == 'seft':
                                print("data loading complete, len(self.dataset_DMP_IDs), len(set(dataset_DMP_IDs)), num_pos_ys", len(self.dataset_DMP_IDs), len(set(self.dataset_DMP_IDs)), len(self.ys), sum(self.ys))
                        else:
                            if self.advanced_model != 'seft':
                                print("data loading complete, len(self.dataset_DMP_IDs), len(set(dataset_DMP_IDs))", len(self.dataset_DMP_IDs), len(set(self.dataset_DMP_IDs)), len(self.X_dates), np.mean(self.X_dates), len(self.ys), sum([y for y, _ in self.ys]))
                            else:
                                print("data loading complete, len(self.dataset_DMP_IDs), len(set(dataset_DMP_IDs))", len(self.dataset_DMP_IDs), len(set(self.dataset_DMP_IDs)), len(self.ys), sum([y for y, _ in self.ys]))
                        # if "Liver_toxicity" in task:
                        #     print("num_sos, num_AST, num_ALT, num_ALP", num_sos, num_AST, num_ALT, num_ALP)

                        if len(self.ys) > 0:
                            # form other_fields dict to be fed into the build_balanced_batches function
                            other_fields = {"cutoffdates": self.cutoffdates, "dates": self.dates}
                            if self.advanced_model in ['default', 'duett', 'raindrop']:
                                other_fields["X_dates"] = self.X_dates
                                other_fields["X_ts"] = self.X_ts
                                if task == 'pretrain' and self.primary_ctype_returned:
                                    other_fields["ctype_l"] = self.ctype_l
                            else:
                                other_fields["static_l"] = self.static_l
                                other_fields["times_l"] = self.times_l
                                other_fields["values_l"] = self.values_l
                                other_fields["measurements_l"] = self.measurements_l
                                other_fields["lengths_l"] = self.lengths_l
                                other_fields["pad_lens"] = self.pad_lens
                            
                            if self.include_gcsf:
                                other_fields["gcsf_groundtruth"] = self.gcsf_groundtruth

                            print("AAAAA", set(self.num_infusions))
                            # if not (set(self.num_infusions)==set([0]) and self.advanced_model in ['seft', 'raindrop']) and force_add_positives == False:
                            if not (set(self.num_infusions)==set([0])) and force_add_positives == True and bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', task)):
                                print("Building balanced batches with force_add_positives=True")
                                self.num_infusions, self.ys, self.dataset_DMP_IDs, other_fields, _ = build_balanced_batches(self.batch_size, self.num_infusions, self.ys, self.dataset_DMP_IDs, other_fields, force_add_positives=force_add_positives)
                            # print("VVVVV", set(self.num_infusions))
                            # repopulate fields from other_fields
                            self.cutoffdates, self.dates = other_fields["cutoffdates"], other_fields["dates"]
                            if self.advanced_model in ['default', 'duett', 'raindrop']:
                                self.X_dates, self.X_ts = other_fields["X_dates"], other_fields["X_ts"]
                                if task == 'pretrain' and self.primary_ctype_returned:
                                    self.ctype_l = other_fields["ctype_l"]
                            else:
                                self.static_l = other_fields["static_l"]
                                self.times_l = other_fields["times_l"]
                                self.values_l = other_fields["values_l"]
                                self.measurements_l = other_fields["measurements_l"]
                                self.lengths_l = other_fields["lengths_l"]
                                self.pad_lens = other_fields["pad_lens"]

                            if self.include_gcsf:
                                self.gcsf_groundtruth = other_fields["gcsf_groundtruth"]
                                
                            if self.advanced_model in ['default', 'duett']:
                                print("data loading complete, len(self.dataset_DMP_IDs), len(set(dataset_DMP_IDs)), self.num_pos_ys", len(self.dataset_DMP_IDs), len(set(self.dataset_DMP_IDs)), len(self.X_dates), np.mean(self.X_dates), len(self.ys))
                                # print the number of pos, total num per num_infusions
                                num_infusion_to_indices_dict = {num_infusion: [i for i, x in enumerate(self.num_infusions) if x == num_infusion] for num_infusion in sorted(list(set(self.num_infusions)))}
                                for num_infusion in sorted(num_infusion_to_indices_dict.keys()):
                                    current_indices = num_infusion_to_indices_dict[num_infusion]
                                    current_ys = [self.ys[i] for i in current_indices]
                                    if 'risk' not in task:
                                        print("num_infusion, num_samples, num_pos_ys", num_infusion, len(current_ys), sum(current_ys))
                                    else:
                                        print("num_infusion, num_samples, num_pos_ys", num_infusion, len(current_ys))
                                if 'risk' in task:
                                    # calculate number of unique patients and unique positives
                                    unique_num_patients = len(set(self.dataset_DMP_IDs))
                                    unique_num_positives = 0
                                    for i in range(len(self.ys)):
                                        unique_num_positives+=self.ys[i][0]
                                    print("unique_num_patients, unique_num_positives", unique_num_patients, unique_num_positives)
                                    print("unique_num_patients, unique_num_positives", unique_num_patients, unique_num_positives)
                                    print("unique_num_patients, unique_num_positives", unique_num_patients, unique_num_positives)
    
                                if self.competing_risk_death_check:
                                    # print infusion_to_death_num
                                    print("infusion_to_death_num", infusion_to_death_num)
                            elif self.advanced_model == 'seft':
                                print("data loading complete, len(self.dataset_DMP_IDs), len(set(dataset_DMP_IDs)), num_pos_ys", len(self.dataset_DMP_IDs), len(set(self.dataset_DMP_IDs)), len(self.ys))
                
    def __len__(self):
        return len(self.dataset_DMP_IDs)
    
    def __getitem__(self, idx):
        DMP_ID = self.dataset_DMP_IDs[idx]
        # print("DMP_ID", DMP_ID)
        if self.task in ['binary', 'pretrain',  'binary_icu_pre', 'binary_risk', 'cancer_typing', 'progression_risk', 'progression'] or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', self.task)):
            if self.data_dict_lab_dict != {}:
                if self.component in ['ts_only', 'all', 'parsimonious']:
                    if self.advanced_model in ['default', 'duett']:
                        if self.med_as_feature:
                            cutoffdate = self.cutoffdates[idx]
                            current_meds = self.MED_DMP_ID_TO_MED_NAMES.get((DMP_ID, cutoffdate), set())
                            current_meds_features = torch.tensor([1 if tuple(med) in current_meds else 0 for med in self.med_l], dtype=torch.float32)
                            if not self.include_gcsf:
                                x = (self.X_ts[idx], self.X_dates[idx], current_meds_features, self.num_infusions[idx], DMP_ID)
                            else:
                                x = (self.X_ts[idx], self.X_dates[idx], current_meds_features, self.num_infusions[idx], DMP_ID, self.gcsf_groundtruth[idx])
                        else:
                            # omit num_infusions when include_infusion_dates is off
                            if not self.include_infusion_dates:
                                x = (self.X_ts[idx], self.X_dates[idx])
                            else:
                                if not self.include_gcsf:
                                    x = (self.X_ts[idx], self.X_dates[idx], self.num_infusions[idx], DMP_ID)
                                else:
                                    x = (self.X_ts[idx], self.X_dates[idx], self.num_infusions[idx], DMP_ID, self.gcsf_groundtruth[idx])
                    elif self.advanced_model == 'raindrop':
                        xs = self.X_ts[idx]
                        xs, static, seq_len = xs
                        x = (xs, static, seq_len, self.X_dates[idx], self.num_infusions[idx])
                    elif self.advanced_model == 'seft':
                        x = (self.static_l[idx], self.times_l[idx], self.values_l[idx], self.measurements_l[idx], self.lengths_l[idx], self.pad_lens[idx], self.num_infusions[idx])
            else:
                # error message
                raise ValueError('Get Item Error - No data to return')
            if self.task in ['binary', 'binary_icu_pre', 'binary_risk', 'cancer_typing', 'progression_risk', 'progression'] or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', self.task)):
                y = self.ys[idx]
            else:
                y = 0
            if self.task == 'pretrain':
                if self.component in ['all']:
                    if not self.pretrain_idx:
                        result = (x, y, self.ctype_l[idx])
                    else:
                        result = ((x, y, self.ctype_l[idx]), DMP_ID)
                else:
                    # For pretrain with component 'ts_only' or other components
                    if self.pretrain_idx:
                        result = ((x, y), DMP_ID)
                    else:
                        result = (x, y)

                return result
            else:
                if not self.primary_ctype_returned:
                    if not self.include_dosage:
                        if self.meds_returned:
                            result = (x, y, self.meds_at_cutoff[idx])
                        else:
                            result = (x, y)
                    else:
                        if self.meds_returned:
                            result = (x, y, (self.DMP_ID_to_dosage_info_dict[DMP_ID], self.cutoffdates[idx]), self.meds_at_cutoff[idx])
                        else:
                            result = (x, y, (self.DMP_ID_to_dosage_info_dict[DMP_ID], self.cutoffdates[idx]))
                else:
                    if not self.include_dosage:
                        if self.meds_returned:
                            result = (x, y, (DMP_ID, self.DMP_ID_to_PRIMARY_CTYPE_dict.get(DMP_ID, 'Unknown'), self.cutoffdates[idx]), self.meds_at_cutoff[idx])
                        else:
                            result = (x, y, (DMP_ID, self.DMP_ID_to_PRIMARY_CTYPE_dict.get(DMP_ID, 'Unknown'), self.cutoffdates[idx]))
                    else:
                        if self.meds_returned:
                            result = (x, y, (DMP_ID, self.DMP_ID_to_PRIMARY_CTYPE_dict.get(DMP_ID, 'Unknown'), self.cutoffdates[idx]), (self.DMP_ID_to_dosage_info_dict[DMP_ID], self.cutoffdates[idx]), self.meds_at_cutoff[idx])
                        else:
                            result = (x, y, (DMP_ID, self.DMP_ID_to_PRIMARY_CTYPE_dict.get(DMP_ID, 'Unknown'), self.cutoffdates[idx]), (self.DMP_ID_to_dosage_info_dict[DMP_ID], self.cutoffdates[idx]))

                return result

def collate_into_seqs(batch, meds_returned=False):
    # Handle different return formats from dataset
    batch_len = len(batch[0])

    if batch_len == 2:
        # Standard format: (xs, ys)
        xs, ys = zip(*batch)
        # Stack each element of xs into batched tensors                                                                                                                                                   
        xs_batched = tuple(torch.stack(x) if isinstance(x[0], torch.Tensor) else x                                                                                                                        
                        for x in zip(*xs))    
        return zip(*xs), ys
        # return xs_batched, ys
    elif batch_len == 3:
        # Format with medications: (xs, ys, medications)
        xs, ys, meds = zip(*batch)
        # # Keep medications as list since they're variable length
        return zip(*xs), ys, list(meds)
        # # Stack each element of xs into batched tensors
        # xs_batched = tuple(torch.stack(x) if isinstance(x[0], torch.Tensor) else x
        #                 for x in zip(*xs))
        # return xs_batched, ys, list(meds)
    elif batch_len == 4:
        # Extended format: (xs, ys, info, medications)
        xs, ys, info, meds = zip(*batch)
        # # Stack each element of xs into batched tensors
        # xs_batched = tuple(torch.stack(x) if isinstance(x[0], torch.Tensor) else x
        #                 for x in zip(*xs))
        # Keep medications as list since they're variable length
        return zip(*xs), ys, info, list(meds)
        # return xs_batched, ys, info, list(meds)
    elif batch_len == 5:
        # Full format: (xs, ys, info, dosage_info, medications)
        xs, ys, info, dosage_info, meds = zip(*batch)
        # Keep medications as list since they're variable length
        return zip(*xs), ys, info, dosage_info, list(meds)
        # # Stack each element of xs into batched tensors
        # xs_batched = tuple(torch.stack(x) if isinstance(x[0], torch.Tensor) else x
        #                 for x in zip(*xs))
        # return xs_batched, ys, info, dosage_info, list(meds)
    else:
        raise ValueError(f"Unexpected batch format: {batch_len} elements per item")


class FlexibleFeatureDataModule(pl.LightningDataModule):
    def __init__(self,
                # dataset-specific
                data_dict_lab_dict, split_dict, dataset = 'MSK', component='ts_only', data_source_path=sparc_paths.DATA_ROOT + '/',
                # hyperparameters
                batch_size = 512, max_time_len = 15, num_workers=1, # num_workers specifies an efficient way to load data
                # task-specific
                task = 'pretrain', binary_days = 180, med='CISPLATIN', med_l = [],
                # med_specific
                med_as_feature = False,
                # first principle predictor or not
                first_principle_predictor = False,
                # model parameters
                advanced_model='default',
                # data processing parameters
                adjust_from_cutoff = True, last_days = True,
                pretrain_idx = True, primary_ctype_returned = False,
                include_dosage = False, control_lack_confounds = False,
                side_effect_censoring = False, premed_only = False, include_infusion_dates = False, competing_risk_death_check = False,
                include_gcsf = False, gcsf_lim = None, max_windows_per_patient = 3, meds_returned = False,
                exclusion_med_l = None, regimen_analysis = False,
                test_shuffle = True,
                **kwargs):
        super().__init__()
        print("TRYING SOMETHING OUTTTTT")
        print("TRYING SOMETHING OUTTTTT")
        print("TRYING SOMETHING OUTTTTT")
        print("TRYING SOMETHING OUTTTTT")
        print("primary_ctype_returned", primary_ctype_returned)
        print("primary_ctype_returned", primary_ctype_returned)
        print("primary_ctype_returned", primary_ctype_returned)
        self.split_dict = split_dict
        # data/task specific parameters
        self.data_dict_lab_dict = data_dict_lab_dict
        self.task = task
        self.binary_days = binary_days
        self.med = med
        if advanced_model == 'rnn_deephit':
            self.advanced_model = 'default'
        self.advanced_model = advanced_model
        self.med_as_feature = med_as_feature
        self.first_principle_predictor = first_principle_predictor
        self.include_dosage = include_dosage
        self.premed_only = premed_only
        self.include_infusion_dates = include_infusion_dates
        self.test_shuffle = test_shuffle

        # Override include_infusion_dates for binary tasks (infusion data only relevant for side-effect tasks)
        if task in ['binary', 'binary_icu_pre', 'binary_risk'] and self.include_infusion_dates:
            print(f"WARNING: include_infusion_dates=True is not applicable for task '{task}'. Setting to False.")
            self.include_infusion_dates = False

        if med_as_feature and med_l == []:
            raise ValueError("med_l must be provided if med_as_feature is True")
        self.med_l = med_l
        self.include_gcsf = include_gcsf
        self.gcsf_lim = gcsf_lim
        self.meds_returned = meds_returned
        self.dataset = dataset
        self.exclusion_med_l = exclusion_med_l
        self.regimen_analysis = regimen_analysis
        # Extract anc flag for CAIA datasets (absolute neutrophil count vs neutrophil percent)
        self.anc = kwargs.get('anc', False)
        print(f"DEBUG: anc flag set to {self.anc} (True=absolute neutrophil count, False=neutrophil percent)")
        print(f"DEBUG: anc flag set to {self.anc} (True=absolute neutrophil count, False=neutrophil percent)")
        print(f"DEBUG: anc flag set to {self.anc} (True=absolute neutrophil count, False=neutrophil percent)")
        if self.first_principle_predictor:
            if 'risk' in task:
                task_fp = '_'.join(task.split('_')[:-1]) if '_' in task else task
            else:
                task_fp = task
            self.MSK_numfeature_dict = {'age': MSK_numfeature_dict['age'],
                                        'gender': MSK_numfeature_dict['gender'],
                                        'basic_lab':MSK_firstprinciple_numfeature_dict[f'{task_fp}_lab'],
                                        'basic_vitals':MSK_firstprinciple_numfeature_dict[f'{task_fp}_vitals']}
            print("Using first principle predictor features", self.MSK_numfeature_dict)
        else:
            self.MSK_numfeature_dict = MSK_numfeature_dict
        print("self.MSK_numfeature_dict", self.MSK_numfeature_dict)

        # extract the data split
        self.train_DMP_IDs = self.split_dict['train']
        self.val_DMP_IDs = self.split_dict['val']
        self.test_DMP_IDs = self.split_dict['test']
        self.shared_data_elements = {}

        if include_gcsf:
            if not 'gcsf_dict' in data_dict_lab_dict:
                raise ValueError("include_gcsf is True but gcsf_dict not in data_dict_lab_dict")
            else:
                self.shared_data_elements['gcsf_dict'] = data_dict_lab_dict['gcsf_dict']

        # hyperparameters
        self.hyperparameters = {
            'batch_size': batch_size,
            'max_time_len': max_time_len
        }
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_windows_per_patient = max_windows_per_patient
        self.data_processing_parameters = {
            'adjust_from_cutoff': adjust_from_cutoff,
            'last_days': last_days,
            'pretrain_idx': pretrain_idx,
            'component': component,
        }

        # demographic data
        # if task == 'pretrain' or dataset == 'MSK':
        if dataset == 'MSK':
            # pretrain uses the MSK model dev set
            path = f'{data_source_path}/MSK'
        
        elif dataset == 'MSK_timeshift':

            path = f'{data_source_path}/MSK_timeshift'
        
        elif dataset == 'stanfordEHRSHOT':
            path = f'{data_source_path}/stanfordEHRSHOT'
        elif dataset == 'MIMICIV_hosp':
            path = f'{data_source_path}/MIMICIV_hosp'
        
        # if pretrain, get cancer types
        if task == 'pretrain' or primary_ctype_returned:
            self.shared_data_elements['DMP_ID_to_PRIMARY_CTYPE_dict'] = data_dict_lab_dict.get('DMP_ID_to_PRIMARY_CTYPE_dict', {})
        # based on data_dict_lab_dict, check temporal modalities
        self.shared_data_elements['temporal_modalities'] = []
        if 'basic_lab' in data_dict_lab_dict or 'extra_lab' in data_dict_lab_dict or 'include_lab' in data_dict_lab_dict:
            self.shared_data_elements['temporal_modalities'].append('lab')
        if 'basic_vitals' in data_dict_lab_dict or 'extra_vitals' in data_dict_lab_dict or 'include_vitals' in data_dict_lab_dict:
            self.shared_data_elements['temporal_modalities'].append('vitals')

        # based on task, load the data
        if task in ['pretrain', 'binary', 'binary_risk']:
            self.shared_data_elements['deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict'] = data_dict_lab_dict.get('deidentified_DMP_ID_to_hospital_admission_discharge_dates_tuple_dict', {})
        if task in ['pretrain', 'binary_icu_pre']:
            self.shared_data_elements['deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict'] = data_dict_lab_dict.get('deidentified_DMP_ID_to_icu_admission_discharge_dates_tuple_dict', {})
            
        if task in ['binary', 'binary_icu_pre', 'binary_risk'] or competing_risk_death_check:
            self.shared_data_elements['DMP_IDs_to_deathdate_dict'] = data_dict_lab_dict.get('DMP_IDs_to_deathdate_dict', {})
        self.shared_data_elements['DMP_IDs_to_lastcontactdate_dict'] = data_dict_lab_dict.get('DMP_IDs_to_lastcontactdate_dict', {})
        
        if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', task)):
            DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT = data_dict_lab_dict['DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT']

            print("len(DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT)", len(DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT))
            print("med", med)
            if med_as_feature:
                if not include_infusion_dates:
                    start_end_dates_dict, self.shared_data_elements['MED_DMP_ID_TO_MED_NAMES'] = find_medication_start_end_dates(DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT, med, as_feature = self.med_as_feature, include_dosage = self.include_dosage, exclusion_meds=self.exclusion_med_l, dataset=self.dataset, regimen_analysis=self.regimen_analysis)
                else:
                    start_end_dates_dict, self.shared_data_elements['MED_DMP_ID_TO_MED_NAMES'], self.shared_data_elements['DMP_ID_TO_INCLUDED_MED_DATES'] = find_medication_start_end_dates(DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT, med, as_feature = self.med_as_feature, include_dosage = self.include_dosage, include_infusion_dates = True, exclusion_meds=self.exclusion_med_l, dataset=self.dataset, regimen_analysis=self.regimen_analysis)
            else:
                if not include_infusion_dates:
                    start_end_dates_dict = find_medication_start_end_dates(DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT, med, include_dosage = self.include_dosage, exclusion_meds=self.exclusion_med_l, dataset=self.dataset, regimen_analysis=self.regimen_analysis)
                else:
                    print("include_infusion_dates", include_infusion_dates)
                    start_end_dates_dict, self.shared_data_elements['DMP_ID_TO_INCLUDED_MED_DATES'] = find_medication_start_end_dates(DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT, med, include_dosage = self.include_dosage, include_infusion_dates = include_infusion_dates, exclusion_meds=self.exclusion_med_l, dataset=self.dataset, regimen_analysis=self.regimen_analysis)
            if include_dosage:
                self.shared_data_elements['DMP_ID_to_dosage_info_dict'] = {DMP_ID: dosage_info for DMP_ID, (start, end, dosage_info) in start_end_dates_dict.items()}
                start_end_dates_dict = {DMP_ID: (start, end) for DMP_ID, (start, end, dosage_info) in start_end_dates_dict.items()}
            self.shared_data_elements['MED_DMP_ID_TO_START_END_DATE_DICT'] = start_end_dates_dict
            # print("self.shared_data_elements['MED_DMP_ID_TO_START_END_DATE_DICT'", self.shared_data_elements['MED_DMP_ID_TO_START_END_DATE_DICT'][:15])
            # use the actual end date + 30 days (30 day wash out period) as the end point by which we check for side effect
            if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI)(_next_risk|_risk)?$', task)):
                self.shared_data_elements['MED_DMP_ID_TO_START_END_DATE_DICT'] = {DMP_ID: (start, end+30) for DMP_ID, (start, end) in self.shared_data_elements['MED_DMP_ID_TO_START_END_DATE_DICT'].items()}
            elif bool(re.match(r'^(Liver_toxicity)(_next_risk|_risk)?$', task)):
                # use the actual end date + 90 days (90 day ICI wash out period) as the end point by which we check for side effect
                self.shared_data_elements['MED_DMP_ID_TO_START_END_DATE_DICT'] = {DMP_ID: (start, end+90) for DMP_ID, (start, end) in self.shared_data_elements['MED_DMP_ID_TO_START_END_DATE_DICT'].items()}
            print("len(self.shared_data_elements['MED_DMP_ID_TO_START_END_DATE_DICT'])", len(self.shared_data_elements['MED_DMP_ID_TO_START_END_DATE_DICT']))
            self.shared_data_elements['DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT'] = DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT
        
        if task == 'progression_risk' or bool (re.match(r'^(progression)(_?\d+days)?$', task)):
            progression_df = pd.read_csv(f'{path}/df_medonc.csv', index_col = 0)
            progression_df['CDD_SERVICE_DTE_since_seq'] = pd.to_timedelta(progression_df['CDD_SERVICE_DTE_since_seq']).dt.days
            # remove nans from CDD_SERVICE_DTE_since_seq
            progression_df = progression_df[progression_df['CDD_SERVICE_DTE_since_seq'].notna()]
            self.shared_data_elements['DMP_ID_to_progression_dates_dict'] = {}
            for dmp_id, group in progression_df.dropna(subset=['CDD_SERVICE_DTE_since_seq']).groupby('DMP_ID'):
                progression_positive_df = group[group['INFER_PROBABILITY_PROGRESSION_POSITIVE'] >= 0.5]
                progression_positive_dates = progression_positive_df['CDD_SERVICE_DTE_since_seq'].dropna().tolist()
                self.shared_data_elements['DMP_ID_to_progression_dates_dict'][dmp_id] = progression_positive_dates

        # construct train/val/test datasets
        print("preparing train set, prior to processing # patients", len(self.train_DMP_IDs))
        self.train_dataset = FlexibleFeatureDataset(data_dict_lab_dict, self.train_DMP_IDs, self.MSK_numfeature_dict, dataset, advanced_model = self.advanced_model, task = self.task, med = self.med, binary_days = binary_days, hyperparameters = self.hyperparameters, data_processing_parameters = self.data_processing_parameters, shared_data_elements = self.shared_data_elements, primary_ctype_returned = primary_ctype_returned, med_as_feature = self.med_as_feature, med_l = self.med_l, include_dosage = self.include_dosage, control_lack_confounds = control_lack_confounds, side_effect_censoring = side_effect_censoring, include_infusion_dates = include_infusion_dates, premed_only = self.premed_only, force_add_positives=True, include_gcsf = include_gcsf, gcsf_lim = gcsf_lim, max_windows_per_patient = self.max_windows_per_patient, val_test_mode = False, meds_returned = self.meds_returned, anc=self.anc)
        print("preparing val set, prior to processing # patients", len(self.val_DMP_IDs))
        self.val_dataset = FlexibleFeatureDataset(data_dict_lab_dict, self.val_DMP_IDs, self.MSK_numfeature_dict, dataset, advanced_model = self.advanced_model, task = self.task, med = self.med, binary_days = binary_days, hyperparameters = self.hyperparameters, data_processing_parameters = self.data_processing_parameters, shared_data_elements = self.shared_data_elements, primary_ctype_returned = primary_ctype_returned, med_as_feature = self.med_as_feature, med_l = self.med_l, include_dosage = self.include_dosage, control_lack_confounds = control_lack_confounds, side_effect_censoring = side_effect_censoring, include_infusion_dates = include_infusion_dates, premed_only = self.premed_only, val_test_mode =True, include_gcsf = include_gcsf, gcsf_lim = gcsf_lim, max_windows_per_patient = self.max_windows_per_patient, meds_returned = self.meds_returned, anc=self.anc)
        print("preparing test set, prior to processing # patients", len(self.test_DMP_IDs))
        self.test_dataset = FlexibleFeatureDataset(data_dict_lab_dict, self.test_DMP_IDs, self.MSK_numfeature_dict, dataset, advanced_model = self.advanced_model, task = self.task, med = self.med, binary_days = binary_days, hyperparameters = self.hyperparameters, data_processing_parameters = self.data_processing_parameters, shared_data_elements = self.shared_data_elements, primary_ctype_returned = primary_ctype_returned, med_as_feature = self.med_as_feature, med_l = self.med_l, include_dosage = self.include_dosage, control_lack_confounds = control_lack_confounds, side_effect_censoring = side_effect_censoring, include_infusion_dates = include_infusion_dates, premed_only = self.premed_only, val_test_mode =True, competing_risk_death_check = False, include_gcsf = include_gcsf, gcsf_lim = gcsf_lim, max_windows_per_patient = self.max_windows_per_patient, meds_returned = self.meds_returned, anc=self.anc, test_shuffle=self.test_shuffle)
        # self.test_dataset = FlexibleFeatureDataset(data_dict_lab_dict, self.test_DMP_IDs, self.MSK_numfeature_dict, dataset, advanced_model = self.advanced_model, task = self.task, med = self.med, binary_days = binary_days, hyperparameters = self.hyperparameters, data_processing_parameters = self.data_processing_parameters, shared_data_elements = self.shared_data_elements, primary_ctype_returned = primary_ctype_returned, med_as_feature = self.med_as_feature, med_l = self.med_l, include_dosage = self.include_dosage, control_lack_confounds = control_lack_confounds, side_effect_censoring = side_effect_censoring, include_infusion_dates = include_infusion_dates, premed_only = self.premed_only, val_test_mode =False, competing_risk_death_check = False)

        # Compute ratio of next-infusion events for hybrid loss weighting
        if bool(__import__('re').match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)$', task)):
            n_pos_events = sum(y_event for y_event, time_to_event in self.train_dataset.ys)
            
            # n_total_events = 0
            # n_next_events = 0

            # for i, (y_event, time_to_event) in enumerate(self.train_dataset.ys):
            #     if y_event == 1:  # Is an event
            #         n_total_events += 1
            #         if time_to_event == 1:  # Next infusion
            #             n_next_events += 1

            # Use unique patients to balance the ratio (avoids bias from patients with many infusions)
            n_unique_patients = len(set(self.train_dataset.dataset_DMP_IDs))
            ratio_next = n_pos_events / n_unique_patients if n_unique_patients > 0 else 0.5
            self.ratio_next_infusion = ratio_next
            print(f"[Hybrid Loss Stats] Training data: {n_pos_events} events / {n_unique_patients} unique patients (ratio={ratio_next:.3f})")
        else:
            self.ratio_next_infusion = None

        # Trim datasets if last batch would be too small (< 1/5 of batch_size)
        min_last_batch_size = self.batch_size // 5 if self.dataset not in ['MIMICIV_hosp', 'stanfordEHRSHOT'] else 2
        for dataset_name in ['train_dataset', 'val_dataset', 'test_dataset']:
            dataset = getattr(self, dataset_name)
            dataset_len = len(dataset)
            remainder = dataset_len % self.batch_size
            if remainder > 0 and remainder < min_last_batch_size:
                print(f"Trimming {dataset_name}: last batch size {remainder} < {min_last_batch_size}, removing {remainder} samples")
                # Trim the dataset by reducing the list lengths
                new_len = dataset_len - remainder
                dataset.dataset_DMP_IDs = dataset.dataset_DMP_IDs[:new_len]
                if hasattr(dataset, 'X_ts'):
                    dataset.X_ts = dataset.X_ts[:new_len]
                if hasattr(dataset, 'X_dates'):
                    dataset.X_dates = dataset.X_dates[:new_len]
                if hasattr(dataset, 'ys'):
                    dataset.ys = dataset.ys[:new_len]
                if hasattr(dataset, 'cutoffdates'):
                    dataset.cutoffdates = dataset.cutoffdates[:new_len]
                dataset.dates = dataset.dates[:new_len]
                dataset.num_infusions = dataset.num_infusions[:new_len]
                if hasattr(dataset, 'gcsf_groundtruth'):
                    dataset.gcsf_groundtruth = dataset.gcsf_groundtruth[:new_len]
                if hasattr(dataset, 'window_metadata'):
                    dataset.window_metadata = dataset.window_metadata[:new_len]
                print(f"  New {dataset_name} length: {len(dataset)}")

        self.dl_args = {'batch_size': self.batch_size, 'collate_fn': collate_into_seqs, 'num_workers': num_workers}


    def train_dataloader(self):
        """Return the training dataloader."""
        dataset = self.train_dataset

        # return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        if self.meds_returned:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_into_seqs)
        else:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.meds_returned:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_into_seqs)
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.meds_returned:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle, num_workers=self.num_workers, collate_fn=collate_into_seqs)
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle, num_workers=self.num_workers)
