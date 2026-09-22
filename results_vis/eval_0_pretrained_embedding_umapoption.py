"""Pretrained SPARC embeddings projected to 2D (t-SNE), one plot per feature.

One time window is sampled per patient, embeddings are averaged across the 10
pretrained folds, and points are coloured by the z-scored value of a given lab
or vital at the last timepoint. Marker shape encodes gender: circle male,
diamond female.

modules='parsimonious' adds the CBC differential (Neut, Mono, Lymph, Baso, Eos)
to the feature order gender, age, CBC differential, basic labs, basic vitals,
and loads the matching pretrained models.

SUBSET_BY_CANCER_TYPE / CANCER_TYPES_TO_INCLUDE restrict the plot to selected
cancer types, taking each patient's earliest annotation and applying the same
collapsing rules as the cancer-type evaluation. QUICK_TEST runs on a small
subset with NUM_FOLDS_QUICK_TEST folds.

Output: {results_path}/pretrained_embeddings/test_set_feature_specific_plots_{dataset}_{subset|all}/
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import pickle
import sys, inspect
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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
from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset  # Use the unified data loading function
from MSK_featurename_list import build_module_index_config
from SPARC_inf_risk_modular import *
import torch
import numpy as np
from sklearn.manifold import TSNE
try:
    import umap  # umap-learn; only required when DIM_REDUCTION == 'umap'
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False
import pytorch_lightning as pl

# CONFIGURATION
option = 'pretrain'
modules = 'parsimonious'  # Set to 'parsimonious' to include CBC differential tests
dataset = 'MSK_CAIA'
caia_portion = 'prior'
dataset = 'MSK_CAIA_timeshift'
caia_portion = 'post'


## HYPERPARAMETERS ##
batch_size = 512
data_source_path=sparc_paths.DATA_ROOT + '/'

# 060726 Architecture parameters
use_module_tokenizer = True   # -umt True
use_continuous_time = True    # -uct True
post_mlp_dropout = 0.2        # -pmd 0.2
missing_mask = True
temporal_aggregation = 'attention'

# Visualization options
SUBSET_BY_CANCER_TYPE = True  # 'all' = all cancer types with colors, True = subset to CANCER_TYPES_TO_INCLUDE, False = all patients no coloring

# Dimensionality reduction method for the 2D embedding plots.
#   'tsne' (default) — sklearn.manifold.TSNE
#   'umap'           — umap-learn (pip install umap-learn); requires the package.
DIM_REDUCTION = 'umap'
# Optional UMAP hyperparameters (only used when DIM_REDUCTION == 'umap').
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Large scatter PDFs: matplotlib writes one vector object per point, so a
# full-cohort embedding (tens of thousands of dots) yields a huge PDF. Rasterizing
# the scatter layer caps the size (one embedded bitmap instead of N paths) while
# keeping axes/text/colorbar vector-crisp. But for SMALL point clouds a pure-vector
# PDF is actually smaller AND sharper, so we rasterize adaptively: only when the
# number of plotted points exceeds RASTER_POINT_THRESHOLD. Set the threshold to 0
# to always rasterize, or a huge number to never rasterize.
#   empirically (plain circle markers): vector wins below ~40k points @200dpi,
#   ~25k @300dpi; rasterized size is roughly flat (~1 MB) above that.
RASTER_POINT_THRESHOLD = 25000
# DPI of the embedded scatter bitmap when rasterized (and of the saved PDF).
# 300 = effectively lossless, 200 = smaller w/ minimal visible loss, 150 = smallest.
SCATTER_RASTER_DPI = 300

CANCER_TYPES_TO_INCLUDE = [
    'Glioma',                        # Brain: CNS effects (n=5,519)
    'Myelodysplastic Syndromes',     # Blood: MDS (n=1,849)
    'Germ Cell Tumor',               # Germ cell (n=3,495)
    'Hepatobiliary Cancer',          # Liver: ALT, AST, bilirubin (n=1,927)
]        
# Quick test mode (for rapid testing)
# QUICK_TEST = False  # Set to True to use small data subset and only first 2 model folds
QUICK_TEST = False  # Set to True to use small data subset and only first 2 model folds
NUM_FOLDS_QUICK_TEST = 5  # Number of folds to use in quick test mode

# DATA LOADING

if QUICK_TEST:
    print("\n" + "="*70)
    print("QUICK TEST MODE ENABLED")
    print(f"- Using small data subset")
    print(f"- Loading only first {NUM_FOLDS_QUICK_TEST} model folds")
    print("="*70 + "\n")

print("Loading dataset using unified data loading function...")
# Use the unified load_dataset function which handles all dataset types including CAIA
data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset,
    data_source_path=data_source_path,
    option=option,
    modules=modules,
    valfold=0,  # Doesn't matter for test-only evaluation
    mode='test' if 'timeshift' not in dataset else 'test_all',
    rseed=42,
    competing_risk_death_check=False,
    include_gcsf=False,
    data_fraction_option='default',
    quick_test=QUICK_TEST,
    caia_portion=caia_portion,
)

print("modular_featuredim_dict", modular_featuredim_dict)

# Build module index config for 060726 models
module_index_cfg = build_module_index_config(modular_featuredim_dict)

# Calculate n_static_features based on module configuration
# Parsimonious: gender + age
if modules == 'parsimonious':
    static_module_names = ['gender', 'age']
else:  # 'all'
    static_module_names = ['gender', 'age', 'primary_ctype', 'genetic_ancestry', 'met_status', 'clinical', 'mutation']
n_static_features = sum(modular_featuredim_dict.get(mod, 0) for mod in static_module_names)
print(f"Calculated n_static_features for {modules} with keys {static_module_names}: {n_static_features}")

# CANCER TYPE HELPER FUNCTIONS (from eval_6)

def get_patient_cancer_type(dmp_id, ctype_dict):
    """
    Get the first (earliest) cancer type for a patient.

    Args:
        dmp_id: Patient DMP ID
        ctype_dict: Dictionary mapping DMP_ID to dates to cancer types

    Returns:
        str or None: The first cancer type for this patient, or None if not found
    """
    if dmp_id not in ctype_dict:
        return None

    date_to_ctypes = ctype_dict[dmp_id]
    if not date_to_ctypes:
        return None

    # Get the earliest date (most negative number)
    earliest_date = min(date_to_ctypes.keys())
    cancer_types = date_to_ctypes[earliest_date]

    if not cancer_types or len(cancer_types) == 0:
        return None

    # Return the first cancer type in the list
    return cancer_types[0]


def collapse_cancer_types(cancer_type):
    """
    Apply cancer type collapsing rules:
    1. Merge "Skin cancer", "Malignant neoplasm of skin", and "Malignant melanoma" -> "Skin cancer"
    2. Remove "Malignant neoplasm of gastrointestinal tract" (return None)
    3. Merge "Mature T-cell AND/OR NK cell neoplasm" and "Non-Hodgkin's lymphoma (clinical)" -> "Non-Hodgkin lymphoma"

    Args:
        cancer_type: Original cancer type string

    Returns:
        str or None: Collapsed cancer type, or None if should be excluded
    """
    if cancer_type is None:
        return None

    # Normalize to lowercase for comparison
    ctype_lower = cancer_type.lower()

    # Rule 1: Collapse skin cancers
    if any(term in ctype_lower for term in ['skin cancer', 'malignant neoplasm of skin', 'malignant melanoma']):
        return "Skin cancer"

    # Rule 2: Exclude GI tract parent category
    if 'malignant neoplasm of gastrointestinal tract' in ctype_lower:
        return None

    # Rule 3: Merge lymphomas
    if any(term in ctype_lower for term in ['mature t-cell and/or nk cell neoplasm', "non-hodgkin's lymphoma"]):
        return "Non-Hodgkin lymphoma"

    # Return original if no rules apply
    return cancer_type

# Load demographic data for visualization metadata
# CAIA datasets don't have a separate demographic CSV - age/gender are in data_dict_lab_dict
print(f"\nDEBUG: dataset = '{dataset}'")
print(f"DEBUG: 'CAIA' in dataset = {'CAIA' in dataset}")
if 'CAIA' in dataset:
    print("\nCAIA dataset detected - using age/gender from loaded data")

    # Debug: Check what gender data looks like
    sample_pids = list(data_dict_lab_dict['gender'].keys())[:3]
    print(f"Sample gender data for first 3 patients:")
    for pid in sample_pids:
        gender_val = data_dict_lab_dict['gender'][pid]
        print(f"  PID {pid}: {gender_val} (type: {type(gender_val)}, value type: {type(gender_val) if not hasattr(gender_val, '__iter__') else type(gender_val[0]) if len(gender_val) > 0 else 'empty'})")

    # For CAIA: gender is one-hot encoded [0]=MALE, [1]=FEMALE
    # Need to decode it back to string labels
    # Note: Use 'Male'/'Female' (not 'MALE'/'FEMALE') to match shape_dict format
    DMP_ID_to_gender_dict = {}
    for pid, gender_oh in data_dict_lab_dict['gender'].items():
        # gender_oh might be a list, tuple, or numpy array
        # Convert to list for consistent comparison
        gender_val = list(gender_oh) if hasattr(gender_oh, '__iter__') else [gender_oh]

        if gender_val == [0]:
            DMP_ID_to_gender_dict[pid] = 'MALE'
        elif gender_val == [1]:
            DMP_ID_to_gender_dict[pid] = 'FEMALE'
        else:
            DMP_ID_to_gender_dict[pid] = 'Unknown'

    # Check gender decoding results
    gender_decode_counts = {}
    for gender in DMP_ID_to_gender_dict.values():
        gender_decode_counts[gender] = gender_decode_counts.get(gender, 0) + 1
    print(f"Gender decoding results: {gender_decode_counts}")

    # Age is already in the correct format [age_bin]
    DMP_ID_to_agegroup_dict = {k: v[0] for k, v in data_dict_lab_dict['age'].items()}

    # Load cancer type data from the comprehensive pickle file (same as eval_6)
    if dataset == 'MSK_CAIA':
        ctype_dict_path = f'{data_source_path}/{dataset}/DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_MSK_CAIA_till2021_050726.pickle'
    elif dataset == 'MSK_CAIA_timeshift':
        ctype_dict_path = f'{data_source_path}/{dataset}/DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_MSK_CAIA_after2021_050726.pickle'
    print(f"\nDEBUG: Checking cancer type pickle at: {ctype_dict_path}")
    print(f"DEBUG: File exists: {os.path.exists(ctype_dict_path)}")
    if os.path.exists(ctype_dict_path):
        with open(ctype_dict_path, 'rb') as f:
            ctype_dict = pickle.load(f)
        print(f"Loaded cancer type data for {len(ctype_dict)} patients from pickle file")

        # Debug: check sample keys from pickle
        sample_pickle_keys = list(ctype_dict.keys())[:5]
        print(f"DEBUG: Sample pickle keys (type={type(sample_pickle_keys[0]) if sample_pickle_keys else 'N/A'}): {sample_pickle_keys}")

        # Build DMP_ID_to_PRIMARY_CTYPE_dict using the helper functions
        DMP_ID_to_PRIMARY_CTYPE_dict = {}
        for dmp_id in ctype_dict.keys():
            cancer_type = get_patient_cancer_type(dmp_id, ctype_dict)
            if cancer_type is not None:
                collapsed_type = collapse_cancer_types(cancer_type)
                if collapsed_type is not None:
                    DMP_ID_to_PRIMARY_CTYPE_dict[dmp_id] = collapsed_type

        print(f"Processed cancer types for {len(DMP_ID_to_PRIMARY_CTYPE_dict)} patients (after collapsing)")

        # Debug: check sample keys from processed dict
        sample_processed_keys = list(DMP_ID_to_PRIMARY_CTYPE_dict.keys())[:5]
        print(f"DEBUG: Sample processed dict keys (type={type(sample_processed_keys[0]) if sample_processed_keys else 'N/A'}): {sample_processed_keys}")
        ctype_counts = {}
        for v in DMP_ID_to_PRIMARY_CTYPE_dict.values():
            ctype_counts[v] = ctype_counts.get(v, 0) + 1
        print(f"Cancer type distribution: {sorted(ctype_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
    else:
        DMP_ID_to_PRIMARY_CTYPE_dict = {}
        print(f"WARNING: cancer type pickle not found at {ctype_dict_path}")
        if SUBSET_BY_CANCER_TYPE:
            print("Setting SUBSET_BY_CANCER_TYPE=False")
            SUBSET_BY_CANCER_TYPE = False
else:
    # For non-CAIA datasets, load from demographic CSV
    dePHIed_demographic_df = pd.read_csv(f'{data_source_path}/{dataset}/dePHIed_demographic_df_{dataset}.csv', index_col=0)

    DMP_ID_to_PRIMARY_CTYPE_dict = dict(zip(dePHIed_demographic_df['DMP_ID'], dePHIed_demographic_df['primary_cancer_type_th']))
    DMP_ID_to_agegroup_dict = dict(zip(dePHIed_demographic_df['DMP_ID'], dePHIed_demographic_df['age_at_first_seq_binned_5']))
    DMP_ID_to_gender_dict = dict(zip(dePHIed_demographic_df['DMP_ID'], dePHIed_demographic_df['GENDER']))

    # Check the number of each cancer type
    ctype_to_num_dict = {}
    for k, v in DMP_ID_to_PRIMARY_CTYPE_dict.items():
        if v not in ctype_to_num_dict:
            ctype_to_num_dict[v] = 0
        ctype_to_num_dict[v] += 1
    print("DMP_ID_to_PRIMARY_CTYPE_dict", len(DMP_ID_to_PRIMARY_CTYPE_dict), ctype_to_num_dict)

print(f"Test set size: {len(split_dict['test'])}")

pl.seed_everything(42)
data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict,
    dataset=dataset, component=component,
    batch_size=512, max_time_len=20, num_workers=1, pretrain_idx=True,
    task='pretrain', binary_days=180, med='CISPLATIN', advanced_model='default'
)

advanced_model = 'default'

all_lasttp_xs_to_predict = []

# EMBEDDING EXTRACTION
# Note: The test_dataloader with pretrain_idx=True returns multiple windows per patient
# (controlled by max_windows_per_patient parameter, typically 2-3 windows).
# For visualization, we want exactly ONE sample per patient to avoid:
# - Overrepresentation of patients with many time windows
# - Duplicate/correlated points in t-SNE space
# Therefore, we track seen patients and only use the FIRST window for each patient.

# load each fold's best model and get the embeddings on the test set
num_folds = NUM_FOLDS_QUICK_TEST if QUICK_TEST else 10
print(f"\nLoading {num_folds} model fold(s)...")
start_all = True
for fold in range(num_folds):
    print("FOLD", fold)
    pl.seed_everything(42)
    if modules == 'practical':
        model_folder = f'{sparc_paths.SPARC_MODEL_ROOT}/practical_MSK_CAIA_060726/practical_pretrain_20tp_augmented_att_modular/fold{fold}/'
    elif modules == 'parsimonious':
        model_folder = f'{sparc_paths.SPARC_MODEL_ROOT}/parsimonious_MSK_CAIA_060726/parsimonious_pretrain_20tp_augmented_att_modular/fold{fold}/'

    existing_files = os.listdir(model_folder)
    model_path = [el for el in existing_files if 'epoch' in el][0]
    optimal_model_path = f'{model_folder}/{model_path}'
    print(f"Loading model from: {optimal_model_path}")

    model = SPARC.load_from_checkpoint(
        optimal_model_path,
        modular_featuredim_dict=modular_featuredim_dict,
        option='pretrain',
        advanced_model=advanced_model,
        max_time_len=20,
        temporal_aggregation=temporal_aggregation,
        missing_mask=missing_mask,
        module_index_cfg=module_index_cfg,
        use_module_tokenizer=use_module_tokenizer,
        use_continuous_time=use_continuous_time,
        n_static_features=n_static_features,
        post_mlp_dropout=post_mlp_dropout
    )
    model.eval()
    basic_feature_trainer = pl.Trainer(max_epochs=1, deterministic=True)
    current_fold_lasttp_xs_to_predict = None

    # get the embeddings from test set - sample one window per patient
    # Note: test_dataloader may return multiple windows per patient (controlled by max_windows_per_patient)
    # We track seen patients to ensure only one sample per patient is used
    all_DMP_IDs = []
    seen_patients = set()
    start = True
    for batch, DMP_IDs in data_module.test_dataloader():
        torch.cuda.empty_cache()

        # Filter to keep only the first window for each patient
        unique_indices = []
        batch_DMP_IDs = []
        for idx, dmp_id in enumerate(DMP_IDs):
            if dmp_id not in seen_patients:
                unique_indices.append(idx)
                seen_patients.add(dmp_id)
                batch_DMP_IDs.append(dmp_id)

        if len(unique_indices) == 0:
            continue

        # Subset batch to unique patients only
        # For pretrain mode with pretrain_idx=True:
        #   - dataloader returns: (batch, DMP_IDs)
        #   - batch structure is: ((xs, times), y)
        #   - data_prep expects: (((xs, times), y), DMP_ID)
        # We need to reconstruct the expected format with filtered data

        # NOTE: Older version with input_len/pred_horizon (for multi-horizon pretraining):

        # Current version (standard pretraining without multi-horizon):
        (xs, times), y = batch

        # Create the batch subset in the format data_prep expects
        batch_subset = (((xs[unique_indices], times[unique_indices]), y[unique_indices]), batch_DMP_IDs)

        xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict = model.data_prep(batch_subset)
        # Note: For pretrain with component='ts_only' (practical modules), xs_combined is just a tensor
        # It would be (xs_combined, ctypes) only if primary_ctype_returned=True (component='all')
        last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]

        # check device of xs_combined, src_key_padding_mask_combined, last_tp_xs_to_predict
        xs_combined = xs_combined.to(model.device)
        src_key_padding_mask_combined = src_key_padding_mask_combined.to(model.device)
        last_tp_times = last_tp_times.to(model.device)
        z, z1 = model(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times)

        # detach unnecessary tensors from gpu
        xs_combined = xs_combined.detach()
        src_key_padding_mask_combined = src_key_padding_mask_combined.detach()
        last_tp_times = last_tp_times.detach()

        if current_fold_lasttp_xs_to_predict is None:
            current_fold_lasttp_xs_to_predict = last_tp_xs_to_predict.detach().cpu()
        else:
            current_fold_lasttp_xs_to_predict = torch.cat((current_fold_lasttp_xs_to_predict, last_tp_xs_to_predict.detach().cpu()), dim=0)

        z = z.detach().cpu()
        if start:
            z_rep_concat = z
            start = False
        else:
            z_rep_concat = torch.cat((z_rep_concat, z), dim=0)
        all_DMP_IDs.extend(batch_DMP_IDs)

    if start_all:
        z_rep_concat_all = z_rep_concat
        start_all = False
    else:
        z_rep_concat_all += z_rep_concat
    all_lasttp_xs_to_predict.append(current_fold_lasttp_xs_to_predict)

    print(f"Fold {fold}: Processed {len(all_DMP_IDs)} unique patients")

# Average embeddings across folds
z_rep_concat_all = z_rep_concat_all / num_folds
z_rep_concat = z_rep_concat_all

print(f"\nAveraged embeddings across {num_folds} fold(s)")
print(f"Total unique patients: {len(all_DMP_IDs)}")
print(f"First 10 patient IDs: {all_DMP_IDs[:10]}")

# Get cancer types and genders for all patients
# For CAIA datasets, DMP_IDs can be integers, tensors, or strings depending on the data loader
# The DMP_ID_to_PRIMARY_CTYPE_dict keys are always the same type as in the original ctype_dict
if 'CAIA' in dataset:
    # Try both integer and original key to handle different formats
    all_cancer_types = []
    for el in all_DMP_IDs:
        # Convert tensor to integer if needed
        if torch.is_tensor(el):
            el_int = el.item()
        else:
            try:
                el_int = int(el)
            except (ValueError, TypeError):
                el_int = el

        # Look up cancer type
        ctype = DMP_ID_to_PRIMARY_CTYPE_dict.get(el_int, None)
        if ctype is None:
            ctype = DMP_ID_to_PRIMARY_CTYPE_dict.get(el, 'Unknown')
        all_cancer_types.append(ctype)
else:
    all_cancer_types = [DMP_ID_to_PRIMARY_CTYPE_dict.get(el, 'Unknown') for el in all_DMP_IDs]

# Similarly for genders - handle tensors
all_genders = []
for el in all_DMP_IDs:
    if torch.is_tensor(el):
        el_int = el.item()
    else:
        try:
            el_int = int(el)
        except (ValueError, TypeError):
            el_int = el
    gender = DMP_ID_to_gender_dict.get(el_int, None)
    if gender is None:
        gender = DMP_ID_to_gender_dict.get(el, 'Unknown')
    all_genders.append(gender)

# Import feature lists from MSK_featurename_list (already imported at top)
from MSK_featurename_list import BASIC_TESTS, BASIC_VITALS, CBC_DIFFERENTIAL_TESTS

# Build included_features list based on module
# IMPORTANT: Feature order must match the data loader!
# In data_loaders_inf_risk_fixed.py, time_series_features are ordered as:
#   ['basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals', 'extra_vitals']
# Then static features (gender, age) are prepended in _process_default_raindrop_model
# So final order is: [gender, age, basic_lab, cbc_differential, basic_vitals]

if modules == 'parsimonious':
    # Parsimonious order: gender, age, basic_lab, cbc_differential, basic_vitals
    # NOTE: cbc_differential comes AFTER basic_lab (not before!)
    included_features = ['gender', 'age'] + BASIC_TESTS + CBC_DIFFERENTIAL_TESTS + BASIC_VITALS
    print(f"\nParsimonious module: Including {len(CBC_DIFFERENTIAL_TESTS)} CBC differential tests")
    print(f"  CBC differential tests: {CBC_DIFFERENTIAL_TESTS}")
    print(f"  Feature order: gender, age, basic_lab ({len(BASIC_TESTS)}), cbc_differential ({len(CBC_DIFFERENTIAL_TESTS)}), basic_vitals ({len(BASIC_VITALS)})")
    print(f"  Total features: {len(included_features)}")
else:
    # Practical includes: gender, age, basic_lab, basic_vitals
    included_features = ['gender', 'age'] + BASIC_TESTS + BASIC_VITALS
    print(f"\nPractical module: Total features: {len(included_features)} (gender=1, age=1, basic_lab={len(BASIC_TESTS)}, basic_vitals={len(BASIC_VITALS)})")

# Color dict and shape dict
shape_dict = {}
shape_dict['MALE'] = 'o'
shape_dict['FEMALE'] = 'D'

# Get subset based on SUBSET_BY_CANCER_TYPE flag
if SUBSET_BY_CANCER_TYPE == 'all':
    # Use all patients and assign colors to all cancer types
    subset_idx = list(range(len(all_DMP_IDs)))
    subset_DMP_IDs = all_DMP_IDs
    subset_ctypes = all_cancer_types
    subset_suffix = '_all_cancer_types'

    # Get unique cancer types and auto-assign colors
    unique_ctypes = sorted(list(set(all_cancer_types)))
    print(f"\nVisualizing all {len(unique_ctypes)} cancer types with distinct colors")

    # Auto-assign colors using tab20 colormap (20 colors) or hsv for more
    if len(unique_ctypes) <= 20:
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i) for i in range(len(unique_ctypes))]
    else:
        cmap = plt.cm.get_cmap('hsv')
        colors = [cmap(i/len(unique_ctypes)) for i in range(len(unique_ctypes))]

    color_dict = {ctype: colors[i] for i, ctype in enumerate(unique_ctypes)}

elif SUBSET_BY_CANCER_TYPE == True:
    # Debug: Print actual cancer types in test set
    unique_ctypes_in_data = sorted(list(set(all_cancer_types)))
    print(f"\nDEBUG: Found {len(unique_ctypes_in_data)} unique cancer types in test set:")
    for ct in unique_ctypes_in_data[:30]:  # Show first 30
        count = all_cancer_types.count(ct)
        print(f"  '{ct}': {count} patients")

    # Subset to specific cancer types from CANCER_TYPES_TO_INCLUDE
    subset_idx = [i for i, el in enumerate(all_cancer_types) if el in CANCER_TYPES_TO_INCLUDE]
    subset_DMP_IDs = [all_DMP_IDs[i] for i in subset_idx]
    subset_ctypes = [all_cancer_types[i] for i in subset_idx]
    subset_suffix = '_subset'
    print(f"\nSubsetting to {len(CANCER_TYPES_TO_INCLUDE)} cancer types: {CANCER_TYPES_TO_INCLUDE}")
    print(f"Found {len(subset_idx)} patients matching these cancer types")

    # Assign specific colors to selected cancer types
    color_dict = {}
    color_dict['Glioma'] = 'blue'
    color_dict['Myelodysplastic Syndromes'] = 'red'
    color_dict['Germ Cell Tumor'] = 'orange'
    color_dict['Hepatobiliary Cancer'] = 'green'

else:
    # Use all patients, no cancer type coloring
    subset_idx = list(range(len(all_DMP_IDs)))
    subset_DMP_IDs = all_DMP_IDs
    subset_ctypes = all_cancer_types
    subset_suffix = '_all'
    color_dict = {}
    print(f"\nUsing all patients (no cancer type subsetting)")


# Get the subset of embeddings and last timepoint features
z_rep_concat_subset = z_rep_concat[subset_idx]
lasttp_features_subset = all_lasttp_xs_to_predict[0][subset_idx]  # Shape: [n_patients, n_features]

# Print embedding checkpoint
print("\n" + "="*70)
print("EMBEDDING CHECKPOINT")
print("="*70)
print(f"Subset mode: {'Cancer type subset' if SUBSET_BY_CANCER_TYPE else 'All patients'}")
print(f"z_rep_concat_subset.shape: {z_rep_concat_subset.shape}")
print(f"First patient embedding (first 5 dims): {z_rep_concat_subset[0, :5]}")
print(f"Last patient embedding (first 5 dims): {z_rep_concat_subset[-1, :5]}")
print(f"Embedding sum: {z_rep_concat_subset.sum():.6f}")
print(f"Embedding mean: {z_rep_concat_subset.mean():.6f}")
print(f"Embedding std: {z_rep_concat_subset.std():.6f}")
print(f"First 10 patient IDs: {subset_DMP_IDs[:10]}")
print(f"Last 10 patient IDs: {subset_DMP_IDs[-10:]}")
if SUBSET_BY_CANCER_TYPE:
    # Count patients by cancer type in subset
    ctype_counts = {}
    for ct in subset_ctypes:
        ctype_counts[ct] = ctype_counts.get(ct, 0) + 1
    print(f"Cancer type distribution in subset: {ctype_counts}")
print("="*70 + "\n")

# Compute 2D embedding once (t-SNE by default, UMAP optional)
_dr = str(DIM_REDUCTION).lower()
if _dr not in ('tsne', 'umap'):
    raise ValueError(f"DIM_REDUCTION must be 'tsne' or 'umap', got {DIM_REDUCTION!r}")
if _dr == 'umap' and not _UMAP_AVAILABLE:
    raise ImportError(
        "DIM_REDUCTION='umap' but the 'umap-learn' package is not installed. "
        "Install it with `pip install umap-learn`, or set DIM_REDUCTION='tsne'."
    )

if _dr == 'umap':
    print("\nComputing UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=0,
    )
    # Keep the historical variable name so all downstream plotting code is unchanged.
    z_rep_concat_tsne = reducer.fit_transform(z_rep_concat_subset)
    print("UMAP complete")
else:
    print("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, random_state=0)
    z_rep_concat_tsne = tsne.fit_transform(z_rep_concat_subset)
    print("t-SNE complete")

# Method-dependent strings used in axis labels, titles, and output filenames.
DR_LABEL = 'UMAP' if _dr == 'umap' else 't-SNE'   # for axis labels / titles
DR_PREFIX = 'umap' if _dr == 'umap' else 'tsne'   # for output filenames

# Decide once whether to rasterize the scatter layers, based on how many points
# will be drawn. Small clouds stay pure vector (smaller + sharper); large clouds
# are rasterized to cap the PDF size. See RASTER_POINT_THRESHOLD above.
_n_points = int(z_rep_concat_tsne.shape[0])
RASTERIZE_SCATTER = _n_points > RASTER_POINT_THRESHOLD
print(f"Scatter rasterization: {'ON' if RASTERIZE_SCATTER else 'OFF'} "
      f"({_n_points} points vs threshold {RASTER_POINT_THRESHOLD}; "
      f"raster DPI {SCATTER_RASTER_DPI})")

# Create output directory
quick_test_suffix = '_quicktest' if QUICK_TEST else ''
output_dir = f'{sparc_paths.SPARC_MODEL_ROOT}/results_vis_060726/eval_pretrained_embedding_feature_specific_plots_{dataset}{subset_suffix}{quick_test_suffix}'
if modules == 'parsimonious':
    output_dir += '_parsimonious'
if SUBSET_BY_CANCER_TYPE in [True, 'all']:
    output_dir += '_CANCERTYPES'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"\nOutput directory: {output_dir}")


# Get number of features
n_features = lasttp_features_subset.shape[1]

# Verify feature count matches
print(f"\n" + "="*70)
print("FEATURE COUNT VERIFICATION")
print("="*70)
print(f"Expected features (from included_features list): {len(included_features)}")
print(f"Actual features (from data tensor): {n_features}")
if n_features != len(included_features):
    print(f"WARNING: Feature count mismatch!")
    print(f"  Expected: {len(included_features)}")
    print(f"  Actual: {n_features}")
    print(f"  Difference: {n_features - len(included_features)}")
    if n_features < len(included_features):
        print(f"  Using only first {n_features} features from included_features list")
        included_features = included_features[:n_features]
    else:
        print(f"  Adding generic feature names for extra features")
        for i in range(len(included_features), n_features):
            included_features.append(f"feature_{i}")
else:
    print(f"ok Feature counts match!")
print("="*70 + "\n")

# Debug: Check gender distribution
gender_counts = {}
for gender in all_genders:
    gender_counts[gender] = gender_counts.get(gender, 0) + 1
print(f"Gender distribution in dataset: {gender_counts}")
print(f"Shape dict keys: {list(shape_dict.keys())}")

# PLOTTING

if SUBSET_BY_CANCER_TYPE in [True, 'all']:
    # --- Single t-SNE plot colored by cancer type (distinct color per type) ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Determine which cancer types to plot
    if SUBSET_BY_CANCER_TYPE == 'all':
        ctypes_to_plot = sorted(set(subset_ctypes))
    else:  # SUBSET_BY_CANCER_TYPE == True
        ctypes_to_plot = CANCER_TYPES_TO_INCLUDE
        # Auto-assign colors for any missing in color_dict
        _tab10 = plt.cm.get_cmap('tab10').colors
        for _idx, _ctype in enumerate(ctypes_to_plot):
            if _ctype not in color_dict:
                color_dict[_ctype] = _tab10[_idx % len(_tab10)]

    for ctype in ctypes_to_plot:
        ctype_indices = [i for i, ct in enumerate(subset_ctypes) if ct == ctype]
        if not ctype_indices:
            print(f"  No patients found for cancer type: {ctype}")
            continue
        ax.scatter(
            z_rep_concat_tsne[ctype_indices, 0],
            z_rep_concat_tsne[ctype_indices, 1],
            c=[color_dict[ctype]] * len(ctype_indices),
            label=f'{ctype} (n={len(ctype_indices)})',
            alpha=0.8,
            s=30,
            edgecolors='none',
            rasterized=RASTERIZE_SCATTER,  # adaptive: only when point count is large
        )
        print(f"  Plotted {len(ctype_indices)} patients for '{ctype}'")

    ax.legend(title='Cancer Type', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9,
              title_fontsize=11, framealpha=0.8)
    ax.set_xlabel(f'{DR_LABEL} 1', fontsize=14)
    ax.set_ylabel(f'{DR_LABEL} 2', fontsize=14)

    if SUBSET_BY_CANCER_TYPE == 'all':
        title = f'SPARC Embedding ({DR_LABEL})\nAll Cancer Types (n={len(subset_DMP_IDs)})'
    else:
        title = f'SPARC Embedding ({DR_LABEL})\n{", ".join(CANCER_TYPES_TO_INCLUDE)}\n(n={len(subset_DMP_IDs)})'

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    if SUBSET_BY_CANCER_TYPE == 'all':
        save_name = f'{output_dir}/{DR_PREFIX}_all_cancer_types_{dataset}_{modules}_test_set.pdf'
    else:
        _safe_ctypes = '_'.join([ct.replace(' ', '_')[:12] for ct in CANCER_TYPES_TO_INCLUDE])
        save_name = f'{output_dir}/{DR_PREFIX}_cancer_types_{_safe_ctypes}_{dataset}_{modules}_test_set.pdf'

    plt.savefig(save_name, bbox_inches='tight', transparent=True, dpi=SCATTER_RASTER_DPI)
    plt.close(fig)
    print(f"Cancer type plot saved to: {save_name}")

print(f"Creating {n_features} plots, one for each feature...")

# Iterate through each feature
for feature_idx in range(n_features):
    # Create a new figure for this feature
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get feature values for all patients
    feature_values = lasttp_features_subset[:, feature_idx]

    # Create mask for non-NaN values
    valid_mask = ~torch.isnan(feature_values)
    valid_mask_np = valid_mask.cpu().numpy() if torch.is_tensor(valid_mask) else valid_mask

    # Convert feature values to numpy
    feature_values_np = feature_values.cpu().numpy() if torch.is_tensor(feature_values) else feature_values

    # Get feature name
    feature_name = included_features[feature_idx]

    # Set color scale limits based on feature type
    if feature_idx == 1:  # Age feature
        # Use actual data range for age
        if valid_mask_np.sum() > 0:
            valid_feature_values = feature_values_np[valid_mask_np]
            vmin, vmax = valid_feature_values.min(), valid_feature_values.max()

            # Handle case where all values are the same
            if vmin == vmax:
                vmin, vmax = vmin - 0.1, vmax + 0.1
        else:
            vmin, vmax = 0, 1

        # Don't clip age values
        feature_values_to_plot = feature_values_np
        colorbar_label = f'{feature_name}'
        colormap = 'viridis'
    else:
        # For all other features, use -1 to +1 (z-score normalized)
        vmin, vmax = -1, 1

        # Clip values to be within [-1, 1] range for visualization
        feature_values_to_plot = np.clip(feature_values_np, vmin, vmax)
        colorbar_label = f'{feature_name} (z-score)'
        colormap = 'coolwarm'  # Red-Blue diverging colormap

    # Two-pass plotting: NaN values first (background), then valid values (foreground)
    # This ensures colored dots are always visible on top of gray transparent dots

    scatter = None  # Initialize scatter for colorbar

    # FIRST PASS: Plot all NaN values (gray background layer)
    for marker_type in set(shape_dict.values()):
        # Get indices for this marker type
        marker_indices = [
            i for i, el in enumerate(subset_DMP_IDs)
            if shape_dict.get(DMP_ID_to_gender_dict.get(int(el), None)) == marker_type
        ]

        if not marker_indices:
            if feature_idx == 0:  # Only print debug for first feature to avoid spam
                print(f"  Warning: No patients found for marker_type='{marker_type}'")
            continue

        # Separate into valid and NaN groups
        valid_indices = [i for i in marker_indices if valid_mask_np[i]]
        nan_indices = [i for i in marker_indices if not valid_mask_np[i]]

        if feature_idx == 0:  # Debug for first feature only
            print(f"  Marker '{marker_type}': {len(marker_indices)} total, {len(valid_indices)} valid, {len(nan_indices)} NaN")

        # Plot NaN values as transparent background (alpha=0.2 makes them faint)
        if nan_indices:
            ax.scatter(
                z_rep_concat_tsne[nan_indices, 0],
                z_rep_concat_tsne[nan_indices, 1],
                c='gray',
                alpha=0.2,
                marker=marker_type,
                s=20,
                edgecolors='none',
                zorder=1,  # Low z-order for background
                rasterized=RASTERIZE_SCATTER,  # adaptive: only when point count is large
            )

    # SECOND PASS: Plot all valid values with colors (foreground layer)
    for marker_type in set(shape_dict.values()):
        # Get indices for this marker type
        marker_indices = [
            i for i, el in enumerate(subset_DMP_IDs)
            if shape_dict.get(DMP_ID_to_gender_dict.get(int(el), None)) == marker_type
        ]

        if not marker_indices:
            continue

        # Get valid indices only
        valid_indices = [i for i in marker_indices if valid_mask_np[i]]

        # Plot valid values with color based on feature value
        if valid_indices:
            # Debug for first feature
            if feature_idx == 0:
                print(f"    Plotting {len(valid_indices)} points with marker '{marker_type}'")
                print(f"    X range: [{z_rep_concat_tsne[valid_indices, 0].min():.2f}, {z_rep_concat_tsne[valid_indices, 0].max():.2f}]")
                print(f"    Y range: [{z_rep_concat_tsne[valid_indices, 1].min():.2f}, {z_rep_concat_tsne[valid_indices, 1].max():.2f}]")
                print(f"    Feature values range: [{feature_values_to_plot[valid_indices].min():.2f}, {feature_values_to_plot[valid_indices].max():.2f}]")


            valid_indices_arr = np.array(valid_indices)
            valid_feature_vals = feature_values_to_plot[valid_indices_arr]

            scatter = ax.scatter(
                z_rep_concat_tsne[valid_indices_arr, 0],
                z_rep_concat_tsne[valid_indices_arr, 1],
                c=valid_feature_vals,
                cmap=colormap,
                alpha=1,
                marker=marker_type,
                s=30,
                vmin=vmin,
                vmax=vmax,
                edgecolors='none',
                zorder=2,
                rasterized=RASTERIZE_SCATTER,  # adaptive: only when point count is large
            )

            if feature_idx == 0:
                print(f"    Scatter created: {scatter is not None}")

    # Add colorbar
    if valid_mask_np.sum() > 0 and scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(colorbar_label, fontsize=12)

        # Set colorbar ticks at meaningful positions
        if feature_idx != 1:  # Not age
            cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    else:
        if feature_idx == 0:  # Debug for first feature
            print(f"  WARNING: No scatter plot created (scatter={scatter}, valid_mask_sum={valid_mask_np.sum()})")

    # Set axis limits explicitly based on t-SNE coordinates
    # Add 10% padding to the range
    x_min, x_max = z_rep_concat_tsne[:, 0].min(), z_rep_concat_tsne[:, 0].max()
    y_min, y_max = z_rep_concat_tsne[:, 1].min(), z_rep_concat_tsne[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    if feature_idx == 0:
        print(f"  Axis limits set: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")

    # Labels and title with feature name
    ax.set_xlabel(f'{DR_LABEL} 1', fontsize=14)
    ax.set_ylabel(f'{DR_LABEL} 2', fontsize=14)

    # Calculate statistics for title
    if valid_mask_np.sum() > 0:
        valid_values = feature_values_np[valid_mask_np]
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)

        if feature_idx == 1:  # Age
            title_text = f'{feature_name}\n(Valid: {valid_mask_np.sum()}/{len(valid_mask_np)}, μ={mean_val:.1f}, σ={std_val:.1f})'
        else:
            title_text = f'{feature_name}\n(Valid: {valid_mask_np.sum()}/{len(valid_mask_np)}, μ={mean_val:.2f}, σ={std_val:.2f})'
    else:
        title_text = f'{feature_name}\n(Valid: {valid_mask_np.sum()}/{len(valid_mask_np)})'

    ax.set_title(title_text, fontsize=16, fontweight='bold')

    # Legend for gender markers
    handles = []
    for key in shape_dict.keys():
        handles.append(plt.Line2D([0], [0], marker=shape_dict[key], color='w',
                                  label=key, markerfacecolor='gray', markersize=10))
    ax.legend(handles=handles, title='Gender', loc='upper right', fontsize=11, title_fontsize=14)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Save figure with feature name in filename
    safe_feature_name = feature_name.replace('/', '_').replace(' ', '_')
    save_name = f'{output_dir}/{DR_PREFIX}_{safe_feature_name}_{feature_idx:03d}_{dataset}_{modules}_test_set.pdf'
    plt.savefig(save_name, bbox_inches='tight', transparent=True, dpi=SCATTER_RASTER_DPI)

    if feature_idx == 0:
        print(f"  First plot saved to: {save_name}")
        # Check if any artists were added to the axes
        print(f"  Number of collections in axes: {len(ax.collections)}")
        print(f"  Axes has data: {ax.has_data()}")

    plt.close(fig)  # Close the specific figure to free memory

    if (feature_idx + 1) % 10 == 0:
        print(f"Processed {feature_idx + 1}/{n_features} features...")

print(f"All {n_features} plots saved to {output_dir}")
print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"Dataset: {dataset}")
print(f"Total unique patients in test set: {len(all_DMP_IDs)}")
if SUBSET_BY_CANCER_TYPE:
    print(f"Patients visualized (subset to {len(CANCER_TYPES_TO_INCLUDE)} cancer types): {len(subset_DMP_IDs)}")
    ctype_counts = {}
    for ct in subset_ctypes:
        ctype_counts[ct] = ctype_counts.get(ct, 0) + 1
    print(f"Cancer type distribution: {ctype_counts}")
else:
    print(f"Patients visualized (all patients): {len(subset_DMP_IDs)}")
print(f"Output directory: {output_dir}")
print("="*70)