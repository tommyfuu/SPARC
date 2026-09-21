
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
with open(sparc_paths.FROZEN_DATA_DIR + '/postbc_DMP_ID_to_DATES_to_TESTSVALUES_DICT_deidentified.pickle', 'rb') as handle:
    DMP_ID_to_DATES_to_TESTSVALUES_DICT = pickle.load(handle)


# randomly split into train test val sets
import random
random.seed(0)
DMP_IDs = list(DMP_ID_to_DATES_to_TESTSVALUES_DICT.keys())
random.shuffle(DMP_IDs)
numDMP_IDs = len(DMP_IDs)
# split into 10 folds; i_th fold for testing, i-1_th fold for validation (if i=0, then i-1=9)
fold_dict = {}
for i in range(11):
    fold_dict[i] = DMP_IDs[int(i/11*numDMP_IDs):int((i+1)/11*numDMP_IDs)]

with open(sparc_paths.CTFM_DIR + '/data/fold_dict_11.pickle', 'wb') as handle:
    pickle.dump(fold_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open(f'{sparc_paths.DATA_ROOT}/MIMICIV_hosp/MIMICIV_subjectid_to_date_to_basic_labs_dict_080625.pkl', 'rb') as handle:
    DMP_ID_to_DATES_to_TESTSVALUES_DICT = pickle.load(handle)

# randomly split into train test val sets (1v1v1)
random.seed(0)
DMP_IDs = list(DMP_ID_to_DATES_to_TESTSVALUES_DICT.keys())
random.shuffle(DMP_IDs)
numDMP_IDs = len(DMP_IDs)
# split into 3 equal length sets
trainDMP_IDs = DMP_IDs[:int(numDMP_IDs/3)]
valDMP_IDs = DMP_IDs[int(numDMP_IDs/3):int(2*numDMP_IDs/3)]
testDMP_IDs = DMP_IDs[int(2*numDMP_IDs/3):]
# save the split
split_dict = {'train': trainDMP_IDs, 'val': valDMP_IDs, 'test': testDMP_IDs}

with open(sparc_paths.DATA_ROOT + '/MIMICIV_hosp/split_dict_1v1v1.pickle', 'wb') as handle:
    pickle.dump(split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
