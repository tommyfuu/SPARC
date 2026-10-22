"""
Create 1:1:1 train/val/test splits for each INSIGHT site (A-F), stratified by medication group
(liver-toxicity immunotherapy > chemotherapy > other), for site-specific and pooled fine-tuning.
Existing split JSONs are overwritten.
Usage: python create_splits.py
"""
import os
import json
import pickle
import random

SITES = ['A', 'B', 'C', 'D', 'E', 'F']
import config
DATA_ROOT  = config.OUTPUT_DIR
SEED  = 42

# Exact uppercase names as used by the training scripts (must match entry[0] in meds dict)
LIVER_MEDS = {
    "IPILIMUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "PEMBROLIZUMAB",
    "AVELUMAB", "CEMIPLIMAB", "DOSTARLIMAB",
}
CHEMO_MEDS = {
    # febrile_neutropenia / anemia / neutropenia / AKI (combined for stratification)
    "DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN",
    "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
    "CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE", "BENDAMUSTINE",
    "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL",
    "IFOSFAMIDE", "PEMETREXED", "METHOTREXATE", "PAMIDRONATE", "ZOLEDRONIC ACID",
}


def load_med_groups(site):
    """Return (all_pids, liver_pids, chemo_pids) for a site."""
    labs_path = os.path.join(DATA_ROOT, site, 'ssid_to_date_to_labs_dict.pickle')
    meds_path = os.path.join(DATA_ROOT, site, 'ssid_to_date_to_nonoral_meds_dict.pickle')
    with open(labs_path, 'rb') as f:
        labs = pickle.load(f)
    with open(meds_path, 'rb') as f:
        meds = pickle.load(f)

    all_pids   = sorted(labs.keys())
    lab_set    = set(all_pids)
    liver_pids = set()
    chemo_pids = set()

    for pid, dates_dict in meds.items():
        if pid not in lab_set:
            continue
        for entries in dates_dict.values():
            for entry in entries:
                if not isinstance(entry[0], str):
                    continue
                med = entry[0].strip().upper()
                if med in LIVER_MEDS:
                    liver_pids.add(pid)
                if med in CHEMO_MEDS:
                    chemo_pids.add(pid)

    return all_pids, liver_pids, chemo_pids


def stratified_split(all_pids, liver_pids, chemo_pids, seed):
    """Split each medication group into equal thirds, then merge."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    assigned = set()

    for group_set in [liver_pids, chemo_pids]:
        group = sorted(group_set - assigned)
        rng.shuffle(group)
        assigned.update(group)
        n     = len(group)
        n_per = n // 3
        train.extend(group[:n_per])
        val.extend(group[n_per: 2 * n_per])
        test.extend(group[2 * n_per:])

    rest = sorted(set(all_pids) - assigned)
    rng.shuffle(rest)
    n     = len(rest)
    n_per = n // 3
    train.extend(rest[:n_per])
    val.extend(rest[n_per: 2 * n_per])
    test.extend(rest[2 * n_per:])

    return train, val, test


def setup_dir(target_root, per_site_root):
    """Create target_root/split/{train,val,test}/ and target_root/{site}/ symlinks to per_site_root/{site}/."""
    os.makedirs(target_root, exist_ok=True)
    for sp in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_root, 'split', sp), exist_ok=True)
    for site in SITES:
        link = os.path.join(target_root, site)
        target = os.path.join(per_site_root, site)
        if os.path.islink(link):
            os.unlink(link)
        if not os.path.exists(link):
            os.symlink(target, link)


def main():
    strat_root  = os.path.join(DATA_ROOT, 'insight_stratified')
    pooled_root = os.path.join(DATA_ROOT, 'insight_pooled_stratified')

    setup_dir(strat_root,  DATA_ROOT)
    setup_dir(pooled_root, DATA_ROOT)

    for site in SITES:
        print(f"\nProcessing site {site} ...")
        all_pids, liver_pids, chemo_pids = load_med_groups(site)
        train, val, test = stratified_split(all_pids, liver_pids, chemo_pids, SEED)

        def _count(split_ids, med_set):
            return sum(1 for p in split_ids if p in med_set)

        print(f"  Total {len(all_pids):,} | train {len(train):,}  val {len(val):,}  test {len(test):,}")
        print(f"  Liver_tox  total={len(liver_pids):4d}  "
              f"train={_count(train, liver_pids):3d}  "
              f"val={_count(val,   liver_pids):3d}  "
              f"test={_count(test,  liver_pids):3d}")
        print(f"  Chemo      total={len(chemo_pids):4d}  "
              f"train={_count(train, chemo_pids):4d}  "
              f"val={_count(val,   chemo_pids):4d}  "
              f"test={_count(test,  chemo_pids):4d}")

        for root in [strat_root, pooled_root]:
            for sp, ids in [('train', train), ('val', val), ('test', test)]:
                out = os.path.join(root, 'split', sp, f'{site}.json')
                with open(out, 'w') as f:
                    json.dump(ids, f)

        print(f"  Splits written to insight_stratified/ and insight_pooled_stratified/")


if __name__ == '__main__':
    main()
