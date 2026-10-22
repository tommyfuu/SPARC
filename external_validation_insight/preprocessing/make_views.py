#!/usr/bin/env python3
"""After run_preprocessing_split.py: build <out>/overall/ (all sites merged) and <out>-fullsite/, symlinks to the same
data without split/, so zero-shot evaluation scores every patient.
Usage: python make_views.py [--output-dir <INSIGHT_OUTPUT_DIR>]
"""
import os, pickle, argparse
import config

SITES = list("ABCDEF")
FILES = ["ssid_to_date_to_labs_dict.pickle", "ssid_to_hospital_adm_dsch_dict.pickle",
         "ssid_to_date_to_nonoral_meds_dict.pickle", "ssid_to_date_to_STAGE_dict.pickle",
         "ssid_to_date_to_vitals_dict.pickle", "ssid_to_demographics_dict.pickle",
         "ssid_to_date_to_oral_meds_dict.pickle"]   # oral meds: optional, the loader does not read it

ap = argparse.ArgumentParser()
ap.add_argument("--output-dir", default=config.OUTPUT_DIR)
args = ap.parse_args()
out = os.path.normpath(args.output_dir)

print(f"[1/2] merging sites into {out}/overall")
os.makedirs(os.path.join(out, "overall"), exist_ok=True)
for f in FILES:
    parts = [os.path.join(out, s, f) for s in SITES if os.path.exists(os.path.join(out, s, f))]
    dst = os.path.join(out, "overall", f)
    if not parts:
        print(f"  - {f}: no per-site files, skipped")
        continue
    merged = {}
    for p in parts:
        with open(p, "rb") as fh:
            merged.update(pickle.load(fh))
    with open(dst + ".partial", "wb") as fh:
        pickle.dump(merged, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(dst + ".partial", dst)
    print(f"  + {f}: {len(merged):,} patients from {len(parts)} site(s)")
    del merged

full = out + "-fullsite"
print(f"[2/2] full-site view {full}")
os.makedirs(full, exist_ok=True)
for s in SITES + ["overall"]:
    link, target = os.path.join(full, s), os.path.join(out, s)
    if os.path.isdir(target) and not os.path.lexists(link):
        os.symlink(target, link)
        print(f"  + {s} -> {target}")
with open(os.path.join(full, "README.md"), "w") as fh:
    fh.write("Full-site view: symlinks to the per-site data WITHOUT a split/ directory, so dataset_loading puts every\n"
             "patient in the test set (zero-shot evaluation). Use the non-fullsite tree for the held-out test third.\n")
print("done")
