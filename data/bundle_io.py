"""
bundle_io.py

Read and write the patient bundle that load_dataset() uses for any dataset whose
name contains 'CAIA'. The bundle is one dict of per-patient dicts, keyed by
integer patient id, with every date given as an integer day offset. It can be
stored as

    DMP_ID_to_dicts_final.json     the original format
    DMP_ID_to_dicts_final.pkl      the same dict, pickled (faster to load)
    patients.csv, labs.csv, ...    tidy tables (CSV or Parquet), see docs/data.md

read_bundle() accepts any of these and returns the same dict.
"""

import hashlib
import itertools
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from MSK_featurename_list import ALL_INCLUDED_TESTS, VITALS  # noqa: E402

BUNDLE_NAME = 'DMP_ID_to_dicts_final'

LABS = 'DMP_ID_to_DATES_to_TESTSVALUES_DICT'
VITALS_KEY = 'DMP_ID_to_DATES_to_VITALSVALUES_DICT'
GENDER = 'DMP_ID_to_GENDER_dict_oh'
AGE = 'DMP_ID_to_firstlabdateagebinned_dict'
DEATH = 'DMP_ID_to_death_days_dict'
LAST_CONTACT = 'DMP_ID_to_last_contact_date_dict_delta'
INPATIENT = 'DMP_ID_to_inpatient_dates_dict_delta'
ICU = 'DMP_ID_to_icu_dates_dict_delta'
FOLD = 'DMP_ID_to_fold_num_dict'
MEDS = 'DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_delta'
CUTOFF = 'DMP_ID_to_cutoff_day_dict'   # only read for dataset == 'MSK_CAIA'

REQUIRED_KEYS = [LABS, VITALS_KEY, GENDER, AGE, DEATH, LAST_CONTACT, INPATIENT, ICU, FOLD, MEDS]

# Required columns of each table (docs/data.md lists the optional ones). Lab and vital tables
# also take one column per test (wide) or 'test' and 'value' columns (long).
REQUIRED_COLUMNS = {
    'patients':    ['patient_id', 'sex', 'age'],
    'labs':        ['patient_id', 'day'],
    'vitals':      ['patient_id', 'day'],
    'medications': ['patient_id', 'day', 'medication'],
    'admissions':  ['patient_id', 'admit_day', 'discharge_day'],
}
# Whole-number columns of each table: (must be filled in, may be empty)
INTEGER_COLUMNS = {
    'patients':    ([], ['fold', 'death_day', 'last_contact_day', 'cutoff_day']),
    'labs':        (['day'], []),
    'vitals':      (['day'], []),
    'medications': (['day'], ['end_day']),
    'admissions':  (['admit_day', 'discharge_day'], []),
}
N_FOLDS = 11   # folds 0-9 are cross-validation folds, fold 10 is the held-out test set
# SPARC_dep package/v1 names: its 'static' table has 'gender' and 'enrollment_date'
V1_COLUMNS = {'gender': 'sex', 'enrollment_date': 'index_date'}

# Usual median of a test in the unit SPARC expects. A median outside this range almost always
# means another unit, which silently shifts task labels (e.g. neutropenia is Neut < 1.0).
TYPICAL_MEDIAN = {
    'HGB': (8.5, 17, 'g/dL'), 'Platelets': (50, 600, '10^3/uL'), 'WBC': (1, 30, '10^3/uL'),
    'Neut': (0.5, 20, '10^3/uL, not %'), 'Lymph': (0.1, 8, '10^3/uL, not %'),
    'Creatinine': (0.3, 5, 'mg/dL'), 'Glucose': (60, 300, 'mg/dL'),
    'Calcium': (6, 13, 'mg/dL'), 'Albumin': (1.5, 6, 'g/dL'), 'Bilirubin, Total': (0.1, 5, 'mg/dL'),
    'Temperature C': (34, 40, 'degrees C'),
}

FILE_FORMATS = {'.json': 'json', '.pkl': 'pickle', '.pickle': 'pickle'}
_BUNDLE_FILES = ([BUNDLE_NAME + ext for ext in FILE_FORMATS]
                 + [f'{name}.{ext}' for name in REQUIRED_COLUMNS for ext in ('csv', 'parquet')])


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def _decode_sets(d):
    if '__set__' in d:
        return set(d['__set__'])
    return d


def read_bundle(path):
    """Load a bundle from a .json/.pkl/.pickle file, or from a directory holding
    DMP_ID_to_dicts_final.{json,pkl,pickle} or CSV/Parquet tables."""
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    if os.path.isdir(path):
        for ext in FILE_FORMATS:
            candidate = os.path.join(path, BUNDLE_NAME + ext)
            if os.path.exists(candidate):
                return read_bundle(candidate)
        if _find_table(path, 'patients') or _find_table(path, 'static'):
            return tables_to_bundle(**read_tables(path))
        raise FileNotFoundError(
            f"No bundle in {path}: expected {BUNDLE_NAME}.json, {BUNDLE_NAME}.pkl, "
            f"or patients.csv/.parquet with labs.csv/.parquet")
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f, object_hook=_decode_sets)
    if path.endswith(('.pkl', '.pickle')):
        with open(path, 'rb') as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported bundle file {path}: use .json, .pkl or .pickle, "
                     f"or the folder that holds the tables")


def write_bundle(bundle, path, fmt=None):
    """Write a bundle. fmt is 'json', 'pickle', 'csv' or 'parquet'. A path ending in .json,
    .pkl or .pickle is written as that file. Any other path is a directory, which gets
    DMP_ID_to_dicts_final.json or .pkl, or one file per table (csv/parquet).
    Returns the path written.
    """
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.csv', '.parquet'):
        raise ValueError(f"{path}: CSV and Parquet bundles are folders of tables; "
                         f"give a folder and the format {ext[1:]!r}")
    if ext in FILE_FORMATS:
        if fmt not in (None, FILE_FORMATS[ext]):
            raise ValueError(f"{path} ends in {ext}, which does not match the format {fmt!r}")
        fmt = FILE_FORMATS[ext]
        if os.path.basename(path) in _BUNDLE_FILES:
            _refuse_other_formats(os.path.dirname(path) or '.', [os.path.basename(path)])
    fmt = fmt or 'json'
    if fmt not in ('json', 'pickle', 'csv', 'parquet'):
        raise ValueError(f"Unknown format {fmt!r}: use json, pickle, csv or parquet")

    if ext not in FILE_FORMATS:
        if fmt in ('csv', 'parquet'):
            files = [f'{name}.{fmt}' for name in REQUIRED_COLUMNS]
        else:
            files = [BUNDLE_NAME + ('.json' if fmt == 'json' else '.pkl')]
        os.makedirs(path, exist_ok=True)
        _refuse_other_formats(path, files)
        if fmt in ('csv', 'parquet'):
            for name, df in bundle_to_tables(bundle).items():
                out = os.path.join(path, f'{name}.{fmt}')
                df.to_csv(out, index=False) if fmt == 'csv' else df.to_parquet(out, index=False)
            return path
        path = os.path.join(path, files[0])

    if fmt == 'json':
        with open(path, 'w') as f:
            json.dump(_jsonable(bundle), f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _refuse_other_formats(folder, files):
    # read_bundle uses one format per folder, so a second one would shadow or be shadowed
    others = [f for f in _BUNDLE_FILES if f not in files and os.path.exists(os.path.join(folder, f))]
    if others:
        raise FileExistsError(f"{folder} already holds {others}; remove them or write to another folder")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(int(k)) if isinstance(k, (float, np.floating)) and float(k).is_integer() else str(k):
                _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return {'__set__': [_jsonable(v) for v in obj]}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    return obj


# ---------------------------------------------------------------------------
# Tables <-> bundle
# ---------------------------------------------------------------------------

def _find_table(directory, name):
    for ext in ('.csv', '.parquet'):
        candidate = os.path.join(directory, name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def read_tables(directory):
    """Read patients/labs/vitals/medications/admissions (.csv or .parquet) from a directory."""
    tables = {}
    for name in REQUIRED_COLUMNS:
        path = _find_table(directory, name)
        if name == 'patients':
            path = path or _find_table(directory, 'static')   # SPARC_dep package/v1 name
        if path:
            tables[name] = (pd.read_csv(path, float_precision='round_trip') if path.endswith('.csv')
                            else pd.read_parquet(path))
    for name in ('patients', 'labs'):
        if name not in tables:
            raise FileNotFoundError(f"{directory} has no {name}.csv or {name}.parquet")
    if _find_table(directory, 'labels'):
        warnings.warn("the labels table is not used: SPARC derives outcomes from death_day, "
                      "last_contact_day, admissions and medications")
    return tables


def tables_to_bundle(patients, labs, vitals=None, medications=None, admissions=None):
    """Build a bundle from tidy DataFrames; the columns are listed in docs/data.md."""
    patients = patients.rename(columns={old: new for old, new in V1_COLUMNS.items()
                                        if old in patients.columns and new not in patients.columns})
    given = {name: df for name, df in [('patients', patients), ('labs', labs), ('vitals', vitals),
                                        ('medications', medications), ('admissions', admissions)]
             if df is not None}
    for name, df in given.items():
        missing = [c for c in REQUIRED_COLUMNS[name]
                   if c not in df.columns and c.replace('day', 'date') not in df.columns]
        if missing:
            raise ValueError(f"{name} table is missing columns {missing}")

    patients = given['patients'].copy()
    patients['patient_id'] = _integer_ids(patients['patient_id'], 'patients')
    if patients['patient_id'].duplicated().any():
        raise ValueError("patients table has duplicate patient_id values")
    index_date = (_dates(patients.set_index('patient_id')['index_date'])
                  if 'index_date' in patients.columns else None)
    tables = {}
    for name, df in given.items():
        df = df.assign(patient_id=_integer_ids(df['patient_id'], name))
        unknown = ~df['patient_id'].isin(patients['patient_id'])
        if unknown.any():
            warnings.warn(f"ignoring {unknown.sum()} {name} rows of patients not in the patients table")
            df = df[~unknown]
        tables[name] = _integer_columns(_dates_to_days(df, name, index_date), name)

    patients = tables['patients'].set_index('patient_id')
    age = pd.to_numeric(patients['age'], errors='coerce')
    for col, values in [('sex', patients['sex']), ('age', age)]:
        bad = values.index[values.isna()]
        if len(bad):
            raise ValueError(f"patients.{col} is empty or not valid for {len(bad)} patients "
                             f"(e.g. patient {bad[0]}: {patients.loc[bad, col].tolist()[0]!r})")
    if 'fold' not in patients.columns or patients['fold'].isna().all():
        patients['fold'] = [_default_fold(pid) for pid in patients.index]
    elif patients['fold'].isna().any():
        raise ValueError("patients.fold is empty for some patients: fill it for all or leave it out")

    bundle = {
        LABS: _measurements(tables['labs'], ALL_INCLUDED_TESTS, 'labs'),
        VITALS_KEY: _measurements(tables['vitals'], VITALS, 'vitals') if 'vitals' in tables else {},
        GENDER: {int(pid): [_female(s)] for pid, s in patients['sex'].items()},
        AGE: {int(pid): int(a // 5) for pid, a in age.items()},
        DEATH: _column(patients, 'death_day'),
        LAST_CONTACT: _column(patients, 'last_contact_day'),
        FOLD: _column(patients, 'fold'),
        INPATIENT: {},
        ICU: {},
        MEDS: {},
    }
    if 'cutoff_day' in patients.columns:
        bundle[CUTOFF] = _column(patients, 'cutoff_day')

    if 'admissions' in tables:
        adm = tables['admissions']
        is_icu = (_flags(adm['icu'], 'admissions.icu') if 'icu' in adm.columns
                  else pd.Series(False, index=adm.index))
        for key, rows in [(INPATIENT, adm[~is_icu]), (ICU, adm[is_icu])]:
            for pid, a, d in zip(rows['patient_id'], rows['admit_day'], rows['discharge_day']):
                bundle[key].setdefault(int(pid), []).append([int(a), int(d)])

    if 'medications' in tables:
        meds = tables['medications']
        names = meds['medication'].astype('string').str.strip().str.upper()
        if names.isna().any() or (names == '').any():
            raise ValueError("medications has rows without a medication name")
        end = meds['end_day'].fillna(meds['day']) if 'end_day' in meds.columns else meds['day']
        for pid, day, name, e in zip(meds['patient_id'], meds['day'], names, end):
            bundle[MEDS].setdefault(int(pid), {}).setdefault(int(day), []).append([name, int(e)])
    return bundle


def bundle_to_tables(bundle):
    """Inverse of tables_to_bundle. Ages are written as the lower bound of their 5-year bin.
    Patients and measurements keep the bundle's order, so the tables load back to an equal bundle."""
    def ids(key):
        return {int(k): v for k, v in bundle.get(key, {}).items()}

    dropped = sorted(set(bundle) - set(REQUIRED_KEYS) - {CUTOFF})
    if dropped:
        warnings.warn(f"these keys have no table and are not written: {dropped}")
    gender, age = ids(GENDER), ids(AGE)
    pids = list(dict.fromkeys([*ids(FOLD), *ids(LABS), *gender, *age]))   # fold order sets the splits' order
    patients = pd.DataFrame({'patient_id': pids})
    patients['sex'] = [('F' if list(gender[p])[0] == 1 else 'M') if p in gender else None for p in pids]
    patients['age'] = pd.array([age[p] * 5 if p in age else None for p in pids], dtype='Int64')
    for col, key in [('fold', FOLD), ('death_day', DEATH), ('last_contact_day', LAST_CONTACT),
                     ('cutoff_day', CUTOFF)]:
        if key in bundle:
            values = ids(key)
            patients[col] = pd.array([values.get(p) for p in pids], dtype='Int64')

    return {
        'patients': patients,
        'labs': _measurement_table(ids(LABS), ALL_INCLUDED_TESTS),
        'vitals': _measurement_table(ids(VITALS_KEY), VITALS),
        'medications': pd.DataFrame(
            [(pid, int(day), m[0], int(m[1])) for pid, days in ids(MEDS).items()
             for day, entries in days.items() for m in entries],
            columns=['patient_id', 'day', 'medication', 'end_day']),
        'admissions': pd.DataFrame(
            [(pid, int(a), int(d), icu) for key, icu in [(INPATIENT, 0), (ICU, 1)]
             for pid, stays in ids(key).items() for a, d in stays],
            columns=['patient_id', 'admit_day', 'discharge_day', 'icu']),
    }


def _integer_ids(ids, table):
    numeric = pd.to_numeric(ids, errors='coerce')
    bad = numeric.isna() | (numeric % 1 != 0) | (numeric.abs() >= 2**63)
    if bad.any():
        raise ValueError(f"{table}.patient_id must be integers (found {ids[bad].tolist()[0]!r}); "
                         f"map ids to integers first")
    return numeric.astype('int64')


def _integer_columns(df, table):
    """Parse INTEGER_COLUMNS as numbers; raise on fractions, text, or blanks where required."""
    required, optional = INTEGER_COLUMNS[table]
    for col in required + [c for c in optional if c in df.columns]:
        values = pd.to_numeric(df[col], errors='coerce')
        bad = (values.notna() & (values % 1 != 0)) | (values.isna() & df[col].notna())
        if col in required:
            bad |= values.isna()
        if bad.any():
            raise ValueError(f"{table}.{col} must be whole numbers{'' if col in required else ' or empty'} "
                             f"(found {df[col][bad].tolist()[0]!r})")
        df[col] = values
    return df


def _default_fold(pid):
    """Fold of a patient without one: fixed by its id, so adding patients moves no one."""
    return int(hashlib.sha256(str(pid).encode()).hexdigest(), 16) % N_FOLDS


def _flags(values, name):
    """0/1, True/False or yes/no (any case) -> bool; empty -> False."""
    text = values.map(lambda v: '' if pd.isna(v) else str(v).strip().lower())
    text = text.replace({'1.0': '1', '0.0': '0'})
    unknown = sorted(set(text) - {'', '0', '1', 'true', 'false', 'yes', 'no'})
    if unknown:
        raise ValueError(f"{name} must be 0/1, true/false or yes/no, not {unknown[:3]}")
    return text.isin(['1', 'true', 'yes'])


def _dates_to_days(df, table, index_date):
    """Replace any '<x>date' column with '<x>day' = calendar days since patients.index_date."""
    df = df.copy()
    for col in [c for c in df.columns if c.endswith('date') and c != 'index_date']:
        if index_date is None:
            raise ValueError(f"{table}.{col} is a date: add index_date to patients, "
                             f"or give {col[:-4]}day instead")
        dates = _dates(df[col])
        anchor = df['patient_id'].map(index_date)
        if (dates.notna() & anchor.isna()).any():
            raise ValueError(f"{table}.{col} has dates for patients without an index_date")
        df[col[:-4] + 'day'] = (dates - anchor).dt.days
        df = df.drop(columns=col)
    return df


def _dates(values):
    """ISO 8601 dates (YYYY-MM-DD, optionally with a time) -> calendar dates."""
    return pd.to_datetime(values, format='ISO8601', utc=True).dt.tz_localize(None).dt.normalize()


def _measurements(df, names, table):
    """Wide (one column per test) or long (test, value) table -> {pid: {day: [values]}}."""
    long_format = {'test', 'value'} <= set(df.columns)
    if long_format:
        if df['test'].isna().any():
            raise ValueError(f"{table} has rows without a test name")
        visits = pd.MultiIndex.from_frame(df[['patient_id', 'day']].drop_duplicates())
        df = (df.assign(value=pd.to_numeric(df['value'], errors='coerce'))
                .groupby(['patient_id', 'day', 'test'])['value'].first()
                .unstack('test').reindex(visits).reset_index())
        df.columns.name = None
    unknown = [c for c in df.columns if c not in ('patient_id', 'day') and c not in names]
    if unknown:
        raise ValueError(f"{table} has unknown columns {unknown}; valid names: {names}")
    tests = [c for c in df.columns if c in names]
    df[tests] = df[tests].apply(pd.to_numeric, errors='coerce')   # e.g. '<0.5' -> missing, as in load_dataset
    # one row per (patient, day), in the table's order; the first non-missing value wins
    df = df.groupby(['patient_id', 'day'], sort=False).first().reindex(columns=names)
    values = df.to_numpy(dtype=float)
    out = {}
    for (pid, day), row in zip(df.index, values):
        out.setdefault(int(pid), {})[int(day)] = [None if np.isnan(v) else float(v) for v in row]
    return out


def _measurement_table(data, names):
    rows = [[pid, int(day)] + [np.nan if v is None else v for v in vals]
            for pid, days in data.items() for day, vals in days.items()]
    df = pd.DataFrame(rows, columns=['patient_id', 'day'] + list(names))
    df[list(names)] = df[list(names)].apply(pd.to_numeric, errors='coerce')
    return df[['patient_id', 'day'] + [c for c in names if df[c].notna().any()]]


def _female(sex):
    return int(str(sex).strip().upper() in ('F', 'FEMALE', '1', '1.0', 'TRUE'))


def _column(df, col):
    if col not in df.columns:
        return {}
    return {int(pid): int(v) for pid, v in df[col].items() if pd.notna(v)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_bundle(bundle):
    """Return a list of problems (empty if the bundle can be loaded). Also warns about
    missing last-contact days, an empty test fold and unusual units."""
    problems = [f"missing key {k}" for k in REQUIRED_KEYS if k not in bundle]
    if problems:
        return problems

    def check(key, ok, what):
        def fails(v):
            try:
                return not ok(v)
            except Exception:
                return True
        bad = [pid for pid, v in bundle[key].items() if fails(v)]
        if bad:
            problems.append(f"{key}: {len(bad)} patients {what} (e.g. {bad[0]})")

    for key, width in [(LABS, len(ALL_INCLUDED_TESTS)), (VITALS_KEY, len(VITALS))]:
        check(key, lambda days, w=width: isinstance(days, dict) and all(len(v) == w for v in days.values()),
              f"do not map day -> list of {width} values")
    check(GENDER, lambda v: list(v) in ([0], [1]), "have gender other than [0] or [1]")
    check(FOLD, lambda v: v in range(N_FOLDS), f"have a fold outside 0-{N_FOLDS - 1}")
    for key in (INPATIENT, ICU):
        check(key, lambda stays: len(stays) > 0 and all(len(s) == 2 and s[0] <= s[1] for s in stays),
              "have an empty stay list or a stay with admit > discharge")
    check(MEDS, lambda days: all(len(m) >= 2 and isinstance(m[0], str) for ms in days.values() for m in ms),
          "have medication entries that are not [NAME, end_day]")

    try:
        for key in REQUIRED_KEYS:
            [int(k) for k in bundle[key]]
        for key in (LABS, VITALS_KEY, MEDS):
            [int(d) for days in bundle[key].values() for d in days]
    except (TypeError, ValueError) as e:
        problems.append(f"patient ids and days must be integers ({e})")
        return problems

    lab_pids = {int(k) for k in bundle[LABS]}
    for key, what in [(GENDER, 'gender'), (AGE, 'age'), (FOLD, 'fold')]:
        missing = lab_pids - {int(k) for k in bundle[key]}
        if missing:
            problems.append(f"{len(missing)} patients with labs have no {what} (e.g. {min(missing)})")
    if problems:
        return problems
    no_contact = lab_pids - {int(k) for k in bundle[LAST_CONTACT]}
    if no_contact:
        warnings.warn(f"{len(no_contact)} patients have no last-contact day; "
                      f"binary and side-effect tasks need it")
    if N_FOLDS - 1 not in set(bundle[FOLD].values()):
        warnings.warn(f"no patient is in fold {N_FOLDS - 1}, so the test split is empty")
    for message in check_units(bundle):
        warnings.warn(message)
    return problems


def check_units(bundle, n_visits=100_000):
    """Messages for tests whose median over the first n_visits visits is unusual in the
    expected unit (see TYPICAL_MEDIAN)."""
    messages = []
    for key, names in [(LABS, ALL_INCLUDED_TESTS), (VITALS_KEY, VITALS)]:
        wanted = [n for n in names if n in TYPICAL_MEDIAN]
        visits = itertools.islice((v for days in bundle[key].values() for v in days.values()), n_visits)
        rows = [[v[names.index(n)] for n in wanted] for v in visits]
        values = pd.DataFrame(rows, columns=wanted, dtype=object).apply(pd.to_numeric, errors='coerce')
        for name, median in values.median().dropna().items():
            low, high, unit = TYPICAL_MEDIAN[name]
            if not low <= median <= high:
                messages.append(f"{name}: median {median:.3g} is unusual for {unit} "
                                f"(usually {low}-{high}); check the units")
    return messages
