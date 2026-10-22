import copy
import math

import pandas as pd
import pytest
from bundle_io import AGE, FOLD, GENDER, ICU, INPATIENT, LABS, LAST_CONTACT, MEDS, VITALS_KEY, check_units
from MSK_featurename_list import VITALS

from sparc_clinical import (
    bundle_to_tables,
    load_dataset,
    read_bundle,
    tables_to_bundle,
    validate_bundle,
    write_bundle,
)


def canonical(bundle):
    """Integer keys everywhere and lists instead of tuples, so bundles compare equal."""
    def conv(v):
        if isinstance(v, dict):
            return {int(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return {key: conv(value) for key, value in bundle.items()}


@pytest.mark.parametrize('fmt', ['json', 'pickle', 'csv', 'parquet'])
def test_roundtrip(bundle, tmp_path, fmt):
    path = write_bundle(bundle, str(tmp_path / 'X_CAIA'), fmt)
    assert canonical(read_bundle(path)) == canonical(bundle)


def test_tables_long_format_and_dates():
    patients = pd.DataFrame({'patient_id': [1, 2], 'sex': ['F', 'male'], 'age': [62, 47],
                             'index_date': ['2020-01-01', '2020-02-01'],
                             'last_contact_date': ['2021-01-01', '2021-02-01']})
    labs = pd.DataFrame({'patient_id': [1, 1, 1, 2],
                         'date': ['2020-01-01', '2020-01-01', '2020-01-11', '2020-02-03'],
                         'test': ['HGB', 'WBC', 'HGB', 'Creatinine'], 'value': [12.5, 6.1, '<0.5', 1.1]})
    meds = pd.DataFrame({'patient_id': [1], 'date': ['2020-01-05'], 'medication': ['cisplatin']})
    admissions = pd.DataFrame({'patient_id': [2, 2], 'admit_day': [1, 3], 'discharge_day': [4, 4],
                               'icu': [0, 1]})
    b = tables_to_bundle(patients, labs, medications=meds, admissions=admissions)

    from MSK_featurename_list import ALL_INCLUDED_TESTS
    hgb, wbc = ALL_INCLUDED_TESTS.index('HGB'), ALL_INCLUDED_TESTS.index('WBC')
    assert sorted(b[LABS][1]) == [0, 10]
    assert b[LABS][1][0][hgb] == 12.5 and b[LABS][1][0][wbc] == 6.1
    assert b[LABS][1][10][hgb] is None                        # '<0.5' is not a number
    assert b[GENDER] == {1: [1], 2: [0]} and b[AGE] == {1: 12, 2: 9}
    assert b[MEDS] == {1: {4: [['CISPLATIN', 4]]}}
    assert b[LAST_CONTACT] == {1: 366, 2: 366}
    assert b[INPATIENT] == {2: [[1, 4]]} and b[ICU] == {2: [[3, 4]]}
    assert set(b[FOLD].values()) <= set(range(11)) and b[VITALS_KEY] == {}


def _string_ids(patients, labs):
    return patients.assign(patient_id=['P1', 'P2']), labs


def _unknown_lab(patients, labs):
    return patients, labs.assign(Hemoglobin=1.0)


def _no_sex(patients, labs):
    return patients.drop(columns='sex'), labs


def _text_age(patients, labs):
    return patients.assign(age=['90+', 60]), labs


def _half_day(patients, labs):
    return patients, labs.assign(day=[0, 0.5])


@pytest.mark.parametrize('change, message', [
    (_string_ids, 'must be integers'),
    (_unknown_lab, 'unknown columns'),
    (_no_sex, 'missing columns'),
    (_text_age, 'patients.age is empty or not valid'),
    (_half_day, 'labs.day must be whole numbers'),
])
def test_table_errors(change, message):
    patients = pd.DataFrame({'patient_id': [1, 2], 'sex': ['F', 'M'], 'age': [50, 60]})
    labs = pd.DataFrame({'patient_id': [1, 2], 'day': [0, 0], 'HGB': [12.0, 13.0]})
    with pytest.raises(ValueError, match=message):
        tables_to_bundle(*change(patients, labs))


def test_dates_must_be_iso():
    patients = pd.DataFrame({'patient_id': [1], 'sex': ['F'], 'age': [50], 'index_date': [20200101]})
    labs = pd.DataFrame({'patient_id': [1], 'date': [20200301], 'HGB': [12.0]})
    assert list(tables_to_bundle(patients, labs)[LABS][1]) == [60]        # YYYYMMDD numbers work
    with pytest.raises(ValueError, match='ISO8601'):
        tables_to_bundle(patients.assign(index_date=['01/02/2020']), labs)


def test_unknown_patients_and_calendar_days():
    patients = pd.DataFrame({'patient_id': [1], 'sex': ['F'], 'age': [50],
                             'index_date': ['2020-01-01 18:00']})
    labs = pd.DataFrame({'patient_id': [1, 2], 'date': ['2020-01-02 08:00', '2020-01-02'],
                         'HGB': [12.0, 11.0]})
    with pytest.warns(UserWarning, match='ignoring 1 labs rows'):
        b = tables_to_bundle(patients, labs)
    assert list(b[LABS]) == [1] and list(b[LABS][1]) == [1]      # the next calendar day, not 14 h later


def test_check_units(bundle):
    assert check_units(bundle) == []
    fahrenheit = copy.deepcopy(bundle)
    t = VITALS.index('Temperature C')
    for days in fahrenheit[VITALS_KEY].values():
        for v in days.values():
            v[t] = v[t] * 9 / 5 + 32
    assert [m.split(':')[0] for m in check_units(fahrenheit)] == ['Temperature C']


def test_write_bundle_formats(bundle, tmp_path):
    write_bundle(bundle, str(tmp_path / 'X_CAIA'))
    with pytest.raises(FileExistsError, match='already holds'):
        write_bundle(bundle, str(tmp_path / 'X_CAIA'), 'csv')       # would be shadowed by the JSON
    with pytest.raises(FileExistsError, match='already holds'):
        write_bundle(bundle, str(tmp_path / 'X_CAIA' / 'DMP_ID_to_dicts_final.pkl'))
    with pytest.raises(ValueError, match='folders of tables'):
        write_bundle(bundle, str(tmp_path / 'out.csv'))
    with pytest.raises(ValueError, match='does not match'):
        write_bundle(bundle, str(tmp_path / 'x.pkl'), 'json')


def test_validate(bundle):
    assert validate_bundle(bundle) == []
    broken = dict(bundle)
    broken[LABS] = {**bundle[LABS], 1000: {0: [1.0, 2.0]}}
    assert any('list of 46 values' in p for p in validate_bundle(broken))
    assert validate_bundle({k: v for k, v in bundle.items() if k != MEDS}) == [f'missing key {MEDS}']


def test_load_dataset_same_for_every_format(bundle, tmp_path):
    """JSON, pickle and CSV bundles give identical load_dataset output."""
    results = {}
    for fmt in ['json', 'pickle', 'csv']:
        write_bundle(bundle, str(tmp_path / f'{fmt.upper()}_CAIA'), fmt)
        data, dims, split, *_ = load_dataset(f'{fmt.upper()}_CAIA', str(tmp_path), 'binary',
                                             'parsimonious', 0, 'train')
        results[fmt] = (data, dims, split)
    ref_data, ref_dims, ref_split = results['json']
    for fmt in ['pickle', 'csv']:
        data, dims, split = results[fmt]
        assert dims == ref_dims and split == ref_split
        for module in ['basic_lab', 'cbc_differential', 'basic_vitals', 'gender', 'age']:
            assert _same(data[module], ref_data[module]), (fmt, module)


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_tables_are_tidy(bundle):
    tables = bundle_to_tables(bundle)
    assert list(tables['patients'].columns[:3]) == ['patient_id', 'sex', 'age']
    assert {'patient_id', 'day', 'HGB', 'Neut'} <= set(tables['labs'].columns)
    assert list(tables['medications'].columns) == ['patient_id', 'day', 'medication', 'end_day']
