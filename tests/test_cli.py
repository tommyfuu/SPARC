import os

from sparc_clinical.cli import _split_config, config_to_args, main


def test_config_to_args():
    config = {'dataset': 'X_CAIA', 'add_num_infusion': True, 'med_of_choice': ['CISPLATIN', 'CARBOPLATIN'],
              'epochs': 3, 'pretrained_model_path': None}
    assert config_to_args(config) == ['--dataset', 'X_CAIA', '--add_num_infusion', 'True',
                                      '--med_of_choice', 'CISPLATIN,CARBOPLATIN', '--epochs', '3']


def test_split_config():
    assert _split_config(['-f', '2', '--config', 'a.yaml', '-e', '1']) == ('a.yaml', ['-f', '2', '-e', '1'])
    assert _split_config(['--config=a.yaml']) == ('a.yaml', [])
    assert _split_config(['-f', '2']) == (None, ['-f', '2'])


def test_synthetic_convert_validate(tmp_path, capsys):
    src, dst = str(tmp_path / 'A_CAIA'), str(tmp_path / 'B_CAIA')
    assert main(['synthetic', src, '--patients', '30']) == 0
    os.makedirs(dst)
    stale = os.path.join(dst, 'normalized_data_dict_lab_dict_B_CAIA_parsimonious.pickle')
    open(stale, 'w').close()
    assert main(['convert', src, dst, '--format', 'csv']) == 0
    assert not os.path.exists(stale)
    tables = ['admissions.csv', 'labs.csv', 'medications.csv', 'patients.csv', 'vitals.csv']
    assert sorted(os.listdir(dst)) == tables
    assert main(['validate', dst]) == 0
    assert 'patients with labs  30' in capsys.readouterr().out


def test_validate_reports_problems(tmp_path, capsys):
    import pandas as pd
    d = tmp_path / 'C_CAIA'
    d.mkdir()
    patients = pd.DataFrame({'patient_id': [1, 2], 'sex': ['F', 'M'], 'age': [50, '90+']})
    patients.to_csv(d / 'patients.csv', index=False)
    labs = pd.DataFrame({'patient_id': [1, 2], 'day': [0, 0], 'HGB': [12.0, 11.0]})
    labs.to_csv(d / 'labs.csv', index=False)
    assert main(['validate', str(d)]) == 1
    assert main(['validate', str(tmp_path / 'missing')]) == 1
    assert main(['train', '--config', str(tmp_path / 'missing.yaml')]) == 1
    out = capsys.readouterr().out
    assert 'ERROR: ' + str(tmp_path / 'missing') + ' does not exist' in out
    assert "ERROR: patients.age is empty or not valid for 1 patients (e.g. patient 2: '90+')" in out


def test_import_is_light():
    import subprocess
    import sys
    code = 'import sparc_clinical, sys; sparc_clinical.read_bundle; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)
