"""End-to-end runs of the training script on the synthetic cohort, on CPU."""

import glob
import math

import pandas as pd
import pytest
import torch

from sparc_clinical import evaluate, train

COMMON = dict(dataset='SYN_CAIA', modules_cat='parsimonious', valfold=1, batch_size=16, epochs=1,
              accelerator='cpu', use_module_tokenizer=True)


def _state_dict(run):
    ckpt = glob.glob(f'{run}/fold0/epoch=*.ckpt')
    assert len(ckpt) == 1, ckpt
    return torch.load(ckpt[0], map_location='cpu', weights_only=False)['state_dict']


@pytest.fixture(scope='module')
def pretrained(data_root, tmp_path_factory):
    run = str(tmp_path_factory.mktemp('runs') / 'pretrain')
    with pytest.MonkeyPatch.context() as mp:   # inside a 2-task allocation, like an OnDemand session
        mp.setenv('SLURM_NTASKS', '2')
        mp.setenv('SLURM_JOB_NAME', 'sys/dashboard/sys/vscode')
        train(volume_path=str(data_root), scheme='pretrain', option='pretrain', output_model_path=run,
              **COMMON)
    return run


def test_pretrain(pretrained):
    state = _state_dict(pretrained)
    assert any(k.startswith('per_module_tokenizer') for k in state)          # -umt reaches training
    assert not any(v.is_floating_point() and torch.isnan(v).any() for v in state.values())


@pytest.mark.parametrize('option, extra, label_columns', [
    ('binary', {}, ['label']),
    ('neutropenia_risk', dict(med_of_choice='CISPLATIN', add_num_infusion=True, loss_option='hybrid'),
     ['infusion_num', 'event', 'time_to_event']),
])
def test_finetune_and_eval(data_root, pretrained, tmp_path, option, extra, label_columns):
    run = str(tmp_path / option)
    flags = dict(volume_path=str(data_root), option=option, output_model_path=run, **COMMON, **extra)
    train(scheme='finetune', pretrained_model_path=pretrained, **flags)
    before = pd.read_csv(f'{run}/fold0/test_predictions.csv')

    evaluate(**flags)
    after = pd.read_csv(f'{run}/fold0/test_predictions.csv')

    assert list(after.columns) == ['patient_id', 'prediction_day', *label_columns, 'score']
    assert len(after) > 0 and after['score'].between(0, 1).all()
    pd.testing.assert_frame_equal(before, after)   # eval reloads the same weights
    assert not math.isnan(after['score'].sum())


def test_eval_needs_trained_fold(data_root, pretrained, tmp_path):
    """eval must not fall back to the pretrained model when the fold was never finetuned."""
    with pytest.raises(FileNotFoundError, match='No trained checkpoint'):
        evaluate(volume_path=str(data_root), option='binary', pretrained_model_path=pretrained,
                 output_model_path=str(tmp_path / 'never_trained'), **COMMON)


def test_batch_first_scores_do_not_depend_on_batch():
    import io
    from contextlib import redirect_stdout

    from MSK_featurename_list import MSK_numfeature_dict
    from SPARC_inf_risk_modular import SPARC

    dims = {'basic_lab': MSK_numfeature_dict['basic_lab'], 'cbc_differential': 5,
            'basic_vitals': MSK_numfeature_dict['basic_vitals'], 'gender': 1, 'age': 1}
    torch.manual_seed(0)
    with redirect_stdout(io.StringIO()):
        model = SPARC(dims, option='binary', temporal_aggregation='attention', batch_first=True,
                      d_model=32, nhead=2, num_encoder_layers=1, dim_feedforward=64).eval()
    src = torch.randn(4, 6, model.total_feature_dim)
    padded = torch.zeros(6, 4, dtype=torch.bool)                                 # (time, batch)
    padded[3:, 1] = True
    with torch.no_grad():
        together = model(src, src_key_padding_mask=padded)
        alone = torch.cat([model(src[i:i + 1], src_key_padding_mask=padded[:, i:i + 1]) for i in range(4)])
    torch.testing.assert_close(together, alone)

