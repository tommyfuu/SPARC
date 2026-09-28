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

