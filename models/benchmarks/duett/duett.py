"""
DuETT benchmark -- standalone LightningModule.

Wraps DuETTEncoder (duett_model.py), which implements the architecture from
Labach et al., "DuETT: Dual Event Time Transformer for Electronic Health
Records" (MLHC 2023): per-variable tokens, continuous value embeddings, and
alternating event-axis / time-axis attention with a learned representation
token.

This module is self-contained -- it does not import or subclass the SPARC
classes. It implements the same interface the other benchmarks expose
(lasttp_MLP, rnn, raindrop) so the training and evaluation scripts can use it
interchangeably:

    __init__(modular_featuredim_dict, option=..., max_time_len=..., lr=...,
             weight_decay=..., num_encoder_layers=..., dropout=...,
             side_effects_num_infusion=...)
    data_prep(batch) -> xs, y, src_key_padding_mask, num_infusion, DMP_ID
    forward(src, src_key_padding_mask=..., num_infusion=...)
    training_step / validation_step / test_step / configure_optimizers

It logs 'val_loss', which is what the trainer's ModelCheckpoint and
EarlyStopping monitor.

Supported `option` values:
    binary, binary_icu_pre   -> BCE on a single label
    side_effects             -> BCE, metrics stratified by infusion number
    side_effects_risk        -> Cox partial likelihood, c-index + next-infusion
                                AUROC stratified by infusion number

data_prep builds the feature row the encoder expects:

    [ values (static | per-module mean,count | time) | missingness | padding ]

which is 2*(2*ts + nts + 1) + 1 wide -- the same width the loader's duett path
produces, so no changes to data_loaders_inf_risk_fixed.py are needed.
"""

# Put the repo root, data/ and models/ on sys.path, from any working directory.
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy, Precision, Recall, AveragePrecision
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex

from duett_model import DuETTEncoder


class DuETT(pl.LightningModule):

    def __init__(self, modular_featuredim_dict, max_time_len=20, d_model=512,
                 nhead=4, num_encoder_layers=4, dim_feedforward=None, dropout=0.1,
                 option='binary', lr=1e-4, weight_decay=1e-5, missing_mask=True,
                 side_effects_num_infusion=False, test_metrics_by_infusion_group=True,
                 d_embedding=None, **kwargs):
        super().__init__()
        if option not in ('binary', 'binary_icu_pre', 'side_effects', 'side_effects_risk'):
            raise ValueError(f"DuETT does not support option={option!r}")
        self.option = option
        self.lr = lr
        self.weight_decay = weight_decay
        self.modular_featuredim_dict = modular_featuredim_dict
        self.side_effects_num_infusion = side_effects_num_infusion
        # Accepted for parity with the other benchmarks, which use it to gate
        # per-infusion-group logging. DuETT logs only the aggregate metrics, so
        # there is nothing here for it to switch off.
        self.test_metrics_by_infusion_group = test_metrics_by_infusion_group

        # The paper's token width. d_model is kept in the signature because the
        # other benchmarks take it, but the transformer runs at d_embedding: with
        # one token per (bin, variable) there are ~370x more tokens than the flat
        # encoder sees, so a 512-wide token is both slower and larger than the
        # architecture it replaces. 64 measured faster than the SPARC variant
        # (112 ms vs 127 ms/batch, 7.5 GiB vs 18.8 GiB at batch 512).
        d_embedding = d_embedding or 64
        # nhead must divide d_embedding
        while d_embedding % nhead != 0 and nhead > 1:
            nhead -= 1

        self.encoder = DuETTEncoder(
            modular_featuredim_dict, max_time_len=max_time_len,
            d_embedding=d_embedding, n_layers=max(1, num_encoder_layers // 2),
            nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            missing_mask=missing_mask)
        self.total_feature_dim = self.encoder.total_feature_dim
        self.missing_mask = missing_mask
        print(f"[DuETT] total_feature_dim {self.total_feature_dim} "
              f"({self.encoder.n_ts_vars} variables, d_embedding {d_embedding}, "
              f"{len(self.encoder.event_layers)} dual-axis layers)")

        self.head = nn.Linear(d_embedding, 1)
        if side_effects_num_infusion:
            self.num_infusion_MLP = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
            self.combine_MLP = nn.Linear(2, 1)

        # losses / metrics
        self.bce = nn.BCELoss()
        self.auroc = AUROC(task='binary')
        self.auprc = AveragePrecision(task='binary')
        self.accuracy = Accuracy(task='binary')
        self.precision_fn = Precision(task='binary')
        self.recall_fn = Recall(task='binary')
        if 'risk' in option:
            self.c_index = ConcordanceIndex()

    # ------------------------------------------------------------------ data
    def data_prep(self, data):
        """Unpack a batch and build the encoder's input row."""
        if self.side_effects_num_infusion:
            (xs, times, num_infusion, DMP_ID), y = data
        else:
            (xs, times), y = data
            num_infusion, DMP_ID = None, None

        if 'risk' not in self.option and self.option != 'side_effects':
            y = torch.as_tensor(y).float().unsqueeze(-1)

        xs = torch.cat([xs, times.unsqueeze(-1)], dim=-1)
        mask = torch.isnan(xs)
        src_key_padding_mask = mask[:, :, -1]            # (batch, time) True where padded

        if self.missing_mask:
            indicators = (~mask).float()
            xs = xs.clone()
            xs[mask] = 0.0
            xs = torch.cat([xs, indicators], dim=-1)
        else:
            xs = xs.clone()
            xs[mask] = -10.0

        padding_indicator = (~src_key_padding_mask).float().unsqueeze(-1)
        xs = torch.cat([xs, padding_indicator], dim=-1)
        return xs, y, src_key_padding_mask, num_infusion, DMP_ID

    # --------------------------------------------------------------- forward
    def forward(self, src, src_key_padding_mask=None, num_infusion=None):
        z = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        z = self.head(z)
        if self.side_effects_num_infusion and num_infusion is not None:
            n = self.num_infusion_MLP(num_infusion.unsqueeze(-1).float())
            z = self.combine_MLP(torch.cat([z, n], dim=-1))
        z = torch.clamp(z, min=-10.0, max=10.0)
        if 'risk' not in self.option:
            z = torch.sigmoid(z)
        return z

    # --------------------------------------------------- per-infusion metrics
    def _metrics_by_infusion(self, y_pred, y_event, time_to_event, num_infusion):
        """Weighted c-index and next-infusion AUROC, stratified by infusion no.

        Mirrors the aggregation the other benchmarks use: metrics are computed
        within each infusion-number group and averaged weighted by group size,
        skipping groups where the metric is undefined.
        """
        out = {}
        risk = y_pred.squeeze(-1) if y_pred.dim() > 1 else y_pred
        ev = torch.as_tensor(y_event, device=risk.device).float()
        tt = torch.as_tensor(time_to_event, device=risk.device).float()
        ni = torch.as_tensor(num_infusion, device=risk.device)
        next_event = ((ev == 1) & (tt == 1)).float()

        c_rows, a_rows = [], []
        for g in torch.unique(ni):
            m = ni == g
            if m.sum() < 2:
                continue
            if 'risk' in self.option and len(torch.unique(ev[m])) > 1:
                try:
                    c_rows.append((self.c_index(risk[m], ev[m].bool(), tt[m]).item(),
                                   int(m.sum())))
                except Exception:
                    pass
            if len(torch.unique(next_event[m])) > 1:
                try:
                    a_rows.append((self.auroc(risk[m], next_event[m].int()).item(),
                                   int(m.sum()), m))
                except Exception:
                    pass

        if c_rows:
            tot = sum(n for _, n in c_rows)
            out['c_index_weighted'] = sum(v * n / tot for v, n in c_rows)
        if a_rows:
            tot = sum(n for _, n, _ in a_rows)
            out['auroc_next_infusion_weighted'] = sum(v * n / tot for v, n, _ in a_rows)
            allm = torch.zeros_like(ni, dtype=torch.bool)
            for _, _, m in a_rows:
                allm |= m
            try:
                out['auroc_next_infusion_overall'] = self.auroc(
                    risk[allm], next_event[allm].int()).item()
            except Exception:
                pass
        if 'risk' in self.option and len(torch.unique(ev)) > 1:
            try:
                out['c_index_overall'] = self.c_index(risk, ev.bool(), tt).item()
            except Exception:
                pass
        return out

    # ----------------------------------------------------------------- steps
    def _shared_step(self, batch, stage):
        xs, y, pad, num_infusion, _ = self.data_prep(batch)
        y_pred = self(xs, src_key_padding_mask=pad, num_infusion=num_infusion)

        if self.option in ('binary', 'binary_icu_pre'):
            loss = self.bce(y_pred, y)
            if stage != 'train':
                yi = y.int()
                self.log(f'{stage}_auroc', self.auroc(y_pred, yi), prog_bar=True)
                self.log(f'{stage}_auprc', self.auprc(y_pred, yi))
                self.log(f'{stage}_accuracy', self.accuracy(y_pred, yi))
                self.log(f'{stage}_precision', self.precision_fn(y_pred, yi))
                self.log(f'{stage}_recall', self.recall_fn(y_pred, yi))

        elif self.option == 'side_effects':
            y_t = torch.as_tensor(y, device=y_pred.device).float().view_as(y_pred)
            loss = self.bce(y_pred, y_t)
            if stage != 'train':
                tt = torch.ones_like(y_t.squeeze(-1))
                m = self._metrics_by_infusion(y_pred, y_t.squeeze(-1), tt, num_infusion)
                for k, v in m.items():
                    self.log(f'{stage}_{k}', v, prog_bar=(k.endswith('weighted')))

        else:  # side_effects_risk
            y_event = torch.as_tensor(y[0], device=y_pred.device)
            tt = torch.as_tensor(y[1], device=y_pred.device).float()
            loss = cox.neg_partial_log_likelihood(y_pred, y_event, tt, ties_method='efron')
            if stage != 'train':
                m = self._metrics_by_infusion(y_pred, y_event, tt, num_infusion)
                for k, v in m.items():
                    self.log(f'{stage}_{k}', v, prog_bar=(k.endswith('weighted')))

        self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    def predict_step(self, batch, batch_idx):
        xs, y, pad, num_infusion, DMP_ID = self.data_prep(batch)
        return self(xs, src_key_padding_mask=pad, num_infusion=num_infusion), y, DMP_ID

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)
