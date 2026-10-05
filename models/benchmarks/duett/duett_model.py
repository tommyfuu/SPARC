"""
DuETT encoder (Labach et al., MLHC 2023; github.com/layer6ai-labs/duett): per-variable tokens,
continuous value embeddings and alternating event/time attention. The LightningModule is in duett.py;
split_features() recovers the [static | per-module means, counts | time] column layout built by data_prep.
"""

import numpy as np
import torch
import torch.nn as nn


# order in which the loader writes the time-series modules (fetch_data_given_dates in
# data/data_loaders_inf_risk_fixed.py); the column layout follows this, not modular_featuredim_dict
TS_MODULES = ('basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals',
              'extra_vitals')


class BatchNormLastDim(nn.Module):
    """BatchNorm1d over the last dimension of an (..., d) tensor."""

    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        shape = x.shape
        return self.bn(x.reshape(-1, shape[-1])).reshape(shape)


class CVE(nn.Module):
    """Continuous value embedding: Linear(1 -> sqrt(d)) -> Tanh -> [BatchNorm] -> Linear(sqrt(d) -> d)."""

    def __init__(self, d_embedding, batch_norm=True):
        super().__init__()
        d_hidden = max(1, int(np.sqrt(d_embedding)))
        layers = [nn.Linear(1, d_hidden), nn.Tanh()]
        if batch_norm:
            layers.append(BatchNormLastDim(d_hidden))
        layers.append(nn.Linear(d_hidden, d_embedding))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.unsqueeze(-1))


def variable_layout(modular_featuredim_dict, missing_mask=True):
    """Column ranges for the flat row data_prep produces: (n_static, ts_slices, n_values, total_width),
        with one (mean_slice, count_slice) pair per time-series feature."""
    nts_keys = [k for k in modular_featuredim_dict if k not in TS_MODULES]
    ts_keys = [k for k in TS_MODULES if k in modular_featuredim_dict]
    n_static = sum(modular_featuredim_dict[k] for k in nts_keys)

    ts_slices, cursor = [], n_static
    for k in ts_keys:
        f = modular_featuredim_dict[k]
        for j in range(f):
            ts_slices.append((cursor + j, cursor + f + j))
        cursor += 2 * f

    n_values = cursor + 1                       # + the time column
    total = 2 * n_values + 1 if missing_mask else n_values + 1
    return n_static, ts_slices, n_values, total


class DuETTEncoder(nn.Module):
    """Dual event/time transformer over per-variable tokens."""

    def __init__(self, modular_featuredim_dict, max_time_len=20, d_embedding=64,
                 n_layers=2, nhead=4, dim_feedforward=None, dropout=0.1,
                 missing_mask=True, batch_norm=True):
        super().__init__()
        # Scale the FFN with the token width. Per-variable tokens mean ~370x more
        # tokens than the flat encoder, so a fixed wide FFN dominates runtime.
        dim_feedforward = dim_feedforward or 4 * d_embedding
        self.modular_featuredim_dict = modular_featuredim_dict
        self.max_time_len = max_time_len
        self.d_embedding = d_embedding
        self.missing_mask = missing_mask

        (self.n_static, self.ts_slices,
         self.n_values, self.total_feature_dim) = variable_layout(
            modular_featuredim_dict, missing_mask)

        self.n_ts_vars = len(self.ts_slices)
        # one token per time-series variable, plus one pooled static token,
        # plus the learned representation token
        self.n_tokens = self.n_ts_vars + 1 + 1

        # embeddings
        self.value_cve = CVE(d_embedding, batch_norm)
        self.count_cve = CVE(d_embedding, batch_norm)
        self.time_cve = CVE(d_embedding, batch_norm)
        # a learned vector per variable, so the event transformer can tell
        # variables apart (the paper's variable/type embedding)
        self.variable_embedding = nn.Embedding(self.n_tokens, d_embedding)
        self.static_projection = (
            nn.Linear(self.n_static, d_embedding) if self.n_static > 0 else None)
        self.rep_token = nn.Parameter(torch.randn(d_embedding) * 0.02)

        # dual-axis transformer stacks
        def _encoder_layer():
            return nn.TransformerEncoderLayer(
                d_model=d_embedding, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, norm_first=True, batch_first=True)

        self.event_layers = nn.ModuleList([_encoder_layer() for _ in range(n_layers)])
        self.time_layers = nn.ModuleList([_encoder_layer() for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_embedding)

    def split_features(self, src):
        """(B, T, total) -> values (B, T, n_values), indicators, padding."""
        values = src[..., :self.n_values]
        if self.missing_mask:
            indicators = src[..., self.n_values:2 * self.n_values]
        else:
            indicators = torch.ones_like(values)
        padding = src[..., -1]
        return values, indicators, padding

    def build_tokens(self, src):
        """(B, T, total) -> tokens (B, T, n_tokens, d), time embedding (B, T, d)."""
        values, indicators, _ = self.split_features(src)
        B, T, _ = values.shape

        mean_idx = torch.tensor([m for m, _ in self.ts_slices], device=src.device)
        count_idx = torch.tensor([c for _, c in self.ts_slices], device=src.device)

        means = values.index_select(-1, mean_idx)      # (B, T, V)
        counts = values.index_select(-1, count_idx)    # (B, T, V)
        observed = indicators.index_select(-1, mean_idx)

        # value + count embeddings, zeroed where nothing was observed so an
        # imputed 0 is not mistaken for a measured 0
        tok = self.value_cve(means) + self.count_cve(counts)
        tok = tok * observed.unsqueeze(-1)             # (B, T, V, d)

        # static features collapse to a single token, constant over time
        extra = []
        if self.static_projection is not None:
            static = values[..., :self.n_static]
            extra.append(self.static_projection(static).unsqueeze(2))
        else:
            extra.append(torch.zeros(B, T, 1, self.d_embedding, device=src.device))

        rep = self.rep_token.view(1, 1, 1, -1).expand(B, T, 1, -1)
        extra.append(rep)

        tokens = torch.cat([tok] + extra, dim=2)       # (B, T, n_tokens, d)
        tokens = tokens + self.variable_embedding.weight.view(1, 1, self.n_tokens, -1)

        time_col = values[..., -1]                     # (B, T)
        return tokens, self.time_cve(time_col)

    def forward(self, src, src_key_padding_mask=None):
        """src (B, T, total_feature_dim) -> representation (B, d_embedding)."""
        tokens, time_embedding = self.build_tokens(src)
        B, T, V, d = tokens.shape

        for event_layer, time_layer in zip(self.event_layers, self.time_layers):
            # event axis: attend across variables, independently per time bin
            z = tokens.reshape(B * T, V, d)
            z = event_layer(z).reshape(B, T, V, d)

            # time axis: attend across time bins, independently per variable,
            # with the bin-time embedding added in
            z = z + time_embedding.unsqueeze(2)
            z = z.permute(0, 2, 1, 3).reshape(B * V, T, d)
            mask = None
            if src_key_padding_mask is not None:
                mask = src_key_padding_mask.repeat_interleave(V, dim=0)
            z = time_layer(z, src_key_padding_mask=mask)
            tokens = z.reshape(B, V, T, d).permute(0, 2, 1, 3)

        tokens = self.norm(tokens)
        # read out the representation token, averaged over unpadded bins
        rep = tokens[:, :, -1, :]                       # (B, T, d)
        if src_key_padding_mask is not None:
            keep = (~src_key_padding_mask).float().unsqueeze(-1)
            return (rep * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return rep.mean(1)
