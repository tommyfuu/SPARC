import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import AUROC, Accuracy, Precision, Recall, F1Score, AveragePrecision
import numpy as np
from MSK_featurename_list import *
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex
import math
import statsmodels.api as sm
import os
import pickle


class WeightedBCELoss(nn.Module):
    """BCELoss with per-sample pos_weight to handle class imbalance."""
    def __init__(self, pos_weight=1.0, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, input, target):
        weight = torch.where(
            target.bool(),
            input.new_full(input.shape, self.pos_weight),
            input.new_ones(input.shape),
        )
        return F.binary_cross_entropy(input, target, weight=weight, reduction=self.reduction)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


class ContinuousTimeEncoding(nn.Module):
    """Learned sin/cos embedding of elapsed time, added to each visit token."""
    def __init__(self, d_model, n_freqs=None, learnable=True):
        super(ContinuousTimeEncoding, self).__init__()
        if n_freqs is None:
            n_freqs = d_model // 2
        self.n_freqs = n_freqs
        self.d_model = d_model
        # log-spaced init like sinusoidal PE, but learnable
        init_freq = torch.exp(
            torch.arange(0, n_freqs, dtype=torch.float) * (-math.log(10000.0) / max(n_freqs, 1))
        )
        if learnable:
            self.freq = nn.Parameter(init_freq)
            self.phase = nn.Parameter(torch.zeros(n_freqs))
        else:
            self.register_buffer('freq', init_freq)
            self.register_buffer('phase', torch.zeros(n_freqs))
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, t):
        """t: (B, T) times, NaNs already filled -> (B, T, d_model)."""
        t = torch.nan_to_num(t, nan=0.0).unsqueeze(-1)          # (B, T, 1)
        angles = t * self.freq + self.phase                      # (B, T, n_freqs)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,T,2F)
        return self.proj(feats)                                  # (B, T, d_model)


class PerModuleTokenizer(nn.Module):
    """Embeds each clinical module separately (gated by its observed fraction) and sums
    them with the static and time embeddings into one token per visit."""
    def __init__(self, module_index_cfg, d_model, dropout=0.1,
                 use_time_encoding=True, n_static=0, time_as_raw_feature=False):
        super(PerModuleTokenizer, self).__init__()
        self.d_model = d_model
        self.modules_cfg = module_index_cfg['modules']  # list of (name, [idx])
        self.static_start = module_index_cfg.get('static_start', None)
        self.n_static = n_static
        self.use_time_encoding = use_time_encoding
        # optional linear embedding of the raw elapsed time
        self.time_as_raw_feature = time_as_raw_feature

        # per-module projection over [values | missingness] -> d_model
        self.module_names = []
        self.module_indices = []          # also registered as buffers below
        self.module_projections = nn.ModuleDict()
        for i, (name, idxs) in enumerate(self.modules_cfg):
            self.module_names.append(name)
            self.register_buffer(f'mod_idx_{i}', torch.tensor(idxs, dtype=torch.long))
            self.module_indices.append(idxs)
            self.module_projections[name] = nn.Sequential(
                nn.Linear(len(idxs) * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # static features (gender, age, ctype one-hot, ...) -> one token
        if n_static > 0:
            self.static_projection = nn.Sequential(
                nn.Linear(n_static * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.static_projection = None

        if use_time_encoding:
            self.time_encoding = ContinuousTimeEncoding(d_model)
        else:
            self.time_encoding = None
        self.raw_time_projection = nn.Linear(1, d_model) if time_as_raw_feature else None

        # module-type embedding so the encoder knows which module a token came from
        self.module_type_embedding = nn.Embedding(len(self.modules_cfg) + 1, d_model)

    def forward(self, values, missingness, times):
        """values, missingness: (B, T, n_base), NaNs filled with 0, 1=observed;
        times: (B, T). Returns (B, T, d_model)."""
        B, T, _ = values.shape
        token = torch.zeros(B, T, self.d_model, device=values.device, dtype=values.dtype)

        for i, name in enumerate(self.module_names):
            idx = getattr(self, f'mod_idx_{i}')
            v = values.index_select(-1, idx)         # (B, T, n_mod)
            m = missingness.index_select(-1, idx)    # (B, T, n_mod)
            mod_in = torch.cat([v, m], dim=-1)        # (B, T, 2*n_mod)
            mod_tok = self.module_projections[name](mod_in)   # (B, T, d_model)
            # gate = fraction of this module's features observed at this visit
            gate = m.mean(dim=-1, keepdim=True)               # (B, T, 1) in [0,1]
            mod_tok = mod_tok * gate
            type_emb = self.module_type_embedding(
                torch.tensor(i, device=values.device)
            )
            mod_tok = mod_tok + type_emb
            token = token + mod_tok

        if self.static_projection is not None and self.static_start is not None:
            s_idx = torch.arange(self.static_start, self.static_start + self.n_static,
                                 device=values.device)
            sv = values.index_select(-1, s_idx)
            sm = missingness.index_select(-1, s_idx)
            s_in = torch.cat([sv, sm], dim=-1)
            token = token + self.static_projection(s_in)

        if self.time_encoding is not None:
            token = token + self.time_encoding(times)
        if self.raw_time_projection is not None:
            t = torch.nan_to_num(times, nan=0.0).unsqueeze(-1)   # (B, T, 1)
            token = token + self.raw_time_projection(t)

        return token


class MissingnessAwareAttentionAggregation(nn.Module):
    """Gated attention pooling over time, combined with the masked mean and std."""
    def __init__(self, d_model, reduction_ratio=4, max_time_len=20):
        super(MissingnessAwareAttentionAggregation, self).__init__()

        # LayerNorm on the scores for sequences longer than 20
        if max_time_len > 20:
            self.attention_scorer = nn.Sequential(
                nn.Linear(d_model, d_model // reduction_ratio),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(d_model // reduction_ratio, 1),
                nn.LayerNorm(1)
            )
        else:
            self.attention_scorer = nn.Sequential(
                nn.Linear(d_model, d_model // reduction_ratio),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(d_model // reduction_ratio, 1)
            )

        # per-timestep reliability score in [0, 1]
        self.quality_gate = nn.Sequential(
            nn.Linear(d_model, d_model // reduction_ratio),
            nn.ReLU(),
            nn.Linear(d_model // reduction_ratio, 1),
            nn.Sigmoid()
        )

        # [attended | mean | std] -> d_model
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, z, mask):
        """z: (B, T, d_model); mask: (B, T), True = padded. Returns (B, d_model)."""
        batch_size, seq_len, d_model = z.shape

        attn_scores = self.attention_scorer(z)

        quality_scores = self.quality_gate(z)

        # down-weight low-quality timesteps
        modulated_scores = attn_scores * quality_scores

        mask_expanded = mask.unsqueeze(-1)  # (batch_size, max_time_len, 1)
        modulated_scores = modulated_scores.masked_fill(mask_expanded, float('-inf'))

        attn_weights = F.softmax(modulated_scores, dim=1)

        z_attended = (z * attn_weights).sum(dim=1)

        # masked mean / std over valid steps
        z_masked = z.masked_fill(mask_expanded.expand_as(z), 0)
        valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1)  # (batch_size, 1)

        z_mean = z_masked.sum(dim=1) / valid_counts  # (batch_size, d_model)

        z_centered = z - z_mean.unsqueeze(1)
        z_centered_masked = z_centered.masked_fill(mask_expanded.expand_as(z), 0)
        z_var = (z_centered_masked ** 2).sum(dim=1) / valid_counts
        z_std = torch.sqrt(z_var + 1e-8)  # (batch_size, d_model)

        z_combined = torch.cat([z_attended, z_mean, z_std], dim=-1)  # (batch_size, 3*d_model)

        z_aggregated = self.output_projection(z_combined)  # (batch_size, d_model)

        return z_aggregated


def nanstd(x):
    return torch.sqrt(torch.nanmean((x - torch.nanmean(x))**2))


class SPARC(pl.LightningModule):
    def __init__(self,
                # task/data-specific
                 modular_featuredim_dict, option='pretrain',
                # model/data choices and manipulations
                 advanced_model='default', usage='standalone', perturbation_index='', normalization=False, med_as_feature=False, meds_l = [],
                # hyperparameters
                 max_time_len=20, weight_decay=1e-6, lr=1e-4,
                 d_model=512, nhead=8, num_encoder_layers=8, dim_feedforward=2048, dropout=0.1, schedule='default',
                # pretrain stuff
                 pretrain_contrastive_weight=0,
                 finetune_part=False, time_calibration=False, primary_ctype_returned=False,
                # time encoding
                 time_encoding = 'explicit',
                # temporal aggregation
                 temporal_aggregation='attention',  # only 'attention' is supported
                # relevant for shapley - avoid using sigmoid
                 use_sigmoid=True, side_effects_num_infusion=False, loss_option = 'default', test_metrics_by_infusion_group=True,
                # relevant for GCSF prediction/analysis
                 include_gcsf=False, use_gcsf=False, include_demographics=False, include_steroid=False, use_steroid=False,
                # whether to concatenate missingness indicators as extra features
                 missing_mask=True,
                # per-feature-category (module) tokenization
                 module_index_cfg=None,        # dict from build_module_index_config, or None to disable
                 use_module_tokenizer=False,   # master switch for per-module input embedding
                 use_continuous_time=True,     # add learned continuous-time embedding
                 time_as_raw_feature=False,    # add a linear map of the raw elapsed-time scalar
                 n_static_features=0,          # # of trailing static scalars (gender/age/ctype...)
                 post_mlp_dropout=None,         # dropout for post-transformer MLP; defaults to `dropout`
                 batch_first=True,             # attend over time; must match how the checkpoint was trained
                # pretraining objective variants
                 binary_pos_weight=None,
                 side_effects_pos_weight=None,
                 ):
        super(SPARC, self).__init__()

        if option not in ['pretrain', 'binary', 'risk', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            raise ValueError('option has to be "pretrain" or "binary" or "risk" or "side_effects" or "side_effects_risk" or "side_effects_next_risk"')
        self.option = option
        self.modular_featuredim_dict = modular_featuredim_dict
        self.primary_ctype_returned = primary_ctype_returned
        self.med_as_feature = med_as_feature
        self.meds_l = meds_l
        self.use_sigmoid = use_sigmoid
        self.loss_option = loss_option
        self.schedule = schedule
        self.test_metrics_by_infusion_group = test_metrics_by_infusion_group
        self.include_gcsf = include_gcsf
        self.use_gcsf = use_gcsf
        self.include_steroid = include_steroid
        self.use_steroid = use_steroid

        if [el for el in modular_featuredim_dict.keys() if el not in MSK_numfeature_dict.keys()]:
            raise ValueError('There are keys of modular_featuredim_dict that are not in the list of allowed features')
        base_features = sum(modular_featuredim_dict.values())
        self.base_features = base_features
        print(f"\n{'='*80}")
        print(f"SPARC MODEL INPUT FEATURE DIMENSIONS")
        print(f"{'='*80}")
        print(f"Modular feature dimensions:")
        for k, v in modular_featuredim_dict.items():
            print(f"  {k:20s}: {v:3d}")
        print(f"  {'Total base features':20s}: {base_features:3d}")
        print(f"Advanced model: {advanced_model}, Missing mask: {missing_mask}")

        if 'vitals' in modular_featuredim_dict.keys() or 'extra_lab' in modular_featuredim_dict.keys() or 'basic_lab' in modular_featuredim_dict.keys():
            if advanced_model == 'default':
                if missing_mask:
                    # (features + time) x 2 for missingness, + padding
                    self.total_feature_dim = 2 * (base_features + 1) + 1
                else:
                    # features + time + padding
                    self.total_feature_dim = (base_features + 1) + 1
            elif advanced_model == 'duett':
                ts_features = sum([el for k, el in modular_featuredim_dict.items() if k in ['basic_vitals', 'extra_vitals', 'extra_lab', 'basic_lab', 'cbc_differential']])
                non_ts_features = sum([el for k, el in modular_featuredim_dict.items() if k not in ['basic_vitals', 'extra_vitals', 'extra_lab', 'basic_lab', 'cbc_differential']])
                if missing_mask:
                    self.total_feature_dim = 2 * (2 * ts_features + non_ts_features + 1) + 1
                else:
                    self.total_feature_dim = (2 * ts_features + non_ts_features + 1) + 1
        else:
            if advanced_model == 'default':
                if missing_mask:
                    self.total_feature_dim = 2 * base_features
                else:
                    self.total_feature_dim = base_features
            elif advanced_model == 'duett':
                ts_features = sum([el for k, el in modular_featuredim_dict.items() if k in ['basic_vitals', 'extra_vitals', 'extra_lab', 'basic_lab', 'cbc_differential']])
                non_ts_features = sum([el for k, el in modular_featuredim_dict.items() if k not in ['basic_vitals', 'extra_vitals', 'extra_lab', 'basic_lab', 'cbc_differential']])
                if missing_mask:
                    self.total_feature_dim = 2 * (2 * ts_features + non_ts_features)
                else:
                    self.total_feature_dim = 2 * ts_features + non_ts_features

        print(f"Final MLP input dimension: {self.total_feature_dim}")
        print(f"{'='*80}\n")

        ## model/data choices and manipulations
        self.perturbation_index = perturbation_index
        self.normalization = normalization
        self.advanced_model = advanced_model
        self.usage = usage

        ## hyperparameters
        self.max_time_len = max_time_len
        self.weight_decay = weight_decay
        self.lr = lr
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.side_effects_num_infusion = side_effects_num_infusion
        self.missing_mask = missing_mask

        self.module_index_cfg = module_index_cfg
        self.use_module_tokenizer = use_module_tokenizer and (module_index_cfg is not None)
        self.use_continuous_time = use_continuous_time
        self.time_as_raw_feature = time_as_raw_feature
        self.batch_first = batch_first
        self.n_static_features = n_static_features
        self.post_mlp_dropout = post_mlp_dropout if post_mlp_dropout is not None else dropout

        if self.use_module_tokenizer and advanced_model == 'duett':
            raise NotImplementedError(
                "Per-module tokenizer is not supported with advanced_model='duett'. "
                "Use advanced_model='default'.")


        # num_infusion only applies to side-effect tasks
        if option in ['binary', 'risk'] and self.side_effects_num_infusion:
            print(f"WARNING: side_effects_num_infusion=True is not applicable for option '{option}'. Setting to False.")
            self.side_effects_num_infusion = False

        # only attention aggregation with the missingness mask is supported
        if temporal_aggregation != 'attention':
            raise ValueError(
                f"temporal_aggregation must be 'attention', got {temporal_aggregation!r}")
        if not missing_mask:
            raise ValueError("missing_mask must be True; the no-mask variant was removed")
        self.temporal_aggregation = temporal_aggregation
        ## pretrain stuff
        self.pretrain_contrastive_weight = pretrain_contrastive_weight
        self.finetune_part = finetune_part
        self.time_calibration = time_calibration


        if self.total_feature_dim == 1:
            raise ValueError('no input features provided')
        else:
            self.time_encoding = time_encoding
            if time_encoding == 'pe':
                self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_time_len)
            if advanced_model == 'duett':
                self.time_cve = nn.Sequential(
                    nn.Linear(1, int(np.sqrt((self.total_feature_dim+1)))),
                    nn.Tanh(),
                    nn.Linear(int(np.sqrt((self.total_feature_dim+1))), 1),
                )

            self.input_projection = nn.Sequential(
                nn.Linear(self.total_feature_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )

            # per-module tokenizer replaces the flat input_projection path
            if self.use_module_tokenizer:
                self.per_module_tokenizer = PerModuleTokenizer(
                    self.module_index_cfg, d_model,
                    dropout=dropout,
                    use_time_encoding=self.use_continuous_time,
                    n_static=self.n_static_features,
                    time_as_raw_feature=self.time_as_raw_feature,
                )
                print(f"Per-module tokenizer ENABLED: "
                      f"{[n for n, _ in self.module_index_cfg['modules']]}, "
                      f"continuous_time={self.use_continuous_time}, "
                      f"n_static={self.n_static_features}")
            else:
                self.per_module_tokenizer = None
                # optional continuous-time embedding on the flat path
                if self.use_continuous_time:
                    self.flat_time_encoding = ContinuousTimeEncoding(d_model)
                else:
                    self.flat_time_encoding = None
            # transformer encoder
            if advanced_model == 'duett':
                num_encoder_layers = num_encoder_layers//2
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, norm_first=True, batch_first=self.batch_first)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
            self.transformer_layer_norm = nn.LayerNorm(d_model)

            if advanced_model == 'duett':
                event_d_model = max_time_len
                # event transformer d_model = max_time_len, so nhead must divide it
                self.event_encoder_layer = nn.TransformerEncoderLayer(d_model=event_d_model, nhead=3, dim_feedforward=dim_feedforward, dropout=dropout, norm_first=True, batch_first=self.batch_first)
                self.event_transformer_encoder = nn.TransformerEncoder(self.event_encoder_layer, num_layers=num_encoder_layers)

            self.temporal_attention = MissingnessAwareAttentionAggregation(d_model, max_time_len=max_time_len)
            aggregation_output_dim = d_model

            self.post_transformer_to_representation_MLP = nn.Sequential(
                nn.Linear(aggregation_output_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(self.post_mlp_dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            # last-timepoint prediction (base features only)
            self.representation_MLP_to_last = nn.Sequential(
                nn.Linear(d_model+1, self.base_features),
            )
            if self.total_feature_dim != 1:
                self.representation_to_binary_MLP = nn.Linear(d_model, 1)
                if option != 'pretrain' and finetune_part:
                    print("finetune_part, only representation_to_binary_MLP should have requires_grad=True")
                    for param in self.parameters():
                        param.requires_grad = False
                    for param in self.representation_to_binary_MLP.parameters():
                        param.requires_grad = True

            if med_as_feature:
                self.meds_to_single_MLP = nn.Sequential(
                    nn.Linear(len(meds_l), 1),
                    nn.ReLU()
                )
                self.medandrepresentation_to_representation_MLP = nn.Sequential(
                    nn.Linear(d_model+1, d_model),
                    nn.ReLU()
                )

            if option in ['side_effects', 'side_effects_risk', 'side_effects_next_risk'] and self.side_effects_num_infusion:
                self.infusion_num_to_single_MLP = nn.Sequential(
                    nn.Linear(1, 1),
                    nn.ReLU()
                )
                add_num = 1
                if self.include_gcsf and self.use_gcsf:
                    self.gcfs_gt_to_single_MLP = nn.Sequential(
                        nn.Linear(1, 1),
                        nn.ReLU()
                    )
                    add_num += 1
                if self.include_steroid and self.use_steroid:
                    self.steroid_gt_to_single_MLP = nn.Sequential(
                        nn.Linear(1, 1),
                        nn.ReLU()
                    )
                    add_num += 1
                self.representation_and_infusionnum_to_representation_MLP = nn.Sequential(
                    nn.Linear(d_model+add_num, d_model),
                    nn.ReLU()
                )

        if option not in ['pretrain', 'binary', 'risk', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            raise ValueError('Output projection error: option has to be "binary" or "pretrain" or "risk"')
        elif option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            if option == 'binary':
                _pw = binary_pos_weight if binary_pos_weight is not None else 1.0
                self.loss_fn = WeightedBCELoss(pos_weight=_pw)
                print(f"Binary loss: WeightedBCELoss(pos_weight={_pw:.2f})")
            elif option in ['side_effects_risk', 'side_effects_next_risk']:
                self.c_index = ConcordanceIndex()
                if self.loss_option == 'hybrid':
                    self.bce_loss_fn = nn.BCELoss()
                    print(f"Hybrid loss component: BCELoss()")
            else:
                # pos_weight None = plain BCE
                _se_pw = side_effects_pos_weight if side_effects_pos_weight is not None else 1.0
                self.loss_fn = WeightedBCELoss(pos_weight=_se_pw, reduction='none')
                if _se_pw == 1.0:
                    print("Side-effects loss: plain BCE (unweighted, pos_weight=1.0), reduction='none'")
                else:
                    print(f"Side-effects loss: WeightedBCELoss(pos_weight={_se_pw:.2f}, reduction='none')")

            self.auroc = AUROC(task='binary')
            self.accuracy = Accuracy(task='binary')
            self.balanced_accuracy = Accuracy(task='binary', average='macro')
            self.precision = Precision(task='binary')
            self.recall = Recall(task='binary')
            self.f1 = F1Score(task='binary')
            self.auprc = AveragePrecision(task='binary')

        elif option == 'pretrain':
            self.loss_fn = nn.MSELoss()
        elif 'risk' in option:
            self.c_index = ConcordanceIndex()

    def forward(self, src, src_key_padding_mask=None, last_tp_times=None, num_infusion=None, gcsf_groundtruth=None, steroid_groundtruth=None):
        if self.total_feature_dim == 1:
            raise ValueError('No data provided')
        if self.med_as_feature:
            src, meds_data = src
        if self.total_feature_dim != 1:
            batch_size, seq_len, _ = src.size() ## src dim: (batch_size, max_time_len, d_features + 1)
            if self.advanced_model == 'duett':
                cved_time = self.time_cve(src[:, :, -2].unsqueeze(-1)).squeeze(-1) # src dim: (batch_size, max_time_len, d_features + 1)
                src = src.clone()
                src[:, :, -2] = cved_time
            if self.per_module_tokenizer is not None:
                # layout: [features | time | missingness | padding], or without missingness
                # (NaNs as -10) when missing_mask=False
                n_base = self.base_features
                values = src[:, :, :n_base]
                times = src[:, :, n_base]
                if self.missing_mask:
                    missingness = src[:, :, n_base + 1: 2 * n_base + 1]
                else:
                    missingness = (values != -10.0).float()
                    values = torch.where(values == -10.0, torch.zeros_like(values), values)
                src = self.per_module_tokenizer(values, missingness, times)  # (B, T, d_model)
            else:
                if getattr(self, 'flat_time_encoding', None) is not None:
                    raw_times = src[:, :, self.base_features]
                src = self.input_projection(src)  ## src dim (batch_size, max_time_len, d_model)
                if getattr(self, 'flat_time_encoding', None) is not None:
                    src = src + self.flat_time_encoding(raw_times)

            if self.advanced_model == 'default' and self.time_encoding == 'pe':
                src = self.positional_encoding(src) # by default this is not used because we have time as a feature

            # padding mask comes as (T, B); the batch_first encoder needs (B, T)
            enc_mask = src_key_padding_mask
            if self.batch_first and enc_mask is not None:
                enc_mask = enc_mask.T

            if self.advanced_model == 'default':
                z = self.transformer_encoder(src, src_key_padding_mask=enc_mask) # z dim (batch_size, max_time_len, d_model)
                if torch.isnan(z).any():
                    z = torch.nan_to_num(z, nan=0.0)
                z = self.transformer_layer_norm(z)

            if self.advanced_model == 'duett':
                for layer_i, (event_layer, time_layer) in enumerate(zip(self.event_transformer_encoder.layers, self.transformer_encoder.layers)):
                    # src shape: (batch_size, max_time_len, d_model) or (batch_size, max_time_len+1, d_model) if cls
                    src = src.permute(0, 2, 1) # src dim (batch_size, d_model, max_time_len) or (batch_size, d_model, max_time_len+1)
                    src = event_layer(src) # src dim (batch_size, d_model, max_time_len) or (batch_size, d_model, max_time_len+1)
                    src = src.permute(0, 2, 1) # src dim (batch_size, max_time_len, d_model) or (batch_size, max_time_len+1, d_model)
                    src = time_layer(src, src_key_padding_mask=enc_mask) # src dim (batch_size, max_time_len, d_model) or (batch_size, max_time_len+1, d_model)

                z = src
                z = self.transformer_layer_norm(z)

            if src_key_padding_mask is not None:
                # (T, B) -> (B, T, 1)
                mask = src_key_padding_mask.T.unsqueeze(-1).bool()
            else:
                # no mask: all positions valid (True = padded)
                mask = torch.zeros((batch_size, seq_len, 1), dtype=torch.bool, device=z.device)

            z = self.temporal_attention(z, mask.squeeze(-1))  # z dim (batch_size, d_model)

            z = self.post_transformer_to_representation_MLP(z)  # z dim: (batch_size, d_model)

            if self.med_as_feature:
                meds_data_representation = self.meds_to_single_MLP(meds_data)  # meds_data_representation dim: (batch_size, 1)
                z = torch.cat([z, meds_data_representation], dim=-1)  # z dim: (batch_size, d_model + 1)
                z = self.medandrepresentation_to_representation_MLP(z)  # z dim: (batch_size, d_model)

            if self.option in ['side_effects', 'side_effects_risk', 'side_effects_next_risk'] and self.side_effects_num_infusion:
                num_infusion_representation = self.infusion_num_to_single_MLP(num_infusion.unsqueeze(-1).float())  # num_infusion_representation dim: (batch_size, 1)
                z = torch.cat([z, num_infusion_representation], dim=-1)  # z dim: (batch_size, d_model + 1)
                if self.include_gcsf and self.use_gcsf and (gcsf_groundtruth is not None):
                    gcsf_representation = self.gcfs_gt_to_single_MLP(gcsf_groundtruth.unsqueeze(-1).float())
                    z = torch.cat([z, gcsf_representation], dim=-1)  # z dim: (batch_size, d_model + 2)
                if self.include_steroid and self.use_steroid and (steroid_groundtruth is not None):
                    steroid_representation = self.steroid_gt_to_single_MLP(steroid_groundtruth.unsqueeze(-1).float())
                    z = torch.cat([z, steroid_representation], dim=-1)
                z = self.representation_and_infusionnum_to_representation_MLP(z)  # z dim: (batch_size, d_model)


        if self.option == 'pretrain' and self.usage == 'standalone':
            z1 = self.representation_MLP_to_last(torch.cat([z, last_tp_times.unsqueeze(-1)], dim=-1))  # z: representation, z1: predicted last timepoint
            return z, z1
        elif self.option == 'pretrain' and self.usage != 'standalone':
            return z

        elif self.option in ['binary', 'risk', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            z = self.representation_to_binary_MLP(z) # z dim (batch_size, 1)

            z = torch.clamp(z, min=-10.0, max=10.0)

            if self.use_sigmoid:
                z = torch.sigmoid(z)

            return z

    @classmethod
    def custom_load_from_checkpoint(cls, checkpoint_path, load_w_meds = False, **kwargs):
        """Load a checkpoint, skipping the med_as_feature layers."""
        print("Loading model from checkpoint using custom loader:", checkpoint_path)
        model = cls(**kwargs)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        missing_keys = ['meds_to_single_MLP', 'medandrepresentation_to_representation_MLP']
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not any(key in k for key in missing_keys)
        }

        model.load_state_dict(filtered_state_dict, strict=False)
        return model

    @classmethod
    def custom_load_from_checkpoint_sideeffect(cls, checkpoint_path, **kwargs):
        """Load a checkpoint, skipping the num_infusion layers."""
        print("Loading model from checkpoint using custom loader:", checkpoint_path)
        model = cls(**kwargs)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        missing_keys = ["infusion_num_to_single_MLP", "representation_and_infusionnum_to_representation_MLP"]
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not any(key in k for key in missing_keys)
        }

        model.load_state_dict(filtered_state_dict, strict=False)
        return model

    def data_prep(self, data, time_calibration=False, perturbation_index='', perturbation_direction='both', primary_ctype_returned=False):
        if primary_ctype_returned:
            data_x, data_y, (DMP_ID, ctype, cutoffdate) = data
            data = data_x, data_y
        if self.total_feature_dim != 1:
            if self.option == 'pretrain' and self.usage == 'standalone':
                if self.primary_ctype_returned:
                    ((xs, times), y, ctypes, *_), DMP_ID = data
                else:
                    ((xs, times), y, *_), DMP_ID = data
            else:
                if not self.med_as_feature and self.side_effects_num_infusion:
                    if not self.include_gcsf and not self.include_steroid:
                        (xs, times, num_infusion, DMP_ID), y = data
                    elif self.include_gcsf and not self.include_steroid:
                        (xs, times, num_infusion, DMP_ID, gcsf_groundtruth), y = data
                    elif not self.include_gcsf and self.include_steroid:
                        (xs, times, num_infusion, DMP_ID, steroid_groundtruth), y = data
                    else:
                        raise ValueError("Currently do not support including both gcsf and steroid groundtruth, please choose one or the other")
                elif not self.side_effects_num_infusion:
                    (xs, times) , y = data
                    num_infusion, DMP_ID = None, None
                else:
                    # med_as_feature=True and side_effects_num_infusion=True
                    if not self.include_gcsf and not self.include_steroid:
                        (xs, times, meds_data, num_infusion, DMP_ID), y = data
                    elif self.include_gcsf and not self.include_steroid:
                        (xs, times, meds_data, num_infusion, DMP_ID, gcsf_groundtruth), y = data
                    elif not self.include_gcsf and self.include_steroid:
                        (xs, times, meds_data, num_infusion, DMP_ID, steroid_groundtruth), y = data
                    else:
                        raise ValueError("Currently do not support including both gcsf and steroid groundtruth, please choose one or the other")
        else:
            raise ValueError('No data provided')

        if 'risk' not in self.option and self.option != 'side_effects':
            y = torch.tensor(y).float()
            y = y.unsqueeze(-1)

        # shift times to start at 0
        if time_calibration:
            times = times - times[:, 0].unsqueeze(-1)

        if self.normalization:
            for i in range(xs.size(-1)):
                mean = xs[:, :, i].nanmean()
                std = nanstd(xs[:, :, i])
                xs[:, :, i] = (xs[:, :, i] - mean) / std


        # permutation importance: shuffle the observed values of one feature across the batch
        if perturbation_index != '':
            if perturbation_index >= xs.size(-1):
                raise ValueError('perturbation_index out of range')

            slice_ = xs[:, :, perturbation_index]
            mask = ~torch.isnan(slice_)
            values = slice_[mask]

            perm = values[torch.randperm(values.size(0))]

            new_slice = slice_.clone()
            new_slice[mask] = perm
            xs[:, :, perturbation_index] = new_slice

        if self.option == 'pretrain' and self.usage == 'standalone':
            # t=0 is the most recent visit; predict it from the rest
            last_tp_xs_to_predict = xs[:, 0, :]
            last_tp_times = times[:, 0]
            xs = xs[:, 1:, :]
            times = times[:, 1:]
            # append time so it can be split off later
            if 'basic_lab' in self.modular_featuredim_dict.keys() or 'extra_lab' in self.modular_featuredim_dict.keys() or 'vital' in self.modular_featuredim_dict.keys():
                last_tp_xs_to_predict = torch.cat([last_tp_xs_to_predict, last_tp_times.unsqueeze(-1)], dim=-1)
        else:
            last_tp_xs_to_predict = None

        # append time before building the mask
        if 'basic_lab' in self.modular_featuredim_dict.keys() or 'extra_lab' in self.modular_featuredim_dict.keys() or 'vital' in self.modular_featuredim_dict.keys():
            xs = torch.cat([xs, times.unsqueeze(-1)], dim=-1)

        mask = torch.isnan(xs) # (B, T, F+1), True = NaN
        src_key_padding_mask = mask[:, :, -1].T # (T, B), True = padded timepoint

        if self.missing_mask:
            # append observed indicators (incl. time) and fill NaNs with 0
            missingness_indicators = (~mask).float()
            xs[mask] = 0.0
            xs = torch.cat([xs, missingness_indicators], dim=-1)
        else:
            # no indicators: fill NaNs with -10 (out of range in z-score space)
            xs[mask] = -10.0

        # padding indicator as an extra feature
        padding_indicator = torch.tensor(~src_key_padding_mask).unsqueeze(-1).permute(1, 0, 2)

        xs = torch.cat([xs, padding_indicator], dim=-1)
        if self.primary_ctype_returned:
            DMP_ID_demo_extended = (DMP_ID, ctype, cutoffdate)
        else:
            DMP_ID_demo_extended = DMP_ID
        if self.option == 'pretrain' and self.usage == 'standalone':
            if self.primary_ctype_returned:
                return (xs, ctypes), y, src_key_padding_mask, last_tp_xs_to_predict
            else:
                return xs, y, src_key_padding_mask, last_tp_xs_to_predict
        else:
            if not self.med_as_feature:
                if not self.include_gcsf and not self.include_steroid:
                    return xs, y, src_key_padding_mask, last_tp_xs_to_predict, num_infusion, DMP_ID_demo_extended
                elif self.include_gcsf and not self.include_steroid:
                    return xs, y, src_key_padding_mask, last_tp_xs_to_predict, num_infusion, DMP_ID_demo_extended, gcsf_groundtruth
                elif not self.include_gcsf and self.include_steroid:
                    return xs, y, src_key_padding_mask, last_tp_xs_to_predict, num_infusion, DMP_ID_demo_extended, steroid_groundtruth
            else:
                if not self.include_gcsf and not self.include_steroid:
                    return (xs, meds_data), y, src_key_padding_mask, last_tp_xs_to_predict, num_infusion, DMP_ID_demo_extended
                elif self.include_gcsf and not self.include_steroid:
                    return (xs, meds_data), y, src_key_padding_mask, last_tp_xs_to_predict, num_infusion, DMP_ID_demo_extended, gcsf_groundtruth
                elif not self.include_gcsf and self.include_steroid:
                    return (xs, meds_data), y, src_key_padding_mask, last_tp_xs_to_predict, num_infusion, DMP_ID_demo_extended, steroid_groundtruth


    def _compute_pretrain_loss(self, xs_rep, xs_last_pred, last_tp_xs_to_predict, ctypes):
        """Compute combined last-token-prediction + contrastive pretraining loss."""
        losses = []
        for i in range(xs_last_pred.shape[0]):
            current_xs_last_gt = last_tp_xs_to_predict[i]
            current_xs_last_gt_non_nan_mask = ~torch.isnan(current_xs_last_gt)

            if current_xs_last_gt_non_nan_mask.sum() == 0:
                continue

            current_xs_last_pred = xs_last_pred[i]
            current_loss = self.loss_fn(
                current_xs_last_pred[current_xs_last_gt_non_nan_mask],
                current_xs_last_gt[current_xs_last_gt_non_nan_mask]
            )
            if not current_loss.isnan():
                losses.append(current_loss / current_xs_last_pred[current_xs_last_gt_non_nan_mask].size(0))

        if losses:
            loss = torch.stack(losses).sum()
        else:
            print("Warning: all samples in batch had invalid loss.")
            print(f"  Batch size: {xs_last_pred.shape[0]}")
            print(f"  last_tp_xs_to_predict shape: {last_tp_xs_to_predict.shape}")
            print(f"  xs_last_pred shape: {xs_last_pred.shape}")
            for i in range(min(3, xs_last_pred.shape[0])):
                current_xs_last_gt = last_tp_xs_to_predict[i]
                current_xs_last_gt_non_nan_mask = ~torch.isnan(current_xs_last_gt)
                print(f"  Sample {i}: valid targets = {current_xs_last_gt_non_nan_mask.sum().item()}/{current_xs_last_gt.numel()}")
                if current_xs_last_gt_non_nan_mask.sum() > 0:
                    print(f"    Target values (first 5): {current_xs_last_gt[current_xs_last_gt_non_nan_mask][:5]}")
                    print(f"    Predicted values (first 5): {xs_last_pred[i][current_xs_last_gt_non_nan_mask][:5]}")
            loss = torch.tensor(0.0, device=xs_last_pred.device, requires_grad=True)

        if self.pretrain_contrastive_weight > 0:
            N = xs_rep.shape[0]
            norms = xs_rep.norm(dim=1, keepdim=True)  # [N, 1]
            normalized_xs_rep = xs_rep / norms  # [N, D]
            cosine_sim_matrix = torch.mm(normalized_xs_rep, normalized_xs_rep.T)  # [N, N]
            i_idx, j_idx = torch.triu_indices(N, N, offset=1)
            pairwise_cosine_sims = cosine_sim_matrix[i_idx, j_idx]
            same_type_mask = (ctypes[i_idx] == ctypes[j_idx])
            contrastive_loss = torch.where(same_type_mask, -pairwise_cosine_sims, pairwise_cosine_sims)
            loss = loss + self.pretrain_contrastive_weight * contrastive_loss.mean()

        return loss

    def _calculate_weighted_c_index_by_infusion(self, y_pred, y_event, time_to_event, num_infusion, mode='train'):
        """C-index and next-infusion metrics, weighted across infusion-number groups."""
        output_dict = {}
        num_infusion_tensor = torch.as_tensor(num_infusion, device=y_pred.device)
        y_event_tensor = torch.as_tensor(y_event, device=y_pred.device)
        time_to_event_tensor = torch.as_tensor(time_to_event, device=y_pred.device)

        unique_infusions = torch.unique(num_infusion_tensor)
        risk_scores = y_pred.squeeze()

        # --- C-index (risk tasks) ---
        if 'risk' in self.option:
            # groups with both events and censoring
            valid_infusions = []
            for infusion in unique_infusions:
                mask = num_infusion_tensor == infusion
                current_events = y_event_tensor[mask]

                if current_events.sum() > 0 and current_events.sum() < len(current_events):
                    valid_infusions.append(infusion)

            # unstratified C-index
            try:
                c_idx_raw = self.c_index(
                    risk_scores.unsqueeze(1),
                    y_event_tensor.bool(),
                    time_to_event_tensor
                ).item()
            except Exception:
                c_idx_raw = 0.5

            c_index_weighted = 0.5
            overall_c_index = 0.5
            num_groups = 0
            total_samples = 0
            computable_results = []
            if len(valid_infusions) > 0:
                for infusion in valid_infusions:
                    mask = num_infusion_tensor == infusion
                    current_risk = risk_scores[mask]
                    current_events = y_event_tensor[mask]
                    current_times = time_to_event_tensor[mask]

                    try:
                        c_idx = self.c_index(
                            current_risk.unsqueeze(1),
                            current_events.bool(),
                            current_times
                        )
                        c_idx_value = c_idx.item()
                        if not np.isnan(c_idx_value):
                            computable_results.append({
                                'infusion': infusion.item(),
                                'c_index': c_idx_value,
                                'num_samples': mask.sum().item(),
                                'mask': mask
                            })
                    except Exception:
                        continue
            if len(computable_results) > 0:
                total_samples = sum(r['num_samples'] for r in computable_results)
                c_index_weighted = sum(
                    r['c_index'] * r['num_samples'] / total_samples
                    for r in computable_results
                )
                # C-index over all valid groups pooled
                all_mask = torch.zeros_like(num_infusion_tensor, dtype=torch.bool)
                for r in computable_results:
                    all_mask |= r['mask']

                try:
                    overall_c_index = self.c_index(
                        risk_scores[all_mask].unsqueeze(1),
                        y_event_tensor[all_mask].bool(),
                        time_to_event_tensor[all_mask]
                    ).item()
                except Exception:
                    overall_c_index = 0.5

                num_groups = len(computable_results)

            output_dict.update({
                'c_index_raw': c_idx_raw,
                'c_index_weighted': torch.tensor(c_index_weighted, device=y_pred.device),
                'c_index_overall': overall_c_index,
                'num_groups': num_groups,
                'num_samples': total_samples,
                'per_infusion': computable_results,
            })
        # --- next-infusion AUROC ---
        # positive = side effect at the next infusion
        time_to_event_tensor = time_to_event_tensor.flatten()
        next_infusion_event = ((y_event_tensor == 1) & (time_to_event_tensor == 1)).float()


        valid_infusions_auroc = []
        for infusion in unique_infusions:
            mask = num_infusion_tensor == infusion
            current_next_events = next_infusion_event[mask]

            # need both positives and negatives
            has_positives = current_next_events.sum() > 0
            has_negatives = current_next_events.sum() < len(current_next_events)

            if has_positives and has_negatives:
                valid_infusions_auroc.append(infusion)

        auroc_next_weighted = 0.5
        auroc_next_overall = 0.5
        accuracy_next_weighted = 0.5
        precision_next_weighted = 0.5
        recall_next_weighted = 0.5
        auprc_next_weighted = 0.5
        auprc_next_overall = 0.5
        num_groups_auroc = 0
        num_samples_auroc = 0
        binary_loss_weighted = None

        if len(valid_infusions_auroc) > 0:
            computable_auroc_results = []

            # per-sample BCE once, then sliced per group
            all_sample_losses = self.loss_fn(
                risk_scores.unsqueeze(-1),
                next_infusion_event.unsqueeze(-1)
            ) if 'risk' not in self.option else None

            for infusion in valid_infusions_auroc:
                mask = num_infusion_tensor == infusion
                current_risk = risk_scores[mask]
                current_next_events = next_infusion_event[mask]
                try:
                    current_res = {'num_samples': mask.sum().item()}
                    if mode != 'train':
                        auroc = self.auroc(current_risk, current_next_events.int())
                        accuracy = self.accuracy(current_risk, current_next_events.int())
                        precision = self.precision(current_risk, current_next_events.int())
                        recall = self.recall(current_risk, current_next_events.int())
                        auprc = self.auprc(current_risk, current_next_events.int())

                        current_res = {
                            'infusion': infusion.item(),
                            'auroc': auroc.item(),
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'auprc': auprc,
                            'num_samples': mask.sum().item(),
                            'num_positives': current_next_events.sum().item(),
                            'mask': mask
                        }

                    # keep as tensor for backprop
                    if 'risk' not in self.option:
                        binary_loss = all_sample_losses[mask].mean()
                        current_res['binary_loss'] = binary_loss

                    computable_auroc_results.append(current_res)

                except Exception as e:
                    print(f"Skipping AUROC for infusion {infusion.item()}: {e}")
                    continue

            if len(computable_auroc_results) > 0:
                total_samples_auroc = sum(r['num_samples'] for r in computable_auroc_results)
                if mode != 'train':
                    auroc_next_weighted = sum(
                        r['auroc'] * r['num_samples'] / total_samples_auroc
                        for r in computable_auroc_results
                    )
                    accuracy_next_weighted = sum(
                        r['accuracy'] * r['num_samples'] / total_samples_auroc
                        for r in computable_auroc_results
                    )
                    precision_next_weighted = sum(
                        r['precision'] * r['num_samples'] / total_samples_auroc
                        for r in computable_auroc_results
                    )
                    recall_next_weighted = sum(
                        r['recall'] * r['num_samples'] / total_samples_auroc
                        for r in computable_auroc_results
                    )
                    auprc_next_weighted = sum(
                        r['auprc'] * r['num_samples'] / total_samples_auroc
                        for r in computable_auroc_results
                    )
                    output_dict.update({
                        'auroc_next_infusion_weighted': auroc_next_weighted,
                        'accuracy_next_infusion_weighted': accuracy_next_weighted,
                        'precision_next_infusion_weighted': precision_next_weighted,
                        'recall_next_infusion_weighted': recall_next_weighted,
                        'auprc_next_infusion_weighted': auprc_next_weighted,
                        'num_groups_next_infusion': num_groups_auroc,
                        'per_infusion_auroc': computable_auroc_results
                    })

                if 'risk' not in self.option:
                    binary_loss_weighted = torch.stack([
                        r['binary_loss'] * r['num_samples'] / total_samples_auroc
                        for r in computable_auroc_results
                    ]).sum()
                if mode != 'train':
                    # pooled AUROC / AUPRC over valid groups
                    all_mask_auroc = torch.zeros_like(num_infusion_tensor, dtype=torch.bool)
                    for r in computable_auroc_results:
                        all_mask_auroc |= r['mask']

                    try:
                        auroc_next_overall = self.auroc(
                            risk_scores[all_mask_auroc],
                            next_infusion_event[all_mask_auroc].int()
                        ).item()
                    except Exception:
                        auroc_next_overall = 0.5
                    output_dict['auroc_next_infusion_overall'] = auroc_next_overall

                    try:
                        auprc_next_overall = self.auprc(
                            risk_scores[all_mask_auroc],
                            next_infusion_event[all_mask_auroc].int()
                        ).item()
                    except Exception:
                        auprc_next_overall = 0.0
                    output_dict['auprc_next_infusion_overall'] = auprc_next_overall

                    num_groups_auroc = len(computable_auroc_results)
                    num_samples_auroc = total_samples_auroc

        else:
            if mode in ['train']:
                return None # skipped by the trainer, no backprop
            else:
                output_dict.update({
                    'auroc_next_infusion_weighted': auroc_next_weighted,
                    'auroc_next_infusion_overall': auroc_next_overall,
                    'accuracy_next_infusion_weighted': accuracy_next_weighted,
                    'precision_next_infusion_weighted': precision_next_weighted,
                    'recall_next_infusion_weighted': recall_next_weighted,
                    'auprc_next_infusion_weighted': auprc_next_weighted,
                    'auprc_next_infusion_overall': auprc_next_overall,
                    'num_groups_next_infusion': num_groups_auroc,
                    'num_samples_next_infusion': num_samples_auroc,
                    'per_infusion_auroc': [],
                })
        if 'risk' not in self.option and binary_loss_weighted is not None:
            output_dict['binary_loss'] = binary_loss_weighted  # tensor, not .item()
            output_dict['num_samples_next_infusion'] = num_samples_auroc

        return output_dict

    def predict_step(self, batch, batch_idx):
        num_infusion = None
        if self.option == 'pretrain' and self.usage == 'standalone':
            xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
        else:
            if not self.include_gcsf and not self.include_steroid:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID_demo_extended = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
            elif self.include_steroid:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID_demo_extended, steroid_groundtruth = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
            else:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID_demo_extended, gcsf_groundtruth = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
        if self.option == 'pretrain':
            if self.primary_ctype_returned:
                xs_combined, ctypes = xs_combined
            else:
                ctypes = None
            last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]
            xs_rep, xs_last_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times)
            return xs_rep, xs_last_pred

        elif self.option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            if not (self.use_gcsf and self.include_gcsf) and not (self.use_steroid and self.include_steroid):
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion)
            elif self.use_steroid and self.include_steroid:
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion, steroid_groundtruth=steroid_groundtruth)
            else:
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)
            platt_scaler = getattr(self, 'platt_scaler', None)
            if platt_scaler is not None:
                y_pred = y_pred.cpu().detach().numpy()
                y_pred = y_pred.reshape(-1, 1)
                y_pred = sm.add_constant(y_pred)  # intercept column
                y_pred = platt_scaler.predict_proba(y_pred)[:, 1]
                y_pred = torch.tensor(y_pred, device=xs_combined[0].device).unsqueeze(1).float()

            if not self.include_gcsf and not self.include_steroid:
                out = {'y_pred': y_pred, 'y_true': y, 'num_infusion': num_infusion}
            elif self.include_steroid:
                out = {'y_pred': y_pred, 'y_true': y, 'num_infusion': num_infusion, 'steroid_groundtruth': steroid_groundtruth}
            else:
                out = {'y_pred': y_pred, 'y_true': y, 'gcsf_groundtruth': gcsf_groundtruth, 'num_infusion': num_infusion}
            if self.primary_ctype_returned:
                out['DMP_ID'] = DMP_ID_demo_extended[0]
                out['ctype'] = DMP_ID_demo_extended[1]
                out['cutoffdate'] = DMP_ID_demo_extended[2]
            return out

    # --- lightning hooks ---
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        print("====TORCH VERSION=====", torch.__version__)

        # 'default': constant learning rate
        if self.schedule == 'default':
            return optimizer

        elif self.schedule == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }
            }
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                }
            }

    def training_step(self, batch, batch_idx):

        num_infusion = None
        if self.option == 'pretrain' and self.usage == 'standalone':
            xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
        else:
            if not self.include_gcsf and not self.include_steroid:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
            elif self.include_steroid and self.use_steroid:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID, steroid_groundtruth = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
            else:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID, gcsf_groundtruth = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)

        if self.option == 'pretrain':
            if self.primary_ctype_returned:
                xs_combined, ctypes = xs_combined
            else:
                ctypes = None
            last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]
            xs_rep, xs_last_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times, num_infusion=num_infusion)

            pretrain_loss = self._compute_pretrain_loss(xs_rep, xs_last_pred, last_tp_xs_to_predict, ctypes)

            loss = pretrain_loss

        elif self.option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            if not (self.use_gcsf and self.include_gcsf) and not (self.use_steroid and self.include_steroid):
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion)
            elif self.use_steroid and self.include_steroid:
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion, steroid_groundtruth=steroid_groundtruth)
            else:
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)
            if self.option == 'binary':
                loss = self.loss_fn(y_pred, y)
            elif self.option in ['side_effects_risk', 'side_effects_next_risk']:
                y_event, time_to_event = torch.tensor(y[0]).to(y_pred.device), torch.tensor(y[1]).float().to(y_pred.device)
                cox_loss = cox.neg_partial_log_likelihood(y_pred, y_event, time_to_event, ties_method = "efron")

                if self.loss_option == 'hybrid':
                    # BCE on the next-infusion event (y_event==1 and time_to_event==1)
                    current_infusion_event = ((y_event == 1) & (time_to_event == 1)).float()
                    bce_loss = self.bce_loss_fn(y_pred.squeeze(), current_infusion_event)
                    loss = cox_loss + 20 * bce_loss
                    self.log('train_cox_loss', cox_loss)
                    self.log('train_bce_loss', bce_loss)
                else:
                    loss = cox_loss
            elif self.option == 'side_effects':
                time_to_event = torch.ones_like(y_pred).float().to(y_pred.device)
                res_dict = self._calculate_weighted_c_index_by_infusion(y_pred, y, time_to_event, num_infusion)
                if res_dict is None:
                    return None
                loss = res_dict.get('binary_loss', torch.tensor(0.0, device=y_pred.device))

        elif 'risk' in self.option:
            y_pred_logits = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion)
            fail_indicators = torch.tensor(y[0]).to(y_pred_logits.device)
            time_to_events = torch.tensor(y[1]).float().to(y_pred_logits.device)
            loss = cox.neg_partial_log_likelihood(y_pred_logits, fail_indicators, time_to_events, ties_method = "efron")

        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        num_infusion = None
        if self.option == 'pretrain' and self.usage == 'standalone':
            xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
        else:
            if not self.include_gcsf and not self.include_steroid:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
            elif self.include_steroid and self.use_steroid:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID, steroid_groundtruth = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
            else:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID, gcsf_groundtruth = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)

        if self.option == 'pretrain':
            if self.primary_ctype_returned:
                xs_combined, ctypes = xs_combined
            else:
                ctypes = None
            last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]
            xs_rep, xs_last_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times, num_infusion=num_infusion)

            pretrain_loss = self._compute_pretrain_loss(xs_rep, xs_last_pred, last_tp_xs_to_predict, ctypes)

            loss = pretrain_loss

        elif self.option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            if not (self.use_gcsf and self.include_gcsf) and not (self.use_steroid and self.include_steroid):
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion)
            elif self.use_steroid and self.include_steroid:
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion, steroid_groundtruth=steroid_groundtruth)
            else:
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)
            if self.option == 'binary' or self.option == 'side_effects':
                if self.option == 'binary':
                    loss = self.loss_fn(y_pred, y)
                    auroc = self.auroc(y_pred, y.int())
                    accuracy = self.accuracy(y_pred, y.int())
                    precision = self.precision(y_pred, y.int())
                    recall = self.recall(y_pred, y.int())
                    auprc = self.auprc(y_pred, y.int())
                    self.log('val_auroc', auroc, prog_bar=True)
                else:
                    time_to_event = torch.ones_like(y_pred).float().to(y_pred.device)
                    res_dict = self._calculate_weighted_c_index_by_infusion(y_pred, y, time_to_event, num_infusion, mode='val')
                    if res_dict is None:
                        return None
                    loss = res_dict.get('binary_loss', torch.tensor(0.0, device=y_pred.device))
                    weighted_auroc, overall_auroc, accuracy, precision, recall, auprc = res_dict.get('auroc_next_infusion_weighted'), res_dict.get('auroc_next_infusion_overall'), res_dict.get('accuracy_next_infusion_weighted'), res_dict.get('precision_next_infusion_weighted'), res_dict.get('recall_next_infusion_weighted'), res_dict.get('auprc_next_infusion_weighted')
                    self.log('val_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
                    self.log('val_auroc_nextinf_overall', overall_auroc, prog_bar=True)
                self.log('val_accuracy', accuracy, prog_bar=True)
                self.log('val_precision', precision, prog_bar=True)
                self.log('val_recall', recall, prog_bar=True)
                self.log('val_auprc', auprc, prog_bar=True)

            elif self.option in ['side_effects_risk', 'side_effects_next_risk']:
                y_event, time_to_event = torch.tensor(y[0]).to(y_pred.device), torch.tensor(y[1]).float().to(y_pred.device)
                cox_loss = cox.neg_partial_log_likelihood(y_pred, y_event, time_to_event, ties_method = "efron")

                if self.loss_option == 'hybrid':
                    # BCE on the next-infusion event
                    current_infusion_event = ((y_event == 1) & (time_to_event == 1)).float()
                    bce_loss = self.bce_loss_fn(y_pred.squeeze(), current_infusion_event)
                    loss = cox_loss + 20 * bce_loss
                    self.log('val_cox_loss', cox_loss)
                    self.log('val_bce_loss', bce_loss)
                else:
                    loss = cox_loss

                c_idx_dict = self._calculate_weighted_c_index_by_infusion(y_pred, y_event, time_to_event, num_infusion, mode='val')
                if c_idx_dict is None:
                    print("c_idx_dict is None, skipping logging")
                    return None
                c_index, c_index_weighted, overall_c_index = c_idx_dict['c_index_raw'], c_idx_dict['c_index_weighted'], c_idx_dict['c_index_overall']
                weighted_auroc, overall_auroc = c_idx_dict['auroc_next_infusion_weighted'], c_idx_dict['auroc_next_infusion_overall']
                weighted_auprc, overall_auprc = c_idx_dict['auprc_next_infusion_weighted'], c_idx_dict['auprc_next_infusion_overall']
                self.log('val_cindex_unfiltered', c_index, prog_bar=True)
                self.log('val_cindex_weighted', c_index_weighted, prog_bar=True)
                self.log('val_cindex_overall_valid', overall_c_index, prog_bar=True)
                self.log('val_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
                self.log('val_auroc_nextinf_overall', overall_auroc, prog_bar=True)
                self.log('val_auprc_nextinf_weighted', weighted_auprc, prog_bar=True)
                self.log('val_auprc_nextinf_overall', overall_auprc, prog_bar=True)

        elif 'risk' in self.option:
            y_pred_logits = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined)
            fail_indicators = torch.tensor(y[0]).to(y_pred_logits.device)
            time_to_events = torch.tensor(y[1]).float().to(y_pred_logits.device)
            loss = cox.neg_partial_log_likelihood(y_pred_logits, fail_indicators, time_to_events, ties_method = "efron")
            c_index = self.c_index(y_pred_logits, fail_indicators, time_to_events)

            self.log('val_c_index', c_index, prog_bar=True)

        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def load_state_dict(self, state_dict, strict=True):
        """Non-strict load that reports missing and unexpected keys."""
        result = super().load_state_dict(state_dict, strict=False)
        missing, unexpected = result.missing_keys, result.unexpected_keys
        if missing or unexpected:
            n_own = len(self.state_dict())
            print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)} "
                  f"(model has {n_own} entries)")
            for label, keys in (("missing", missing), ("unexpected", unexpected)):
                if keys:
                    print(f"  {label}: {keys[:8]}{' ...' if len(keys) > 8 else ''}")
            if n_own and len(missing) > 0.5 * n_own:
                print("WARNING: over half of the model's weights are missing from the checkpoint; "
                      "check use_module_tokenizer / batch_first / module_index_cfg.")
        return result

    # architecture flags saved with each checkpoint (batch_first can't be inferred from the weights)
    ARCH_STAMP_KEYS = ('use_module_tokenizer', 'use_continuous_time', 'batch_first',
                       'n_static_features', 'missing_mask', 'temporal_aggregation')

    def _arch_stamp(self):
        return {k: getattr(self, k, None) for k in self.ARCH_STAMP_KEYS}

    def on_save_checkpoint(self, checkpoint):
        """Save the architecture flags with the checkpoint."""
        checkpoint['sparc_arch'] = self._arch_stamp()
        super().on_save_checkpoint(checkpoint)

    def check_arch_stamp(self, checkpoint, strict=False):
        """Architecture flags that differ from the checkpoint's, or None if it has none saved."""
        stamp = checkpoint.get('sparc_arch') if isinstance(checkpoint, dict) else None
        if not stamp:
            print("[arch] checkpoint has no saved architecture flags; batch_first and "
                  "use_module_tokenizer can't be checked against it")
            return None
        mine = self._arch_stamp()
        diff = [k for k in self.ARCH_STAMP_KEYS if stamp.get(k) != mine.get(k)]
        if diff:
            print("WARNING: model architecture differs from the checkpoint:")
            for k in diff:
                print(f"  {k}: checkpoint={stamp.get(k)!r}  model={mine.get(k)!r}")
            if strict:
                raise RuntimeError(f"checkpoint/model architecture mismatch on {diff}")
        else:
            print(f"[arch] stamp verified: {mine}")
        return diff

    def on_load_checkpoint(self, checkpoint):
        """Refuse a batch_first mismatch; drop optimizer state if param counts differ."""
        _diff = self.check_arch_stamp(checkpoint)
        if _diff and 'batch_first' in _diff:
            raise RuntimeError(
                "checkpoint was trained with batch_first="
                f"{checkpoint['sparc_arch'].get('batch_first')!r} but the model was built with "
                f"batch_first={self.batch_first!r}; pass the matching -bf / batch_first")
        # drop optimizer state saved for a different set of parameters
        if 'optimizer_states' in checkpoint:
            num_current_params = len(list(p for p in self.parameters() if p.requires_grad))

            has_mismatch = False
            for opt_state in checkpoint['optimizer_states']:
                if 'state' in opt_state and len(opt_state['state']) > 0:
                    max_old_idx = max(k for k in opt_state['state'].keys() if isinstance(k, int))
                    if max_old_idx >= num_current_params:
                        has_mismatch = True
                        break

            if has_mismatch:
                checkpoint['optimizer_states'] = []
                if 'lr_schedulers' in checkpoint:
                    checkpoint['lr_schedulers'] = []

                print(f"\nDetected parameter mismatch in optimizer state")
                print(f"  Dropping old optimizer state to allow training to continue")
                print(f"  Model weights and epoch counter preserved")
                print(f"  Optimizer will restart fresh (losing momentum/adaptive LR state)")
                print(f"  Current model has {num_current_params} trainable parameters\n")

    def test_step(self, batch, batch_idx):

        num_infusion = None
        if self.option == 'pretrain' and self.usage == 'standalone':
            xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
        else:
            if not self.include_gcsf and not self.include_steroid:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration, perturbation_index=self.perturbation_index, primary_ctype_returned=self.primary_ctype_returned)
            elif self.include_steroid and self.use_steroid:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID, steroid_groundtruth = self.data_prep(batch, time_calibration=self.time_calibration, perturbation_index=self.perturbation_index, primary_ctype_returned=self.primary_ctype_returned)
            else:
                xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID, gcsf_groundtruth = self.data_prep(batch, time_calibration=self.time_calibration, primary_ctype_returned=self.primary_ctype_returned)
        if self.option == 'pretrain':
            if self.primary_ctype_returned:
                xs_combined, ctypes = xs_combined
            else:
                ctypes = None
            last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]
            xs_rep, xs_last_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times, num_infusion=num_infusion)
            loss = self._compute_pretrain_loss(xs_rep, xs_last_pred, last_tp_xs_to_predict, ctypes)

        elif self.option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            if not (self.use_gcsf and self.include_gcsf) and not (self.use_steroid and self.include_steroid):
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion)
            elif self.use_steroid and self.include_steroid:
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion, steroid_groundtruth=steroid_groundtruth)
            else:
                y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)
            platt_scaler = getattr(self, "platt_scaler", None)
            if platt_scaler is not None:
                y_pred = y_pred.cpu().detach().numpy()
                y_pred = y_pred.reshape(-1, 1)
                y_pred = sm.add_constant(y_pred)  # intercept column
                y_pred = platt_scaler.predict_proba(y_pred)[:, 1]
                y_pred = torch.tensor(y_pred, device=xs_combined[0].device).unsqueeze(1).float()
            if self.option == 'binary' or self.option == 'side_effects':
                if self.option == 'binary':
                    loss = self.loss_fn(y_pred, y)
                    auroc = self.auroc(y_pred, y.int())
                    accuracy = self.accuracy(y_pred, y.int())
                    precision = self.precision(y_pred, y.int())
                    recall = self.recall(y_pred, y.int())
                    auprc = self.auprc(y_pred, y.int())
                    self.log('test_auroc', auroc, prog_bar=True)
                else:
                    time_to_event = torch.ones_like(y_pred).float().to(y_pred.device)
                    res_dict = self._calculate_weighted_c_index_by_infusion(y_pred, y, time_to_event, num_infusion, mode='test')
                    if res_dict is None:
                        return None
                    loss = res_dict.get('binary_loss', torch.tensor(0.0, device=y_pred.device))
                    weighted_auroc, overall_auroc, accuracy, precision, recall, auprc = res_dict.get('auroc_next_infusion_weighted'), res_dict.get('auroc_next_infusion_overall'), res_dict.get('accuracy_next_infusion_weighted'), res_dict.get('precision_next_infusion_weighted'), res_dict.get('recall_next_infusion_weighted'), res_dict.get('auprc_next_infusion_weighted')
                    if self.test_metrics_by_infusion_group:
                        per_infusion_auroc = res_dict.get('per_infusion_auroc')
                        for res in per_infusion_auroc:
                            infusion_id = res['infusion']
                            self.log(f'test_auroc_infusion_{infusion_id}', res['auroc'], prog_bar=False)
                            self.log(f'infusion_num_samples_{infusion_id}', res['num_samples'], prog_bar=False)

                    self.log('test_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
                    self.log('test_auroc_nextinf_overall', overall_auroc, prog_bar=True)
                self.log('test_accuracy', accuracy, prog_bar=True)
                self.log('test_precision', precision, prog_bar=True)
                self.log('test_recall', recall, prog_bar=True)
                self.log('test_auprc', auprc, prog_bar=True)
            else:
                # side_effects_risk / side_effects_next_risk
                y_event, time_to_event = torch.tensor(y[0]).to(y_pred.device), torch.tensor(y[1]).float().to(y_pred.device)
                loss = cox.neg_partial_log_likelihood(y_pred, y_event, time_to_event, ties_method = "efron")

                c_idx_dict = self._calculate_weighted_c_index_by_infusion(y_pred, y_event, time_to_event, num_infusion, mode='test')
                if c_idx_dict is None:
                    return None
                c_index, c_index_weighted, overall_c_index = c_idx_dict['c_index_raw'], c_idx_dict['c_index_weighted'], c_idx_dict['c_index_overall']
                weighted_auroc, overall_auroc = c_idx_dict['auroc_next_infusion_weighted'], c_idx_dict['auroc_next_infusion_overall']
                weighted_auprc, overall_auprc = c_idx_dict['auprc_next_infusion_weighted'], c_idx_dict['auprc_next_infusion_overall']
                self.log('test_cindex_unfiltered', c_index, prog_bar=True)
                self.log('test_cindex_weighted', c_index_weighted, prog_bar=True)
                self.log('test_cindex_overall_valid', overall_c_index, prog_bar=True)
                self.log('test_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
                self.log('test_auroc_nextinf_overall', overall_auroc, prog_bar=True)
                self.log('test_auprc_nextinf_weighted', weighted_auprc, prog_bar=True)
                self.log('test_auprc_nextinf_overall', overall_auprc, prog_bar=True)

        elif 'risk' in self.option:
            y_pred_logits = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined)
            fail_indicators = torch.tensor(y[0]).to(y_pred_logits.device)
            time_to_events = torch.tensor(y[1]).float().to(y_pred_logits.device)
            loss = cox.neg_partial_log_likelihood(y_pred_logits, fail_indicators, time_to_events, ties_method = "efron")
            c_index = self.c_index(y_pred_logits, fail_indicators, time_to_events)
            self.log('test_c_index', c_index, prog_bar=True)

        self.log('test_loss', loss, prog_bar=True)
        return loss