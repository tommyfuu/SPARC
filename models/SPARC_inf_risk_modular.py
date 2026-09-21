import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import AUROC, Accuracy, Precision, Recall, F1Score, AveragePrecision
import numpy as np
# from MSK_featurename_list import *
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
        encoding = encoding.unsqueeze(0)  # Add batch dimension
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


class ContinuousTimeEncoding(nn.Module):
    """
    Learned Fourier-feature encoding of *actual* time values (e.g. days since
    treatment start), as opposed to integer visit-index positional encoding.

    Two visits a few days apart get similar embeddings; visits far apart get
    very different ones. Works with irregular spacing and variable #visits.
    The embedding is ADDED to each visit token (BERT-style), and the raw time
    scalar is also kept as an input feature (handled elsewhere).
    """
    def __init__(self, d_model, n_freqs=None, learnable=True):
        super(ContinuousTimeEncoding, self).__init__()
        if n_freqs is None:
            n_freqs = d_model // 2
        self.n_freqs = n_freqs
        self.d_model = d_model
        # Initialize frequencies on a log scale (like sinusoidal PE) but learnable
        init_freq = torch.exp(
            torch.arange(0, n_freqs, dtype=torch.float) * (-math.log(10000.0) / max(n_freqs, 1))
        )
        if learnable:
            self.freq = nn.Parameter(init_freq)
            self.phase = nn.Parameter(torch.zeros(n_freqs))
        else:
            self.register_buffer('freq', init_freq)
            self.register_buffer('phase', torch.zeros(n_freqs))
        # Project the 2*n_freqs sin/cos features to d_model if needed
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, t):
        """
        Args:
            t: (batch, seq_len) tensor of time values (NaNs should be pre-filled)
        Returns:
            (batch, seq_len, d_model) time embedding
        """
        t = torch.nan_to_num(t, nan=0.0).unsqueeze(-1)          # (B, T, 1)
        angles = t * self.freq + self.phase                      # (B, T, n_freqs)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,T,2F)
        return self.proj(feats)                                  # (B, T, d_model)


class PerModuleTokenizer(nn.Module):
    """
    Tokenizes each visit into one token per clinical module (erythroid, myeloid,
    renal, hepatic, vitals, ...) plus tokens for static features and time, then
    sums them into a single d_model visit token (or returns per-module tokens).

    Crucially, missingness is used to GATE each module projection: a module whose
    constituent labs are all missing contributes ~0 to the visit token, rather
    than a learned "missing" sentinel that competes with real signal. This is the
    quality-gate idea applied at the input.

    Expects the assembled base feature layout (per timestep), WITHOUT the
    missingness-doubling / padding columns:
        [ dynamic_feats (labs[+vitals]) | static_feats | time ]
    and a matching missingness mask of the same width (1=observed, 0=missing).
    """
    def __init__(self, module_index_cfg, d_model, dropout=0.1,
                 use_time_encoding=True, n_static=0):
        super(PerModuleTokenizer, self).__init__()
        self.d_model = d_model
        self.modules_cfg = module_index_cfg['modules']  # list of (name, [idx])
        self.static_start = module_index_cfg.get('static_start', None)
        self.n_static = n_static
        self.use_time_encoding = use_time_encoding

        # One small projection per module: (n_feat_in_module * 2) -> d_model
        # (*2 because we feed [values | missingness] for that module's columns)
        self.module_names = []
        self.module_indices = []          # python lists; registered as buffers below
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

        # Static features (gender, age, ctype one-hot, ...) -> one token
        if n_static > 0:
            self.static_projection = nn.Sequential(
                nn.Linear(n_static * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.static_projection = None

        # Continuous-time embedding
        if use_time_encoding:
            self.time_encoding = ContinuousTimeEncoding(d_model)
        else:
            self.time_encoding = None

        # Learned per-module type embedding (so the encoder knows which axis a
        # token came from), added to each module token.
        self.module_type_embedding = nn.Embedding(len(self.modules_cfg) + 1, d_model)

    def forward(self, values, missingness, times):
        """
        Args:
            values:      (B, T, n_base)  feature values, NaNs pre-filled with 0
            missingness: (B, T, n_base)  1=observed, 0=missing (same width)
            times:       (B, T)          time values for continuous-time encoding
        Returns:
            visit_tokens: (B, T, d_model) summed module/static/time token
        """
        B, T, _ = values.shape
        token = torch.zeros(B, T, self.d_model, device=values.device, dtype=values.dtype)

        for i, name in enumerate(self.module_names):
            idx = getattr(self, f'mod_idx_{i}')
            v = values.index_select(-1, idx)         # (B, T, n_mod)
            m = missingness.index_select(-1, idx)    # (B, T, n_mod)
            mod_in = torch.cat([v, m], dim=-1)        # (B, T, 2*n_mod)
            mod_tok = self.module_projections[name](mod_in)   # (B, T, d_model)
            # Quality gate: fraction of this module's features observed at this visit
            gate = m.mean(dim=-1, keepdim=True)               # (B, T, 1) in [0,1]
            mod_tok = mod_tok * gate
            # Add module-type embedding
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

        return token


# class TemporalAttentionAggregation(nn.Module):
#     """Attention-based aggregation over the temporal dimension."""
#     def __init__(self, d_model):
#         super(TemporalAttentionAggregation, self).__init__()
#         # Linear layer to compute attention scores
#         self.attention = nn.Linear(d_model, 1)

#     def forward(self, z, mask):
#         """
#         Args:
#             z: (batch_size, max_time_len, d_model) - transformer output
#             mask: (batch_size, max_time_len) - True for padded, False for valid
#         Returns:
#             z_attended: (batch_size, d_model) - attention-weighted aggregation
#         """
#         # Compute attention scores: (batch_size, max_time_len, 1)
#         attn_scores = self.attention(z)

#         # Mask padded positions with -inf before softmax
#         attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), float('-inf'))

#         # Apply softmax to get attention weights: (batch_size, max_time_len, 1)
#         attn_weights = F.softmax(attn_scores, dim=1)

#         # Weighted sum over time: (batch_size, d_model)
#         z_attended = (z * attn_weights).sum(dim=1)

#         return z_attended


class MissingnessAwareAttentionAggregation(nn.Module):
    """
    Attention-based temporal aggregation that explicitly models data quality
    and missingness patterns. Downweights attention from timesteps with heavy
    missingness and combines attended features with uncertainty-aware statistics.
    """
    def __init__(self, d_model, reduction_ratio=4, max_time_len=20):
        super(MissingnessAwareAttentionAggregation, self).__init__()

        # Attention scoring network - deeper for better temporal pattern learning
        # Add LayerNorm for long sequences (>20 timepoints) to prevent numerical instability
        if max_time_len > 20:
            self.attention_scorer = nn.Sequential(
                nn.Linear(d_model, d_model // reduction_ratio),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(d_model // reduction_ratio, 1),
                nn.LayerNorm(1)  # Stabilize attention scores for long sequences
            )
        else:
            self.attention_scorer = nn.Sequential(
                nn.Linear(d_model, d_model // reduction_ratio),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(d_model // reduction_ratio, 1)
            )

        # Quality gate network - learns to assess representation reliability
        # This will output a quality score per timestep based on the representation
        self.quality_gate = nn.Sequential(
            nn.Linear(d_model, d_model // reduction_ratio),
            nn.ReLU(),
            nn.Linear(d_model // reduction_ratio, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

        # Combine attended representation with statistical features
        # attended + mean + std (3 representations total)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, z, mask):
        """
        Args:
            z: (batch_size, max_time_len, d_model) - transformer output
            mask: (batch_size, max_time_len) - True for padded, False for valid
        Returns:
            z_aggregated: (batch_size, d_model) - missingness-aware aggregated representation
        """
        batch_size, seq_len, d_model = z.shape

        # Compute base attention scores: (batch_size, max_time_len, 1)
        attn_scores = self.attention_scorer(z)

        # Compute quality gates (representation reliability): (batch_size, max_time_len, 1)
        quality_scores = self.quality_gate(z)

        # Modulate attention by quality - timesteps with poor quality get lower attention
        # This implements the "trust complete data more" principle
        modulated_scores = attn_scores * quality_scores

        # Mask padded positions with -inf before softmax
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, max_time_len, 1)
        modulated_scores = modulated_scores.masked_fill(mask_expanded, float('-inf'))

        # Apply softmax to get final attention weights: (batch_size, max_time_len, 1)
        attn_weights = F.softmax(modulated_scores, dim=1)

        # Quality-aware weighted sum over time: (batch_size, d_model)
        z_attended = (z * attn_weights).sum(dim=1)

        # Compute uncertainty-aware statistics
        # Mask out padded positions for statistical computation
        z_masked = z.masked_fill(mask_expanded.expand_as(z), 0)
        valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1)  # (batch_size, 1)

        # Mean over valid positions
        z_mean = z_masked.sum(dim=1) / valid_counts  # (batch_size, d_model)

        # Standard deviation over valid positions (measures temporal variability)
        z_centered = z - z_mean.unsqueeze(1)
        z_centered_masked = z_centered.masked_fill(mask_expanded.expand_as(z), 0)
        z_var = (z_centered_masked ** 2).sum(dim=1) / valid_counts
        z_std = torch.sqrt(z_var + 1e-8)  # (batch_size, d_model)

        # Combine: attended (quality-weighted) + mean (central tendency) + std (variability)
        z_combined = torch.cat([z_attended, z_mean, z_std], dim=-1)  # (batch_size, 3*d_model)

        # Project back to d_model with normalization
        z_aggregated = self.output_projection(z_combined)  # (batch_size, d_model)

        return z_aggregated


class MLPTemporalAggregation(nn.Module):
    """
    MLP-based temporal aggregation for DuETT.
    Combines mean and max pooling, then applies an MLP to learn
    a flexible aggregation function.
    """
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super(MLPTemporalAggregation, self).__init__()

        if hidden_dim is None:
            hidden_dim = d_model

        # MLP to process pooled features
        # Input: concatenated mean and max pooling (2 * d_model)
        # Output: d_model
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, z, mask):
        """
        Args:
            z: (batch_size, max_time_len, d_model) - transformer output
            mask: (batch_size, max_time_len) - True for padded, False for valid
        Returns:
            z_aggregated: (batch_size, d_model) - MLP-aggregated representation
        """
        batch_size, seq_len, d_model = z.shape
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, max_time_len, 1)

        # Mean pooling over valid positions
        z_masked = z.masked_fill(mask_expanded.expand_as(z), 0)
        valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1)  # (batch_size, 1)
        z_mean = z_masked.sum(dim=1) / valid_counts  # (batch_size, d_model)

        # Max pooling over valid positions
        z_max = z.masked_fill(mask_expanded.expand_as(z), float('-inf'))
        z_max, _ = z_max.max(dim=1)  # (batch_size, d_model)

        # Concatenate pooled features
        z_pooled = torch.cat([z_mean, z_max], dim=-1)  # (batch_size, 2*d_model)

        # Apply MLP to learn aggregation
        z_aggregated = self.aggregation_mlp(z_pooled)  # (batch_size, d_model)

        return z_aggregated


def nanstd(x):
    return torch.sqrt(torch.nanmean((x - torch.nanmean(x))**2))

def symmetrical_loss_calc(preds, gt, loss_fn, option = 'pairwise_logistic'):
    pos = preds[gt == 1]
    neg = preds[gt == 0]

    # pairwise differences z_ij = s_pos_i - s_neg_j
    z = pos.unsqueeze(1) - neg.unsqueeze(0)   # shape (P, Q)

    if option == 'pairwise_barrier':
        loss_matrix = loss_fn.barrier_hinge_loss(z)   # element-wise
    elif option == 'pairwise_logistic':
        loss_matrix = F.softplus(-z)   # element-wise logistic loss
    auc_loss = loss_matrix.mean()                 # aggregate over pairs
    return auc_loss

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
                 temporal_aggregation='minmaxmean',  # Options: 'minmaxmean', 'attention', 'mlp', or 'cls' (rep_token style - appended at end)
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
                 n_static_features=0,          # # of trailing static scalars (gender/age/ctype...)
                 post_mlp_dropout=None,         # dropout for post-transformer MLP; defaults to `dropout`
                # pretraining objective variants
                 binary_pos_weight=None,
                 side_effects_pos_weight=None,
                 ):
        super(SPARC, self).__init__()

        ## task/data-specific
        # check to make sure option is in ['pretrain', 'binary', 'risk']
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

        # check if there are keys of modular_featuredim_dict that are not in the list of allowed features
        if [el for el in modular_featuredim_dict.keys() if el not in MSK_numfeature_dict.keys()]:
            raise ValueError('There are keys of modular_featuredim_dict that are not in the list of allowed features')
        # Account for missingness indicators and padding: 2 * (base_features + time) + padding_indicator
        base_features = sum(modular_featuredim_dict.values())
        self.base_features = base_features  # Store for later use in pretrain prediction
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
                    # base_features + 1 (time), then double for missingness, then +1 for padding
                    self.total_feature_dim = 2 * (base_features + 1) + 1
                else:
                    # base_features + 1 (time) + 1 (padding indicator), no missingness doubling
                    self.total_feature_dim = (base_features + 1) + 1
            elif advanced_model == 'duett':
                ts_features = sum([el for k, el in modular_featuredim_dict.items() if k in ['basic_vitals', 'extra_vitals', 'extra_lab', 'basic_lab', 'cbc_differential']])
                non_ts_features = sum([el for k, el in modular_featuredim_dict.items() if k not in ['basic_vitals', 'extra_vitals', 'extra_lab', 'basic_lab', 'cbc_differential']])
                if missing_mask:
                    # DuETT: 2*ts + non_ts, then +1 time, then double for missingness, then +1 padding
                    self.total_feature_dim = 2 * (2 * ts_features + non_ts_features + 1) + 1
                else:
                    self.total_feature_dim = (2 * ts_features + non_ts_features + 1) + 1
        else:
            if advanced_model == 'default':
                if missing_mask:
                    # No time column for non lab/vitals features, just double for missingness
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

        # Per-module tokenizer config
        self.module_index_cfg = module_index_cfg
        self.use_module_tokenizer = use_module_tokenizer and (module_index_cfg is not None)
        self.use_continuous_time = use_continuous_time
        self.n_static_features = n_static_features
        self.post_mlp_dropout = post_mlp_dropout if post_mlp_dropout is not None else dropout

        if self.use_module_tokenizer and advanced_model == 'duett':
            raise NotImplementedError(
                "Per-module tokenizer is not supported with advanced_model='duett'. "
                "Use advanced_model='default'.")


        # Override side_effects_num_infusion for binary tasks (only relevant for side-effect tasks)
        if option in ['binary', 'risk'] and self.side_effects_num_infusion:
            print(f"WARNING: side_effects_num_infusion=True is not applicable for option '{option}'. Setting to False.")
            self.side_effects_num_infusion = False

        # Auto-select temporal_aggregation for DuETT
        if advanced_model == 'duett' and temporal_aggregation != 'cls':
            print(f"Using DuETT model. Automatically setting temporal_aggregation to be 'cls' (rep_token style).")
            temporal_aggregation = 'cls'
        self.temporal_aggregation = temporal_aggregation
        ## pretrain stuff
        self.pretrain_contrastive_weight = pretrain_contrastive_weight
        self.finetune_part = finetune_part
        self.time_calibration = time_calibration

        #############################

        # model modules
        if self.total_feature_dim == 1:
            raise ValueError('no input features provided')
        else:
            # time processing
            self.time_encoding = time_encoding
            if time_encoding == 'pe':
                self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_time_len)
            if advanced_model == 'duett':
                self.time_cve = nn.Sequential(
                    nn.Linear(1, int(np.sqrt((self.total_feature_dim+1)))),
                    nn.Tanh(),
                    nn.Linear(int(np.sqrt((self.total_feature_dim+1))), 1),
                )

            # input projection
            self.input_projection = nn.Sequential(
                nn.Linear(self.total_feature_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )

            # Per-module tokenizer (optional). When enabled it REPLACES the flat
            # input_projection path with one token per clinical module + static +
            # continuous time, with missingness used to gate each module.
            if self.use_module_tokenizer:
                self.per_module_tokenizer = PerModuleTokenizer(
                    self.module_index_cfg, d_model,
                    dropout=dropout,
                    use_time_encoding=self.use_continuous_time,
                    n_static=self.n_static_features,
                )
                print(f"Per-module tokenizer ENABLED: "
                      f"{[n for n, _ in self.module_index_cfg['modules']]}, "
                      f"continuous_time={self.use_continuous_time}, "
                      f"n_static={self.n_static_features}")
            else:
                self.per_module_tokenizer = None
                # Standalone continuous-time encoding for the flat path (added on
                # top of the projected token) if requested.
                if self.use_continuous_time:
                    self.flat_time_encoding = ContinuousTimeEncoding(d_model)
                else:
                    self.flat_time_encoding = None
            # transformer encoder
            if advanced_model == 'duett':
                num_encoder_layers = num_encoder_layers//2
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, norm_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
            self.transformer_layer_norm = nn.LayerNorm(d_model)

            if advanced_model == 'duett':
                # Adjust event transformer d_model to account for rep token if using cls aggregation
                event_d_model = max_time_len + 1 if temporal_aggregation == 'cls' else max_time_len
                ## since max_time_len is typically 20, nhead should be 3 to be a divisor of event_d_model for the event transformer (21)
                self.event_encoder_layer = nn.TransformerEncoderLayer(d_model=event_d_model, nhead=3, dim_feedforward=dim_feedforward, dropout=dropout, norm_first=True)
                self.event_transformer_encoder = nn.TransformerEncoder(self.event_encoder_layer, num_layers=num_encoder_layers)

            # temporal aggregation module
            if temporal_aggregation == 'attention':
                self.temporal_attention = MissingnessAwareAttentionAggregation(d_model, max_time_len=max_time_len)
                aggregation_output_dim = d_model
            elif temporal_aggregation == 'minmaxmean':
                aggregation_output_dim = 3 * d_model
            elif temporal_aggregation == 'mlp':
                self.temporal_mlp = MLPTemporalAggregation(d_model, hidden_dim=d_model, dropout=dropout)
                aggregation_output_dim = d_model
            elif temporal_aggregation == 'cls':
                # CLS token: learnable token appended to sequence (similar to rep_token in DuETT)
                # Use special embeddings module for initialization
                self.special_embeddings = nn.Embedding(2, d_model)  # 2 embeddings: one for masked, one for representation
                self.register_buffer("REPRESENTATION_EMBEDDING_KEY", torch.tensor(1))
                aggregation_output_dim = d_model
            else:
                raise ValueError(f"temporal_aggregation must be 'minmaxmean', 'attention', 'mlp', or 'cls', got {temporal_aggregation}")

            # output projection
            self.post_transformer_to_representation_MLP = nn.Sequential(
                nn.Linear(aggregation_output_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(self.post_mlp_dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            # last time point prediction (predicts only base features, not expanded with missingness indicators)
            self.representation_MLP_to_last = nn.Sequential(
                nn.Linear(d_model+1, self.base_features),
            )
            # representation to binary classification
            if self.total_feature_dim != 1:
                self.representation_to_binary_MLP = nn.Linear(d_model, 1)
                # if finetune_part: only representation_to_binary_MLP should have requires_grad=True
                if option != 'pretrain' and finetune_part:
                    print("finetune_part, only representation_to_binary_MLP should have requires_grad=True")
                    for param in self.parameters():
                        param.requires_grad = False
                    for param in self.representation_to_binary_MLP.parameters():
                        param.requires_grad = True

            # if med_as_feature is True, two MLPs
            if med_as_feature:
                # MLP to map the representation to the medication features
                self.meds_to_single_MLP = nn.Sequential(
                    nn.Linear(len(meds_l), 1),
                    nn.ReLU()
                )
                # MLP to combine the representation and the medication features
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
                # MLP to combine the representation and the num_infusion features
                self.representation_and_infusionnum_to_representation_MLP = nn.Sequential(
                    nn.Linear(d_model+add_num, d_model),
                    nn.ReLU()
                )

        # loss functions
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
                # -wbce False leaves side_effects_pos_weight None -> pos_weight 1.0,
                # which is plain BCE. Checkpoints store no hyperparameters, so the
                # flag is the only record of which loss a run used.
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
        ### TS TRANSFORMER SECTION
        if self.total_feature_dim != 1:
            batch_size, seq_len, _ = src.size() ## src dim: (batch_size, max_time_len, d_features + 1)
            if self.advanced_model == 'duett':
                # on the time dim, do the time_cve
                cved_time = self.time_cve(src[:, :, -2].unsqueeze(-1)).squeeze(-1) # src dim: (batch_size, max_time_len, d_features + 1)
                src = src.clone()
                src[:, :, -2] = cved_time
            # Project inputs to model dimension
            if self.per_module_tokenizer is not None:
                # Deconstruct the assembled flat vector back into parts.
                # Layout (missing_mask=True): [base(n_base) | time(1) | miss_base(n_base) | miss_time(1) | padding(1)]
                # Layout (missing_mask=False): [base(n_base) | time(1) | padding(1)] (NaNs -> -10 sentinel)
                n_base = self.base_features
                values = src[:, :, :n_base]
                times = src[:, :, n_base]
                if self.missing_mask:
                    missingness = src[:, :, n_base + 1: 2 * n_base + 1]
                else:
                    # Reconstruct missingness from the -10 sentinel used in data_prep
                    missingness = (values != -10.0).float()
                    values = torch.where(values == -10.0, torch.zeros_like(values), values)
                src = self.per_module_tokenizer(values, missingness, times)  # (B, T, d_model)
            else:
                # Capture raw time (column n_base) before projecting, for optional
                # continuous-time embedding on the flat path.
                if getattr(self, 'flat_time_encoding', None) is not None:
                    raw_times = src[:, :, self.base_features]
                src = self.input_projection(src)  ## src dim (batch_size, max_time_len, d_model)
                if getattr(self, 'flat_time_encoding', None) is not None:
                    src = src + self.flat_time_encoding(raw_times)

            # If using CLS token, append it to the sequence (rep_token style)
            if self.temporal_aggregation == 'cls':
                batch_size = src.size(0)
                # Get representation embedding token: (batch_size, 1, d_model)
                rep_tokens = self.special_embeddings(self.REPRESENTATION_EMBEDDING_KEY).unsqueeze(0).expand(batch_size, -1, -1)
                # Append rep token: (batch_size, max_time_len+1, d_model)
                src = torch.cat([src, rep_tokens], dim=1)

                # Update padding mask to include rep token (rep token is never masked)
                if src_key_padding_mask is not None:
                    # src_key_padding_mask is (max_time_len, batch_size)
                    # Add False for rep token position (not masked)
                    rep_mask = torch.zeros(1, batch_size, dtype=torch.bool, device=src_key_padding_mask.device)
                    src_key_padding_mask = torch.cat([src_key_padding_mask, rep_mask], dim=0)
                    # Now shape: (max_time_len+1, batch_size)

            # add positional encoding
            if self.advanced_model == 'default' and self.time_encoding == 'pe':
                src = self.positional_encoding(src) # by default this is not used because we have time as a feature

            # src dim: (batch_size, max_time_len, d_model) or (batch_size, max_time_len+1, d_model) if CLS
            if self.advanced_model == 'default':
                z = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask) # z dim (batch_size, max_time_len, d_model)
                # Small batches can cause LayerNorm instability - avoid
                if torch.isnan(z).any():
                    z = torch.nan_to_num(z, nan=0.0)
                z = self.transformer_layer_norm(z)

            # duett: + event transformer
            if self.advanced_model == 'duett':
                for layer_i, (event_layer, time_layer) in enumerate(zip(self.event_transformer_encoder.layers, self.transformer_encoder.layers)):
                    # Rep token flows through both transformers naturally (no separation needed)
                    # src shape: (batch_size, max_time_len, d_model) or (batch_size, max_time_len+1, d_model) if cls
                    src = src.permute(0, 2, 1) # src dim (batch_size, d_model, max_time_len) or (batch_size, d_model, max_time_len+1)
                    src = event_layer(src) # src dim (batch_size, d_model, max_time_len) or (batch_size, d_model, max_time_len+1)
                    src = src.permute(0, 2, 1) # src dim (batch_size, max_time_len, d_model) or (batch_size, max_time_len+1, d_model)
                    src = time_layer(src, src_key_padding_mask=src_key_padding_mask) # src dim (batch_size, max_time_len, d_model) or (batch_size, max_time_len+1, d_model)

                z = src
                z = self.transformer_layer_norm(z)

            # Temporal aggregation
            if self.temporal_aggregation == 'cls':
                # Extract rep token (last position) - no need for masking or pooling
                # z shape: (batch_size, max_time_len+1, d_model)
                z = z[:, -1, :]  # z dim: (batch_size, d_model)
            else:
                # For other aggregation methods, need to handle masking
                if src_key_padding_mask is not None:
                    # src_key_padding_mask is of shape (max_time_len, batch_size) or (max_time_len+1, batch_size) if CLS
                    # We need to transpose it to (batch_size, max_time_len) and unsqueeze to (batch_size, max_time_len, 1)
                    mask = src_key_padding_mask.T.unsqueeze(-1).bool()
                else:
                    # No padding mask provided - all positions are valid (False = valid, True = padded)
                    mask = torch.zeros((batch_size, seq_len, 1), dtype=torch.bool, device=z.device)

                if self.temporal_aggregation == 'minmaxmean':
                    z_max = z.masked_fill(mask, float('-inf'))  # Mask padded positions with -inf for max
                    z_max_over_time, _ = torch.max(z_max, dim=1)  # Max along the `max_time_len` dimension
                    z_min = z.masked_fill(mask, float('inf'))  # Mask padded positions with +inf for min
                    z_min_over_time, _ = torch.min(z_min, dim=1)  # Min along the `max_time_len` dimension

                    z_sum = z.masked_fill(mask, 0).sum(dim=1)  # Sum over `max_time_len`
                    valid_counts = (~mask).sum(dim=1).clamp(min=1)  # Count VALID (non-padded) entries per batch
                    z_mean_over_time = z_sum / valid_counts  # Element-wise division
                    z = torch.cat([z_max_over_time, z_min_over_time, z_mean_over_time], dim=-1) # z dim (batch_size, 3*d_model)
                elif self.temporal_aggregation == 'attention':
                    z = self.temporal_attention(z, mask.squeeze(-1))  # z dim (batch_size, d_model)
                elif self.temporal_aggregation == 'mlp':
                    z = self.temporal_mlp(z, mask.squeeze(-1))  # z dim (batch_size, d_model)

            # Pass through MLP to map to representation dimension
            z = self.post_transformer_to_representation_MLP(z)  # z dim: (batch_size, d_model)

            # if med_as_feature is True, we need to process the medication features
            if self.med_as_feature:
                meds_data_representation = self.meds_to_single_MLP(meds_data)  # meds_data_representation dim: (batch_size, 1)
                # concatenate the representation with the medication features
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
            # concatenate the last time point representation with the last time point time as input
            z1 = self.representation_MLP_to_last(torch.cat([z, last_tp_times.unsqueeze(-1)], dim=-1)) # z for contrastive learning loss, z1 for last token prediction loss
            # z1 dim (batch_size, total_feature_dim-1)
            return z, z1
        elif self.option == 'pretrain' and self.usage != 'standalone':
            # just directly use as features (plugin)
            return z

        # if the goal is binary classification after pretraining layers or risk prediction
        elif self.option in ['binary', 'risk', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            z = self.representation_to_binary_MLP(z) # z dim (batch_size, 1)

            # Clamp logits to prevent sigmoid saturation
            z = torch.clamp(z, min=-10.0, max=10.0)

            if self.use_sigmoid:
                z = torch.sigmoid(z)

            return z

    @classmethod # TOCHECK - see if it works
    def custom_load_from_checkpoint(cls, checkpoint_path, load_w_meds = False, **kwargs):
        """
        Load the model from a checkpoint file.
        """
        print("Loading model from checkpoint using custom loader:", checkpoint_path)
        model = cls(**kwargs)
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Filter out med_as_feature-specific layers if they exist in current model
        missing_keys = ['meds_to_single_MLP', 'medandrepresentation_to_representation_MLP']
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not any(key in k for key in missing_keys)
        }

        model.load_state_dict(filtered_state_dict, strict=False)
        return model

    @classmethod # TOCHECK - see if it works
    def custom_load_from_checkpoint_sideeffect(cls, checkpoint_path, **kwargs):
        """
        Load the model from a checkpoint file.
        """
        print("Loading model from checkpoint using custom loader:", checkpoint_path)
        model = cls(**kwargs)
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Filter out med_as_feature-specific layers if they exist in current model
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
                # num_infusion, DMP_ID = None, None  # Initialize for pretraining
            else:
                if not self.med_as_feature and self.side_effects_num_infusion:
                    # TODO: just added DMP_ID here for side_effects option, to be removed later
                    if not self.include_gcsf and not self.include_steroid:
                        (xs, times, num_infusion, DMP_ID), y = data
                    elif self.include_gcsf and not self.include_steroid:
                        (xs, times, num_infusion, DMP_ID, gcsf_groundtruth), y = data
                    elif not self.include_gcsf and self.include_steroid:
                        (xs, times, num_infusion, DMP_ID, steroid_groundtruth), y = data
                    else:
                        # error case for now
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

        # turn times such that it starts from 0 instead
        if time_calibration:
            times = times - times[:, 0].unsqueeze(-1)

        ### TS TRANSFORMER DATA SECTION
        if self.normalization:
            for i in range(xs.size(-1)):
                mean = xs[:, :, i].nanmean()
                std = nanstd(xs[:, :, i])
                xs[:, :, i] = (xs[:, :, i] - mean) / std

        # # concatenate xs and times
        # if 'basic_lab' in self.modular_featuredim_dict.keys() or 'extra_lab' in self.modular_featuredim_dict.keys() or 'vital' in self.modular_featuredim_dict.keys():
        #     xs = torch.cat([xs, times.unsqueeze(-1)], dim=-1)

        # perturb a feature given the index if necessary, perturb it to randomly generated values at normal distribution around (mu=0, std=1) per z-score normalization
        if perturbation_index != '':
            if perturbation_index >= xs.size(-1):
                raise ValueError('perturbation_index out of range')

            ## we fetch all non-nan values from that perturbation_index and randomly place them back
            # extract the slice and flatten
            slice_ = xs[:, :, perturbation_index]
            mask = ~torch.isnan(slice_)
            values = slice_[mask]

            # permute the non-NaN values
            perm = values[torch.randperm(values.size(0))]

            # create new slice and assign permuted values
            new_slice = slice_.clone()
            new_slice[mask] = perm
            xs[:, :, perturbation_index] = new_slice

        if self.option == 'pretrain' and self.usage == 'standalone':
            # the data has already been naturally order, the left most is the last time point
            last_tp_xs_to_predict = xs[:, 0, :]  # extract last timepoint features
            last_tp_times = times[:, 0]  # extract last timepoint time
            xs = xs[:, 1:, :]  # rest of the time points
            times = times[:, 1:]  # also slice times to match xs dimensions
            # Concatenate time to last_tp features so it can be split later
            if 'basic_lab' in self.modular_featuredim_dict.keys() or 'extra_lab' in self.modular_featuredim_dict.keys() or 'vital' in self.modular_featuredim_dict.keys():
                last_tp_xs_to_predict = torch.cat([last_tp_xs_to_predict, last_tp_times.unsqueeze(-1)], dim=-1)
        else:
            last_tp_xs_to_predict = None

        # Concatenate xs and times before creating mask
        if 'basic_lab' in self.modular_featuredim_dict.keys() or 'extra_lab' in self.modular_featuredim_dict.keys() or 'vital' in self.modular_featuredim_dict.keys():
            xs = torch.cat([xs, times.unsqueeze(-1)], dim=-1)

        # Now create mask from xs+times combined
        mask = torch.isnan(xs) # xs.shape (batch_size, max_time_len, feature_dim+1); True if the value is NaN
        src_key_padding_mask = mask[:, :, -1].T # dim: (max_time_len, batch_size); True if the time point is padded

        if self.missing_mask:
            # Concatenate missingness indicators: 1 if present, 0 if missing (for ALL features including time)
            # This doubles the feature dimension: [features+time, missingness_indicators]
            # Fill NaN with 0 — the indicator tells the model which 0s are imputed vs real.
            missingness_indicators = (~mask).float()
            xs[mask] = 0.0
            xs = torch.cat([xs, missingness_indicators], dim=-1)
        else:
            # No missingness indicator: fill NaN with -10 (a clear out-of-distribution sentinel
            # in z-score space, so the model treats it as missing rather than as the mean).
            xs[mask] = -10.0

        # attach the src_key_padding_mask as a feature so as to tell the model which time points are padded
        padding_indicator = torch.tensor(~src_key_padding_mask).unsqueeze(-1).permute(1, 0, 2)

        # calculate # of non-zero time points per patient
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

            # Skip if target has no valid values
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
            for i in range(min(3, xs_last_pred.shape[0])):  # Print first 3 samples
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
        """
        Calculate weighted C-index stratified by infusion number.
        Only includes infusions where C-index can be computed.

        Returns:
            dict with keys: c_index_weighted, c_index_overall, num_groups, num_samples,
                        auroc metrics, and optionally binary_loss (as tensor for backprop)
        """
        output_dict = {}
        num_infusion_tensor = torch.as_tensor(num_infusion, device=y_pred.device)
        y_event_tensor = torch.as_tensor(y_event, device=y_pred.device)
        time_to_event_tensor = torch.as_tensor(time_to_event, device=y_pred.device)

        unique_infusions = torch.unique(num_infusion_tensor)
        risk_scores = y_pred.squeeze()

        # ========================================================================
        # Part 1: Risk prediction metrics (C-index)
        # ========================================================================
        if 'risk' in self.option:
            # First pass: identify valid infusions (has both events and censoring)
            valid_infusions = []
            for infusion in unique_infusions:
                mask = num_infusion_tensor == infusion
                current_events = y_event_tensor[mask]

                if current_events.sum() > 0 and current_events.sum() < len(current_events):
                    valid_infusions.append(infusion)

            # Calculate raw C-index (no stratification)
            try:
                c_idx_raw = self.c_index(
                    risk_scores.unsqueeze(1),
                    y_event_tensor.bool(),
                    time_to_event_tensor
                ).item()
            except Exception:
                c_idx_raw = 0.5

            # Initialize with defaults
            c_index_weighted = 0.5
            overall_c_index = 0.5
            num_groups = 0
            total_samples = 0
            computable_results = []
            if len(valid_infusions) > 0:
                # Second pass: compute C-index for valid infusions
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
                # Calculate weighted C-index
                total_samples = sum(r['num_samples'] for r in computable_results)
                c_index_weighted = sum(
                    r['c_index'] * r['num_samples'] / total_samples
                    for r in computable_results
                )
                # Calculate overall C-index
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
        # ========================================================================
        # Part 2: Next-infusion side effect prediction (AUROC)
        # ========================================================================

        # Create binary labels: 1 if side effect occurs in next infusion, 0 otherwise
        ### flattened tensors
        time_to_event_tensor = time_to_event_tensor.flatten()
        next_infusion_event = ((y_event_tensor == 1) & (time_to_event_tensor == 1)).float()
        # next_infusion_event = ((y_event_tensor == 1) & (time_to_event_tensor == 0)).float()

        # # DEBUG: Print statistics about the labels
        # if mode in ['test', 'val']:
        #     n_y_event_1 = (y_event_tensor == 1).sum().item()
        #     n_time_1 = (time_to_event_tensor == 1).sum().item()
        #     n_next_inf_1 = next_infusion_event.sum().item()
            # print(f"[{mode}] AUROC labels: y_event=1: {n_y_event_1}, time_to_event=1: {n_time_1}, next_infusion_event=1: {n_next_inf_1}, total: {len(next_infusion_event)}")

        valid_infusions_auroc = []
        for infusion in unique_infusions:
            mask = num_infusion_tensor == infusion
            current_next_events = next_infusion_event[mask]

            # Valid if has both positives and negatives
            has_positives = current_next_events.sum() > 0
            has_negatives = current_next_events.sum() < len(current_next_events)

            if has_positives and has_negatives:
                valid_infusions_auroc.append(infusion)

        # Initialize defaults
        auroc_next_weighted = 0.5
        auroc_next_overall = 0.5
        accuracy_next_weighted = 0.5
        precision_next_weighted = 0.5
        recall_next_weighted = 0.5
        auprc_next_weighted = 0.5
        auprc_next_overall = 0.5
        num_groups_auroc = 0
        num_samples_auroc = 0
        binary_loss_weighted = None  # Will be tensor if binary mode

        if len(valid_infusions_auroc) > 0:
            # Second pass: compute AUROC for valid infusions
            computable_auroc_results = []

            # calculate binary loss across all samples for vectorized comparison
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
                        # Calculate metrics
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

                    # Calculate binary loss (keep as tensor for backprop!)
                    if 'risk' not in self.option:
                        binary_loss = all_sample_losses[mask].mean()
                        current_res['binary_loss'] = binary_loss  # Keep as tensor

                    computable_auroc_results.append(current_res)

                except Exception as e:
                    print(f"Skipping AUROC for infusion {infusion.item()}: {e}")
                    continue

            if len(computable_auroc_results) > 0:
                # Calculate weighted metrics
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
                        # Next-infusion prediction metrics
                        'auroc_next_infusion_weighted': auroc_next_weighted,
                        'accuracy_next_infusion_weighted': accuracy_next_weighted,
                        'precision_next_infusion_weighted': precision_next_weighted,
                        'recall_next_infusion_weighted': recall_next_weighted,
                        'auprc_next_infusion_weighted': auprc_next_weighted,
                        'num_groups_next_infusion': num_groups_auroc,
                        'per_infusion_auroc': computable_auroc_results
                    })

                # Calculate weighted binary loss (keep as tensor!)
                if 'risk' not in self.option:
                    binary_loss_weighted = torch.stack([
                        r['binary_loss'] * r['num_samples'] / total_samples_auroc
                        for r in computable_auroc_results
                    ]).sum()
                if mode != 'train':
                    # Calculate overall AUROC across all valid samples
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

                    # Calculate overall AUPRC across all valid samples
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
                return None # to be skipped in trainer loop so that no backprop is done
            else:
                output_dict.update({
                    # Next-infusion prediction metrics
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
        # Add binary loss as tensor (for backprop) if in binary mode
        if 'risk' not in self.option and binary_loss_weighted is not None:
            output_dict['binary_loss'] = binary_loss_weighted  # TENSOR, not .item()!
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
            # apply platt scaling when available
            platt_scaler = getattr(self, 'platt_scaler', None)
            if platt_scaler is not None:
                y_pred = y_pred.cpu().detach().numpy()  # move to CPU and convert to numpy for platt scaler
                y_pred = y_pred.reshape(-1, 1)  # reshape if necessary for platt scaler
                y_pred = sm.add_constant(y_pred)  # add intercept term for platt scaler
                y_pred = platt_scaler.predict_proba(y_pred)[:, 1]  # get probability of positive class
                # push back to same device
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

    ### BELOW IS ALL THE PYTORCH LIGHTNING THINGIS
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        print("====TORCH VERSION=====", torch.__version__)

        # The original code used a 'scheduler' key instead of 'lr_scheduler',
        # which caused PyTorch Lightning to silently ignore the ReduceLROnPlateau scheduler.
        # The model actually trained better with constant LR, so we're keeping that behavior.
        # To use a scheduler, uncomment one of the options below:

        if self.schedule == 'default':
            # Option 1: No scheduler (constant LR - matches original working behavior)
            return optimizer

        elif self.schedule == 'exponential':
            # Option 2: Gentle ExponentialLR decay (uncomment to use)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }
            }
        else:
            # Option 3: ReduceLROnPlateau with gentler settings (uncomment to use)
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

            # Compute base pretraining loss (MSE on autoregressive/delta target)
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
                    # Calculate BCE loss for current infusion event prediction
                    # Current infusion event: y_event==1 AND time_to_event==1 (event at next infusion)
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

        # Periodic GPU cache clearing to prevent memory fragmentation slowdown
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
            # print(f"DEBUG validation_step: before split last_tp_xs_to_predict shape={last_tp_xs_to_predict.shape}")
            last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]
            # print(f"DEBUG validation_step: after split last_tp_xs_to_predict shape={last_tp_xs_to_predict.shape}, last_tp_times shape={last_tp_times.shape}")
            xs_rep, xs_last_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times, num_infusion=num_infusion)
            # print(f"DEBUG validation_step: xs_last_pred shape={xs_last_pred.shape}")

            # Compute base pretraining loss
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
                    # Calculate BCE loss for current infusion event prediction
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

        # Periodic GPU cache clearing to prevent memory fragmentation slowdown
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def load_state_dict(self, state_dict, strict=True):
        """Load non-strictly, tolerating benign key mismatches between a
        checkpoint and the current architecture."""
        return super().load_state_dict(state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint):
        """Drop optimizer state when a checkpoint's architecture does not match."""
        # For checkpoints predating an architectural change,
        # drop the optimizer state completely to avoid parameter mismatch errors.
        # The model weights and epoch counter will be preserved, but optimizer starts fresh.
        if 'optimizer_states' in checkpoint:
            # Check if there's a parameter count mismatch
            num_current_params = len(list(p for p in self.parameters() if p.requires_grad))

            # Try to get the old parameter count from optimizer state
            has_mismatch = False
            for opt_state in checkpoint['optimizer_states']:
                if 'state' in opt_state and len(opt_state['state']) > 0:
                    # Get max parameter index in old optimizer state
                    max_old_idx = max(k for k in opt_state['state'].keys() if isinstance(k, int))
                    if max_old_idx >= num_current_params:
                        has_mismatch = True
                        break

            if has_mismatch:
                # Drop optimizer state completely - safest approach
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
                y_pred = y_pred.cpu().detach().numpy()  # move to CPU and convert to numpy for platt scaler
                y_pred = y_pred.reshape(-1, 1)  # reshape if necessary for platt scaler
                y_pred = sm.add_constant(y_pred)  # add intercept term for platt scaler
                y_pred = platt_scaler.predict_proba(y_pred)[:, 1]  # get probability of positive class
                # push back to same device
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
                        # log per-infusion group metrics
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
                # it's now a risk prediction task with side effects as outcome
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