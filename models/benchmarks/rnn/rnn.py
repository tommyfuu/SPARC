import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import AUROC, Accuracy, Precision, Recall, F1Score, ROC, AveragePrecision
import torch
import torch.nn as nn
import time
import numpy as np
from itertools import combinations
from MSK_featurename_list import *
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex
import math
import re



def nanstd(x): 
    return torch.sqrt(torch.nanmean((x - torch.nanmean(x))**2))

# # def deephit_log_likelihood_loss(predictions, time_to_events, fail_indicators, categories, num_categories):
# def deephit_log_likelihood_loss(predictions, time_to_events, fail_indicators):
#     """
#     DeepHit negative log-likelihood loss function
    
#     Args:
#         predictions: torch.Tensor [batch_size, num_Event, num_Category]
#             - Predicted PMF (probability mass function) over time bins/categories
#             - Should be output of softmax (probabilities sum to ~1)
#             - num_Event: number of competing risk types
#             - num_Category: number of discrete time bins/categories
        
#         time_to_events: torch.Tensor [batch_size]
#             - The time bin index when event occurred or censoring happened
#             - Integer values from 0 to num_categories-1
#             - e.g., [4, 7, 2, ...] means patient 1 had event at bin 4, etc.
        
#         fail_indicators: torch.Tensor [batch_size]
#             - Which risk type occurred (0 = censored, 1 = risk type 1, 2 = risk type 2, etc.)
#             - Integer values from 0 to num_Event
#             - e.g., [1, 0, 2, ...] means patient 1 had risk type 1, patient 2 censored, patient 3 had risk type 2
        
#         categories: torch.Tensor [batch_size, num_Category]
#             - One-hot encoded vector of which time bin the event occurred
#             - All zeros if censored
#             - e.g., [0,0,0,0,1,0,0] means event at bin 4 (0-indexed)
        
#         num_categories: int
#             - Total number of time bins/categories
    
#     Returns:
#         torch.Tensor: Scalar loss value (negative log-likelihood)
    
#     # note: no need for rank loss since no competing risks in this implementations
#     """
#     ####### task specific
#     num_categories = 4
#     categories = torch.zeros_like(time_to_events).long().to(predictions.device) 
#     categories[(time_to_events >=1) & (time_to_events <=3)] = 0
#     categories[(time_to_events >=4) & (time_to_events <=10)] = 1
#     categories[(time_to_events >=11) & (time_to_events <=20)] = 2
#     categories[time_to_events >20] = 3
#     ## one hot encode categories to batch_size x num_categories
#     categories = nn.functional.one_hot(categories, num_classes=num_categories).float().to(predictions.device)
#     print("categories", categories.shape)
#     for i in range(categories.shape[0]):
#         print("categories[i]", categories[i], "time_to_events", time_to_events[i])
#     ####### task specific
#     batch_size = predictions.shape[0]
#     num_Event = predictions.shape[1]
    
#     # Convert fail_indicators to one-hot encoding for mask1
#     # mask1: [batch_size, num_Event] - which risk type occurred (or all zeros if censored)
#     mask1 = torch.zeros(batch_size, num_Event, device=predictions.device)
#     for i in range(batch_size):
#         if fail_indicators[i] > 0:  # Not censored
#             risk_idx = int(fail_indicators[i]) - 1  # Convert to 0-indexed
#             mask1[i, risk_idx] = 1.0
    
#     # Create mask2: survival mask indicating which time bins patient was at risk
#     # mask2: [batch_size, num_Category] - 1 for bins where patient was at risk
#     mask2 = torch.zeros(batch_size, num_categories, device=predictions.device)
#     for i in range(batch_size):
#         event_time_idx = int(time_to_events[i])
#         mask2[i, :event_time_idx + 1] = 1.0  # At risk from bin 0 to event_time_idx
    
#     # -------------------------------------------------------------------------
#     # PART 1: Log-likelihood for OBSERVED EVENTS
#     # -------------------------------------------------------------------------
    
#     # Sum probabilities across all risk types for each time bin
#     # I_1: [batch_size, num_Category]
#     # I_1[i, t] = P(any event type at time bin t | patient i)
#     I_1 = torch.sum(predictions, dim=1)
    
#     # Element-wise multiply with ground truth categories (one-hot encoded)
#     # This isolates the probability at the actual event time
#     # tmp1: [batch_size, 1]
#     tmp1 = torch.sum(I_1 * categories, dim=1, keepdim=True)
#     # tmp1[i] = P(event at actual time bin | patient i) for those with events
    
#     # Apply mask1 to only include patients with observed events
#     # Take log for log-likelihood
#     # Add small epsilon for numerical stability
#     tmp1 = mask1 * torch.log(tmp1 + 1e-8)
#     # Shape: [batch_size, num_Event]
#     # Non-zero only for patients with observed events
    
#     # -------------------------------------------------------------------------
#     # PART 2: Log-likelihood for CENSORED PATIENTS
#     # -------------------------------------------------------------------------
    
#     # Sum all probabilities across risk types and time bins
#     # tmp2: [batch_size, 1, 1]
#     # tmp2[i] = Total probability mass = sum of P(event at any time, any risk)
#     tmp2 = torch.sum(predictions, dim=[1, 2], keepdim=True)
    
#     # For censored patients: likelihood is P(T > censoring time)
#     # P(T > c) = 1 - P(T <= c) = 1 - sum(all allocated probabilities)
#     # (1.0 - mask1): identifies censored patients (where mask1 is all zeros)
#     tmp2 = (1.0 - mask1) * torch.log(1.0 - tmp2 + 1e-8)
#     # Shape: [batch_size, num_Event, 1]
#     # Non-zero only for censored patients
    
#     # Squeeze to match dimensions
#     tmp2 = tmp2.squeeze(dim=2)  # [batch_size, num_Event]
    
#     # -------------------------------------------------------------------------
#     # PART 3: Combine and return negative log-likelihood
#     # -------------------------------------------------------------------------
    
#     # Combine log-likelihoods from observed events and censored patients
#     # tmp1: contribution from observed events
#     # tmp2: contribution from censored patients
#     total_log_likelihood = tmp1 + tmp2
    
#     # Average over batch and risk types
#     # Return negative log-likelihood (to minimize)
#     neg_log_likelihood = -torch.mean(total_log_likelihood)
    
#     return neg_log_likelihood

def deephit_log_likelihood_loss_single_risk(predictions, time_to_events, fail_indicators):
    """
    DeepHit loss for single risk (no competing risks)
    
    Args:
        predictions: [batch_size, num_Category] - PMF over time bins
        time_to_events: [batch_size] - which time bin event/censoring occurred
        fail_indicators: [batch_size] - 1 if event, 0 if censored
    """
    batch_size = predictions.shape[0]
    num_categories = 4
    
    # Map raw time to category bins
    category_indices = torch.zeros_like(time_to_events).long().to(predictions.device)
    category_indices[(time_to_events <= 1)] = 0
    category_indices[(time_to_events >= 1) & (time_to_events <= 5)] = 1
    category_indices[(time_to_events >= 5) & (time_to_events <= 10)] = 2
    category_indices[time_to_events > 10] = 3
    # for i in range(category_indices.shape[0]):
    #     print("category_indices[i]", category_indices[i], "time_to_events", time_to_events[i])
    # One-hot encode
    # print("num_categories", num_categories, 'category_indices', category_indices)
    # categories = nn.functional.one_hot(category_indices, num_classes=num_categories).float()
    ## one-hot encode manually to avoid potential issues
    categories = torch.zeros(batch_size, num_categories, device=predictions.device)
    for i in range(batch_size):
        categories[i, category_indices[i]] = 1.0

    # Mask for observed events (1) vs censored (0)
    event_mask = fail_indicators.float().unsqueeze(1)  # [batch, 1]
    censor_mask = (1.0 - fail_indicators.float()).unsqueeze(1)  # [batch, 1]
    
    # -------------------------------------------------------------------------
    # PART 1: Log-likelihood for OBSERVED EVENTS
    # -------------------------------------------------------------------------
    
    # Probability at actual event time
    prob_at_event = torch.sum(predictions * categories, dim=1, keepdim=True)  # [batch, 1]
    
    # Log-likelihood for events (only for uncensored patients)
    ll_events = event_mask * torch.log(prob_at_event + 1e-8)
    
    # -------------------------------------------------------------------------
    # PART 2: Log-likelihood for CENSORED PATIENTS
    # -------------------------------------------------------------------------
    
    # Total probability mass (should be ≤ 1)
    total_prob = torch.sum(predictions, dim=1, keepdim=True)  # [batch, 1]
    
    # P(T > censoring time) = 1 - P(T ≤ censoring time)
    survival_prob = 1.0 - total_prob
    
    # Log-likelihood for censored (only for censored patients)
    ll_censored = censor_mask * torch.log(survival_prob + 1e-8)
    
    # -------------------------------------------------------------------------
    # PART 3: Combine
    # -------------------------------------------------------------------------
    
    total_ll = ll_events + ll_censored
    neg_log_likelihood = -torch.mean(total_ll)
    
    return neg_log_likelihood


class RNN(pl.LightningModule):
    def __init__(self, modular_featuredim_dict, max_time_len=15, d_model=512,
                 nhead=8, num_encoder_layers=2, dim_feedforward=2048, dropout=0.1,
                 option='binary', pretrain_contrastive_weight=0.5, 
                 representation_dim=512, time_calibration=False,
                 lr=1e-4, weight_decay=1e-5, perturbation_index='', pos_weight=None, normalization=False,
                 #side effects
                 side_effects_num_infusion=False, test_metrics_by_infusion_group=True, advanced_model = 'default', deephitloss_num_category=4):
        super(RNN, self).__init__()
        self.modular_featuredim_dict = modular_featuredim_dict
        # check if there are keys of modular_featuredim_dict that are not in the list of allowed features
        if [el for el in modular_featuredim_dict.keys() if el not in MSK_numfeature_dict.keys()]:
            # raise error
            raise ValueError('There are keys of modular_featuredim_dict that are not in the list of allowed features')
        elif 'basic_vitals' in modular_featuredim_dict.keys() or 'extra_vitals' in modular_featuredim_dict.keys() or 'extra_lab' in modular_featuredim_dict.keys() or 'basic_lab' in modular_featuredim_dict.keys() or 'cbc_differential' in modular_featuredim_dict.keys():
            self.total_feature_dim = sum(modular_featuredim_dict.values()) +2 # +1 for time and another +1 for padding indicator
        else:
            self.total_feature_dim = sum(modular_featuredim_dict.values())
        self.perturbation_index = perturbation_index
        self.normalization = normalization

        self.side_effects_num_infusion = side_effects_num_infusion
        self.test_metrics_by_infusion_group = test_metrics_by_infusion_group
        self.advanced_model = advanced_model # either default or DeepHit loss
        # print("advanced_model", self.advanced_model)
        # print("advanced_model", self.advanced_model)
        # print("advanced_model", self.advanced_model)
        # print("advanced_model", self.advanced_model)
        if self.advanced_model not in ['default', 'rnn_deephit']:
            raise ValueError('advanced_model must be either "default" or "deephitloss"')
        if self.advanced_model == 'rnn_deephit':
            last_dim = 4
        else:
            last_dim = 1
        self.deephitloss_num_category = deephitloss_num_category # 4 categories by default for DeepHit loss (1-3 for early, 4-10 for middle, 11-20 for late, 20+ for really late)

        print("self.total_feature_dim", self.total_feature_dim)

        # hyperparameters
        self.d_model = d_model
        self.nhead = nhead
        self.option = option
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrain_contrastive_weight = pretrain_contrastive_weight
        self.time_calibration = time_calibration

        # Override side_effects_num_infusion for binary tasks (only relevant for side-effect tasks)
        if option in ['binary', 'risk'] and self.side_effects_num_infusion:
            print(f"WARNING: side_effects_num_infusion=True is not applicable for option '{option}'. Setting to False.")
            self.side_effects_num_infusion = False

        # model modules
        if self.total_feature_dim == 1:
            raise ValueError('no input features provided')
        else:
            print("total_feature_dim", self.total_feature_dim)
            print("total_feature_dim", self.total_feature_dim)
            print("total_feature_dim", self.total_feature_dim)

            self.rnn = nn.RNN(input_size=self.total_feature_dim, hidden_size=self.d_model, num_layers=num_encoder_layers, batch_first=True, dropout=dropout)
            self.post_rnn_to_flatten_d_time = nn.Sequential(
                nn.Linear(max_time_len, 1),
                nn.ReLU(),
            )
            self.post_rnn_to_binary_MLP = nn.Linear(representation_dim, last_dim)

            # in the case where infusion number is also provided as a feature for side effects prediction
            if self.side_effects_num_infusion:
                self.num_infusion_MLP = nn.Sequential(
                    nn.Linear(1, 1),
                    nn.ReLU()
                )
                # MLP to combine both
                self.combine_MLP = nn.Sequential(
                    nn.Linear(last_dim+1, 1),
                    # nn.ReLU(),
                )

                
        # loss functions
        if option not in ['pretrain', 'binary', 'risk', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            raise ValueError('Output projection error: option has to be "binary" or "pretrain" or "anemia" or "binary_icu" or "binary_icu_pre" or "binary_risk" or "neutropenia" or "AKI" or "anemia_2wks" or "neutropenia_2wks" or "AKI_2wks" or "Pneumonitis" or "Pneumonitis_2wks" or "Colitis" or "Colitis_2wks" or "Thyroid_dysfunction" or "Thyroid_dysfunction_2wks"')
        # elif option in ['binary']:
        #     if pos_weight is not None:
        #         self.loss_fn = nn.BCELoss(weight=torch.tensor([1-pos_weight]))
        #     else:
        #         self.loss_fn = nn.BCELoss()
        #     self.auroc = AUROC(task='binary')
        #     self.accuracy = Accuracy(task='binary')
        #     self.precision = Precision(task='binary')
        #     self.recall = Recall(task='binary')
        #     self.f1 = F1Score(task='binary')
        #     self.auprc = AveragePrecision(task='binary')
        #     print("OBJECTIVE IS BINARY CLASSIFICATION")
        #     print("OBJECTIVE IS BINARY CLASSIFICATION")
        #     print("OBJECTIVE IS BINARY CLASSIFICATION")
        #     print("OBJECTIVE IS BINARY CLASSIFICATION")
        # elif 'risk' in option:
        #     self.c_index = ConcordanceIndex()
        #     # self.td_auroc = Auc()
        elif option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            if option == 'binary':
                self.loss_fn = nn.BCELoss()
            elif option in ['side_effects_risk', 'side_effects_next_risk']:
                self.c_index = ConcordanceIndex()
            else:
                print("Using default BCELoss for side effects prediction")
                self.loss_fn = nn.BCELoss(reduction='none')
            
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

    def forward(self, src, src_key_padding_mask=None, last_tp_times=None, num_infusion=None):
        if self.total_feature_dim == 1:
            raise ValueError('No data provided')
        
        ### TS TRANSFORMER SECTION
        if self.total_feature_dim != 1:
            batch_size, seq_len, _ = src.size() ## src dim: (batch_size, max_time_len, d_features + 1)

            # pass into RNN
            z, _ = self.rnn(src) # z dim (batch_size, max_time_len, d_model)
                        
            # Pass through MLP to flatten the time dimension
            z = z.permute(0, 2, 1)  # Change to (batch_size, d_model, max_time_len)
            z = self.post_rnn_to_flatten_d_time(z)  # z dim (batch_size, d_model, 1)
            z = z.squeeze(-1)  # Remove the last dimension, z dim (batch_size, d_model)
            z = self.post_rnn_to_binary_MLP(z)  # z dim (batch_size, 1)
            if self.side_effects_num_infusion and num_infusion is not None:
                infusion_num_feat = num_infusion.unsqueeze(-1).float()
                infusion_num_proj = self.num_infusion_MLP(infusion_num_feat) # dim (batch_size, 1)
                z = torch.cat([z, infusion_num_proj], dim=-1) # dim (batch_size, 2)
                z = self.combine_MLP(z) # dim (batch_size, 1)
            if not 'risk' in self.option:
                z = torch.sigmoid(z) # only apply sigmoid for binary classification tasks
        return z

    
    def data_prep(self, data, time_calibration=False, perturbation_index='', normalization=False):
        if self.total_feature_dim != 1:
            if self.option == 'pretrain':
                (xs, times), y, ctypes = data
            elif not self.side_effects_num_infusion:
                (xs, times), y = data
                num_infusion, DMP_ID = None, None
            else:
                # (xs, times), y = data
                (xs, times, num_infusion, DMP_ID), y = data
        else:
            raise ValueError('No data provided')
        
        if 'risk' not in self.option and self.option != 'side_effects':
            y = torch.tensor(y).float()
            y = y.unsqueeze(-1)
        
        # # turn times such that it starts from 0 instead
        if time_calibration:
            times = times - times[:, 0].unsqueeze(-1)

        # ### TS TRANSFORMER DATA SECTION
        # # conduct standard scaler normalization to each feature in X_ts
        # if normalization:
        #     for i in range(xs.size(-1)):
        #         mean = xs[:, :, i].nanmean()
        #         std = nanstd(xs[:, :, i])
        #         xs[:, :, i] = (xs[:, :, i] - mean) / std

        # concatenate xs and times
        if 'basic_lab' in self.modular_featuredim_dict.keys() or 'extra_lab' in self.modular_featuredim_dict.keys() or 'basic_vitals' in self.modular_featuredim_dict.keys():
            xs = torch.cat([xs, times.unsqueeze(-1)], dim=-1)

        # perturb a feature given the index if necessary, perturb it to randomly generated values at normal distribution around (mu=0, std=1) per z-score normalization
        if perturbation_index != '':
            if perturbation_index >= xs.size(-1):
                raise ValueError('perturbation_index out of range')
            # xs[:, :, perturbation_index] = torch.randn(xs[:, :, perturbation_index].shape) ## OG: generate from a random distribution
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


        ## TODO: 120924 debugging - Tom
        if self.option == 'pretrain':
            # the data has already been naturally order, the left most is the last time point
            last_tp_xs_to_predict = xs[:, 0, :] # this also includes the time
            xs = xs[:, 1:, :] # rest of the time points
        else:
            last_tp_xs_to_predict = None

        # create nan mask
        mask = torch.isnan(xs)
        src_key_padding_mask = mask[:, :, -1].T # dim: (max_time_len, batch_size); True if the time point is padded
        xs[mask] = -10
        # attach the src_key_padding_mask as a feature so as to tell the model which time points are padded
        padding_indicator = torch.tensor(~src_key_padding_mask).unsqueeze(-1).permute(1, 0, 2)
        
        # # calculate # of non-zero time points per patient
        # non_zero_time_points = (~src_key_padding_mask).sum(dim=0)
        # print("non_zero_time_points", non_zero_time_points.shape, non_zero_time_points)

        xs = torch.cat([xs, padding_indicator], dim=-1)
        if self.option != 'pretrain':
            return xs, y, src_key_padding_mask, last_tp_xs_to_predict, num_infusion, DMP_ID
        else:
            return (xs, ctypes), y, src_key_padding_mask, last_tp_xs_to_predict

    # def _calculate_weighted_c_index_by_infusion(self, y_pred, y_event, time_to_event, num_infusion, mode='train'):
    #     """
    #     Calculate weighted C-index stratified by infusion number.
    #     Only includes infusions where C-index can be computed.
        
    #     Returns:
    #         dict with keys: c_index_weighted, c_index_overall, num_groups, num_samples,
    #                     auroc metrics, and optionally binary_loss (as tensor for backprop)
    #     """
    #     output_dict = {}
    #     num_infusion_tensor = torch.as_tensor(num_infusion, device=y_pred.device)
    #     y_event_tensor = torch.as_tensor(y_event, device=y_pred.device)
    #     time_to_event_tensor = torch.as_tensor(time_to_event, device=y_pred.device)
        
    #     unique_infusions = torch.unique(num_infusion_tensor)
    #     risk_scores = y_pred.squeeze()
        
    #     # ========================================================================
    #     # Part 1: Risk prediction metrics (C-index)
    #     # ========================================================================
    #     if 'risk' in self.option:
    #         # First pass: identify valid infusions (has both events and censoring)
    #         valid_infusions = []
    #         for infusion in unique_infusions:
    #             mask = num_infusion_tensor == infusion
    #             current_events = y_event_tensor[mask]
                
    #             if current_events.sum() > 0 and current_events.sum() < len(current_events):
    #                 valid_infusions.append(infusion)
            
    #         # Calculate raw C-index (no stratification)
    #         try:
    #             c_idx_raw = self.c_index(
    #                 risk_scores.unsqueeze(1),
    #                 y_event_tensor.bool(),
    #                 time_to_event_tensor
    #             ).item()
    #         except:
    #             c_idx_raw = 0.5
            
    #         # Initialize with defaults
    #         c_index_weighted = 0.5
    #         overall_c_index = 0.5
    #         num_groups = 0
    #         total_samples = 0
    #         computable_results = []
            
    #         if len(valid_infusions) > 0:
    #             # Second pass: compute C-index for valid infusions
    #             for infusion in valid_infusions:
    #                 mask = num_infusion_tensor == infusion
    #                 current_risk = risk_scores[mask]
    #                 current_events = y_event_tensor[mask]
    #                 current_times = time_to_event_tensor[mask]
                    
    #                 try:
    #                     c_idx = self.c_index(
    #                         current_risk.unsqueeze(1),
    #                         current_events.bool(),
    #                         current_times
    #                     )
    #                     # computable_results.append({
    #                     #     'infusion': infusion.item(),
    #                     #     'c_index': c_idx.item(),
    #                     #     'num_samples': mask.sum().item(),
    #                     #     'mask': mask
    #                     # })
    #                     c_idx_value = c_idx.item()
    #                     if not np.isnan(c_idx_value):
    #                         computable_results.append({
    #                             'infusion': infusion.item(),
    #                             'c_index': c_idx_value,
    #                             'num_samples': mask.sum().item(),
    #                             'mask': mask
    #                         })
    #                 except Exception as e:
    #                     continue
            
    #         if len(computable_results) > 0:
    #             # Calculate weighted C-index
    #             total_samples = sum(r['num_samples'] for r in computable_results)
    #             c_index_weighted = sum(
    #                 r['c_index'] * r['num_samples'] / total_samples
    #                 for r in computable_results
    #             )
                
    #             # Calculate overall C-index
    #             all_mask = torch.zeros_like(num_infusion_tensor, dtype=torch.bool)
    #             for r in computable_results:
    #                 all_mask |= r['mask']
                
    #             try:
    #                 overall_c_index = self.c_index(
    #                     risk_scores[all_mask].unsqueeze(1),
    #                     y_event_tensor[all_mask].bool(),
    #                     time_to_event_tensor[all_mask]
    #                 ).item()
    #             except:
    #                 overall_c_index = 0.5
                
    #             num_groups = len(computable_results)
            
    #         output_dict.update({
    #             'c_index_raw': c_idx_raw,
    #             'c_index_weighted': c_index_weighted,
    #             'c_index_overall': overall_c_index,
    #             'num_groups': num_groups,
    #             'num_samples': total_samples,
    #             'per_infusion': computable_results,
    #         })
        
    #     # ========================================================================
    #     # Part 2: Next-infusion side effect prediction (AUROC)
    #     # ========================================================================
        
    #     # Create binary labels: 1 if side effect occurs in next infusion, 0 otherwise
    #     ### flattened tensors
    #     time_to_event_tensor = time_to_event_tensor.flatten()
    #     next_infusion_event = ((y_event_tensor == 1) & (time_to_event_tensor == 1)).float()

    #     # print("time_to_event_tensor", time_to_event_tensor.shape, time_to_event_tensor)
    #     # print("y_event_tensor", y_event_tensor.shape, y_event_tensor)
    #     # print("next_infusion_event:", next_infusion_event.shape, next_infusion_event)
    #     # print("unique_infusions:", unique_infusions)
    #     # First pass: identify valid infusions for AUROC (need both classes)
    #     valid_infusions_auroc = []
    #     for infusion in unique_infusions:
    #         mask = num_infusion_tensor == infusion
    #         current_next_events = next_infusion_event[mask]
            
    #         # Valid if has both positives and negatives
    #         has_positives = current_next_events.sum() > 0
    #         has_negatives = current_next_events.sum() < len(current_next_events)
            
    #         if has_positives and has_negatives:
    #             valid_infusions_auroc.append(infusion)
    #     # print("valid_infusions_auroc 2", valid_infusions_auroc, "defined 2")
        
    #     # Initialize defaults
    #     auroc_next_weighted = 0.5
    #     auroc_next_overall = 0.5
    #     accuracy_next_weighted = 0.5
    #     precision_next_weighted = 0.5
    #     recall_next_weighted = 0.5
    #     auprc_next_weighted = 0.5
    #     num_groups_auroc = 0
    #     num_samples_auroc = 0
    #     binary_loss_weighted = None  # Will be tensor if binary mode
        
    #     if len(valid_infusions_auroc) > 0:
    #         # Second pass: compute AUROC for valid infusions
    #         computable_auroc_results = []
            
    #         # TODO: testing - calculate binary loss across all samples for vectorized comparison
    #         all_sample_losses = self.loss_fn(
    #             risk_scores.unsqueeze(-1),
    #             next_infusion_event.unsqueeze(-1)
    #         ) if 'risk' not in self.option else None

    #         for infusion in valid_infusions_auroc:
    #             mask = num_infusion_tensor == infusion
    #             current_risk = risk_scores[mask]
    #             current_next_events = next_infusion_event[mask]
    #             # print("infusion", infusion.item(), "current_risk", current_risk, "current_next_events", current_next_events)
    #             try:
    #                 current_res = {'num_samples': mask.sum().item()}
    #                 if mode != 'train':
    #                     # Calculate metrics
    #                     auroc = self.auroc(current_risk, current_next_events.int())
    #                     accuracy = self.accuracy(current_risk, current_next_events.int())
    #                     precision = self.precision(current_risk, current_next_events.int())
    #                     recall = self.recall(current_risk, current_next_events.int())
    #                     auprc = self.auprc(current_risk, current_next_events.int())
                        
    #                     current_res = {
    #                         'infusion': infusion.item(),
    #                         'auroc': auroc.item(),
    #                         'accuracy': accuracy,
    #                         'precision': precision,
    #                         'recall': recall,
    #                         'auprc': auprc,
    #                         'num_samples': mask.sum().item(),
    #                         'num_positives': current_next_events.sum().item(),
    #                         'mask': mask
    #                     }
                        
    #                 # Calculate binary loss (keep as tensor for backprop!)
    #                 if 'risk' not in self.option:
    #                     # binary_loss = self.loss_fn(
    #                     #     current_risk.unsqueeze(-1),
    #                     #     current_next_events.unsqueeze(-1)
    #                     # )
    #                     binary_loss = all_sample_losses[mask].mean()
    #                     # print("infusion", infusion.item(), "binary_loss", binary_loss)
    #                     current_res['binary_loss'] = binary_loss  # Keep as tensor
    #                     # current_res['binary_loss_value'] = binary_loss.item()  # For logging
                    
    #                 computable_auroc_results.append(current_res)
                    
    #             except Exception as e:
    #                 print(f"Skipping AUROC for infusion {infusion.item()}: {e}")
    #                 continue
            
    #         if len(computable_auroc_results) > 0:
    #             # Calculate weighted metrics
    #             total_samples_auroc = sum(r['num_samples'] for r in computable_auroc_results)
    #             if mode != 'train':
    #                 auroc_next_weighted = sum(
    #                     r['auroc'] * r['num_samples'] / total_samples_auroc
    #                     for r in computable_auroc_results
    #                 )
    #                 accuracy_next_weighted = sum(
    #                     r['accuracy'] * r['num_samples'] / total_samples_auroc
    #                     for r in computable_auroc_results
    #                 )
    #                 precision_next_weighted = sum(
    #                     r['precision'] * r['num_samples'] / total_samples_auroc
    #                     for r in computable_auroc_results
    #                 )
    #                 recall_next_weighted = sum(
    #                     r['recall'] * r['num_samples'] / total_samples_auroc
    #                     for r in computable_auroc_results
    #                 )
    #                 auprc_next_weighted = sum(
    #                     r['auprc'] * r['num_samples'] / total_samples_auroc
    #                     for r in computable_auroc_results
    #                 )
    #                 output_dict.update({
    #                     # Next-infusion prediction metrics
    #                     'auroc_next_infusion_weighted': auroc_next_weighted,
    #                     # 'auroc_next_infusion_overall': auroc_next_overall,
    #                     'accuracy_next_infusion_weighted': accuracy_next_weighted,
    #                     'precision_next_infusion_weighted': precision_next_weighted,
    #                     'recall_next_infusion_weighted': recall_next_weighted,
    #                     'auprc_next_infusion_weighted': auprc_next_weighted,
    #                     'num_groups_next_infusion': num_groups_auroc,
    #                     # 'num_samples_next_infusion': num_samples_auroc,
    #                     'per_infusion_auroc': computable_auroc_results
    #                 })
    #                 # print("accuracy_next_weighted:", accuracy_next_weighted, "precision_next_weighted:", precision_next_weighted, "recall_next_weighted:", recall_next_weighted)
                
    #             # Calculate weighted binary loss (keep as tensor!)
    #             if 'risk' not in self.option:
    #                 # binary_loss_weighted = sum(
    #                 #     r['binary_loss'] * r['num_samples'] / total_samples_auroc
    #                 #     for r in computable_auroc_results
    #                 # )
    #                 binary_loss_weighted = torch.stack([
    #                     r['binary_loss'] * r['num_samples'] / total_samples_auroc
    #                     for r in computable_auroc_results
    #                 ]).sum()
    #             if mode != 'train':
    #                 # Calculate overall AUROC across all valid samples
    #                 all_mask_auroc = torch.zeros_like(num_infusion_tensor, dtype=torch.bool)
    #                 for r in computable_auroc_results:
    #                     all_mask_auroc |= r['mask']
                    
    #                 try:
    #                     auroc_next_overall = self.auroc(
    #                         risk_scores[all_mask_auroc],
    #                         next_infusion_event[all_mask_auroc].int()
    #                     ).item()
    #                 except:
    #                     auroc_next_overall = 0.5
    #                 output_dict['auroc_next_infusion_overall'] = auroc_next_overall
                    
    #                 num_groups_auroc = len(computable_auroc_results)
    #                 num_samples_auroc = total_samples_auroc

    #     else:
    #         return None # to be skipped in trainer loop so that no backprop is done
        
    #     # Add binary loss as tensor (for backprop) if in binary mode
    #     if 'risk' not in self.option and binary_loss_weighted is not None:
    #         output_dict['binary_loss'] = binary_loss_weighted  # TENSOR, not .item()!
    #         output_dict['num_samples_next_infusion'] = num_samples_auroc
        
    #     return output_dict

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
        # xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict = self.data_prep(batch, time_calibration=self.time_calibration, perturbation_index=self.perturbation_index)
        xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration)

        y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion)
        # if not 'risk' in self.option:
        #     return {'y_pred': y_pred, 'y_true': y}
        # else:
        #     return {'y_pred': y_pred, 'y_true': y, 'last_tp_xs_to_predict': last_tp_xs_to_predict}
        return {'y_pred': y_pred, 'y_true': y, 'num_infusion': num_infusion}
            
    ### BELOW IS ALL THE PYTORCH LIGHTNING THINGIS
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
        return {
           'optimizer': optimizer,
           'scheduler': scheduler,
           'monitor': 'val_loss'
       }
    
    def training_step(self, batch, batch_idx):
        num_infusion = None
        xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration)
        # xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict = self.data_prep(batch, time_calibration=self.time_calibration)
        # y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask, num_infusion=num_infusion)
        if self.option == 'pretrain':
            loss = 0
            xs_combined, ctypes = xs_combined
            last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]
            xs_rep, xs_last_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times)

            if self.total_feature_dim!=1:
                losses = []
                # contrastive_losses = []
                ctypes_combo_l = []
                # for pretraining: calculate the patient-average loss based on the last time point embedding prediction only
                for i in range(batch[0][0].shape[0]):
                    ## last token prediction part
                    current_xs_last_gt = last_tp_xs_to_predict[i]
                    current_xs_last_gt_non_nan_mask = ~torch.isnan(current_xs_last_gt)
                    current_xs_last_pred = xs_last_pred[i]
                    current_loss = self.loss_fn(current_xs_last_pred[current_xs_last_gt_non_nan_mask], current_xs_last_gt[current_xs_last_gt_non_nan_mask])
                    losses.append(current_loss/current_xs_last_pred.size(0))
                loss += torch.stack(losses).sum()
                ## contrastive loss part
                ### iterate through pairs of patients efficiently
                if self.pretrain_contrastive_weight > 0:
                    N = xs_rep.shape[0]

                    # Compute pairwise cosine similarities
                    norms = xs_rep.norm(dim=1, keepdim=True)  # [N, 1]
                    normalized_xs_rep = xs_rep / norms  # Normalize each vector, [N, D]

                    # Compute pairwise cosine similarity matrix [N, N]
                    cosine_sim_matrix = torch.mm(normalized_xs_rep, normalized_xs_rep.T)

                    # Get upper triangular indices without the diagonal (to match combinations behavior)
                    i, j = torch.triu_indices(N, N, offset=1)

                    # Extract corresponding cosine similarities
                    pairwise_cosine_sims = cosine_sim_matrix[i, j]

                    # Determine whether to negate based on ctypes
                    same_type_mask = (ctypes[i] == ctypes[j])  # Boolean mask for matching types
                    contrastive_loss = torch.where(same_type_mask, -pairwise_cosine_sims, pairwise_cosine_sims)

                    # Append to the list if needed
                    loss += self.pretrain_contrastive_weight*contrastive_loss.mean()
                
        # elif self.option == 'binary':
        #     y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined)
        #     loss = self.loss_fn(y_pred, y)
        
        # elif 'risk' in self.option:
        #     y_pred_logits = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined)
        #     fail_indicators = torch.tensor(y[0]).to(y_pred_logits.device)
        #     time_to_events = torch.tensor(y[1]).float().to(y_pred_logits.device)
        #     loss = cox.neg_partial_log_likelihood(y_pred_logits, fail_indicators, time_to_events, ties_method = "efron")

        # self.log('train_loss', loss)
        # return loss
        elif self.option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion)
            # print("current DMP_ID", DMP_ID)
            if self.option == 'binary':
                loss = self.loss_fn(y_pred, y)
            elif self.option in ['side_effects_risk', 'side_effects_next_risk']:
                # it's now a risk prediction task with side effects as outcome and current_num_durations_to_event_or_cutoff as time to event
                y_event, time_to_event = torch.tensor(y[0]).to(y_pred.device), torch.tensor(y[1]).float().to(y_pred.device)
                # loss = deephit_log_likelihood_loss(y_pred, y_event, time_to_event)
                if self.advanced_model == 'default':
                    loss = cox.neg_partial_log_likelihood(y_pred, y_event, time_to_event, ties_method = "efron")
                else:
                    # predictions, time_to_events, fail_indicators
                    loss = deephit_log_likelihood_loss_single_risk(y_pred, time_to_event, y_event)
            elif self.option == 'side_effects':
                # set it to all 1s and fit the number of patients
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

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict = self.data_prep(batch, time_calibration=self.time_calibration)
        num_infusion = None
        xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration)

        if self.option == 'pretrain':
            loss = 0
            xs_combined, ctypes = xs_combined
            last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]
            xs_rep, xs_last_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times)

            if self.total_feature_dim!=1:
                # xs_last_pred = xs_last_pred[:, :batch[0][0].shape[-1]]
                losses = []
                # contrastive_losses = []
                ctypes_combo_l = []
                # for pretraining: calculate the patient-average loss based on the last time point embedding prediction only
                for i in range(batch[0][0].shape[0]):
                    ## last token prediction part
                    current_xs_last_gt = last_tp_xs_to_predict[i]
                    current_xs_last_gt_non_nan_mask = ~torch.isnan(current_xs_last_gt)
                    current_xs_last_pred = xs_last_pred[i]
                    current_loss = self.loss_fn(current_xs_last_pred[current_xs_last_gt_non_nan_mask], current_xs_last_gt[current_xs_last_gt_non_nan_mask])
                    losses.append(current_loss/current_xs_last_pred.size(0))
                loss += torch.stack(losses).sum()
                ## contrastive loss part
                ### iterate through pairs of patients efficiently
                if self.pretrain_contrastive_weight > 0:
                    N = xs_rep.shape[0]

                    # Compute pairwise cosine similarities
                    norms = xs_rep.norm(dim=1, keepdim=True)  # [N, 1]
                    normalized_xs_rep = xs_rep / norms  # Normalize each vector, [N, D]

                    # Compute pairwise cosine similarity matrix [N, N]
                    cosine_sim_matrix = torch.mm(normalized_xs_rep, normalized_xs_rep.T)

                    # Get upper triangular indices without the diagonal (to match combinations behavior)
                    i, j = torch.triu_indices(N, N, offset=1)

                    # Extract corresponding cosine similarities
                    pairwise_cosine_sims = cosine_sim_matrix[i, j]

                    # Determine whether to negate based on ctypes
                    same_type_mask = (ctypes[i] == ctypes[j])  # Boolean mask for matching types
                    contrastive_loss = torch.where(same_type_mask, -pairwise_cosine_sims, pairwise_cosine_sims)

                    # Append to the list if needed
                    loss += self.pretrain_contrastive_weight*contrastive_loss.mean()
                
        elif self.option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion)
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
                # it's now a risk prediction task with side effects as outcome and current_num_durations_to_event_or_cutoff as time to event
                y_event, time_to_event = torch.tensor(y[0]).to(y_pred.device), torch.tensor(y[1]).float().to(y_pred.device)
                # loss = deephit_log_likelihood_loss(y_pred, y_event, time_to_event)
                if self.advanced_model == 'default':
                    loss = cox.neg_partial_log_likelihood(y_pred, y_event, time_to_event, ties_method = "efron")
                else:
                    loss = deephit_log_likelihood_loss_single_risk(y_pred, time_to_event, y_event)
                    y_pred = y_pred[ :, 0]  # for c-index calculation
                c_idx_dict = self._calculate_weighted_c_index_by_infusion(y_pred, y_event, time_to_event, num_infusion, mode='val')
                if c_idx_dict is None:
                    return None
                c_index, c_index_weighted, overall_c_index = c_idx_dict['c_index_raw'], c_idx_dict['c_index_weighted'], c_idx_dict['c_index_overall']
                weighted_auroc, overall_auroc = c_idx_dict['auroc_next_infusion_weighted'], c_idx_dict['auroc_next_infusion_overall']
                self.log('val_cindex_unfiltered', c_index, prog_bar=True)
                self.log('val_cindex_weighted', c_index_weighted, prog_bar=True)
                self.log('val_cindex_overall_valid', overall_c_index, prog_bar=True)
                self.log('val_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
                self.log('val_auroc_nextinf_overall', overall_auroc, prog_bar=True)
                    
        elif 'risk' in self.option:
            y_pred_logits = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined)
            fail_indicators = torch.tensor(y[0]).to(y_pred_logits.device)
            time_to_events = torch.tensor(y[1]).float().to(y_pred_logits.device)
            loss = cox.neg_partial_log_likelihood(y_pred_logits, fail_indicators, time_to_events, ties_method = "efron")
            c_index = self.c_index(y_pred_logits, fail_indicators, time_to_events)
            
            self.log('val_c_index', c_index, prog_bar=True)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        num_infusion = None
        xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration, perturbation_index=self.perturbation_index)
        # xs_combined, y, src_key_padding_mask_combined, last_tp_xs_to_predict = self.data_prep(batch, time_calibration=self.time_calibration, perturbation_index=self.perturbation_index)

        if self.option == 'pretrain':
            loss = 0
            xs_combined, ctypes = xs_combined
            last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]
            xs_rep, xs_last_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times)

            if self.total_feature_dim!=1:
                losses = []
                contrastive_losses = []
                ctypes_combo_l = []
                # for pretraining: calculate the patient-average loss based on the last time point embedding prediction only
                for i in range(batch[0][0].shape[0]):
                    ## last token prediction part
                    current_xs_last_gt = last_tp_xs_to_predict[i]
                    current_xs_last_gt_non_nan_mask = ~torch.isnan(current_xs_last_gt)
                    current_xs_last_pred = xs_last_pred[i]
                    current_loss = self.loss_fn(current_xs_last_pred[current_xs_last_gt_non_nan_mask], current_xs_last_gt[current_xs_last_gt_non_nan_mask])
                    losses.append(current_loss/current_xs_last_pred.size(0))
                loss += torch.stack(losses).sum()
                ## contrastive loss part
                ### iterate through pairs of patients efficiently
                if self.pretrain_contrastive_weight > 0:
                    N = xs_rep.shape[0]

                    # Compute pairwise cosine similarities
                    norms = xs_rep.norm(dim=1, keepdim=True)  # [N, 1]
                    normalized_xs_rep = xs_rep / norms  # Normalize each vector, [N, D]

                    # Compute pairwise cosine similarity matrix [N, N]
                    cosine_sim_matrix = torch.mm(normalized_xs_rep, normalized_xs_rep.T)

                    # Get upper triangular indices without the diagonal (to match combinations behavior)
                    i, j = torch.triu_indices(N, N, offset=1)

                    # Extract corresponding cosine similarities
                    pairwise_cosine_sims = cosine_sim_matrix[i, j]

                    # Determine whether to negate based on ctypes
                    same_type_mask = (ctypes[i] == ctypes[j])  # Boolean mask for matching types
                    contrastive_loss = torch.where(same_type_mask, -pairwise_cosine_sims, pairwise_cosine_sims)

                    # Append to the list if needed
                    loss += self.pretrain_contrastive_weight*contrastive_loss.mean()
                
        # elif self.option == 'binary':   
        #     y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined)
        #     loss = self.loss_fn(y_pred, y)
        #     auroc = self.auroc(y_pred, y.int())
        #     accuracy = self.accuracy(y_pred, y.int())
        #     precision = self.precision(y_pred, y.int())
        #     recall = self.recall(y_pred, y.int())
        #     auprc = self.auprc(y_pred, y.int())

        #     self.log('test_auroc', auroc, prog_bar=True)
        #     self.log('test_accuracy', accuracy, prog_bar=True)
        #     self.log('test_precision', precision, prog_bar=True)
        #     self.log('test_recall', recall, prog_bar=True)
        #     self.log('test_f1', self.f1(y_pred, y.int()), prog_bar=True)
        #     self.log('test_auprc', auprc, prog_bar=True)

        # elif 'risk' in self.option:
        #     y_pred_logits = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined)
        #     fail_indicators = torch.tensor(y[0]).to(y_pred_logits.device)
        #     time_to_events = torch.tensor(y[1]).float().to(y_pred_logits.device)
        #     print("fail_indicators", fail_indicators.device)
        #     print("time_to_events", time_to_events.device)
        #     loss = cox.neg_partial_log_likelihood(y_pred_logits, fail_indicators, time_to_events, ties_method = "efron")
        #     c_index = self.c_index(y_pred_logits, fail_indicators, time_to_events)
        #     # calculate whether patient fails at binary days based on the time_to_events and fail_indicators (the latest)
        #     # binary_days_fail = (time_to_events <= self.binary_days) & fail_indicators
        #     # brier_score = self.brier_score(y_pred_logits, binary_days_fail , self.binary_days)
        #     # brier_score = self.brier_score(y_pred_logits, fail_indicators, time_to_events, new_time=self.binary_days)
        #     # print("fail_indicators", fail_indicators.device)
        #     # print("time_to_events", time_to_events.device)
        #     # ipcw = get_ipcw(fail_indicators.to('cpu'), time_to_events.to('cpu'))
        #     # td_auroc = self.td_auroc(y_pred_logits, fail_indicators, time_to_events, weight=ipcw)
        #     self.log('test_c_index', c_index, prog_bar=True)
        #     # self.log('test_td_auroc', td_auroc, prog_bar=True)

        # self.log('test_loss', loss, prog_bar=True)
        # return loss
        elif self.option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            y_pred = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, num_infusion=num_infusion)
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
                            # current_res = {
                            #     'infusion': infusion.item(),
                            #     'auroc': auroc.item(),
                            #     'accuracy': accuracy,
                            #     'precision': precision,
                            #     'recall': recall,
                            #     'auprc': auprc,
                            #     'num_samples': mask.sum().item(),
                            #     'num_positives': current_next_events.sum().item(),
                            #     'mask': mask
                            # }
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
                # it's now a risk prediction task with side effects as outcome and current_num_durations_to_event_or_cutoff as time to event
                y_event, time_to_event = torch.tensor(y[0]).to(y_pred.device), torch.tensor(y[1]).float().to(y_pred.device)
                # loss = deephit_log_likelihood_loss(y_pred, y_event, time_to_event)
                if self.advanced_model == 'default':
                    loss = cox.neg_partial_log_likelihood(y_pred, y_event, time_to_event, ties_method = "efron")
                else:
                    loss = deephit_log_likelihood_loss_single_risk(y_pred, time_to_event, y_event)
                    y_pred = y_pred[ :, 0]  # for c-index calculation
                c_idx_dict = self._calculate_weighted_c_index_by_infusion(y_pred, y_event, time_to_event, num_infusion, mode='test')
                if c_idx_dict is None:
                    return None
                # if self.test_metrics_by_infusion_group:
                #     per_infusion_auroc = c_idx_dict.get('per_infusion_auroc')
                #     # log per-infusion group metrics
                #     for res in per_infusion_auroc:
                #         # current_res = {
                #         #     'infusion': infusion.item(),
                #         #     'auroc': auroc.item(),
                #         #     'accuracy': accuracy,
                #         #     'precision': precision,
                #         #     'recall': recall,
                #         #     'auprc': auprc,
                #         #     'num_samples': mask.sum().item(),
                #         #     'num_positives': current_next_events.sum().item(),
                #         #     'mask': mask
                #         # }
                #         infusion_id = res['infusion']
                #         self.log(f'test_auroc_infusion_{infusion_id}', res['auroc'], prog_bar=False)
                #         self.log(f'infusion_num_samples_{infusion_id}', res['num_samples'], prog_bar=False)
                    # self.log()
                c_index, c_index_weighted, overall_c_index = c_idx_dict['c_index_raw'], c_idx_dict['c_index_weighted'], c_idx_dict['c_index_overall']
                weighted_auroc, overall_auroc = c_idx_dict['auroc_next_infusion_weighted'], c_idx_dict['auroc_next_infusion_overall']
                self.log('test_cindex_unfiltered', c_index, prog_bar=True)
                self.log('test_cindex_weighted', c_index_weighted, prog_bar=True)
                self.log('test_cindex_overall_valid', overall_c_index, prog_bar=True)
                self.log('test_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
                self.log('test_auroc_nextinf_overall', overall_auroc, prog_bar=True)

        elif 'risk' in self.option:
            y_pred_logits = self(xs_combined, src_key_padding_mask=src_key_padding_mask_combined)
            fail_indicators = torch.tensor(y[0]).to(y_pred_logits.device)
            time_to_events = torch.tensor(y[1]).float().to(y_pred_logits.device)
            # print("fail_indicators", fail_indicators.device)
            # print("time_to_events", time_to_events.device)
            loss = cox.neg_partial_log_likelihood(y_pred_logits, fail_indicators, time_to_events, ties_method = "efron")
            c_index = self.c_index(y_pred_logits, fail_indicators, time_to_events)
            self.log('test_c_index', c_index, prog_bar=True)


        self.log('test_loss', loss, prog_bar=True)
        return loss

   