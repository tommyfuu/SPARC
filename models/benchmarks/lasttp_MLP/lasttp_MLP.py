import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import AUROC, Accuracy, Precision, Recall, F1Score, AveragePrecision
import torch
import torch.nn as nn
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex
import sys
import re
# Put the repo root, data/ and models/ on sys.path, from any working directory.
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import math
from MSK_featurename_list import *


def nanstd(x): 
    return torch.sqrt(torch.nanmean((x - torch.nanmean(x))**2))

class lasttp_MLP(pl.LightningModule):
    def __init__(self, modular_featuredim_dict, d_model=512, num_layers=1,
                 lr=1e-5, weight_decay=1e-5, pos_weight=None,
                 option='binary', time_calibration=False, primary_ctype_returned=False,
                 #side effects
                 side_effects_num_infusion=False, test_metrics_by_infusion_group=True):
        super(lasttp_MLP, self).__init__()
        # check if there are keys of modular_featuredim_dict that are not in the list of allowed features
        if [el for el in modular_featuredim_dict.keys() if el not in MSK_numfeature_dict.keys()]:
            # raise error
            raise ValueError('There are keys of modular_featuredim_dict that are not in the list of allowed features')
        elif 'basic_vitals' in modular_featuredim_dict.keys() or 'extra_vitals' in modular_featuredim_dict.keys() or 'extra_lab' in modular_featuredim_dict.keys() or 'basic_lab' in modular_featuredim_dict.keys() or 'cbc_differential' in modular_featuredim_dict.keys():
            self.total_feature_dim = sum(modular_featuredim_dict.values()) + 1 # +1 for time
        else:
            self.total_feature_dim = sum(modular_featuredim_dict.values())
        print("TRYING MODEL")
        print("TRYING MODEL")
        print("TRYING MODEL")
        print("TRYING MODEL")
        print("modular_featuredim_dict: ", modular_featuredim_dict)
        print("model running option", option)
        self.d_model = d_model
        self.option = option
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.time_calibration = time_calibration
        self.primary_ctype_returned = primary_ctype_returned

        self.side_effects_num_infusion = side_effects_num_infusion
        self.test_metrics_by_infusion_group = test_metrics_by_infusion_group

        # Override side_effects_num_infusion for binary tasks (only relevant for side-effect tasks)
        if option in ['binary', 'risk'] and self.side_effects_num_infusion:
            print(f"WARNING: side_effects_num_infusion=True is not applicable for option '{option}'. Setting to False.")
            self.side_effects_num_infusion = False

        # create a num_layer layer MLP
        if num_layers < 1:
            raise ValueError('num_layers must be greater than 0')
        elif num_layers == 1:
            self.MLP = nn.Sequential(
                nn.Linear(self.total_feature_dim, 1),
                nn.ReLU()
            )
        elif num_layers == 2:
            self.MLP = nn.Sequential(
                nn.Linear(self.total_feature_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.ReLU()
            )
        else:
            # create a MLP with num_layers layers
            # first layer
            self.MLP = nn.Sequential(
                nn.Linear(self.total_feature_dim, d_model),
                nn.ReLU(),
            )
            # middle layers
            for i in range(num_layers-2):
                self.MLP.append(nn.Linear(d_model, d_model))
                self.MLP.append(nn.ReLU())
            # last layer
            self.MLP.append(nn.Linear(d_model, 1))
            # self.MLP.append(nn.ReLU())
        
        if option in ['side_effects', 'side_effects_risk', 'side_effects_next_risk'] and self.side_effects_num_infusion:
            self.infusion_num_to_single_MLP = nn.Sequential(
                    nn.Linear(1, 1),
                    nn.ReLU()
                )
            # MLP to combine the two
            self.combine_MLP = nn.Sequential(
                nn.Linear(2, 1),
                # nn.ReLU()
            )


        if option not in ['pretrain', 'binary', 'risk', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            print("option: ", option)
            raise ValueError('Output projection error: option not permitted')
        elif option in ['binary', 'side_effects', 'side_effects_risk', 'side_effects_next_risk']:
            if option == 'binary':
                self.loss_fn = nn.BCELoss()
            elif option in ['side_effects_risk', 'side_effects_next_risk']:
                self.c_index = ConcordanceIndex()
            else:
                print("Using default BCELoss for side effects prediction")
                self.loss_fn = nn.BCELoss(reduction='none')
            self.MLP_model = nn.Sequential(
                self.MLP,
                nn.Sigmoid()
            )
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

        

    def forward(self, src, src_key_padding_mask=None, num_infusion=None):
        # if self.option in ['binary', 'side_effects']:
        #     # project to binary
        #     z = self.MLP_model(src)
        #     return z
        # elif 'risk' in self.option:
        #     # project to risk
        #     z = self.MLP(src)
        #     if self.side_effects_num_infusion and num_infusion is not None:
        #         # print("infusion used")
        #         infusion_num_feat = num_infusion.unsqueeze(-1).float()
        #         infusion_num_proj = self.infusion_num_to_single_MLP(infusion_num_feat)
        #         z = torch.cat([z, infusion_num_proj], dim=-1)
        #         z = self.combine_MLP(z)
        #     # Apply sigmoid for side_effects_risk to match the training behavior
        #     # (models trained with BCE loss need sigmoid applied)
        #     if self.option in ['side_effects_risk', 'side_effects_next_risk']:
        #         print("Applying sigmoid to risk output for option:", self.option)
        #         z = torch.sigmoid(z)
        #     return z
        z=self.MLP_model(src)
        return z
        
    
    def data_prep(self, data, time_calibration=False):
        if self.total_feature_dim != 1:
            if self.option == 'pretrain':
                (xs, times), y, ctypes = data
            # elif not self.primary_ctype_returned:
            #     (xs, times), y = data
            # else:
            #     (xs, times), y, ctypes = data
            elif not self.side_effects_num_infusion:
                (xs, times), y = data
                num_infusion, DMP_ID = None, None
            else:
                (xs, times, num_infusion, DMP_ID), y = data
            
        else:
            raise ValueError('No data provided')

        if 'risk' not in self.option and self.option != 'side_effects':
            y = torch.tensor(y).float()
            y = y.unsqueeze(-1)

        if time_calibration:
            times = times - times.min()

        # # conduct standard scaler normalization to each feature in X_ts
        # for i in range(xs.size(-1)):
        #     mean = xs[:, :, i].nanmean()
        #     std = nanstd(xs[:, :, i])
        #     xs[:, :, i] = (xs[:, :, i] - mean) / std
            
        xs_last, times_last = xs[:, 0, :], times[:, 0]

        xs = torch.cat([xs_last, times_last.unsqueeze(-1)], dim=-1)
        
        # create nan mask
        mask = torch.isnan(xs)
        src_key_padding_mask = torch.isnan(xs).all(dim=-1).T
        xs[mask] = -1

        return xs, y, src_key_padding_mask, num_infusion, DMP_ID
        
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
                        
    #                     c_idx_value = c_idx.item()
    #                     # print("infusion", infusion.item(), "c_idx", c_idx_value, "math.isnan(c_idx_value)", math.isnan(c_idx_value), "num_samples", mask.sum().item())
    #                     if not math.isnan(c_idx_value):  # Option 1
    #                         # print("appending infusion", infusion.item(), "c_idx", c_idx_value)
    #                         computable_results.append({
    #                             'infusion': infusion.item(),
    #                             'c_index': c_idx_value,
    #                             'num_samples': mask.sum().item(),
    #                             'mask': mask
    #                         })
    #                 except Exception as e:
    #                     continue
    #         # print("computable_results:", len(computable_results), computable_results)
    #         if len(computable_results) > 0:
    #             # Calculate weighted C-index
    #             total_samples = sum(r['num_samples'] for r in computable_results)
    #             # print("total_samples for c-index:", total_samples)
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
    #     valid_infusions_auroc = []
    #     for infusion in unique_infusions:
    #         mask = num_infusion_tensor == infusion
    #         current_next_events = next_infusion_event[mask]
            
    #         # Valid if has both positives and negatives
    #         has_positives = current_next_events.sum() > 0
    #         has_negatives = current_next_events.sum() < len(current_next_events)
            
    #         if has_positives and has_negatives:
    #             valid_infusions_auroc.append(infusion)
        
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
    #     # print("valid_infusions_auroc:", valid_infusions_auroc)
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
    #                     # print("current_res for infusion", infusion.item(), ":", current_res)
                        
    #                 # binary loss, kept as a tensor for backprop
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
                
    #             # weighted binary loss (tensor)
    #             if 'risk' not in self.option:
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
        """Weighted C-index and next-infusion metrics stratified by infusion number,
        skipping infusions where they can't be computed."""
        output_dict = {}
        num_infusion_tensor = torch.as_tensor(num_infusion, device=y_pred.device)
        y_event_tensor = torch.as_tensor(y_event, device=y_pred.device)
        time_to_event_tensor = torch.as_tensor(time_to_event, device=y_pred.device)

        unique_infusions = torch.unique(num_infusion_tensor)
        risk_scores = y_pred.squeeze()

        # part 1: risk prediction metrics (C-index)
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
        # part 2: next-infusion side effect prediction (AUROC)

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

                    # binary loss, kept as a tensor for backprop
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

                # weighted binary loss (tensor)
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
            output_dict['binary_loss'] = binary_loss_weighted  # tensor, not .item()
            output_dict['num_samples_next_infusion'] = num_samples_auroc

        return output_dict

    def predict_step(self, batch, batch_idx):
        # xs, y, src_key_padding_mask = self.data_prep(batch)
        xs, y, src_key_padding_mask, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration)
        y_pred = self(xs, src_key_padding_mask=src_key_padding_mask, num_infusion=num_infusion)
        # if not 'risk' in self.option:
        #     return {'y_pred': y_pred, 'y_true': y}
        # else:
        #     fail_indicators = torch.tensor(y[0]).to(y_pred.device)
        #     time_to_events = torch.tensor(y[1]).float().to(y_pred.device)
        #     return cox.neg_partial_log_likelihood(y_pred, fail_indicators, time_to_events, ties_method = 'efron')
        return {'y_pred': y_pred, 'y_true': y, 'num_infusion': num_infusion}

    ### BELOW IS ALL THE PYTORCH LIGHTNING THINGIS
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
           'optimizer': optimizer,
           'monitor': 'val_loss'
       }
    
    def training_step(self, batch, batch_idx):
        # xs, y, src_key_padding_mask = self.data_prep(batch)
        # y_pred = self(xs, src_key_padding_mask=src_key_padding_mask)
        # if not 'risk' in self.option:
        #     loss = self.loss_fn(y_pred, y)
        # else:
        #     fail_indicators = torch.tensor(y[0]).to(y_pred.device)
        #     time_to_events = torch.tensor(y[1]).float().to(y_pred.device)
        #     loss = cox.neg_partial_log_likelihood(y_pred, fail_indicators, time_to_events, ties_method = 'efron')
        # self.log('train_loss', loss)
        # return loss
        num_infusion = None
        # xs, y, src_key_padding_mask, num_infusion, DMP_ID
        xs, y, src_key_padding_mask, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration)
        y_pred = self(xs, src_key_padding_mask=src_key_padding_mask, num_infusion=num_infusion)
        if self.option == 'binary':
            loss = self.loss_fn(y_pred, y)
        elif self.option in ['side_effects_risk', 'side_effects_next_risk']:
            # it's now a risk prediction task with side effects as outcome and current_num_durations_to_event_or_cutoff as time to event
            y_event, time_to_event = torch.tensor(y[0]).to(y_pred.device), torch.tensor(y[1]).float().to(y_pred.device)
            # loss = deephit_log_likelihood_loss(y_pred, y_event, time_to_event)
            loss = cox.neg_partial_log_likelihood(y_pred, y_event, time_to_event, ties_method = "efron")
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
        # xs, y, src_key_padding_mask = self.data_prep(batch)
        # y_pred = self(xs, src_key_padding_mask=src_key_padding_mask)
        # if not 'risk' in self.option:
        #     loss = self.loss_fn(y_pred, y)
        #     auroc = self.auroc(y_pred, y.int())
        #     accuracy = self.accuracy(y_pred, y.int())
        #     precision = self.precision(y_pred, y.int())
        #     recall = self.recall(y_pred, y.int())
        #     auprc = self.auprc(y_pred, y.int())
        #     self.log('val_auroc', auroc, prog_bar=True)
        #     self.log('val_accuracy', accuracy, prog_bar=True)
        #     self.log('val_precision', precision, prog_bar=True)
        #     self.log('val_recall', recall, prog_bar=True)
        #     self.log('val_f1', self.f1(y_pred, y.int()), prog_bar=True)
        #     self.log('val_auprc', auprc, prog_bar=True)
        # else:
        #     fail_indicators = torch.tensor(y[0]).to(y_pred.device)
        #     time_to_events = torch.tensor(y[1]).float().to(y_pred.device)
        #     loss = cox.neg_partial_log_likelihood(y_pred, fail_indicators, time_to_events, ties_method = 'efron')
        #     self.log('val_c_index', self.c_index(y_pred, fail_indicators, time_to_events), prog_bar=True)
        
        # self.log('val_loss', loss, prog_bar=True)
        # return loss
        num_infusion = None
        xs, y, src_key_padding_mask, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration)
        y_pred = self(xs, src_key_padding_mask=src_key_padding_mask, num_infusion=num_infusion)
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
                # set it to all 1s and fit the number of patients
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
            loss = cox.neg_partial_log_likelihood(y_pred, y_event, time_to_event, ties_method = "efron")
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
        # xs, y, src_key_padding_mask = self.data_prep(batch)
        # y_pred = self(xs, src_key_padding_mask=src_key_padding_mask)
        # if not 'risk' in self.option:
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
        #     self.log('test_auprc', self.auprc(y_pred, y.int()), prog_bar=True)
        # else:
        #     fail_indicators = torch.tensor(y[0]).to(y_pred.device)
        #     time_to_events = torch.tensor(y[1]).float().to(y_pred.device)
        #     loss = cox.neg_partial_log_likelihood(y_pred, fail_indicators, time_to_events, ties_method = 'efron')
        #     self.log('test_c_index', self.c_index(y_pred, fail_indicators, time_to_events), prog_bar=True)

        # self.log('test_loss', loss, prog_bar=True)
        # return loss

        # print the model structure
        # print("check model", self)
        num_infusion = None
        xs, y, src_key_padding_mask, num_infusion, DMP_ID = self.data_prep(batch, time_calibration=self.time_calibration)
        y_pred = self(xs, src_key_padding_mask=src_key_padding_mask, num_infusion=num_infusion)
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
                # set it to all 1s and fit the number of patients
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

        elif self.option in ['side_effects_risk', 'side_effects_next_risk']:
            # it's now a risk prediction task with side effects as outcome and current_num_durations_to_event_or_cutoff as time to event
            y_event, time_to_event = torch.tensor(y[0]).to(y_pred.device), torch.tensor(y[1]).float().to(y_pred.device)
            # loss = deephit_log_likelihood_loss(y_pred, y_event, time_to_event)
            loss = cox.neg_partial_log_likelihood(y_pred, y_event, time_to_event, ties_method = "efron")
            c_idx_dict = self._calculate_weighted_c_index_by_infusion(y_pred, y_event, time_to_event, num_infusion, mode='test')
            if c_idx_dict is None:
                return None
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
            loss = cox.neg_partial_log_likelihood(y_pred_logits, fail_indicators, time_to_events, ties_method = "efron")
            c_index = self.c_index(y_pred_logits, fail_indicators, time_to_events)
            
            self.log('test_c_index', c_index, prog_bar=True)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    
