import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
# import metrics: AUROC, AUPRC, F1, etc.
from torchmetrics.classification import AUROC, Accuracy, Precision, Recall, F1Score, ROC, AveragePrecision
import torch.optim as optim
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from torch.nn.parameter import Parameter
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from ob_propagation import Observation_progation
import numpy as np
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex


class PositionalEncoding(nn.Module):
    def __init__(self, max_time=20000, n_dim=10):
        super(PositionalEncoding, self).__init__()
        self.max_time = max_time
        self.n_dim = n_dim
        self.num_timescales = n_dim // 2
        
        # Precompute timescales
        self.timescales = max_time ** torch.linspace(0, 1, self.num_timescales)

    def forward(self, times):
        scaled_time = times.unsqueeze(-1) / self.timescales.to(times.device)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return signal


class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000,):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        P_time = P_time.float()

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)

        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe

def nanstd(x): 
    return torch.sqrt(torch.nanmean((x - torch.nanmean(x))**2))
def obprop_safe(ob_layer, stepdata, edge_index, edge_weights=None, edge_attr=None):
    """
    Safe wrapper for an Observation_progation layer: handles NaNs, all-zero
    inputs, or invalid shapes. (Same safety behavior as the old
    transconv_safe; renamed/adapted since the graph layer is now
    Raindrop_v2's Observation_progation instead of a plain TransformerConv.)
    """
    if torch.isnan(stepdata).any():
        print("⚠️ NaNs in stepdata — replacing with zeros")
        stepdata = torch.nan_to_num(stepdata, nan=0.0)

    if torch.all(stepdata == 0):
        print("⚠️ stepdata is all zeros — skipping ob_propagation")
        dummy_out = torch.zeros_like(stepdata)
        dummy_weights = torch.zeros(edge_index.shape[1], 1, device=stepdata.device)
        return dummy_out, (edge_index, dummy_weights)

    try:
        out, attn = ob_layer(
            stepdata,
            edge_index=edge_index,
            edge_weights=edge_weights,
            edge_attr=edge_attr,
            return_attention_weights=True
        )
    except Exception as e:
        print("⚠️ ob_propagation failed — replacing with zeros")
        print("Error:", e)
        out = torch.zeros_like(stepdata)
        attn = (edge_index, torch.zeros(edge_index.shape[1], 1, device=stepdata.device))

    # Final safety net
    if torch.isnan(out).any():
        print("⚠️ NaNs in ob_propagation output — zeroing")
        out = torch.nan_to_num(out, nan=0.0)

    return out, attn
    
class Raindrop(pl.LightningModule):
    ""
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, d_inp=30, d_model=60, nhead=2, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9, n_classes=1, lr=1e-3, weight_decay=1e-5,
                 MAX=100, perc=0.5, aggreg='mean', global_structure = None, static=True,
                 #side effects
                 option = 'binary',
                 side_effects_num_infusion=False, test_metrics_by_infusion_group=True, advanced_model = 'default'):
        super(Raindrop, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.lr = lr
        self.weight_decay = weight_decay
        self.option = option
        self.side_effects_num_infusion = side_effects_num_infusion
        self.test_metrics_by_infusion_group = test_metrics_by_infusion_group

        # Override side_effects_num_infusion for binary tasks (only relevant for side-effect tasks)
        if option in ['binary', 'risk'] and self.side_effects_num_infusion:
            print(f"WARNING: side_effects_num_infusion=True is not applicable for option '{option}'. Setting to False.")
            self.side_effects_num_infusion = False

        self.global_structure = torch.ones(d_inp, max_len).cuda()  # complete graph

        d_pe = d_inp
        d_enc = d_inp

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_model+d_inp, nhead, nhid, dropout)  # nhid is the number of hidden, 36 is the dim of timestamp

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.gcs = nn.ModuleList()
        conv_name = 'dense_hgt' # 'hgt' # 'dense_hgt',  'gcn', 'dense_hgt'
        # num_types, num_relations = 36, 1
        num_types, num_relations = d_inp, 1

        self.edge_type_train = torch.ones([d_inp*d_inp*2], dtype=torch.int64).cuda()  # 2 times fully-connected graph
        self.adj = torch.ones([d_inp, d_inp]).cuda()  # complete graph

        """For GIN"""
        self.dim = int(d_model/d_inp)  # the output dim of each node in graph
        in_D, hidden_D = 1, self.dim  # each timestamp input 2 value, map to self.dim dimension feature, output is self.dim
        self.GINstep1 = GINConv(
            Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
                       Linear(hidden_D, hidden_D), ReLU()))

        # Raindrop_v2's graph step: two chained Observation_progation layers.
        # Layer 1 runs on the fixed adjacency (edge_weights below); layer 2
        # runs on the edge weights layer 1 produced -- that hand-off is what
        # makes the sensor graph adaptive rather than fixed, which is the
        # thing that distinguishes "v2" from the plain-TransformerConv v1
        # baseline this used to be.
        self.ob_propagation = Observation_progation(
            in_channels=max_len*self.dim, out_channels=max_len*self.dim,
            heads=1, n_nodes=d_inp, ob_dim=self.dim, dropout=dropout)
        self.ob_propagation_layer2 = Observation_progation(
            in_channels=max_len*self.dim, out_channels=max_len*self.dim,
            heads=1, n_nodes=d_inp, ob_dim=self.dim, dropout=dropout)

        self.GINmlp = nn.Sequential(
            nn.Linear(self.dim+d_static, self.dim+d_static),  # self.dim for observation, d_static for static info
            nn.ReLU(),
            nn.Linear(self.dim+d_static, n_classes),
        )

        if static == False:
            d_final = d_enc + d_pe
        else:
            d_final = d_enc + d_pe + d_inp + d_model

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.d_inp = d_inp
        self.d_model = d_model
        self.encoder = nn.Linear(d_inp, d_enc)
        self.static = static
        if self.static:
            self.emb = nn.Linear(d_static, d_model)

        # self.MLP_replace_transformer = nn.Linear(72, d_inp)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        # if n_classes == 1:
        #     # add sigmoid
        #     self.end_activation = nn.Sigmoid()
        if n_classes == 1 and option in ['binary', 'side_effects']:
            # add sigmoid
            self.end_activation = nn.Sigmoid()
        elif n_classes == 1 and option == 'side_effects_risk':
            self.end_activation = None
            # num_infusion added if side_effects_num_infusion is True
            if self.side_effects_num_infusion:
                self.num_infusion_MLP = nn.Sequential(
                    nn.Linear(1, 1),
                    nn.ReLU(),
                )
                self.combine_MLP = nn.Sequential(
                    nn.Linear(n_classes + 1, n_classes),
                )
            self.c_index = ConcordanceIndex()

        self.aggreg = aggreg

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

         # validation metrics
        self.accuracy = Accuracy(task='binary').to('cuda')
        self.precision = Precision(task='binary').to('cuda')
        self.recall = Recall(task='binary').to('cuda')
        self.f1 = F1Score(task='binary').to('cuda')
        self.auroc = AUROC(task='binary').to('cuda')
        self.auprc = AveragePrecision(task='binary').to('cuda')
        self.loss = nn.BCELoss()
        self.loss_fn = nn.BCELoss(reduction='none')


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
    #                 time_to_event_tensor.float()
    #             ).item()
    #         except:
    #             c_idx_raw = 0.5
            
    #         # Initialize with defaults
    #         c_index_weighted = 0.5
    #         overall_c_index = 0.5
    #         num_groups = 0
    #         total_samples = 0
    #         computable_results = []
    #         # print("valid_infusions:", valid_infusions)
    #         if len(valid_infusions) > 0:
    #             # Second pass: compute C-index for valid infusions
    #             for infusion in valid_infusions:
    #                 mask = num_infusion_tensor == infusion
    #                 current_risk = risk_scores[mask]
    #                 current_events = y_event_tensor[mask]
    #                 current_times = time_to_event_tensor[mask]
    #                 # print("infusion", infusion.item(), "current_events", current_events, "current_times", current_times)
    #                 try:
    #                     c_idx = self.c_index(
    #                         current_risk.unsqueeze(1),
    #                         current_events.bool(),
    #                         current_times.float()
    #                     )
    #                     # # print("infusion", infusion.item(), "c_idx", c_idx.item())
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
    #                     time_to_event_tensor[all_mask].float()
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

    def forward(self, src, static, times, lengths, num_infusions=None):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        # input shape (batch_size, seq_len, num_features) ->  [seq_len, batch_size, num_features]
        src = src.permute(1, 0, 2) # [seq_len, batch_size, num_features]
        times = times.permute(1, 0) # [seq_len, batch_size]
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :self.d_inp]
        maxlen, batch_size = src.shape[0], src.shape[1]
        src = self.encoder(src) * math.sqrt(self.d_model)
        pe = self.pos_encoder(times)
        src = self.dropout(src)

        if static is not None:
            emb = self.emb(static)
            # print("emb shape", emb.shape)

        withmask = False
        if withmask==True:
            x = torch.cat([src, missing_mask], dim=-1)
        else:
            x = src
        
        mask = torch.arange(maxlen).unsqueeze(0).expand(batch_size, -1).cuda() >= lengths.unsqueeze(1)
        

        step2 = True  # If skip step 2, direct process the input data
        if step2 == False:
            output = x
            distance = 0
        elif step2 == True:
            adj = self.global_structure.cuda()
            adj[torch.eye(self.d_inp).byte()] = 1 # TODO: vulnerable when max_len < d_inp
            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            # graph message passing (Raindrop_v2: two chained
            # Observation_progation hops, layer 2 using layer 1's own
            # edge weights, so the sensor graph adapts between hops)
            output = torch.zeros([maxlen, src.shape[1], self.d_inp*self.dim]).cuda()
            alpha_all = torch.zeros([edge_index.shape[1],  src.shape[1] ]).cuda()

            # expand each sensor's scalar reading into self.dim ("d_ob")
            # channels -- Raindrop_v2's per-sensor observation embedding
            x_expanded = torch.repeat_interleave(x, self.dim, dim=-1)  # [maxlen, batch, d_inp*dim]

            for i, unit in enumerate(range(0, x.shape[1])):
                stepdata = x_expanded[:, unit, :]  # [maxlen, d_inp*dim]
                # give each of the d_inp sensor *nodes* its full time series
                # (length maxlen*dim) as its node feature, instead of one
                # scalar per node per call
                stepdata = stepdata.reshape([maxlen, self.d_inp, self.dim]).permute(1, 0, 2)
                stepdata = stepdata.reshape([self.d_inp, maxlen*self.dim])

                stepdata, attentionweights = obprop_safe(
                    self.ob_propagation, stepdata,
                    edge_index=edge_index,
                    edge_weights=edge_weights,
                )
                edge_index_layer2 = attentionweights[0]
                edge_weights_layer2 = attentionweights[1].squeeze(-1)

                stepdata, attentionweights = obprop_safe(
                    self.ob_propagation_layer2, stepdata,
                    edge_index=edge_index_layer2,
                    edge_weights=edge_weights_layer2,
                )

                stepdata = stepdata.view([self.d_inp, maxlen, self.dim]).permute(1, 0, 2)
                stepdata = stepdata.reshape([-1, self.d_inp*self.dim])
                output[:, unit, :] = stepdata

                alpha_all[:, unit] = attentionweights[1].squeeze(-1)


            # calculate Euclidean distance between alphas
            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        # fill nans with 0s in output
        output = torch.nan_to_num(output, nan=0.0)
        # use transformer to aggregate temporal information
        output = torch.cat([output, pe], dim=-1)
        output = torch.nan_to_num(output, nan=0.0)
        step3 = True
        if step3==True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step3==False:
            r_out = output
        # print("r_out shape", r_out.shape)
        masked_agg = True
        if masked_agg ==True:
            mask2 = mask.permute(1, 0).unsqueeze(2).float()
            if self.aggreg == 'mean':
                lengths2 = lengths.unsqueeze(1)
                # print("NaNs in r_out:", torch.isnan(r_out).any().item())
                output = torch.sum(r_out * (1 - mask2), dim=0)
                # print("r_out * (1 - mask2)", output[3:])
                # print("lengths2 [3:]", lengths2[3:])
                output = output / (lengths2+1)
                # output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
                # output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2.clamp(min=1).float())
                # print("output shape after mean [3:]", output[3:])
        elif masked_agg ==False:
            # Without masked aggregation across rows
            output = r_out[-1, :, :].squeeze(0)

        if static is not None:
            output = torch.cat([output, emb], dim=1)
        # print("output shape after concat", output.shape)
        output = self.mlp_static(output)
        # print("output shape after mlp_static [3:]", output[3:])
        # final activation
        if self.end_activation is not None:
            output = self.end_activation(output)
        if self.option == 'side_effects_risk' and self.side_effects_num_infusion and num_infusions is not None:
            # process num_infusions
            # print("going to process num_infusions", num_infusions)
            num_infusion = num_infusions.unsqueeze(1).float()
            num_infusion_emb = self.num_infusion_MLP(num_infusion)
            # combine with output
            output = self.combine_MLP(torch.cat([output, num_infusion_emb], dim=1))
        # print("output shape after mlp_static", output.shape)

        # return output, distance, None
        return output

    def predict_step(self, batch, batch_idx):
        # (demo, times, values, measurements, lengths), pad_lens, ys = batch
        # (demo, times, values, measurements, lengths, pad_lens), ys = batch
        # y_hat = self(demo, times, values, measurements, lengths).squeeze().float()
        (xs, static, seq_len, times, num_infusions), ys = batch
        # print("values", values)
        xs = self.normalization(xs)
        # stack the list into a tensor
        static = torch.stack(static, dim=1).float()
        # print("static shape", static.shape)
        y_hat = self(xs, static, times, seq_len).squeeze().float()
        if 'risk' not in self.option:
            ys = ys.to('cuda')
        y_hat = y_hat.to('cuda')
        return {'y_pred': y_hat, 'y_true': ys, 'num_infusion': num_infusions}
    # def data_prep(self, data):
    #     (demo, times, values, measurements, lengths), pad_lens, ys = data
    #     return demo, times, values, measurements, lengths

    ### BELOW IS ALL THE PYTORCH LIGHTNING THINGIS
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
        return {
           'optimizer': optimizer,
           'scheduler': scheduler,
           'monitor': 'val_loss'
       }

    def normalization(self, xs):
        # for i in range(xs.size(-1)):
        #     mean = xs[:, :, i].nanmean()
        #     std = nanstd(xs[:, :, i])
        #     xs[:, :, i] = (xs[:, :, i] - mean) / std
        # fill nans with -10s
        xs = torch.nan_to_num(xs, nan=-10)
        return xs
        
    def training_step(self, batch, batch_idx):
        # print("prior to training data loading")
        # (demo, times, values, measurements, lengths), pad_lens, ys = batch
        (xs, static, seq_len, times, num_infusions), ys = batch
        # print("values", values)
        xs = self.normalization(xs)
        # stack the list into a tensor
        static = torch.stack(static, dim=1).float()
        # print("static shape", static.shape)
        y_hat = self(xs, static, times, seq_len, num_infusions).squeeze().float()
        # ys, y_hat = ys.to('cuda'), y_hat.to('cuda')
        # loss = self.loss(y_hat, ys.float())
        # # Log metrics
        # self.log('train_loss', loss)
        # Log metrics
        if self.option == 'binary':
            ys, y_hat = ys.to('cuda'), y_hat.to('cuda')
            loss = self.loss(y_hat, ys.float())
            self.log('train_loss', loss)
        elif self.option in ['side_effects_risk', 'side_effects_next_risk']:
            y_hat = y_hat.to('cuda')
            y_event, time_to_event = torch.tensor(ys[0]).to(y_hat.device), torch.tensor(ys[1]).to(y_hat.device)
            loss = cox.neg_partial_log_likelihood(y_hat, y_event, time_to_event, ties_method = "efron")
        else:
            # set it to all 1s and fit the number of patients
            time_to_event = torch.ones_like(y_hat).float().to(y_hat.device)
            y_event, num_infusion = torch.tensor(ys).to(y_hat.device), torch.tensor(num_infusions).to(y_hat.device)
            res_dict = self._calculate_weighted_c_index_by_infusion(y_hat, ys, time_to_event, num_infusions, mode='train')
            if res_dict is None:
                return None
            loss = res_dict.get('binary_loss', torch.tensor(0.0, device=y_hat.device))
        return loss

    def validation_step(self, batch, batch_idx):
        # (demo, times, values, measurements, lengths), pad_lens, ys = batch
        # (demo, times, values, measurements, lengths, pad_lens), ys = batch
        # y_hat = self(demo, times, values, measurements, lengths).squeeze().float()
        (xs, static, seq_len, times, num_infusions), ys = batch
        # print("values", values)
        xs = self.normalization(xs)
        # stack the list into a tensor
        static = torch.stack(static, dim=1).float()
        # print("static shape", static.shape)
        y_hat = self(xs, static, times, seq_len, num_infusions).squeeze().float()
        # ys, y_hat = ys.to('cuda'), y_hat.to('cuda')
        # loss = self.loss(y_hat, ys.float())
        # # ys = ys.float()
        # val_auprc, val_auroc, val_acc, val_precision, val_recall, val_f1 = self.auprc(y_hat, ys), self.auroc(y_hat, ys), self.accuracy(y_hat, ys), self.precision(y_hat, ys), self.recall(y_hat, ys), self.f1(y_hat, ys)
        # # Log metrics
        # self.log('val_loss', loss, prog_bar=True)
        # self.log('val_auprc', val_auprc, prog_bar=True)
        # self.log('val_auroc', val_auroc, prog_bar=True)
        # self.log('val_acc', val_acc, prog_bar=True)
        # self.log('val_precision', val_precision, prog_bar=True)
        # self.log('val_recall', val_recall, prog_bar=True)
        # self.log('val_f1', val_f1, prog_bar=True)
        if self.option == 'binary':
            ys, y_hat = ys.to('cuda'), y_hat.to('cuda')
            loss = self.loss(y_hat, ys.float())
            self.log('train_loss', loss)
            # ys = ys.float()
            val_auprc, val_auroc, val_acc, val_precision, val_recall, val_f1 = self.auprc(y_hat, ys), self.auroc(y_hat, ys), self.accuracy(y_hat, ys), self.precision(y_hat, ys), self.recall(y_hat, ys), self.f1(y_hat, ys)
            # Log metrics
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_auprc', val_auprc, prog_bar=True)
            self.log('val_auroc', val_auroc, prog_bar=True)
            self.log('val_acc', val_acc, prog_bar=True)
            self.log('val_precision', val_precision, prog_bar=True)
            self.log('val_recall', val_recall, prog_bar=True)
            self.log('val_f1', val_f1, prog_bar=True)
        elif self.option in ['side_effects_risk', 'side_effects_next_risk']:
            y_hat = y_hat.to('cuda')
            y_event, time_to_event = torch.tensor(ys[0]).to(y_hat.device), torch.tensor(ys[1]).to(y_hat.device)
            loss = cox.neg_partial_log_likelihood(y_hat, y_event, time_to_event, ties_method = "efron")
            c_idx_dict = self._calculate_weighted_c_index_by_infusion(y_hat, y_event, time_to_event, num_infusions, mode='val')
            if c_idx_dict is None:
                return None
            c_index, c_index_weighted, overall_c_index = c_idx_dict['c_index_raw'], c_idx_dict['c_index_weighted'], c_idx_dict['c_index_overall']
            weighted_auroc, overall_auroc = c_idx_dict['auroc_next_infusion_weighted'], c_idx_dict['auroc_next_infusion_overall']
            self.log("val_loss", loss, prog_bar=True)
            self.log('val_cindex_unfiltered', c_index, prog_bar=True)
            self.log('val_cindex_weighted', c_index_weighted, prog_bar=True)
            self.log('val_cindex_overall_valid', overall_c_index, prog_bar=True)
            self.log('val_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
            self.log('val_auroc_nextinf_overall', overall_auroc, prog_bar=True)
        else: # side effects
            # set it to all 1s and fit the number of patients
            time_to_event = torch.ones_like(y_hat).float().to(y_hat.device)
            y_event, num_infusion = torch.tensor(ys).to(y_hat.device), torch.tensor(num_infusions).to(y_hat.device)
            res_dict = self._calculate_weighted_c_index_by_infusion(y_hat, ys, time_to_event, num_infusions, mode='val')
            if res_dict is None:
                return None
            loss = res_dict.get('binary_loss', torch.tensor(0.0, device=y_hat.device))
            self.log('val_loss', loss, prog_bar=True)
            weighted_auroc, overall_auroc, accuracy, precision, recall, auprc = res_dict.get('auroc_next_infusion_weighted'), res_dict.get('auroc_next_infusion_overall'), res_dict.get('accuracy_next_infusion_weighted'), res_dict.get('precision_next_infusion_weighted'), res_dict.get('recall_next_infusion_weighted'), res_dict.get('auprc_next_infusion_weighted')
            if self.test_metrics_by_infusion_group:
                per_infusion_auroc = res_dict.get('per_infusion_auroc')
                for res in per_infusion_auroc:
                    infusion_id = res['infusion']
                    self.log(f'val_auroc_infusion_{infusion_id}', res['auroc'], prog_bar=False)
                    self.log(f'val_infusion_num_samples_{infusion_id}', res['num_samples'], prog_bar=False)
            self.log('val_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
            self.log('val_auroc_nextinf_overall', overall_auroc, prog_bar=True)
            
        return loss

    def test_step(self, batch, batch_idx):
        # (demo, times, values, measurements, lengths), pad_lens, ys = batch
        # (demo, times, values, measurements, lengths, pad_lens), ys = batch
        # y_hat = self(demo, times, values, measurements, lengths).squeeze().float()
        (xs, static, seq_len, times, num_infusions), ys = batch
        # print("values", values)
        xs = self.normalization(xs)
        # stack the list into a tensor
        static = torch.stack(static, dim=1).float()  # make sure static is float
        # print("static shape", static.shape)
        y_hat = self(xs, static, times, seq_len, num_infusions).squeeze().float()
        # ys, y_hat = ys.to('cuda'), y_hat.to('cuda')
        # # ys = ys.float()
        # # print("ys", ys)
        # # print("y_hat", y_hat)
        # loss = self.loss(y_hat, ys.float())
        # test_auprc, test_auroc, test_acc, test_precision, test_recall, test_f1 = self.auprc(y_hat, ys), self.auroc(y_hat, ys), self.accuracy(y_hat, ys), self.precision(y_hat, ys), self.recall(y_hat, ys), self.f1(y_hat, ys)
        # # Log metrics
        # self.log('test_loss', loss, prog_bar=True)
        # self.log('test_auprc', test_auprc, prog_bar=True)
        # self.log('test_auroc', test_auroc, prog_bar=True)
        # self.log('test_acc', test_acc, prog_bar=True)
        # self.log('test_precision', test_precision, prog_bar=True)
        # self.log('test_recall', test_recall, prog_bar=True)
        # self.log('test_f1', test_f1, prog_bar=True)
        if self.option == 'binary':
            ys, y_hat = ys.to('cuda'), y_hat.to('cuda')
            loss = self.loss(y_hat, ys.float())
            self.log('train_loss', loss)
            # ys = ys.float()
            test_auprc, test_auroc, test_acc, test_precision, test_recall, test_f1 = self.auprc(y_hat, ys), self.auroc(y_hat, ys), self.accuracy(y_hat, ys), self.precision(y_hat, ys), self.recall(y_hat, ys), self.f1(y_hat, ys)
            # Log metrics
            self.log('test_loss', loss, prog_bar=True)
            self.log('test_auprc', test_auprc, prog_bar=True)
            self.log('test_auroc', test_auroc, prog_bar=True)
            self.log('test_acc', test_acc, prog_bar=True)
            self.log('test_precision', test_precision, prog_bar=True)
            self.log('test_recall', test_recall, prog_bar=True)
            self.log('test_f1', test_f1, prog_bar=True)
        elif self.option in ['side_effects_risk', 'side_effects_next_risk']:
            y_hat = y_hat.to('cuda')
            y_event, time_to_event = torch.tensor(ys[0]).to(y_hat.device), torch.tensor(ys[1]).to(y_hat.device)
            loss = cox.neg_partial_log_likelihood(y_hat, y_event, time_to_event, ties_method = "efron")
            c_idx_dict = self._calculate_weighted_c_index_by_infusion(y_hat, y_event, time_to_event, num_infusions, mode='val')
            if c_idx_dict is None:
                return None
            c_index, c_index_weighted, overall_c_index = c_idx_dict['c_index_raw'], c_idx_dict['c_index_weighted'], c_idx_dict['c_index_overall']
            weighted_auroc, overall_auroc = c_idx_dict['auroc_next_infusion_weighted'], c_idx_dict['auroc_next_infusion_overall']
            self.log("test_loss", loss, prog_bar=True)
            self.log('test_cindex_unfiltered', c_index, prog_bar=True)
            self.log('test_cindex_weighted', c_index_weighted, prog_bar=True)
            self.log('test_cindex_overall_valid', overall_c_index, prog_bar=True)
            self.log('test_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
            self.log('test_auroc_nextinf_overall', overall_auroc, prog_bar=True)
        else: # side effects
            time_to_event = torch.ones_like(y_hat).float().to(y_hat.device)
            y_event, num_infusion = torch.tensor(ys).to(y_hat.device), torch.tensor(num_infusions).to(y_hat.device)
            res_dict = self._calculate_weighted_c_index_by_infusion(y_hat, y_event, time_to_event, num_infusion, mode='test')
            if res_dict is None:
                return None
            loss = res_dict.get('binary_loss', torch.tensor(0.0, device=y_hat.device))
            weighted_auroc, overall_auroc, accuracy, precision, recall, auprc = res_dict.get('auroc_next_infusion_weighted'), res_dict.get('auroc_next_infusion_overall'), res_dict.get('accuracy_next_infusion_weighted'), res_dict.get('precision_next_infusion_weighted'), res_dict.get('recall_next_infusion_weighted'), res_dict.get('auprc_next_infusion_weighted')
            if self.test_metrics_by_infusion_group:
                per_infusion_auroc = res_dict.get('per_infusion_auroc')
                # log per-infusion group metrics
                for res in per_infusion_auroc:
                    infusion_id = res['infusion']
                    self.log(f'test_auroc_infusion_{infusion_id}', res['auroc'], prog_bar=False)
                    self.log(f'test_infusion_num_samples_{infusion_id}', res['num_samples'], prog_bar=False)

            self.log('test_auroc_nextinf_weighted', weighted_auroc, prog_bar=True)
            self.log('test_auroc_nextinf_overall', overall_auroc, prog_bar=True)
                
        return loss
