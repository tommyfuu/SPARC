"""Shared metric helpers for the eval scripts (weighted C-index / next-infusion AUROC by infusion number)."""

import torch
from torchsurv.metrics.cindex import ConcordanceIndex
from torchmetrics import AUROC, AveragePrecision, Accuracy, Precision, Recall


def calculate_weighted_c_index_by_infusion(y_pred, y_event, time_to_event, num_infusion,
                                           mode='train', option='AKI_risk',
                                           extended_metrics=False):
    """C-index and next-infusion AUROC/AUPRC, weighted over infusion-number groups.

    Only groups where the metric is computable are used. extended_metrics=True also
    returns auprc_next_infusion_overall (written by eval_1).
    """
    auroc_fn = AUROC(task='binary')
    accuracy_fn = Accuracy(task='binary')
    precision_fn = Precision(task='binary')
    recall_fn = Recall(task='binary')
    auprc_fn = AveragePrecision(task='binary')
    c_index = ConcordanceIndex()
    output_dict = {}

    device = y_pred.device if torch.is_tensor(y_pred) else 'cpu'
    num_infusion_tensor = torch.as_tensor(num_infusion, device=device)
    y_event_tensor = torch.as_tensor(y_event, device=device)
    time_to_event_tensor = torch.as_tensor(time_to_event, device=device).float()

    unique_infusions = torch.unique(num_infusion_tensor)
    risk_scores = y_pred.squeeze()

    # C-index
    if 'risk' in option:
        # groups with both events and censored samples
        valid_infusions = []
        for infusion in unique_infusions:
            mask = num_infusion_tensor == infusion
            current_events = y_event_tensor[mask]

            if current_events.sum() > 0 and current_events.sum() < len(current_events):
                valid_infusions.append(infusion)

        # unstratified C-index
        try:
            c_idx_raw = c_index(
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
                current_times = time_to_event_tensor[mask].float()

                try:
                    c_idx_val = c_index(
                        current_risk.unsqueeze(1),
                        current_events.bool(),
                        current_times
                    )
                    c_idx_value = c_idx_val.item()
                    if not torch.isnan(c_idx_val):
                        computable_results.append({
                            'infusion': infusion.item(),
                            'c_index': c_idx_value,
                            'num_samples': mask.sum().item(),
                            'mask': mask
                        })
                except Exception as e:
                    continue

        if len(computable_results) > 0:
            total_samples = sum(r['num_samples'] for r in computable_results)
            c_index_weighted = sum(
                r['c_index'] * r['num_samples'] / total_samples
                for r in computable_results
            )

            # overall C-index pooled over the valid groups
            all_mask = torch.zeros_like(num_infusion_tensor, dtype=torch.bool)
            for r in computable_results:
                all_mask |= r['mask']

            try:
                overall_c_index = c_index(
                    risk_scores[all_mask].unsqueeze(1),
                    y_event_tensor[all_mask].bool(),
                    time_to_event_tensor[all_mask]
                ).item()
            except Exception:
                overall_c_index = 0.5

            num_groups = len(computable_results)

        output_dict.update({
            'c_index_raw': c_idx_raw,
            'c_index_weighted': c_index_weighted,
            'c_index_overall': overall_c_index,
            'num_groups': num_groups,
            'num_samples': total_samples,
            'per_infusion': computable_results,
        })

    # Next-infusion AUROC: label = event at the next infusion (time_to_event == 1)
    time_to_event_tensor = time_to_event_tensor.flatten()
    next_infusion_event = ((y_event_tensor == 1) & (time_to_event_tensor == 1)).float()

    valid_infusions_auroc = []
    for infusion in unique_infusions:
        mask = num_infusion_tensor == infusion
        current_next_events = next_infusion_event[mask]

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
    num_groups_auroc = 0
    num_samples_auroc = 0

    if len(valid_infusions_auroc) > 0:
        computable_auroc_results = []

        for infusion in valid_infusions_auroc:
            mask = num_infusion_tensor == infusion
            current_risk = risk_scores[mask]
            current_next_events = next_infusion_event[mask]
            try:
                current_res = {'num_samples': mask.sum().item()}
                if mode != 'train':
                    current_risk = current_risk.float()
                    auroc = auroc_fn(current_risk, current_next_events.int())
                    accuracy = accuracy_fn(current_risk, current_next_events.int())
                    precision = precision_fn(current_risk, current_next_events.int())
                    recall = recall_fn(current_risk, current_next_events.int())
                    auprc = auprc_fn(current_risk, current_next_events.int())

                    current_res = {
                        'infusion': infusion.item(),
                        'auroc': auroc.item(),
                        'accuracy': accuracy.item(),
                        'precision': precision.item(),
                        'recall': recall.item(),
                        'auprc': auprc.item(),
                        'num_samples': mask.sum().item(),
                        'num_positives': current_next_events.sum().item(),
                        'mask': mask
                    }

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

            if mode != 'train':
                # overall AUROC pooled over the valid groups
                all_mask_auroc = torch.zeros_like(num_infusion_tensor, dtype=torch.bool)
                for r in computable_auroc_results:
                    all_mask_auroc |= r['mask']

                try:
                    auroc_next_overall = auroc_fn(
                        risk_scores[all_mask_auroc],
                        next_infusion_event[all_mask_auroc].int()
                    ).item()
                except Exception:
                    auroc_next_overall = 0.5
                output_dict['auroc_next_infusion_overall'] = auroc_next_overall

                if extended_metrics:
                    try:
                        auprc_next_overall = auprc_fn(
                            risk_scores[all_mask_auroc],
                            next_infusion_event[all_mask_auroc].int()
                        ).item()
                    except Exception:
                        auprc_next_overall = 0.5
                    output_dict['auprc_next_infusion_overall'] = auprc_next_overall

                num_groups_auroc = len(computable_auroc_results)
                num_samples_auroc = total_samples_auroc
    else:
        if mode in ['test', 'val']:
            output_dict.update({
                'auroc_next_infusion_weighted': auroc_next_weighted,
                'auroc_next_infusion_overall': auroc_next_overall,
                'accuracy_next_infusion_weighted': accuracy_next_weighted,
                'precision_next_infusion_weighted': precision_next_weighted,
                'recall_next_infusion_weighted': recall_next_weighted,
                'auprc_next_infusion_weighted': auprc_next_weighted,
                'num_groups_next_infusion': num_groups_auroc,
                'num_samples_next_infusion': num_samples_auroc,
            })
            if extended_metrics:
                # eval_1 mirrors these defaults when no infusion group is computable
                output_dict['auprc_next_infusion_overall'] = 0.5

    return output_dict


def nextinf_auroc_weighted_overall(y_pred, next_event, num_infusion):
    """Numpy version of the weighted and overall next-infusion AUROC (same definition as above).

    Used for the bootstrap CIs in eval_1. Returns (nan, nan) if no group has both classes.
    """
    import numpy as _np
    from sklearn.metrics import roc_auc_score as _auc
    y_pred = _np.asarray(y_pred, dtype=float).ravel()
    next_event = _np.asarray(next_event).ravel().astype(int)
    num_infusion = _np.asarray(num_infusion).ravel()
    num, den, keep = 0.0, 0, _np.zeros(len(next_event), dtype=bool)
    for g in _np.unique(num_infusion):
        m = num_infusion == g
        npos = next_event[m].sum()
        if 0 < npos < m.sum():
            num += _auc(next_event[m], y_pred[m]) * m.sum()
            den += int(m.sum())
            keep |= m
    if den == 0:
        return float('nan'), float('nan')
    return num / den, _auc(next_event[keep], y_pred[keep])
