"""
Summary exporter for the design-choice figures: writes <stem>_summary_long.csv (one row per
task/model/fold) and <stem>_summary_wide.csv (mean, std, n_folds) from the same dataframes the figure plots.
"""

import os
import numpy as np
import pandas as pd


def build_design_choice_summary(
    *,
    design_choice,
    task_data,
    available_tasks,
    models,
    metrics,
    save_dir,
    out_stem,
    task_display=None,
    model_display=None,
    to_percent=True,
    extra_cols=None,
    verbose=True,
):
    """Write long and wide summary CSVs from the figure's per-task dataframes.

    metrics maps metric_name -> source column, or -> {task: column} when the column varies by task.
    Returns (long_df, wide_df, long_path, wide_path).
    """
    os.makedirs(save_dir, exist_ok=True)
    extra_cols = extra_cols or []

    def _resolve_col(metric_col, task):
        if isinstance(metric_col, dict):
            return metric_col.get(task)
        return metric_col

    long_rows = []
    for task in available_tasks:
        df = task_data.get(task)
        if df is None or 'model' not in df.columns:
            continue
        for model in models:
            rows = df[df['model'] == model]
            if rows.empty:
                continue
            for metric_name, metric_col in metrics.items():
                col = _resolve_col(metric_col, task)
                if col is None or col not in rows.columns:
                    continue
                sub = rows[['val_fold', col]].dropna(subset=[col]) if 'val_fold' in rows.columns \
                    else rows[[col]].dropna(subset=[col])
                if sub.empty:
                    continue
                vals = sub[col].astype(float).values
                # Match get_vals(): convert to percent only if it looks like a proportion.
                if to_percent and vals.size and np.nanmax(vals) <= 1.0:
                    vals = vals * 100.0
                folds = sub['val_fold'].values if 'val_fold' in sub.columns else np.arange(len(vals))
                for fold, v in zip(folds, vals):
                    row = {
                        'design_choice': design_choice,
                        'task': task,
                        'task_display': (task_display or {}).get(task, task),
                        'model': model,
                        'model_display': (model_display or {}).get(model, model),
                        'val_fold': fold,
                        'metric': metric_name,
                        'metric_value': v,
                    }
                    # carry passthrough columns from the matching source row
                    if extra_cols and 'val_fold' in rows.columns:
                        src = rows[rows['val_fold'] == fold]
                        if not src.empty:
                            for ec in extra_cols:
                                if ec in src.columns:
                                    row[ec] = src.iloc[0][ec]
                    long_rows.append(row)

    long_df = pd.DataFrame(long_rows)

    # Wide roll-up: mean/std/n across folds per (task, model, metric).
    if not long_df.empty:
        wide_df = (
            long_df
            .groupby(['design_choice', 'task', 'task_display', 'model', 'model_display', 'metric'],
                     dropna=False)['metric_value']
            .agg(mean='mean', std='std', n_folds='count')
            .reset_index()
        )
        wide_df['std'] = wide_df['std'].fillna(0.0)
    else:
        wide_df = pd.DataFrame(
            columns=['design_choice', 'task', 'task_display', 'model',
                     'model_display', 'metric', 'mean', 'std', 'n_folds']
        )

    long_path = os.path.join(save_dir, f'{out_stem}_summary_long.csv')
    wide_path = os.path.join(save_dir, f'{out_stem}_summary_wide.csv')
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    if verbose:
        n_tasks = long_df['task'].nunique() if not long_df.empty else 0
        n_models = long_df['model'].nunique() if not long_df.empty else 0
        print("\n" + "=" * 70)
        print(f"DATA SUMMARY EXPORT — {design_choice}")
        print("=" * 70)
        print(f"  tasks: {n_tasks}, models: {n_models}, "
              f"metrics: {list(metrics.keys())}, rows(long): {len(long_df)}")
        print(f"  long: {long_path}")
        print(f"  wide: {wide_path}")
        print("=" * 70 + "\n")

    return long_df, wide_df, long_path, wide_path
