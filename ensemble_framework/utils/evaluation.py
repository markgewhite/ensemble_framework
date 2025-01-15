from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score, confusion_matrix)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> Dict:
    """
    Compute comprehensive set of evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)

    return metrics


def compare_models(results: List[Dict[str, Dict]],
                   model_names: List[str]) -> pd.DataFrame:
    """
    Compare performance metrics across multiple models.

    Args:
        results: List of result dictionaries from different models
        model_names: Names of the models for identification

    Returns:
        DataFrame with comparative metrics
    """
    comparison = []

    for result, name in zip(results, model_names):
        # Get sample-level metrics
        sample_metrics = result.get('sample_level', {})
        patient_metrics = result.get('patient_level', {})

        row = {
            'model': name,
            'sample_auc': sample_metrics.get('auc', np.nan),
            'sample_accuracy': sample_metrics.get('accuracy', np.nan),
            'sample_sensitivity': sample_metrics.get('sensitivity', np.nan),
            'sample_specificity': sample_metrics.get('specificity', np.nan),
            'sample_f1': sample_metrics.get('f1', np.nan),
            'patient_auc': patient_metrics.get('auc', np.nan),
            'patient_accuracy': patient_metrics.get('accuracy', np.nan),
            'patient_sensitivity': patient_metrics.get('sensitivity', np.nan),
            'patient_specificity': patient_metrics.get('specificity', np.nan),
            'patient_f1': patient_metrics.get('f1', np.nan)
        }
        comparison.append(row)

    return pd.DataFrame(comparison)


def aggregate_ground_truth_and_predictions(dataset,
                                           results: dict,
                                           level: str = 'patient'):
    """
    Returns (y_true, y_pred, y_prob) at either the 'sample' or 'patient' level.

    Args:
        dataset: a Dataset object with .X, .y, .patient_ids, etc.
        results: dict from model.predict_patients(...), containing
                 'sample_level' and 'patient_level' predictions.
        level: 'sample' or 'patient'.

    Returns:
        tuple of (y_true, y_pred, y_prob), each a 1D np.ndarray.
    """

    if level == 'sample':
        # In this scenario, ground truth is just dataset.y,
        # and predictions come directly from 'sample_level'
        y_true = dataset.y
        y_pred = results['sample_level']['y_pred']
        y_prob = results['sample_level']['y_prob']
        return y_true, y_pred, y_prob

    elif level == 'patient':
        # 1) Aggregate ground truth to patient level
        #    Assuming dataset has a method like "aggregate_patient_labels()"
        #    which returns (unique_patients, aggregated_y)
        unique_pids_true, y_true_agg = dataset.aggregate_patient_labels(aggregator='max')
        # or 'mean', 'majority', etc. -- up to you

        # 2) The predictions at patient-level presumably already exist in 'results'
        #    from predict_patients(...). So we just retrieve them.
        #    'results' gives us:
        #      'patient_ids': shape (n_patients,)
        #      'y_pred': shape (n_patients,)
        #      'y_prob': shape (n_patients,)
        unique_pids_pred = results['patient_level']['patient_ids']
        y_pred_agg = results['patient_level']['y_pred']
        y_prob_agg = results['patient_level']['y_prob']

        # 3) Now, we must ensure that the order of unique_pids_true matches unique_pids_pred
        #    If they are guaranteed to be the same order, we can just return them.
        #    Otherwise, we can do a mapping.
        #    Here's the "simple" case if both arrays are guaranteed to match in order:
        if np.array_equal(unique_pids_true, unique_pids_pred):
            return y_true_agg, y_pred_agg, y_prob_agg
        else:
            # We need to align them by patient ID
            # Convert them to dictionary or do a reindex
            pid_to_truth = {pid: val for pid, val in zip(unique_pids_true, y_true_agg)}
            pid_to_pred = {pid: val for pid, val in zip(unique_pids_pred, y_pred_agg)}
            pid_to_prob = {pid: val for pid, val in zip(unique_pids_pred, y_prob_agg)}

            # Rebuild in the order of unique_pids_pred, for instance
            new_y_true = np.array([pid_to_truth[pid] for pid in unique_pids_pred])
            new_y_pred = np.array([pid_to_pred[pid] for pid in unique_pids_pred])
            new_y_prob = np.array([pid_to_prob[pid] for pid in unique_pids_pred])

            return new_y_true, new_y_pred, new_y_prob
    else:
        raise ValueError("level must be 'sample' or 'patient'")


def aggregate_feature_importance(fi_list: List[pd.DataFrame]) -> pd.DataFrame:
    # For example, we assume each fi DF has the same columns and rows in the same order
    # Then we can average the numeric columns across them
    # Return a final DF with mean importance over repeats
    if not fi_list:
        return None

    # Concatenate
    fi_concat = pd.concat(fi_list, axis=0, ignore_index=True)
    # Group by 'feature_name' and compute mean
    fi_mean = fi_concat.groupby('feature_name', as_index=False).mean()
    # Or you can do .agg(['mean','std']) if you want both
    fi_mean.sort_values('importance_mean', ascending=False, inplace=True)
    return fi_mean



def summarize_feature_importance(importance_df: pd.DataFrame,
                                 top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Summarize feature importance results.

    Args:
        importance_df: DataFrame with feature importance results
        top_n: Number of top features to include (optional)

    Returns:
        Summary DataFrame
    """
    summary = importance_df.sort_values('importance_mean', ascending=False)

    if top_n is not None:
        summary = summary.head(top_n)

    return summary[['feature_name', 'selection_rate', 'importance_mean', 'importance_std']]