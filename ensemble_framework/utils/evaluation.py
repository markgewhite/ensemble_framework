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