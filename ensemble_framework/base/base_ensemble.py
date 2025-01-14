from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score, confusion_matrix)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


class BaseEnsemble(BaseEstimator, ClassifierMixin):
    """Base class for all ensemble methods"""

    def __init__(self, random_state: Optional[int] = None, base_pipeline: Optional[Pipeline] = None):
        self.random_state = random_state
        self.base_pipeline = base_pipeline
        self.models_ = []
        self.feature_names_ = None

    def predict_patients(self, X: np.ndarray, patient_ids: np.ndarray) -> Dict:
        """Make predictions at both sample and patient levels."""
        # Get sample-level predictions
        y_pred, y_prob = self.predict(X)

        # Aggregate to patient level
        unique_patients = np.unique(patient_ids)
        patient_probs = np.zeros(len(unique_patients))

        for i, patient_id in enumerate(unique_patients):
            patient_mask = (patient_ids == patient_id)
            patient_probs[i] = np.mean(y_prob[patient_mask])

        patient_preds = (patient_probs > 0.5).astype(int)

        return {
            'sample_level': {
                'y_pred': y_pred,
                'y_prob': y_prob
            },
            'patient_level': {
                'patient_ids': unique_patients,
                'y_pred': patient_preds,
                'y_prob': patient_probs
            }
        }

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: np.ndarray) -> Dict:
        """Evaluate predictions using multiple metrics"""
        return {
            'auc': roc_auc_score(y_true, y_prob),
            'accuracy': accuracy_score(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred),
            'specificity': precision_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    def feature_importance(self, X: np.ndarray, y: np.ndarray,
                           n_repeats: int = 10) -> pd.DataFrame:
        """Compute feature selection rates and permutation importance."""
        n_features = X.shape[1]
        n_models = len(self.models_)

        # Track feature selection per fold
        selection_matrix = np.zeros((n_features, n_models), dtype=bool)

        for i, model in enumerate(self.models_):
            if hasattr(model, 'named_steps') and 'feature_selection' in model.named_steps:
                selection_matrix[:, i] = model.named_steps['feature_selection'].get_support()

        # Calculate selection statistics
        selection_counts = np.sum(selection_matrix, axis=1)
        selection_rates = selection_counts / n_models

        # Compute permutation importance
        r = permutation_importance(
            self, X, y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring='roc_auc'
        )

        # Combine results
        importance_df = pd.DataFrame({
            'feature_name': self.feature_names_,
            'selection_count': selection_counts,
            'selection_rate': selection_rates,
            'importance_mean': r.importances_mean,
            'importance_std': r.importances_std
        })

        return importance_df.sort_values('importance_mean', ascending=False)