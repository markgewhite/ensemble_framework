from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import clone

from ..base.base_ensemble import BaseEnsemble
from ..base.data_structures import RepeatedStratifiedGroupCV


class RepeatedNestedCV(BaseEnsemble):
    """Ensemble classifier using repeated nested cross-validation."""

    def __init__(self,
                 n_outer_splits: int = 5,
                 n_outer_repeats: int = 1,
                 random_state: Optional[int] = None,
                 base_pipeline: Optional['Pipeline'] = None):

        super().__init__(random_state=random_state, base_pipeline=base_pipeline)
        self.n_outer_splits = n_outer_splits
        self.n_outer_repeats = n_outer_repeats
        self.cv_splits_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'RepeatedNestedCV':
        """Fit the repeated nested CV ensemble."""
        self.feature_names_ = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Generate outer CV splits
        cv = RepeatedStratifiedGroupCV(
            n_splits=self.n_outer_splits,
            n_repeats=self.n_outer_repeats,
            random_state=self.random_state
        )
        self.cv_splits_ = cv.split(X, y, groups)

        # Train a model for each split
        self.models_ = []
        for split in self.cv_splits_:
            # Fit model on training data
            model = clone(self.base_pipeline)
            model.fit(X[split.train_idx], y[split.train_idx])
            self.models_.append(model)

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the ensemble."""
        n_samples = X.shape[0]
        n_models = len(self.models_)

        # Initialize arrays for predictions
        predictions = np.zeros((n_samples, n_models))
        probabilities = np.zeros((n_samples, n_models))
        prediction_counts = np.zeros(n_samples)

        # Get predictions from each model
        for i, (model, split) in enumerate(zip(self.models_, self.cv_splits_)):
            # Only predict on legitimate test samples
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[split.test_idx] = True

            if np.any(test_mask):
                X_subset = X[test_mask]
                predictions[test_mask, i] = model.predict(X_subset)
                probabilities[test_mask, i] = model.predict_proba(X_subset)[:, 1]
                prediction_counts[test_mask] += 1

        # Average probabilities
        with np.errstate(divide='ignore', invalid='ignore'):
            y_prob = np.sum(probabilities, axis=1) / prediction_counts

        # Majority voting
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            valid_predictions = predictions[i, predictions[i, :] != 0]
            if len(valid_predictions) > 0:
                y_pred[i] = np.round(np.mean(valid_predictions))

        return y_pred, y_prob