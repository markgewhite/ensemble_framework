from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import clone

from ..base.base_ensemble import BaseEnsemble


class GradientBoostingEnsemble(BaseEnsemble):
    """
    Ensemble classifier using gradient boosting.

    Features:
    - Creates sequential ensemble focusing on hard examples
    - Maintains consistent pipeline with other ensembles
    - Supports both sample and patient-level predictions
    """

    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 random_state: Optional[int] = None,
                 base_pipeline: Optional['Pipeline'] = None):

        super().__init__(random_state=random_state, base_pipeline=base_pipeline)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> 'GradientBoostingEnsemble':
        """Fit the gradient boosting ensemble."""
        self.feature_names_ = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        n_samples = X.shape[0]

        rng = np.random.RandomState(self.random_state)

        # Initialize predictions with zeros
        F = np.zeros(n_samples)
        self.models_ = []

        for _ in range(self.n_estimators):
            # Compute pseudo-residuals
            p = 1 / (1 + np.exp(-F))  # sigmoid
            residuals = y - p

            # Subsample if specified
            if self.subsample < 1:
                sample_mask = rng.rand(n_samples) < self.subsample
                X_subset = X[sample_mask]
                residuals_subset = residuals[sample_mask]
            else:
                X_subset = X
                residuals_subset = residuals

            # Train model on pseudo-residuals
            model = clone(self.base_pipeline)
            model.fit(X_subset, residuals_subset)
            self.models_.append(model)

            # Update F with predictions
            F += self.learning_rate * model.predict(X)

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the gradient boosting ensemble."""
        n_samples = X.shape[0]
        F = np.zeros(n_samples)

        # Sum up predictions from all models
        for model in self.models_:
            F += self.learning_rate * model.predict(X)

        # Convert to probabilities using sigmoid
        y_prob = 1 / (1 + np.exp(-F))
        y_pred = (y_prob > 0.5).astype(int)

        return y_pred, y_prob