from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import clone

from ..base.base_ensemble import BaseEnsemble


class BaggingEnsemble(BaseEnsemble):
    """Ensemble classifier using bootstrap aggregation (bagging)."""

    def __init__(self,
                 n_estimators: int = 50,
                 max_samples: float = 1.0,
                 random_state: Optional[int] = None,
                 base_pipeline: Optional['Pipeline'] = None):

        super().__init__(random_state=random_state, base_pipeline=base_pipeline)
        self.n_estimators = n_estimators
        self.max_samples = max_samples

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> 'BaggingEnsemble':
        """Fit the bagging ensemble."""
        self.feature_names_ = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        n_samples = X.shape[0]

        rng = np.random.RandomState(self.random_state)

        # Clear any existing models
        self.models_ = []

        # Number of samples to draw for each estimator
        n_samples_draw = int(n_samples * self.max_samples)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = rng.randint(0, n_samples, n_samples_draw)
            X_boot = X[indices]
            y_boot = y[indices]

            # Train model on bootstrap sample
            model = clone(self.base_pipeline)
            model.fit(X_boot, y_boot)
            self.models_.append(model)

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the bagging ensemble."""
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.models_)))
        probabilities = np.zeros((n_samples, len(self.models_)))

        # Get predictions from all models
        for i, model in enumerate(self.models_):
            predictions[:, i] = model.predict(X)
            probabilities[:, i]