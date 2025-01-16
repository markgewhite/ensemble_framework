from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import clone

from ..base.base_ensemble import BaseEnsemble


class GradientBoostingEnsemble(BaseEnsemble):
    """
    Ensemble classifier using gradient boosting.

    This implementation uses binary logistic loss for classification,
    fitting each model to predict the negative gradient of the loss
    with respect to F (the current predictions in log odds space).
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
        # Ensure y is binary
        y = y.astype(int)
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("y must be binary with classes 0 and 1")

        self.feature_names_ = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        n_samples = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # Initialize log odds with zeros
        F = np.zeros(n_samples)
        self.models_ = []
        self.initial_prediction_ = np.log(np.mean(y) / (1 - np.mean(y)))  # log odds of base rate
        F += self.initial_prediction_

        for _ in range(self.n_estimators):
            # Current predictions (probabilities)
            p = 1 / (1 + np.exp(-F))

            # Compute negative gradient (for logistic loss)
            # This gives us what we want each tree to predict
            negative_gradient = y - p

            # Subsample if specified
            if self.subsample < 1:
                sample_mask = rng.rand(n_samples) < self.subsample
                if np.sum(sample_mask) == 0:  # Ensure at least one sample
                    sample_mask[rng.randint(0, n_samples)] = True
                X_subset = X[sample_mask]
                negative_gradient_subset = negative_gradient[sample_mask]
            else:
                X_subset = X
                negative_gradient_subset = negative_gradient

            # Train model to predict the negative gradient
            model = clone(self.base_pipeline)
            model.fit(X_subset, negative_gradient_subset)
            self.models_.append(model)

            # Update F with predictions scaled by learning rate
            F += self.learning_rate * model.predict(X)

        return self


    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the gradient boosting ensemble."""
        # Start with initial prediction (log odds of base rate)
        F = np.full(X.shape[0], self.initial_prediction_)

        # Add up predictions from all models
        for model in self.models_:
            F += self.learning_rate * model.predict(X)

        # Convert log odds to probabilities using sigmoid
        y_prob = 1 / (1 + np.exp(-F))
        y_pred = (y_prob > 0.5).astype(int)

        return y_pred, y_prob