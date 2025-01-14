from typing import List, Optional, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

from ..base.base_ensemble import BaseEnsemble
from ..base.data_structures import RepeatedStratifiedGroupCV


class RepeatedNestedCV(BaseEnsemble):
    def __init__(self,
                 n_outer_splits=5,
                 n_outer_repeats=1,
                 n_inner_splits=2,
                 n_inner_repeats=5,
                 random_state=None,
                 base_pipeline=None):

        self.n_outer_splits = n_outer_splits
        self.n_outer_repeats = n_outer_repeats
        self.n_inner_splits = n_inner_splits
        self.n_inner_repeats = n_inner_repeats
        self.random_state = random_state
        self.base_pipeline = base_pipeline

        self.models_ = []
        self.outer_splits_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'RepeatedNestedCV':
        """Fit the repeated nested CV ensemble."""
        self.feature_names_ = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Generate outer CV splits
        outer_cv = RepeatedStratifiedGroupCV(
            n_splits=self.n_outer_splits,
            n_repeats=self.n_outer_repeats,
            random_state=self.random_state
        )
        self.outer_splits_ = outer_cv.split(X, y, groups)

        # For each outer split
        for split in self.outer_splits_:
            train_idx = split.train_idx
            test_idx = split.test_idx
            X_train, y_train = X[train_idx], y[train_idx]
            train_groups = groups[train_idx]

            # Inner CV for model selection
            inner_cv = RepeatedStratifiedGroupCV(
                n_splits=self.n_inner_splits,
                n_repeats=self.n_inner_repeats,
                random_state=self.random_state
            )

            # Grid search with inner CV
            grid_search = GridSearchCV(
                estimator=clone(self.base_pipeline),
                param_grid=self.param_grid,
                cv=inner_cv.split(X_train, y_train, train_groups),
                scoring=self.scoring,
                n_jobs=-1
            )

            # Fit grid search
            grid_search.fit(X_train, y_train)

            # Store best model and results
            self.models_.append(grid_search.best_estimator_)
            self.cv_results_.append(grid_search.cv_results_)
            self.outer_splits_.append({
                'train_idx': train_idx,
                'test_idx': test_idx
            })

            # Print best parameters for this fold
            print(f"Best parameters for fold: {grid_search.best_params_}")
            print(f"Best score for fold: {grid_search.best_score_:.3f}")

        return self

    def predict(self, X):
        """Aggregate predictions from all models in ensemble"""

        n_samples = X.shape[0]
        y_preds = np.zeros((n_samples, len(self.models_)))
        y_probs = np.zeros((n_samples, len(self.models_)))

        # Get predictions from each model
        for i, (model, split) in enumerate(zip(self.models_, self.outer_splits_)):
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[split['test_idx']] = True

            if np.any(test_mask):
                pred = model.predict(X[test_mask])
                prob = model.predict_proba(X[test_mask])[:, 1]

                y_preds[test_mask, i] = pred
                y_probs[test_mask, i] = prob

        # Aggregate predictions
        y_pred = np.zeros(n_samples)
        y_prob = np.zeros(n_samples)

        for i in range(n_samples):
            valid_preds = y_preds[i, y_preds[i, :] != 0]
            valid_probs = y_probs[i, y_probs[i, :] != 0]

            if len(valid_preds) > 0:
                y_pred[i] = np.round(np.mean(valid_preds))
                y_prob[i] = np.mean(valid_probs)

        return y_pred, y_prob