from typing import List, Optional, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.metrics import make_scorer, recall_score

from ..base.base_ensemble import BaseEnsemble
from ..base.data_structures import RepeatedStratifiedGroupCV


class RepeatedNestedCV(BaseEnsemble):
    def __init__(self,
                 n_outer_splits=5,
                 n_outer_repeats=1,
                 n_inner_splits=2,
                 n_inner_repeats=5,
                 random_state=None,
                 base_pipeline=None,
                 param_grid=None,  # Add param_grid parameter
                 scoring='roc_auc'):

        super().__init__(random_state=random_state, base_pipeline=base_pipeline)
        self.n_outer_splits = n_outer_splits
        self.n_outer_repeats = n_outer_repeats
        self.n_inner_splits = n_inner_splits
        self.n_inner_repeats = n_inner_repeats
        self.param_grid = param_grid
        self.outer_splits_ = None
        self.n_train_samples_ = None

        # Define scoring metrics mapping
        self.scoring_metrics = {
            'roc_auc': 'roc_auc',
            'accuracy': 'accuracy',
            'f1': 'f1',
            'sensitivity': make_scorer(recall_score, pos_label=1)
        }
        self.scoring = self.scoring_metrics.get(scoring, scoring)


    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'RepeatedNestedCV':
        """Fit the repeated nested CV ensemble."""

        if self.param_grid is None:
            raise ValueError("param_grid must be specified")

        self.feature_names_ = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Generate outer CV splits
        outer_cv = RepeatedStratifiedGroupCV(
            n_splits=self.n_outer_splits,
            n_repeats=self.n_outer_repeats,
            random_state=self.random_state
        )
        # Create list to store full split information
        self.train_n_samples_ = X.shape[0]
        self.outer_splits_ = []

        # For each outer split
        for train_idx, test_idx in outer_cv.split(X, y, groups):
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
                cv=inner_cv,
                scoring=self.scoring,
                n_jobs=-1
            )

            # Fit grid search
            grid_search.fit(X_train, y_train, groups=train_groups)

            # Store best model and results
            self.models_.append(grid_search.best_estimator_)
            self.outer_splits_.append({
                'train_idx': train_idx,
                'test_idx': test_idx
            })

            # Print best parameters for this fold
            print(f"Best parameters for fold: {grid_search.best_params_}")
            print(f"Best score for fold: {grid_search.best_score_:.3f}")

        return self

    def predict(self, X: np.ndarray, indices: np.ndarray = None):
        """
        Make ensemble predictions on data.

        If X has the same shape (number of rows) as the dataset used in fit,
        we assume the user wants "OOF predictions" using stored test splits.
        Otherwise, all models predict all samples.
        """
        n_samples = X.shape[0]
        n_models = len(self.models_)

        if indices is None:
            # Check if we have an attribute storing the training set size
            # and if X matches that size exactly
            if hasattr(self, 'train_n_samples_') and n_samples == self.train_n_samples_:
                # => OOF scenario: re-build the mask using outer_splits_
                indices = np.zeros((n_samples, n_models), dtype=bool)
                for i, split_dict in enumerate(self.outer_splits_):
                    test_idx = split_dict["test_idx"]
                    indices[test_idx, i] = True
            else:
                # => This must be new data or a smaller subset, so let all models predict
                indices = np.ones((n_samples, n_models), dtype=bool)

        # Prepare arrays to hold predictions/probs from each model
        # We use NaN for “no prediction” so we can safely average ignoring it
        y_preds = np.full((n_samples, n_models), np.nan, dtype=float)
        y_probs = np.full((n_samples, n_models), np.nan, dtype=float)

        # Populate the arrays with predictions from each model
        for i, model in enumerate(self.models_):
            # This column in 'indices' tells us which rows
            # the i-th model is responsible for
            mask = indices[:, i]
            if not np.any(mask):
                # No samples for this model
                continue

            # Subset the data rows for this model
            X_sub = X[mask, :]
            preds = model.predict(X_sub)
            probas = model.predict_proba(X_sub)[:, 1]

            # Place them into the big arrays
            y_preds[mask, i] = preds
            y_probs[mask, i] = probas

        # Average predictions & probabilities across models that actually predicted
        with np.errstate(invalid='ignore'):
            mean_preds = np.nanmean(y_preds, axis=1)
            mean_probs = np.nanmean(y_probs, axis=1)

        # If a sample is never predicted by any model (all-NaN in that row),
        # nanmean returns NaN. Decide how to handle it. Here we default to 0:
        mean_preds = np.where(np.isnan(mean_preds), 0, mean_preds)
        mean_probs = np.where(np.isnan(mean_probs), 0, mean_probs)

        # Convert continuous predictions to integer class labels
        y_pred = np.round(mean_preds).astype(int)

        return y_pred, mean_probs
