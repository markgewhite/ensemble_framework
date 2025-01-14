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

    def predict(self, X):
        """
        Aggregate predictions from all models in ensemble using row-wise averaging
        (ignoring models that did not predict on each sample).
        """
        n_samples = X.shape[0]
        n_models = len(self.models_)

        # Prepare arrays to hold model-by-model predictions
        # We use NaN so that "missing" predictions don't skew the average
        y_preds = np.full((n_samples, n_models), fill_value=np.nan)
        y_probs = np.full((n_samples, n_models), fill_value=np.nan)

        # Fill in predictions/probs for each model
        for i, (model, split) in enumerate(zip(self.models_, self.outer_splits_)):
            # 'test_idx' are the samples that this model was actually tested on
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[split['test_idx']] = True

            if np.any(test_mask):
                X_subset = X[test_mask]
                preds = model.predict(X_subset)
                probs = model.predict_proba(X_subset)[:, 1]

                # Store them in the y_preds / y_probs arrays
                y_preds[test_mask, i] = preds
                y_probs[test_mask, i] = probs

        # Compute row-wise mean ignoring NaN values
        with np.errstate(invalid='ignore'):
            mean_preds = np.nanmean(y_preds, axis=1)
            mean_probs = np.nanmean(y_probs, axis=1)

        # If a given sample wasn't predicted by *any* model (all NaN),
        # np.nanmean returns NaN. You can set those to 0 or some other default.
        mean_preds = np.where(np.isnan(mean_preds), 0, mean_preds)
        mean_probs = np.where(np.isnan(mean_probs), 0, mean_probs)

        # Convert continuous predictions into integer class labels
        y_pred = np.round(mean_preds).astype(int)

        return y_pred, mean_probs