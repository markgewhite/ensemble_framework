import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from typing import List, Dict, Tuple


def cross_validate_pipeline(pipeline: Pipeline,
                            X: np.ndarray,
                            y: np.ndarray,
                            splits: List[Dict],
                            scoring: str = 'accuracy') -> Tuple[List[Pipeline], Dict]:
    """
    Perform cross-validation of a pipeline.

    Args:
        pipeline: Pipeline to validate
        X: Training data
        y: Target values
        splits: CV splits
        scoring: Metric to use for evaluation

    Returns:
        Tuple of (trained models, validation results)
    """
    models = []
    results = {
        'train_scores': [],
        'test_scores': [],
        'feature_selection': []
    }

    for split in splits:
        # Train model
        model = clone(pipeline)
        X_train = X[split['train_idx']]
        y_train = y[split['train_idx']]
        X_test = X[split['test_idx']]
        y_test = y[split['test_idx']]

        model.fit(X_train, y_train)
        models.append(model)

        # Record scores
        results['train_scores'].append(model.score(X_train, y_train))
        results['test_scores'].append(model.score(X_test, y_test))

        # Record feature selection if available
        if 'feature_selection' in model.named_steps:
            results['feature_selection'].append(
                model.named_steps['feature_selection'].get_support()
            )

    return models, results