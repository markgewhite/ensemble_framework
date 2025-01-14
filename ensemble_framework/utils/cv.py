from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline


def create_cv_folds(X: np.ndarray,
                    y: np.ndarray,
                    groups: np.ndarray,
                    n_splits: int = 5,
                    n_repeats: int = 1,
                    stratified: bool = True,
                    random_state: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
    """
    Create cross-validation folds maintaining group integrity.

    Args:
        X: Training data
        y: Target values
        groups: Group labels (e.g., patient IDs)
        n_splits: Number of folds
        n_repeats: Number of times to repeat CV
        stratified: Whether to maintain class distribution
        random_state: Random seed

    Returns:
        List of dictionaries containing train/test indices for each fold
    """
    all_folds = []
    rng = np.random.RandomState(random_state)

    for repeat in range(n_repeats):
        # Choose CV splitter based on stratification requirement
        if stratified:
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=rng.randint(0, 1000000)
            )
        else:
            cv = GroupKFold(n_splits=n_splits)

        # Generate splits
        for train_idx, test_idx in cv.split(X, y, groups):
            fold = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_groups': np.unique(groups[train_idx]),
                'test_groups': np.unique(groups[test_idx])
            }
            all_folds.append(fold)

    return all_folds


def validate_splits(splits: List[Dict],
                    y: np.ndarray,
                    groups: np.ndarray) -> bool:
    """
    Validate cross-validation splits for potential issues.

    Args:
        splits: List of split dictionaries
        y: Target values
        groups: Group labels

    Returns:
        True if splits are valid, raises ValueError otherwise
    """
    for i, split in enumerate(splits):
        train_y = y[split['train_idx']]
        test_y = y[split['test_idx']]

        # Check if both classes are present in train and test
        if len(np.unique(train_y)) < 2:
            raise ValueError(f"Split {i}: Training set missing some classes")
        if len(np.unique(test_y)) < 2:
            raise ValueError(f"Split {i}: Test set missing some classes")

        # Check group overlap
        train_groups = split['train_groups']
        test_groups = split['test_groups']
        overlap = np.intersect1d(train_groups, test_groups)
        if len(overlap) > 0:
            raise ValueError(f"Split {i}: Groups overlap between train and test")

    return True


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