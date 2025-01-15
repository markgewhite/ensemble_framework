from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC, SVC

# Define standard parameter grids
SVM_PARAM_GRID = {
    'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'classifier__C': [0.1, 0.5, 1, 5, 10, 100],
    'classifier__gamma': [0.1, 0.01, 0.001],
}

XGB_PARAM_GRID = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.01, 0.1],
}


def get_default_param_grid(model_type: str) -> Dict:
    """Get default parameter grid for given model type"""
    if model_type == 'svc':
        return SVM_PARAM_GRID
    elif model_type == 'xgb':
        return XGB_PARAM_GRID
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_pipeline(model_type: str = 'svc',
                    include_feature_selection: bool = True,
                    include_scaling: bool = True,
                    **model_params) -> Pipeline:
    """
    Create a pipeline with optional scaling and feature selection.

    Args:
        model_type: Type of model ('svc', 'xgb', etc.)
        include_feature_selection: Whether to include feature selection
        include_scaling: Whether to include scaling
        model_params: Additional parameters for the classifier
    """
    steps = []

    if include_scaling:
        steps.append(('scaler', RobustScaler()))

    if include_feature_selection:
        steps.append(('feature_selection',
                      SelectFromModel(LinearSVC(penalty='l1',
                                                dual=False,
                                                max_iter=10000,
                                                random_state=42))))

    if model_type == 'svc':
        clf = SVC(probability=True, max_iter=10000, random_state=42)
        clf.set_params(**model_params)
        steps.append(('classifier', clf))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline(steps)