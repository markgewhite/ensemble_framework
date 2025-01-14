from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC, SVC


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