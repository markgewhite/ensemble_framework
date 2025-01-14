from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

from .dataset import Dataset


def load_from_csv(
        filepath: Union[str, Path],
        feature_cols: List[str],
        target_col: str,
        patient_id_col: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
) -> Dataset:
    """
    Load dataset from CSV file.

    Args:
        filepath: Path to CSV file
        feature_cols: Names of feature columns
        target_col: Name of target column
        patient_id_col: Name of patient ID column
        name: Dataset name
        description: Dataset description
        metadata: Additional metadata
    """
    df = pd.read_csv(filepath)

    # Validate columns
    missing_cols = set(feature_cols + [target_col, patient_id_col]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    # Extract components
    X = df[feature_cols].values
    y = df[target_col].values
    patient_ids = df[patient_id_col].values

    return Dataset(
        X=X,
        y=y,
        patient_ids=patient_ids,
        feature_names=feature_cols,
        name=name or Path(filepath).stem,
        description=description or "",
        metadata=metadata or {}
    )


def load_from_dataframe(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        patient_id_col: str,
        name: str = "unnamed_dataset",
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
) -> Dataset:
    """
    Create dataset from pandas DataFrame.

    Args:
        df: Input DataFrame
        feature_cols: Names of feature columns
        target_col: Name of target column
        patient_id_col: Name of patient ID column
        name: Dataset name
        description: Dataset description
        metadata: Additional metadata
    """
    # Validate columns
    missing_cols = set(feature_cols + [target_col, patient_id_col]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Extract components
    X = df[feature_cols].values
    y = df[target_col].values
    patient_ids = df[patient_id_col].values

    return Dataset(
        X=X,
        y=y,
        patient_ids=patient_ids,
        feature_names=feature_cols,
        name=name,
        description=description or "",
        metadata=metadata or {}
    )


def split_dataset(
        dataset: Dataset,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: Optional[int] = None
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and test sets.

    Args:
        dataset: Input dataset
        test_size: Proportion of dataset to include in test split
        stratify: Whether to maintain class distribution
        random_state: Random seed

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    from sklearn.model_selection import train_test_split

    # Get unique patients
    unique_patients = dataset.unique_patients

    # Get labels for stratification
    if stratify:
        patient_labels = np.array([
            dataset.y[dataset.patient_ids == pid][0]
            for pid in unique_patients
        ])
    else:
        patient_labels = None

    # Split patients
    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        stratify=patient_labels,
        random_state=random_state
    )

    # Create masks
    train_mask = np.isin(dataset.patient_ids, train_patients)
    test_mask = np.isin(dataset.patient_ids, test_patients)

    # Create datasets
    train_dataset = Dataset(
        X=dataset.X[train_mask],
        y=dataset.y[train_mask],
        patient_ids=dataset.patient_ids[train_mask],
        feature_names=dataset.feature_names,
        name=f"{dataset.name}_train",
        description=dataset.description,
        metadata=dataset.metadata
    )

    test_dataset = Dataset(
        X=dataset.X[test_mask],
        y=dataset.y[test_mask],
        patient_ids=dataset.patient_ids[test_mask],
        feature_names=dataset.feature_names,
        name=f"{dataset.name}_test",
        description=dataset.description,
        metadata=dataset.metadata
    )

    return train_dataset, test_dataset