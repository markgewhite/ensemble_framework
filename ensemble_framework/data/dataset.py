from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """
    Dataset container with features and metadata

    Attributes:
        X: Feature matrix
        y: Target labels
        patient_ids: Patient identifiers for each sample
        feature_names: Names of features
        name: Dataset name
        description: Dataset description
        metadata: Additional metadata
    """
    X: np.ndarray
    y: np.ndarray
    patient_ids: np.ndarray
    feature_names: List[str]
    name: str = "unnamed_dataset"
    description: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate dataset attributes after initialization"""
        if self.X.shape[0] != len(self.y):
            raise ValueError(
                f"Number of samples in X ({self.X.shape[0]}) "
                f"does not match length of y ({len(self.y)})"
            )

        if self.X.shape[0] != len(self.patient_ids):
            raise ValueError(
                f"Number of samples in X ({self.X.shape[0]}) "
                f"does not match length of patient_ids ({len(self.patient_ids)})"
            )

        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Number of features ({self.X.shape[1]}) does not match "
                f"number of feature names ({len(self.feature_names)})"
            )

    @property
    def n_samples(self) -> int:
        """Number of samples in dataset"""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features in dataset"""
        return self.X.shape[1]

    @property
    def unique_patients(self) -> np.ndarray:
        """Unique patient identifiers"""
        return np.unique(self.patient_ids)

    def get_patient_data(self, patient_id: str) -> 'Dataset':
        """Extract data for a specific patient"""
        mask = (self.patient_ids == patient_id)
        return Dataset(
            X=self.X[mask],
            y=self.y[mask],
            patient_ids=self.patient_ids[mask],
            feature_names=self.feature_names,
            name=f"{self.name}_{patient_id}",
            description=self.description,
            metadata=self.metadata
        )

    def get_feature_subset(self, feature_indices: List[int]) -> 'Dataset':
        """Create new dataset with subset of features"""
        return Dataset(
            X=self.X[:, feature_indices],
            y=self.y,
            patient_ids=self.patient_ids,
            feature_names=[self.feature_names[i] for i in feature_indices],
            name=f"{self.name}_subset",
            description=self.description,
            metadata=self.metadata
        )

    def summary(self) -> Dict:
        """Get summary statistics of the dataset"""
        return {
            'name': self.name,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_patients': len(self.unique_patients),
            'class_distribution': np.bincount(self.y),
            'samples_per_patient': pd.Series(self.patient_ids).value_counts().describe(),
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }