from typing import Optional, Tuple
import numpy as np
from sklearn.datasets import load_breast_cancer

from .dataset import Dataset


def load_wisconsin_data(random_state: Optional[int] = None) -> Dataset:
    """
    Load the Wisconsin Breast Cancer dataset and format it for our framework.

    The original dataset doesn't have patient IDs, so we'll simulate them by
    assuming every 5 samples come from the same patient (this simulates multiple
    measurements per patient).

    Args:
        random_state: Random seed for reproducibility

    Returns:
        Dataset object containing the Wisconsin data
    """
    # Load the raw data
    raw_data = load_breast_cancer()
    X = raw_data.data
    y = raw_data.target
    feature_names = raw_data.feature_names

    # Create simulated patient IDs (5 samples per patient)
    n_samples = X.shape[0]
    samples_per_patient = 5
    n_patients = n_samples // samples_per_patient
    patient_ids = np.repeat(
        [f'P{i:03d}' for i in range(n_patients)],
        samples_per_patient
    )

    # If there are any remaining samples, assign them to new patients
    remaining_samples = n_samples % samples_per_patient
    if remaining_samples > 0:
        extra_patients = [f'P{i:03d}' for i in range(n_patients, n_patients + remaining_samples)]
        patient_ids = np.concatenate([patient_ids, extra_patients])

    # If random_state is provided, shuffle the data while keeping patient samples together
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        unique_patients = np.unique(patient_ids)
        patient_order = rng.permutation(len(unique_patients))
        shuffled_patients = unique_patients[patient_order]

        # Create new arrays in shuffled order
        new_X = []
        new_y = []
        new_patient_ids = []

        for patient in shuffled_patients:
            mask = patient_ids == patient
            new_X.append(X[mask])
            new_y.append(y[mask])
            new_patient_ids.extend([patient] * np.sum(mask))

        X = np.vstack(new_X)
        y = np.concatenate(new_y)
        patient_ids = np.array(new_patient_ids)

    # Create Dataset object
    dataset = Dataset(
        X=X,
        y=y,
        patient_ids=patient_ids,
        feature_names=list(feature_names),
        name="Wisconsin Breast Cancer",
        description="Diagnostic breast cancer data with multiple measurements per patient",
        metadata={
            'source': 'UCI ML Repository',
            'target_names': ['malignant', 'benign'],
            'samples_per_patient': samples_per_patient
        }
    )

    return dataset