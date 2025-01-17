import pandas as pd

from .dataset import Dataset


def load_simpreg1_data(datafile='/Users/markgewhite/Library/CloudStorage/OneDrive-SwanseaUniversity/PatientData/Export/simpreg1.csv') -> Dataset:
    """
    Load the Simpreg1 dataset and format it for our framework.

    Args:
        None

    Returns:
        Dataset object containing the Wisconsin data
    """
    # Load the entire dataset
    df = pd.read_csv(datafile)

    # Split back into components
    patient_ids = df['patient_id'].values  # Get patient IDs
    y = df['status'].values  # Get class labels
    X = df.drop(['patient_id', 'status'], axis=1).values  # Get feature matrix

    # If you need the feature names
    feature_names = df.drop(['patient_id', 'status'], axis=1).columns.tolist()

    # Create Dataset object
    dataset = Dataset(
        X=X,
        y=y,
        patient_ids=patient_ids,
        feature_names=list(feature_names),
        name="Simpreg1",
        description="Diagnostic pregnancy disease data with multiple measurements per patient",
        metadata={
            'source': 'Swansea University/Edinbugh University',
            'target_names': ['normal', 'diseased'],
        }
    )

    return dataset