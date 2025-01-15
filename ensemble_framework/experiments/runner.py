from typing import Dict,  Union
from pathlib import Path
import json
import time

from config import ExperimentConfig
from ..data import Dataset, load_from_csv, split_dataset
from ..models import RepeatedNestedCV, BaggingEnsemble, GradientBoostingEnsemble
from ..utils import create_pipeline, compute_metrics, compare_models


class ExperimentRunner:
    """Class to run and manage experiments"""

    def __init__(self, config: ExperimentConfig, dataset=None, train_dataset=None, test_dataset=None):
        self.config = config
        self.model = None
        self.results = {}

        if dataset is not None:
            # Single dataset provided - split it now
            self.train_dataset, self.test_dataset = split_dataset(
                dataset,
                test_size=0.2,  # you might want this as a config parameter
                random_state=42  # and this too
            )
            self.dataset = dataset  # keep original if needed
        elif train_dataset is not None and test_dataset is not None:
            # Pre-split datasets provided
            self.dataset = None
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
        else:
            raise ValueError("Must provide either a single dataset or train/test datasets")

    def load_data(self) -> None:
        """Load and split dataset"""
        # Load dataset
        self.dataset = load_from_csv(
            self.config.data_path,
            self.config.feature_cols,
            self.config.target_col,
            self.config.patient_id_col,
            name=self.config.name
        )

        # Split into train/test
        self.train_dataset, self.test_dataset = split_dataset(
            self.dataset,
            test_size=self.config.test_size,
            stratify=True,
            random_state=self.config.model.random_state
        )

    def setup_model(self) -> None:
        """Create and configure model based on config"""
        # Create base pipeline
        pipeline = create_pipeline(
            model_type=self.config.pipeline.base_model,
            include_feature_selection=self.config.pipeline.include_feature_selection,
            include_scaling=self.config.pipeline.include_scaling,
            **self.config.pipeline.model_params
        )

        # Get appropriate param grid from config or defaults
        param_grid = self.config.pipeline.param_grid
        if param_grid is None:
            from ..utils import SVM_PARAM_GRID, XGB_PARAM_GRID
            param_grid = SVM_PARAM_GRID if self.config.pipeline.base_model == 'svc' else XGB_PARAM_GRID

        # Create appropriate model type
        if self.config.model.model_type == "repeated_ncv":
            self.model = RepeatedNestedCV(
                n_outer_splits=self.config.model.n_outer_splits,
                n_outer_repeats=self.config.model.n_outer_repeats,
                random_state=self.config.model.random_state,
                base_pipeline=pipeline,
                param_grid=param_grid
            )
        elif self.config.model.model_type == "bagging":
            self.model = BaggingEnsemble(
                n_estimators=self.config.model.n_estimators,
                max_samples=self.config.model.max_samples,
                random_state=self.config.model.random_state,
                base_pipeline=pipeline,
                param_grid=param_grid
            )
        elif self.config.model.model_type == "boosting":
            self.model = GradientBoostingEnsemble(
                n_estimators=self.config.model.n_estimators,
                learning_rate=self.config.model.learning_rate,
                subsample=self.config.model.subsample,
                random_state=self.config.model.random_state,
                base_pipeline=pipeline,
                param_grid=param_grid
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model.model_type}")

    def run(self) -> Dict:
        """Run complete experiment"""

        # Record the start time
        start_time = time.time()

        # Load data if not already loaded
        if self.dataset is None:
            self.load_data()

        # Setup model if not already setup
        if self.model is None:
            self.setup_model()

        # Train model
        self.model.fit(
            self.train_dataset.X,
            self.train_dataset.y,
            self.train_dataset.patient_ids,
            feature_names=self.train_dataset.feature_names
        )

        # Get predictions
        train_results = self.model.predict_patients(
            self.train_dataset.X,
            self.train_dataset.patient_ids
        )
        test_results = self.model.predict_patients(
            self.test_dataset.X,
            self.test_dataset.patient_ids
        )

        # Aggregate the ground truth to patient-level
        train_pids, y_train_patient = self.train_dataset.aggregate_patient_labels(aggregator='max')
        test_pids, y_test_patient = self.test_dataset.aggregate_patient_labels(aggregator='max')

        # Compute metrics
        self.results = {
            'train': {
                'sample_level': compute_metrics(
                    self.train_dataset.y,
                    train_results['sample_level']['y_pred'],
                    train_results['sample_level']['y_prob']
                ),
                'patient_level': compute_metrics(
                    y_train_patient,
                    train_results['patient_level']['y_pred'],
                    train_results['patient_level']['y_prob']
                )
            },
            'test': {
                'sample_level': compute_metrics(
                    self.test_dataset.y,
                    test_results['sample_level']['y_pred'],
                    test_results['sample_level']['y_prob']
                ),
                'patient_level': compute_metrics(
                    y_test_patient,
                    test_results['patient_level']['y_pred'],
                    test_results['patient_level']['y_prob']
                )
            },
            'feature_importance': self.model.feature_importance(
                self.test_dataset.X,
                self.test_dataset.y,
                n_repeats=self.config.n_permutation_repeats
            ),
            'runtime': time.time() - start_time
        }

        return self.results

    def save_results(self, filepath: Union[str, Path]):
        """Save results to file"""
        results_dict = {
            'config': {
                'name': self.config.name,
                'description': self.config.description,
                'model_type': self.config.model.model_type,
                'feature_cols': self.config.feature_cols
            },
            'results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=4)