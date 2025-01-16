import numpy as np
from typing import Dict,  Union
from pathlib import Path
import json
import time
from collections import defaultdict

from .config import ExperimentConfig
from ..data import Dataset, load_from_csv, split_dataset
from ..models import RepeatedNestedCV, BaggingEnsemble, GradientBoostingEnsemble
from ..utils import create_pipeline, compute_metrics, aggregate_ground_truth_and_predictions, aggregate_feature_importance


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

        # For boosting, we need regression models as base estimators
        if self.config.model.model_type == "boosting":
            # Create regression pipeline
            pipeline = create_pipeline(
                model_type='svr' if self.config.pipeline.base_model == 'svc' else 'xgb_regressor',
                include_feature_selection=self.config.pipeline.include_feature_selection,
                include_scaling=self.config.pipeline.include_scaling,
                **self.config.pipeline.model_params
            )
        else:
            # For other models (bagging, repeated_ncv), use classification pipeline
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
                base_pipeline=pipeline
            )
        elif self.config.model.model_type == "boosting":
            self.model = GradientBoostingEnsemble(
                n_estimators=self.config.model.n_estimators,
                learning_rate=self.config.model.learning_rate,
                subsample=self.config.model.subsample,
                random_state=self.config.model.random_state,
                base_pipeline=pipeline
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model.model_type}")


    def run(self) -> Dict:
        """
        If self.config.use_holdout is True, do repeated holdout splits.
        Otherwise, do "unleashed" repeated nested CV on the entire dataset.
        """
        if self.dataset is None:
            self.load_data()

        if self.config.use_holdout:
            return self._run_with_repeated_holdout(
                n_repeats=self.config.n_holdout_repeats
            )
        else:
            return self._run_unleashed_repeated_cv()


    def _run_unleashed_repeated_cv(self) -> Dict:

        start_time = time.time()

        # 1) Setup model (fresh if needed)
        self.setup_model()

        # 2) Fit using the entire dataset
        self.model.fit(
            self.dataset.X,
            self.dataset.y,
            self.dataset.patient_ids,
            feature_names=self.dataset.feature_names
        )

        # 3) Predict on the same dataset (to get OOF predictions)
        results = self.model.predict_patients(
            self.dataset.X,
            self.dataset.patient_ids
        )

        # 4) Evaluate at sample-level
        sample_metrics = compute_metrics(
            self.dataset.y,
            results['sample_level']['y_pred'],
            results['sample_level']['y_prob']
        )

        # 5) Evaluate at patient-level
        patient_metrics = compute_metrics(
            *aggregate_ground_truth_and_predictions(self.dataset, results, level='patient')
        )

        # 6) Feature importance etc.
        #fi = self.model.feature_importance(
        #    self.dataset.X,
        #    self.dataset.y,
        #    n_repeats=self.config.n_permutation_repeats
        #)

        # 7) Package up results
        final_results = {
            'oof': {
                'sample_level': sample_metrics,
                'patient_level': patient_metrics
            },
            'feature_importance': fi,
            'runtime': time.time() - start_time
        }

        self.results = final_results
        return final_results


    def _run_with_repeated_holdout(self, n_repeats: int = 10) -> Dict:
        """
        Run the experiment with repeated holdout splits (default 10 repeats).
        """
        # If dataset not loaded, load it once
        if self.dataset is None:
            self.load_data()

        # Prepare to accumulate metrics across repeats
        # e.g. train_metrics_sample_level["accuracy"] will be a list of 10 accuracy values
        train_metrics_sample_level = defaultdict(list)
        train_metrics_patient_level = defaultdict(list)
        test_metrics_sample_level = defaultdict(list)
        test_metrics_patient_level = defaultdict(list)

        # We might accumulate feature_importance across runs too, or just collect the final
        # For example, store them in a list
        fi_list = []

        # Time tracking
        overall_start_time = time.time()

        # Loop over the number of repeats
        for i in range(n_repeats):
            # 1) Split dataset into train/test
            #    - use a unique random_state each time
            #      (e.g. self.config.model.random_state + i)
            #    - or rely on randomness if you prefer
            train_ds, test_ds = split_dataset(
                self.dataset,
                test_size=self.config.test_size,
                stratify=True,
                random_state=self.config.model.random_state + i
            )

            # 2) Setup model (important to create a fresh model each repeat)
            self.setup_model()  # this populates self.model

            # 3) Fit model
            self.model.fit(
                train_ds.X,
                train_ds.y,
                train_ds.patient_ids,
                feature_names=train_ds.feature_names
            )

            # 4) Predict on train + test
            train_results = self.model.predict_patients(
                train_ds.X,
                train_ds.patient_ids
            )
            test_results = self.model.predict_patients(
                test_ds.X,
                test_ds.patient_ids
            )

            # 5) Compute metrics
            # sample-level
            train_sample_metrics = compute_metrics(
                train_ds.y,
                train_results['sample_level']['y_pred'],
                train_results['sample_level']['y_prob']
            )
            test_sample_metrics = compute_metrics(
                test_ds.y,
                test_results['sample_level']['y_pred'],
                test_results['sample_level']['y_prob']
            )

            # patient-level
            train_patient_metrics = compute_metrics(
                # aggregated ground truth at patient-level
                *aggregate_ground_truth_and_predictions(train_ds, train_results, level='patient')
            )
            test_patient_metrics = compute_metrics(
                *aggregate_ground_truth_and_predictions(test_ds, test_results, level='patient')
            )

            # Save metrics into lists
            for m, v in train_sample_metrics.items():
                train_metrics_sample_level[m].append(v)
            for m, v in test_sample_metrics.items():
                test_metrics_sample_level[m].append(v)
            for m, v in train_patient_metrics.items():
                train_metrics_patient_level[m].append(v)
            for m, v in test_patient_metrics.items():
                test_metrics_patient_level[m].append(v)

            # 6) Feature importance (optional)
            #fi = self.model.feature_importance(
            #    test_ds.X, test_ds.y,
            #    n_repeats=self.config.n_permutation_repeats
            #)
            #fi_list.append(fi)

        # Now we have repeated runs. Let's average metrics.
        # e.g. 'accuracy' across 10 runs
        final_results = {
            'train': {
                'sample_level': {},
                'patient_level': {}
            },
            'test': {
                'sample_level': {},
                'patient_level': {}
            },
            'feature_importance': None,  # or aggregated somehow
            'runtime': time.time() - overall_start_time
        }

        # Compute means
        for metric_name, values_list in train_metrics_sample_level.items():
            final_results['train']['sample_level'][metric_name] = float(np.mean(values_list))
        for metric_name, values_list in train_metrics_patient_level.items():
            final_results['train']['patient_level'][metric_name] = float(np.mean(values_list))

        for metric_name, values_list in test_metrics_sample_level.items():
            final_results['test']['sample_level'][metric_name] = float(np.mean(values_list))
        for metric_name, values_list in test_metrics_patient_level.items():
            final_results['test']['patient_level'][metric_name] = float(np.mean(values_list))

        # If you want to store standard deviations as well, you can do that, too
        # e.g. final_results['test']['sample_level'][metric_name+'_std'] = float(np.std(values_list))

        # Aggregate feature importance across repeats
        final_results['feature_importance'] = aggregate_feature_importance(fi_list)

        self.results = final_results
        return final_results


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