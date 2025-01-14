from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import time

from ..data import Dataset, load_from_csv, split_dataset
from ..models import RepeatedNestedCV, BaggingEnsemble, GradientBoostingEnsemble
from ..utils import create_pipeline, compute_metrics, compare_models


@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    model_type: str = "repeated_ncv"  # or "bagging" or "boosting"
    n_estimators: int = 50
    random_state: int = 42

    # RepeatedNestedCV specific
    n_outer_splits: int = 5
    n_outer_repeats: int = 1

    # Bagging specific
    max_samples: float = 1.0

    # Boosting specific
    learning_rate: float = 0.1
    subsample: float = 1.0


@dataclass
class PipelineConfig:
    """Configuration for pipeline components"""
    include_scaling: bool = True
    include_feature_selection: bool = True
    base_model: str = "svc"  # base model type
    model_params: Dict = field(default_factory=lambda: {
        "kernel": "rbf",
        "C": 1.0,
        "probability": True
    })


@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment"""
    # Experiment identification
    name: str
    description: Optional[str] = None
    output_dir: Union[str, Path] = Path("results")

    # Data configuration
    data_path: Union[str, Path] = Path("data")
    feature_cols: List[str] = field(default_factory=list)
    target_col: str = "target"
    patient_id_col: str = "patient_id"
    test_size: float = 0.2

    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Evaluation configuration
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "auc", "sensitivity", "specificity", "f1"
    ])
    n_permutation_repeats: int = 10

    def __post_init__(self):
        """Convert paths to Path objects and create output directory"""
        self.output_dir = Path(self.output_dir)
        self.data_path = Path(self.data_path)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filepath: Union[str, Path]):
        """Save configuration to file"""
        # Convert to dictionary
        config_dict = {
            'name': self.name,
            'description': self.description,
            'output_dir': str(self.output_dir),
            'data_path': str(self.data_path),
            'feature_cols': self.feature_cols,
            'target_col': self.target_col,
            'patient_id_col': self.patient_id_col,
            'test_size': self.test_size,
            'model': vars(self.model),
            'pipeline': vars(self.pipeline),
            'metrics': self.metrics,
            'n_permutation_repeats': self.n_permutation_repeats
        }

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from file"""
        with open(filepath) as f:
            config_dict = json.load(f)

        # Convert nested dictionaries back to dataclasses
        model_config = ModelConfig(**config_dict.pop('model'))
        pipeline_config = PipelineConfig(**config_dict.pop('pipeline'))

        return cls(
            **config_dict,
            model=model_config,
            pipeline=pipeline_config
        )


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

        # Create appropriate model type
        if self.config.model.model_type == "repeated_ncv":
            self.model = RepeatedNestedCV(
                n_outer_splits=self.config.model.n_outer_splits,
                n_outer_repeats=self.config.model.n_outer_repeats,
                random_state=self.config.model.random_state,
                base_pipeline=pipeline
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

        # Compute metrics
        self.results = {
            'train': {
                'sample_level': compute_metrics(
                    self.train_dataset.y,
                    train_results['sample_level']['y_pred'],
                    train_results['sample_level']['y_prob']
                ),
                'patient_level': compute_metrics(
                    self.train_dataset.y,
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
                    self.test_dataset.y,
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