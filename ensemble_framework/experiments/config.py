from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json


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
    param_grid: Optional[Dict] = None


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
