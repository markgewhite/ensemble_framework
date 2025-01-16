from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
import pandas as pd

from ..data import registry as dataset_registry
from ..experiments import ExperimentRunner


@dataclass
class ExperimentSweepConfig:
    """Complete configuration for parameter sweeps including dataset selection"""
    # Parameter ranges to sweep (can include dataset parameters)
    parameter_ranges: Dict[str, List[Any]] = field(default_factory=dict)

    # Fixed parameters that don't change during sweep
    fixed_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dataset configurations"""
        # Check if dataset name is specified either in ranges or fixed params
        dataset_in_ranges = 'dataset.name' in self.parameter_ranges
        dataset_in_fixed = 'dataset.name' in self.fixed_parameters

        if not (dataset_in_ranges or dataset_in_fixed):
            raise ValueError("Dataset name must be specified either in parameter_ranges or fixed_parameters")

        # Validate all dataset names
        if dataset_in_ranges:
            for dataset_name in self.parameter_ranges['dataset.name']:
                self._validate_dataset(dataset_name)
        else:
            self._validate_dataset(self.fixed_parameters['dataset.name'])

    def _validate_dataset(self, dataset_name: str):
        """Validate a dataset name and its parameters"""
        if dataset_name not in dataset_registry.get_available_datasets():
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def save(self, filepath: Union[str, Path]):
        """Save sweep configuration to file"""
        config_dict = {
            'parameter_ranges': self.parameter_ranges,
            'fixed_parameters': self.fixed_parameters
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentSweepConfig':
        """Load sweep configuration from file"""
        with open(filepath) as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class SweepRunner:
    """Enhanced sweep runner with dataset support"""

    def __init__(self, base_config: 'ExperimentConfig', sweep_config: ExperimentSweepConfig):
        self.base_config = base_config
        self.sweep_config = sweep_config
        self.results = []

    def run(self) -> pd.DataFrame:
        """Run all parameter combinations and collect results"""
        combinations = self._generate_combinations()

        for params in combinations:
            # Extract dataset parameters
            dataset_params = {}
            model_params = {}

            for key, value in params.items():
                if key.startswith('dataset.'):
                    # Strip 'dataset.' prefix
                    dataset_params[key[8:]] = value
                else:
                    model_params[key] = value

            # Load dataset for this combination
            dataset = dataset_registry.load_dataset(
                dataset_params.pop('name'),  # Remove name from params
                **dataset_params
            )

            # Create a new config with model parameters
            config = self._update_config(model_params)

            # Run experiment with this config
            runner = ExperimentRunner(config, dataset=dataset)
            result = runner.run()

            # Add all parameters to result dictionary
            result_with_params = {
                'parameters': params,
                'metrics': result
            }
            self.results.append(result_with_params)

        return self._create_results_dataframe()

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for the sweep"""
        param_names = list(self.sweep_config.parameter_ranges.keys())
        param_values = list(self.sweep_config.parameter_ranges.values())

        combinations = []
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            params.update(self.sweep_config.fixed_parameters)

            # Get default dataset parameters if not specified
            dataset_name = params.get('dataset.name')
            if dataset_name:
                default_params = dataset_registry.get_dataset_params(dataset_name)
                for param_name, default_value in default_params.items():
                    full_param_name = f'dataset.{param_name}'
                    if full_param_name not in params:
                        params[full_param_name] = default_value

            combinations.append(params)

        return combinations

    def _update_config(self, params: Dict[str, Any]) -> 'ExperimentConfig':
        """Create new config with updated parameters"""
        config = self.base_config

        for param_name, value in params.items():
            # Skip dataset parameters as they're handled separately
            if param_name.startswith('dataset.'):
                continue

            path = param_name.split('.')
            target = config
            for component in path[:-1]:
                target = getattr(target, component)
            setattr(target, path[-1], value)

        return config

    def _create_results_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame"""
        flat_results = []

        for result in self.results:
            row = {}
            # Add all parameters (including dataset parameters)
            for param_name, value in result['parameters'].items():
                row[f'param_{param_name}'] = value

            # Add metrics
            metrics = result['metrics']
            for level in ['sample', 'patient']:
                for metric, value in metrics['test'][f'{level}_level'].items():
                    if metric != 'confusion_matrix':
                        row[f'{level}_{metric}'] = value

            flat_results.append(row)

        return pd.DataFrame(flat_results)

    def save_results(self, filepath: Union[str, Path]):
        """Save results to file"""
        results_df = self._create_results_dataframe()
        results_df.to_csv(filepath, index=False)