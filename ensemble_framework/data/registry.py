from typing import Dict, Callable, Optional, Type
from pathlib import Path
import importlib
from dataclasses import dataclass
from .dataset import Dataset


@dataclass
class DatasetInfo:
    """Information about a registered dataset"""
    name: str
    loader_func: Callable
    description: str
    default_params: dict = None


class DatasetRegistry:
    """Registry for managing available datasets and their loaders"""

    def __init__(self):
        self._datasets: Dict[str, DatasetInfo] = {}

    def register(self, name: str, loader_func: Callable,
                 description: str, default_params: Optional[dict] = None):
        """Register a new dataset"""
        self._datasets[name] = DatasetInfo(
            name=name,
            loader_func=loader_func,
            description=description,
            default_params=default_params or {}
        )

    def load_dataset(self, name: str, **kwargs) -> Dataset:
        """Load a registered dataset with optional parameters"""
        if name not in self._datasets:
            raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(self._datasets.keys())}")

        dataset_info = self._datasets[name]

        # Merge default parameters with provided kwargs
        params = dataset_info.default_params.copy()
        params.update(kwargs)

        return dataset_info.loader_func(**params)

    def get_available_datasets(self) -> Dict[str, str]:
        """Get dictionary of available datasets and their descriptions"""
        return {name: info.description for name, info in self._datasets.items()}

    def get_dataset_params(self, name: str) -> dict:
        """Get default parameters for a dataset"""
        if name not in self._datasets:
            raise ValueError(f"Unknown dataset: {name}")
        return self._datasets[name].default_params.copy()


# Create global registry instance
registry = DatasetRegistry()

# Initialize with Wisconsin dataset
from .wisconsin_loader import load_wisconsin_data
registry.register(
    name="wisconsin",
    loader_func=load_wisconsin_data,
    description="Wisconsin Breast Cancer dataset with simulated patient IDs",
    default_params={"random_state": 42}
)