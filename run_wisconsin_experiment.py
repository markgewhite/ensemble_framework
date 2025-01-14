import argparse
from pathlib import Path
from datetime import datetime

from ensemble_framework.data.wisconsin_loader import load_wisconsin_data
from ensemble_framework.experiments.config import (
    ExperimentConfig, ModelConfig, PipelineConfig, ExperimentRunner
)


def create_wisconsin_config(args):
    """Create configuration for Wisconsin breast cancer experiment"""
    model_config = ModelConfig(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_outer_splits=args.n_outer_splits,
        n_outer_repeats=args.n_outer_repeats,
        learning_rate=args.learning_rate if args.model_type == 'boosting' else 0.1,
        max_samples=args.max_samples if args.model_type == 'bagging' else 1.0
    )

    pipeline_config = PipelineConfig(
        include_scaling=True,  # Wisconsin data needs scaling
        include_feature_selection=args.feature_selection,
        base_model='svc',
        model_params={
            'kernel': 'rbf',
            'C': 1.0,
            'probability': True
        },
        param_grid=None
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"wisconsin_{args.model_type}_{timestamp}"

    config = ExperimentConfig(
        name=f"wisconsin_{args.model_type}",
        description=(
            f"Wisconsin Breast Cancer experiment using {args.model_type} ensemble "
            f"with {args.n_estimators} estimators"
        ),
        output_dir=output_dir,
        model=model_config,
        pipeline=pipeline_config
    )

    return config


def main():
    parser = argparse.ArgumentParser(description='Run Wisconsin Breast Cancer experiment')

    # Model selection
    parser.add_argument('--model-type', required=True,
                        choices=['repeated_ncv', 'bagging', 'boosting'],
                        help='Type of ensemble model')

    # General parameters
    parser.add_argument('--n-estimators', type=int, default=50,
                        help='Number of estimators')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory')
    parser.add_argument('--feature-selection', action='store_true',
                        help='Enable feature selection')

    # Model-specific parameters
    parser.add_argument('--n-outer-splits', type=int, default=5,
                        help='Number of outer CV splits (for repeated_ncv)')
    parser.add_argument('--n-outer-repeats', type=int, default=1,
                        help='Number of outer CV repeats (for repeated_ncv)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate (for boosting)')
    parser.add_argument('--max-samples', type=float, default=1.0,
                        help='Sample size ratio (for bagging)')

    args = parser.parse_args()

    # Load Wisconsin data
    print("Loading Wisconsin Breast Cancer dataset...")
    dataset = load_wisconsin_data(random_state=args.random_state)
    print(f"Loaded {dataset.n_samples} samples from {len(dataset.unique_patients)} patients")
    print(f"Class distribution: {np.bincount(dataset.y)}")

    # Create configuration
    config = create_wisconsin_config(args)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config.save(config.output_dir / "config.json")

    # Run experiment
    print(f"\nRunning {args.model_type} experiment...")
    runner = ExperimentRunner(config, dataset)

    # Run experiment and get results
    results = runner.run()

    # Save results
    runner.save_results(config.output_dir / "results.json")

    print(f"\nExperiment completed. Results saved to {config.output_dir}")

    # Print summary metrics
    print("\nTest Set Results:")
    print("Sample-level metrics:")
    for metric, value in results['test']['sample_level'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.3f}")

    print("\nPatient-level metrics:")
    for metric, value in results['test']['patient_level'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.3f}")

    print(f"\nRuntime: {results['runtime']:.2f} seconds")

    # Print top features if feature selection was enabled
    if args.feature_selection:
        print("\nTop 10 features by importance:")
        importance_df = results['feature_importance']
        print(importance_df.head(10)[['feature_name', 'selection_rate', 'importance_mean']])


if __name__ == "__main__":
    import numpy as np

    main()