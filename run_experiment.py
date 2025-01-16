import argparse
from pathlib import Path
from datetime import datetime
import json

from ensemble_framework.experiments import (
    ExperimentConfig, ModelConfig, PipelineConfig, ExperimentRunner,
    ExperimentSweepConfig, SweepRunner
)
from ensemble_framework.data.registry import registry as dataset_registry


def parse_args():
    parser = argparse.ArgumentParser(description='Run ensemble learning experiment')

    # Experiment identification
    parser.add_argument('--name', required=True, help='Experiment name')
    parser.add_argument('--description', help='Experiment description')
    parser.add_argument('--output-dir', default='results', help='Output directory')

    # Parameter sweep configuration
    parser.add_argument('--sweep-config', help='Path to parameter sweep configuration file')

    # Dataset configuration (used when no sweep is specified)
    parser.add_argument('--dataset',
                        choices=list(dataset_registry.get_available_datasets().keys()),
                        help='Dataset to use')
    parser.add_argument('--dataset-params', type=json.loads, default={},
                        help='Dataset parameters as JSON string')

    # Base model configuration (used when no sweep is specified)
    parser.add_argument('--model-type', default='repeated_ncv',
                        choices=['repeated_ncv', 'bagging', 'boosting'],
                        help='Type of ensemble model')
    parser.add_argument('--n-estimators', type=int, default=50,
                        help='Number of estimators')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n-outer-splits', type=int, default=5,
                        help='Number of outer CV splits (for repeated_ncv)')
    parser.add_argument('--n-outer-repeats', type=int, default=1,
                        help='Number of outer CV repeats (for repeated_ncv)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate (for boosting)')

    return parser.parse_args()


def create_base_config(args):
    """Create base experiment configuration"""
    model_config = ModelConfig(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_outer_splits=args.n_outer_splits,
        n_outer_repeats=args.n_outer_repeats,
        learning_rate=args.learning_rate
    )

    pipeline_config = PipelineConfig()  # Use defaults

    config = ExperimentConfig(
        name=args.name,
        description=args.description,
        output_dir=args.output_dir,
        model=model_config,
        pipeline=pipeline_config
    )

    return config


def main():
    parser = argparse.ArgumentParser(description='Run ensemble learning experiment')
    args = parse_args()

    if not args.sweep_config and not args.dataset:
        parser.error("--dataset is required when not using --sweep-config")

    # Create timestamp for output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create base configuration
    config = create_base_config(args)

    if args.sweep_config:
        # Load parameter sweep configuration
        sweep_config = ExperimentSweepConfig.load(args.sweep_config)

        # Run parameter sweep
        sweep_runner = SweepRunner(config, sweep_config)
        results_df = sweep_runner.run()

        # Save sweep results
        sweep_runner.save_results(output_dir / "sweep_results.csv")
        sweep_config.save(output_dir / "sweep_config.json")

        print(f"\nParameter sweep completed. Results saved to {output_dir}")

        # Print summary
        print("\nSweep Summary:")
        print(f"Total configurations tested: {len(results_df)}")

        # Get unique datasets if multiple were used
        datasets = results_df['param_dataset.name'].unique()
        for dataset in datasets:
            df_dataset = results_df[results_df['param_dataset.name'] == dataset]
            print(f"\nResults for dataset: {dataset}")
            print("Top 5 configurations by patient AUC:")
            top_configs = df_dataset.nlargest(5, 'patient_auc')

            # Get parameter columns (excluding dataset name)
            param_cols = [col for col in df_dataset.columns
                          if col.startswith('param_') and col != 'param_dataset.name']

            print(top_configs[['patient_auc'] + param_cols])

    else:
        if not args.dataset:
            parser.error("--dataset is required when not using --sweep-config")

        # Load dataset
        try:
            dataset = dataset_registry.load_dataset(args.dataset, **args.dataset_params)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        print(f"\nLoaded dataset: {args.dataset}")
        print(f"Number of samples: {dataset.n_samples}")
        print(f"Number of features: {dataset.n_features}")
        print(f"Number of patients: {len(dataset.unique_patients)}")

        # Run single experiment
        runner = ExperimentRunner(config, dataset=dataset)
        results = runner.run()

        # Save configuration and results
        config.save(output_dir / "config.json")
        runner.save_results(output_dir / "results.json")

        print(f"\nExperiment completed. Results saved to {output_dir}")

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


if __name__ == "__main__":
    main()