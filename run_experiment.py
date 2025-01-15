import argparse
from pathlib import Path
from datetime import datetime

from ensemble_framework.experiments import (
    ExperimentConfig, ModelConfig, PipelineConfig, ExperimentRunner
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run ensemble learning experiment')

    # Experiment identification
    parser.add_argument('--name', required=True, help='Experiment name')
    parser.add_argument('--description', help='Experiment description')
    parser.add_argument('--output-dir', default='results', help='Output directory')

    # Data configuration
    parser.add_argument('--data-path', required=True, help='Path to data file')
    parser.add_argument('--feature-file', help='File containing feature column names')
    parser.add_argument('--target-col', default='target', help='Target column name')
    parser.add_argument('--patient-id-col', default='patient_id', help='Patient ID column name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')

    # Model configuration
    parser.add_argument('--model-type', default='repeated_ncv',
                        choices=['repeated_ncv', 'bagging', 'boosting'],
                        help='Type of ensemble model')
    parser.add_argument('--n-estimators', type=int, default=50,
                        help='Number of estimators')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')

    # Model-specific parameters
    parser.add_argument('--n-outer-splits', type=int, default=5,
                        help='Number of outer CV splits (for repeated_ncv)')
    parser.add_argument('--n-outer-repeats', type=int, default=1,
                        help='Number of outer CV repeats (for repeated_ncv)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate (for boosting)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load feature columns if provided
    if args.feature_file:
        with open(args.feature_file) as f:
            feature_cols = f.read().strip().split('\n')
    else:
        feature_cols = []  # Will use all columns except target and patient_id

    # Create model configuration
    model_config = ModelConfig(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_outer_splits=args.n_outer_splits,
        n_outer_repeats=args.n_outer_repeats,
        learning_rate=args.learning_rate
    )

    # Create pipeline configuration
    pipeline_config = PipelineConfig()  # Use defaults

    # Create experiment configuration
    config = ExperimentConfig(
        name=args.name,
        description=args.description,
        output_dir=args.output_dir,
        data_path=args.data_path,
        feature_cols=feature_cols,
        target_col=args.target_col,
        patient_id_col=args.patient_id_col,
        test_size=args.test_size,
        model=model_config,
        pipeline=pipeline_config
    )

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config.save(output_dir / "config.json")

    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run()

    # Save results
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