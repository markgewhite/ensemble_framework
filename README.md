# Ensemble Learning Framework

A Python framework for ensemble learning with support for patient-level predictions and feature importance analysis. The framework implements three ensemble approaches:
1. Repeated Nested Cross-Validation
2. Bagging
3. Gradient Boosting

## Features
- Patient-level prediction aggregation
- Feature selection tracking
- Permutation feature importance
- Comprehensive evaluation metrics
- Support for group-based cross-validation

## Installation
```bash
git clone [repository-url]
cd ensemble_framework
pip install -e .
```

## Quick Start
```bash
python run_experiment.py \
    --name "example_experiment" \
    --data-path "data/example_dataset.csv" \
    --model-type "repeated_ncv" \
    --n-outer-splits 5 \
    --n-outer-repeats 10
```

## Data Format
The framework expects CSV files with the following structure:
- A column for patient IDs
- Feature columns
- A target column with binary labels (0/1)

Example:
```csv
patient_id,feature1,feature2,feature3,target
P001,0.5,1.2,0.8,0
P002,0.7,0.9,1.1,1
...
```

## Running Experiments
See all available options:
```bash
python run_experiment.py --help
```

Common configurations:
1. Repeated Nested CV:
```bash
python run_experiment.py \
    --name "ncv_experiment" \
    --data-path "data/my_dataset.csv" \
    --model-type "repeated_ncv" \
    --n-outer-splits 5 \
    --n-outer-repeats 10
```

2. Bagging:
```bash
python run_experiment.py \
    --name "bagging_experiment" \
    --data-path "data/my_dataset.csv" \
    --model-type "bagging" \
    --n-estimators 50
```

3. Gradient Boosting:
```bash
python run_experiment.py \
    --name "boosting_experiment" \
    --data-path "data/my_dataset.csv" \
    --model-type "boosting" \
    --n-estimators 100 \
    --learning-rate 0.1
```

## Output
Results are saved in the specified output directory with:
- Configuration file (config.json)
- Results file (results.json) containing:
  - Sample-level metrics
  - Patient-level metrics
  - Feature importance scores
  - Runtime information

## Citation
If you use this framework in your research, please cite:
[Your paper citation]