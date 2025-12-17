# Results Directory Structure

This directory contains all outputs from model training, evaluation, and analysis.

## Directory Organization

```
results/
├── nb/                    # Naive Bayes model results
│   ├── metrics/           # Evaluation metrics (CSV files)
│   ├── visualizations/   # Plots and charts (PNG files)
│   ├── errors/           # Misclassified examples (CSV files)
│   └── tuning/           # Hyperparameter tuning results
│
├── nn/                    # Neural Network model results
│   ├── metrics/           # Evaluation metrics (CSV files)
│   ├── visualizations/   # Plots and charts (PNG files)
│   ├── errors/           # Misclassified examples (CSV files)
│   └── tuning/           # Hyperparameter tuning results
│
└── comparison/            # Model comparison results
    └── (comparison plots, tables, etc.)
```

## File Descriptions

### Naive Bayes (nb/)

**metrics/**
- `evaluation_metrics.csv` - Accuracy, precision, recall, F1 scores
- `all_predictions.csv` - All test predictions with ground truth

**visualizations/**
- `confusion_matrix.png` - Confusion matrix heatmap

**errors/**
- `misclassified.csv` - Examples that were incorrectly classified

**tuning/**
- `best_params.json` - Best hyperparameters from grid search
- `all_results.csv` - All hyperparameter combinations tested

### Neural Network (nn/)

Same structure as `nb/` (will be populated when NN model is trained)

### Comparison

Results comparing both models side-by-side (to be added)