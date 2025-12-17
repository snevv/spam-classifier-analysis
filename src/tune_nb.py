"""
Hyperparameter tuning for Naive Bayes spam classifier.

Uses cross-validation to find best parameters for:
- TF-IDF: max_features, ngram_range
- Naive Bayes: alpha (smoothing parameter)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
import json


def load_processed_data(train_path='../data/processed/train.csv'):
    """Load training data for hyperparameter tuning."""
    base_dir = Path(__file__).parent.parent
    train_df = pd.read_csv(base_dir / train_path.lstrip('../'))
    return train_df


def tune_hyperparameters(train_df, cv_folds=5, n_jobs=-1):
    """
    Tune hyperparameters using grid search with cross-validation.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with 'message' and 'label' columns
    cv_folds : int
        Number of cross-validation folds
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
        
    Returns:
    --------
    dict
        Best parameters and scores
    """
    print("=" * 80)
    print("HYPERPARAMETER TUNING - NAIVE BAYES")
    print("=" * 80)
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
        ('nb', MultinomialNB())
    ])
    
    # Define parameter grid
    # Note: max_features capped at 10000 for fair comparison with NN model
    param_grid = {
        'tfidf__max_features': [2500, 5000, 7500, 10000],  # Tests up to cap with intermediate values
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],  # unigrams, bigrams, trigrams
        'nb__alpha': [0.1, 0.5, 1.0, 2.0]  # Smoothing parameter
    }
    
    print(f"\nParameter grid:")
    print(f"  - max_features: {param_grid['tfidf__max_features']}")
    print(f"  - ngram_range: {param_grid['tfidf__ngram_range']}")
    print(f"  - alpha: {param_grid['nb__alpha']}")
    print(f"  - Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Use F1 score for spam class as the scoring metric
    scorer = make_scorer(f1_score, pos_label='spam')
    
    # Stratified K-Fold to maintain class balance
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    print(f"\nUsing {cv_folds}-fold stratified cross-validation...")
    print("This may take a few minutes...\n")
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Fit on training data
    grid_search.fit(train_df['message'], train_df['label'])
    
    # Results
    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    print(f"\nBest F1 Score (CV): {grid_search.best_score_:.4f}")
    print(f"\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Show top 5 configurations
    print("\nTop 5 configurations:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]
    for idx, row in top_5.iterrows():
        print(f"\n  Rank {len(top_5) - list(top_5.index).index(idx)}:")
        print(f"    F1 Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f})")
        print(f"    Params: {row['params']}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': results_df
    }


def save_tuning_results(results, output_dir='../results/nb/tuning'):
    """Save tuning results to JSON and CSV."""
    base_dir = Path(__file__).parent.parent
    output_path = base_dir / output_dir.lstrip('../')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save best parameters as JSON
    best_params = results['best_params'].copy()
    # Convert tuple to list for JSON serialization
    if 'tfidf__ngram_range' in best_params:
        best_params['tfidf__ngram_range'] = list(best_params['tfidf__ngram_range'])
    
    with open(output_path / 'best_params.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_cv_score': float(results['best_score'])
        }, f, indent=2)
    
    # Save all CV results
    results['cv_results'].to_csv(output_path / 'all_results.csv', index=False)
    
    print(f"\nTuning results saved to {output_path}/")
    print(f"  - best_params.json")
    print(f"  - all_results.csv")


def main():
    """Main tuning pipeline."""
    # Load data
    print("\n1. Loading training data...")
    train_df = load_processed_data()
    print(f"   Loaded {len(train_df)} training samples")
    
    # Tune hyperparameters
    print("\n2. Tuning hyperparameters...")
    results = tune_hyperparameters(train_df, cv_folds=5)
    
    # Save results
    print("\n3. Saving results...")
    save_tuning_results(results)
    
    print("\n" + "=" * 80)
    print("TUNING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review results/nb/tuning/best_params.json for best parameters")
    print("2. Update train_nb.py to use these parameters")
    print("3. Train final model on full training set")
    print("4. Evaluate on test set")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    results = main()

