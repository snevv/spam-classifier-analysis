"""
Hyperparameter tuning for Neural Network spam classifier.

Uses cross-validation to find best parameters for:
- Architecture: number of layers, hidden units
- Regularization: dropout rate, L2 regularization
- Training: learning rate, batch size
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
warnings.filterwarnings('ignore')

# Disable GPU to avoid CUDA errors (use CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Cap max_features to match Naive Bayes (for fair comparison)
MAX_FEATURES_CAP = 10000


def load_processed_data(train_path='../data/processed/train.csv'):
    """Load training data for hyperparameter tuning."""
    base_dir = Path(__file__).parent.parent
    train_df = pd.read_csv(base_dir / train_path.lstrip('../'))
    return train_df


def create_features(messages, vectorizer=None, fit=False):
    """
    Create TF-IDF features and convert to dense arrays.
    
    Parameters:
    -----------
    messages : pd.Series or list
        Messages to vectorize
    vectorizer : TfidfVectorizer, optional
        Fitted vectorizer (if None, creates new one)
    fit : bool
        Whether to fit the vectorizer (True for training, False for test)
        
    Returns:
    --------
    tuple : (vectorizer, X_dense)
        Vectorizer and dense feature matrix
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES_CAP,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english'
        )
    
    if fit:
        X_sparse = vectorizer.fit_transform(messages)
    else:
        X_sparse = vectorizer.transform(messages)
    
    # Convert sparse matrix to dense array for neural network
    X_dense = X_sparse.toarray()
    
    return vectorizer, X_dense


def encode_labels(labels):
    """
    Encode labels as binary (0 = ham, 1 = spam).
    
    Parameters:
    -----------
    labels : array-like
        String labels ('ham' or 'spam')
        
    Returns:
    --------
    np.array : Binary encoded labels
    """
    return np.array([1 if label == 'spam' else 0 for label in labels])


def create_model(input_dim, 
                 hidden_units=128,
                 num_layers=1,
                 dropout_rate=0.5,
                 l2_reg=0.01,
                 learning_rate=0.001):
    """
    Create a feedforward neural network model.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_units : int
        Number of units in each hidden layer
    num_layers : int
        Number of hidden layers (1 or 2)
    dropout_rate : float
        Dropout rate for regularization
    l2_reg : float
        L2 regularization strength
    learning_rate : float
        Learning rate for optimizer
        
    Returns:
    --------
    keras.Model : Compiled model
    """
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Dense(
        hidden_units,
        input_dim=input_dim,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    ))
    model.add(layers.Dropout(dropout_rate))
    
    # Additional hidden layer if specified
    if num_layers == 2:
        model.add(layers.Dense(
            hidden_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer (binary classification)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def evaluate_model_cv(X, y, 
                     hidden_units=128,
                     num_layers=1,
                     dropout_rate=0.5,
                     l2_reg=0.01,
                     learning_rate=0.001,
                     batch_size=32,
                     cv_folds=5,
                     epochs=50,
                     verbose=0):
    """
    Evaluate model using cross-validation.
    
    Parameters:
    -----------
    X : np.array
        Feature matrix
    y : np.array
        Binary labels
    hidden_units : int
        Number of hidden units
    num_layers : int
        Number of hidden layers
    dropout_rate : float
        Dropout rate
    l2_reg : float
        L2 regularization
    learning_rate : float
        Learning rate
    batch_size : int
        Batch size
    cv_folds : int
        Number of CV folds
    epochs : int
        Maximum epochs
    verbose : int
        Verbosity level
        
    Returns:
    --------
    float : Mean F1 score across CV folds
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create and train model
        model = create_model(
            input_dim=X.shape[1],
            hidden_units=hidden_units,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            learning_rate=learning_rate
        )
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        # Predict and evaluate
        y_pred_proba = model.predict(X_val_fold, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        f1 = f1_score(y_val_fold, y_pred)
        f1_scores.append(f1)
        
        # Clear memory
        del model
        keras.backend.clear_session()
    
    return np.mean(f1_scores)


def tune_hyperparameters(train_df, cv_folds=5):
    """
    Tune hyperparameters using grid search with cross-validation.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with 'message' and 'label' columns
    cv_folds : int
        Number of CV folds
        
    Returns:
    --------
    dict : Best parameters and scores
    """
    print("=" * 80)
    print("HYPERPARAMETER TUNING - NEURAL NETWORK")
    print("=" * 80)
    
    # Create features
    print("\n1. Creating TF-IDF features...")
    vectorizer, X = create_features(train_df['message'], fit=True)
    y = encode_labels(train_df['label'])
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Max features: {MAX_FEATURES_CAP}")
    
    # Define parameter grid (reduced for efficiency)
    # Start with key parameters, can expand later if needed
    param_grid = {
        'hidden_units': [64, 128],  # Reduced from [64, 128, 256]
        'num_layers': [1, 2],
        'dropout_rate': [0.3, 0.5],
        'l2_reg': [0.001, 0.01],
        'learning_rate': [0.001],  # Reduced from [0.001, 0.0001] - 0.001 is usually good
        'batch_size': [32]  # Reduced from [32, 64] - 32 is standard
    }
    
    print(f"\n2. Parameter grid:")
    for param, values in param_grid.items():
        print(f"   - {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n   Total combinations: {total_combinations}")
    print(f"   Using {cv_folds}-fold cross-validation")
    print(f"   Total model trainings: {total_combinations * cv_folds}")
    print("\n   This will take a while... Starting grid search...\n")
    
    # Grid search
    best_score = -1
    best_params = None
    results = []
    
    from itertools import product
    
    for idx, params in enumerate(product(*param_grid.values()), 1):
        param_dict = dict(zip(param_grid.keys(), params))
        
        print(f"[{idx}/{total_combinations}] Testing: {param_dict}")
        
        try:
            score = evaluate_model_cv(
                X, y,
                cv_folds=cv_folds,
                epochs=50,
                verbose=0,
                **param_dict
            )
            
            print(f"         CV F1 Score: {score:.4f}\n")
            
            results.append({
                **param_dict,
                'cv_f1_score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = param_dict.copy()
                print(f"         *** New best! F1 = {best_score:.4f} ***\n")
        
        except Exception as e:
            print(f"         Error: {e}\n")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    print(f"\nBest CV F1 Score: {best_score:.4f}")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Show top 5 configurations
    if len(results_df) > 0:
        print("\nTop 5 configurations:")
        top_5 = results_df.nlargest(5, 'cv_f1_score')
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            print(f"\n  Rank {i}:")
            print(f"    F1 Score: {row['cv_f1_score']:.4f}")
            params_str = ", ".join([f"{k}={v}" for k, v in row.items() if k != 'cv_f1_score'])
            print(f"    Params: {params_str}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': results_df,
        'vectorizer': vectorizer
    }


def save_tuning_results(results, output_dir='../results/nn/tuning'):
    """Save tuning results to JSON and CSV."""
    base_dir = Path(__file__).parent.parent
    output_path = base_dir / output_dir.lstrip('../')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save best parameters as JSON
    best_params = results['best_params'].copy()
    
    with open(output_path / 'best_params.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_cv_score': float(results['best_score'])
        }, f, indent=2)
    
    # Save all CV results
    if len(results['cv_results']) > 0:
        results['cv_results'].to_csv(output_path / 'all_results.csv', index=False)
    
    print(f"\nTuning results saved to {output_path}/")
    print(f"  - best_params.json")
    print(f"  - all_results.csv")


def main():
    """Main tuning pipeline."""
    # Load data
    print("\nLoading training data...")
    train_df = load_processed_data()
    print(f"   Loaded {len(train_df)} training samples")
    
    # Tune hyperparameters
    results = tune_hyperparameters(train_df, cv_folds=5)
    
    # Save results
    print("\nSaving results...")
    save_tuning_results(results)
    
    print("\n" + "=" * 80)
    print("TUNING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review results/nn/tuning/best_params.json for best parameters")
    print("2. Run train_nn.py to train final model with best parameters")
    print("3. Evaluate on test set and compare with Naive Bayes")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    results = main()

