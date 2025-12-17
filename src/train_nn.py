"""
Train and evaluate Neural Network spam classifier.

This module:
- Loads processed train/test data
- Creates TF-IDF features (same as Naive Bayes for fair comparison)
- Trains feedforward neural network
- Evaluates on test set
- Saves model and results
- Generates metrics and visualizations
"""

import os
# Disable GPU to avoid CUDA errors (must be set before importing TensorFlow)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Cap max_features to match Naive Bayes (for fair comparison)
MAX_FEATURES_CAP = 10000


def load_processed_data(train_path='../data/processed/train.csv',
                        test_path='../data/processed/test.csv'):
    """
    Load processed train and test datasets.
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV
    test_path : str
        Path to test CSV
        
    Returns:
    --------
    tuple
        (train_df, test_df)
    """
    base_dir = Path(__file__).parent.parent
    
    train_df = pd.read_csv(base_dir / train_path.lstrip('../'))
    test_df = pd.read_csv(base_dir / test_path.lstrip('../'))
    
    print(f"Loaded training data: {len(train_df)} samples")
    print(f"Loaded test data: {len(test_df)} samples")
    print(f"\nTraining label distribution:\n{train_df['label'].value_counts()}")
    print(f"\nTest label distribution:\n{test_df['label'].value_counts()}")
    
    return train_df, test_df


def create_features(train_messages, test_messages, max_features=None, ngram_range=None):
    """
    Create TF-IDF features and convert to dense arrays.
    
    Parameters:
    -----------
    train_messages : pd.Series or list
        Training messages
    test_messages : pd.Series or list
        Test messages
    max_features : int, optional
        Maximum number of features (capped at MAX_FEATURES_CAP)
    ngram_range : tuple, optional
        Range of n-grams to use
        
    Returns:
    --------
    tuple
        (vectorizer, X_train, X_test)
    """
    # Use defaults if not provided
    if max_features is None:
        max_features = 5000
    if ngram_range is None:
        ngram_range = (1, 2)
    
    # Enforce cap
    max_features = min(max_features, MAX_FEATURES_CAP)
    
    print("\nCreating TF-IDF features...")
    print(f"  - Max features: {max_features} (capped at {MAX_FEATURES_CAP})")
    print(f"  - N-gram range: {ngram_range}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words='english'
    )
    
    # Fit on training, transform both
    X_train_sparse = vectorizer.fit_transform(train_messages)
    X_test_sparse = vectorizer.transform(test_messages)
    
    # Convert to dense arrays for neural network
    X_train = X_train_sparse.toarray()
    X_test = X_test_sparse.toarray()
    
    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Test features shape: {X_test.shape}")
    
    return vectorizer, X_train, X_test


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


def train_model(X_train, y_train, X_val, y_val,
                hidden_units=128,
                num_layers=1,
                dropout_rate=0.5,
                l2_reg=0.01,
                learning_rate=0.001,
                batch_size=32,
                epochs=100):
    """
    Train neural network model.
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    y_train : np.array
        Training labels (binary)
    X_val : np.array
        Validation features
    y_val : np.array
        Validation labels (binary)
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
    epochs : int
        Maximum epochs
        
    Returns:
    --------
    keras.Model : Trained model
    keras.History : Training history
    """
    print(f"\nTraining Neural Network model...")
    print(f"  Architecture: {num_layers} hidden layer(s), {hidden_units} units each")
    print(f"  Dropout: {dropout_rate}, L2: {l2_reg}, LR: {learning_rate}")
    print(f"  Batch size: {batch_size}, Max epochs: {epochs}")
    
    # Create model
    model = create_model(
        input_dim=X_train.shape[1],
        hidden_units=hidden_units,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        learning_rate=learning_rate
    )
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("Training complete!")
    
    return model, history


def evaluate_model(model, X_test, y_test, y_train=None):
    """
    Evaluate model and print metrics.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_test : np.array
        Test features
    y_test : np.array
        Test labels (binary)
    y_train : np.array, optional
        Training labels (for baseline comparison)
        
    Returns:
    --------
    dict : Dictionary of metrics
    np.array : Predictions
    """
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Convert back to string labels for metrics
    y_test_str = ['spam' if label == 1 else 'ham' for label in y_test]
    y_pred_str = ['spam' if label == 1 else 'ham' for label in y_pred]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_str, y_pred_str)
    precision = precision_score(y_test_str, y_pred_str, pos_label='spam', zero_division=0)
    recall = recall_score(y_test_str, y_pred_str, pos_label='spam', zero_division=0)
    f1 = f1_score(y_test_str, y_pred_str, pos_label='spam', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_str, y_pred_str, labels=['ham', 'spam'])
    
    # Print results
    print("\n" + "=" * 80)
    print("NEURAL NETWORK MODEL EVALUATION")
    print("=" * 80)
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("              Ham    Spam")
    print(f"Actual Ham   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Spam  {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_str, y_pred_str, target_names=['Ham', 'Spam']))
    
    # Baseline comparison
    if y_train is not None:
        y_train_str = ['spam' if label == 1 else 'ham' for label in y_train]
        majority_class = pd.Series(y_train_str).value_counts().index[0]
        baseline_accuracy = (pd.Series(y_test_str) == majority_class).mean()
        print(f"\nBaseline (always predict '{majority_class}'): {baseline_accuracy:.4f}")
        print(f"Improvement over baseline: {accuracy - baseline_accuracy:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return metrics, y_pred_str


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy).
    
    Parameters:
    -----------
    history : keras.History
        Training history
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.title('Neural Network Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
    
    plt.show()


def save_model_and_vectorizer(model, vectorizer, model_dir='../models'):
    """
    Save trained model and vectorizer.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    model_dir : str
        Directory to save models
    """
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / model_dir.lstrip('../')
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Save model (Keras format)
    model.save(model_path / 'nn_model.h5')
    
    # Save vectorizer
    with open(model_path / 'nn_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nModel and vectorizer saved to {model_path}/")
    print(f"  - nn_model.h5")
    print(f"  - nn_vectorizer.pkl")


def main():
    """
    Main training pipeline.
    """
    print("=" * 80)
    print("NEURAL NETWORK SPAM CLASSIFIER - TRAINING")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading processed data...")
    train_df, test_df = load_processed_data()
    
    # Try to load best parameters from tuning, otherwise use defaults
    base_dir = Path(__file__).parent.parent
    best_params_path = base_dir / 'results' / 'nn' / 'tuning' / 'best_params.json'
    
    if best_params_path.exists():
        print("\n2. Loading best parameters from tuning...")
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)['best_params']
        print(f"   Using tuned parameters: {best_params}")
    else:
        print("\n2. Using default parameters (run tune_nn.py first for optimal parameters)...")
        best_params = {
            'hidden_units': 128,
            'num_layers': 1,
            'dropout_rate': 0.5,
            'l2_reg': 0.01,
            'learning_rate': 0.001,
            'batch_size': 32
        }
    
    # Create features (use same settings as tuning or defaults)
    print("\n3. Creating features...")
    vectorizer, X_train_full, X_test = create_features(
        train_df['message'],
        test_df['message'],
        max_features=MAX_FEATURES_CAP,
        ngram_range=(1, 2)
    )
    
    # Encode labels
    y_train_full = encode_labels(train_df['label'])
    y_test = encode_labels(test_df['label'])
    
    # Create validation split from training data
    print("\n4. Creating train/validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15,
        random_state=42,
        stratify=y_train_full
    )
    print(f"   Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model
    print("\n5. Training model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        hidden_units=best_params.get('hidden_units', 128),
        num_layers=best_params.get('num_layers', 1),
        dropout_rate=best_params.get('dropout_rate', 0.5),
        l2_reg=best_params.get('l2_reg', 0.01),
        learning_rate=best_params.get('learning_rate', 0.001),
        batch_size=best_params.get('batch_size', 32),
        epochs=100
    )
    
    # Evaluate
    print("\n6. Evaluating model...")
    metrics, y_pred = evaluate_model(
        model, X_test, y_test, y_train=y_train
    )
    
    # Visualize
    print("\n7. Generating visualizations...")
    results_dir = base_dir / 'results' / 'nn'
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'visualizations').mkdir(exist_ok=True)
    (results_dir / 'metrics').mkdir(exist_ok=True)
    (results_dir / 'errors').mkdir(exist_ok=True)
    
    plot_training_history(history, save_path=results_dir / 'visualizations' / 'training_history.png')
    plot_confusion_matrix(metrics['confusion_matrix'],
                          save_path=results_dir / 'visualizations' / 'confusion_matrix.png')
    
    # Save metrics
    print("\n8. Saving metrics...")
    metrics_df = pd.DataFrame([{
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }])
    metrics_df.to_csv(results_dir / 'metrics' / 'evaluation_metrics.csv', index=False)
    print(f"Metrics saved to {results_dir / 'metrics' / 'evaluation_metrics.csv'}")
    
    # Save predictions for error analysis
    test_df_with_preds = test_df.copy()
    test_df_with_preds['prediction'] = y_pred
    test_df_with_preds['correct'] = (test_df_with_preds['label'] == test_df_with_preds['prediction'])
    
    # Save misclassified examples
    misclassified = test_df_with_preds[~test_df_with_preds['correct']]
    misclassified.to_csv(results_dir / 'errors' / 'misclassified.csv', index=False)
    print(f"Misclassified examples saved to {results_dir / 'errors' / 'misclassified.csv'}")
    print(f"Total misclassified: {len(misclassified)} out of {len(test_df)}")
    
    # Save all predictions
    test_df_with_preds.to_csv(results_dir / 'metrics' / 'all_predictions.csv', index=False)
    
    # Save model
    print("\n9. Saving model...")
    save_model_and_vectorizer(model, vectorizer)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNote: Neural networks are less interpretable than Naive Bayes.")
    print("Consider using SHAP/LIME for interpretability analysis if needed.")
    print("=" * 80)
    
    return model, vectorizer, metrics


if __name__ == '__main__':
    model, vectorizer, metrics = main()

