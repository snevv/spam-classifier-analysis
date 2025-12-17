"""
Train and evaluate Naive Bayes spam classifier.

This module:
- Loads processed train/test data
- Creates TF-IDF features
- Trains Multinomial Naive Bayes model
- Evaluates on test set
- Saves model and vectorizer
- Generates metrics and visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


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


def create_features(train_messages, test_messages, max_features=5000, ngram_range=(1, 2)):
    """
    Create TF-IDF features from messages.
    
    Parameters:
    -----------
    train_messages : pd.Series or list
        Training messages
    test_messages : pd.Series or list
        Test messages
    max_features : int
        Maximum number of features (vocabulary size)
    ngram_range : tuple
        Range of n-grams to use (e.g., (1, 2) for unigrams and bigrams)
        
    Returns:
    --------
    tuple
        (vectorizer, X_train, X_test)
    """
    print("\nCreating TF-IDF features...")
    print(f"  - Max features: {max_features}")
    print(f"  - N-gram range: {ngram_range}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words='english'  # Remove common English stopwords
    )
    
    X_train = vectorizer.fit_transform(train_messages)
    X_test = vectorizer.transform(test_messages)
    
    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Test features shape: {X_test.shape}")
    
    return vectorizer, X_train, X_test


def train_model(X_train, y_train, alpha=1.0):
    """
    Train Multinomial Naive Bayes model.
    
    Parameters:
    -----------
    X_train : scipy.sparse matrix
        Training features
    y_train : array-like
        Training labels
    alpha : float
        Smoothing parameter (Laplace smoothing)
        
    Returns:
    --------
    MultinomialNB
        Trained model
    """
    print(f"\nTraining Naive Bayes model (alpha={alpha})...")
    
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    
    print("Training complete!")
    
    return model


def evaluate_model(model, X_test, y_test, y_train=None):
    """
    Evaluate model and print metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : scipy.sparse matrix
        Test features
    y_test : array-like
        Test labels
    y_train : array-like, optional
        Training labels (for baseline comparison)
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam', zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label='spam', zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label='spam', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    
    # Print results
    print("\n" + "=" * 80)
    print("NAIVE BAYES MODEL EVALUATION")
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
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Baseline comparison (if training labels provided)
    if y_train is not None:
        majority_class = y_train.value_counts().index[0]
        baseline_accuracy = (y_test == majority_class).mean()
        print(f"\nBaseline (always predict '{majority_class}'): {baseline_accuracy:.4f}")
        print(f"Improvement over baseline: {accuracy - baseline_accuracy:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return metrics, y_pred


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
    plt.title('Naive Bayes Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
    
    plt.show()


def get_top_features(model, vectorizer, n=20, class_label='spam'):
    """
    Get top features (words) for a given class.
    
    Parameters:
    -----------
    model : MultinomialNB
        Trained model
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    n : int
        Number of top features to return
    class_label : str
        Class to get features for ('spam' or 'ham')
        
    Returns:
    --------
    list
        List of (feature, score) tuples
    """
    # Get class index
    class_idx = list(model.classes_).index(class_label)
    
    # Get feature log probabilities
    feature_log_probs = model.feature_log_prob_[class_idx]
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top features
    top_indices = np.argsort(feature_log_probs)[-n:][::-1]
    top_features = [(feature_names[i], feature_log_probs[i]) for i in top_indices]
    
    return top_features


def save_model_and_vectorizer(model, vectorizer, model_dir='../models'):
    """
    Save trained model and vectorizer.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    model_dir : str
        Directory to save models
    """
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / model_dir.lstrip('../')
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(model_path / 'nb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save vectorizer
    with open(model_path / 'nb_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nModel and vectorizer saved to {model_path}/")


def main():
    """
    Main training pipeline.
    """
    print("=" * 80)
    print("NAIVE BAYES SPAM CLASSIFIER - TRAINING")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading processed data...")
    train_df, test_df = load_processed_data()
    
    # Create features
    print("\n2. Creating features...")
    vectorizer, X_train, X_test = create_features(
        train_df['message'],
        test_df['message'],
        max_features=5000,
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    
    # Train model
    print("\n3. Training model...")
    model = train_model(X_train, train_df['label'], alpha=1.0)
    
    # Evaluate
    print("\n4. Evaluating model...")
    metrics, y_pred = evaluate_model(
        model, X_test, test_df['label'], y_train=train_df['label']
    )
    
    # Visualize
    print("\n5. Generating visualizations...")
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results' / 'nb'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (results_dir / 'visualizations').mkdir(exist_ok=True)
    (results_dir / 'errors').mkdir(exist_ok=True)
    (results_dir / 'metrics').mkdir(exist_ok=True)
    
    plot_confusion_matrix(metrics['confusion_matrix'],
                          save_path=results_dir / 'visualizations' / 'confusion_matrix.png')
    
    # Show top features
    print("\n6. Top features for spam classification:")
    top_spam = get_top_features(model, vectorizer, n=20, class_label='spam')
    print("\nTop 20 spam-indicative words:")
    for word, score in top_spam:
        print(f"  {word:20s} {score:8.4f}")
    
    print("\nTop 20 ham-indicative words:")
    top_ham = get_top_features(model, vectorizer, n=20, class_label='ham')
    for word, score in top_ham:
        print(f"  {word:20s} {score:8.4f}")
    
    # Save model
    print("\n6. Saving model...")
    save_model_and_vectorizer(model, vectorizer)
    
    # Save metrics
    print("\n7. Saving metrics...")
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
    
    # Save all predictions for analysis
    test_df_with_preds.to_csv(results_dir / 'metrics' / 'all_predictions.csv', index=False)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    return model, vectorizer, metrics


if __name__ == '__main__':
    model, vectorizer, metrics = main()

