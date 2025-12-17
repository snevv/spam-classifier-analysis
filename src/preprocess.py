"""
Preprocessing module for spam classification project.

This module handles:
- Loading raw data
- Basic cleaning (remove empty messages)
- Train/test splitting
- Saving processed data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_raw_data(data_path='../data/raw/spam.csv'):
    """
    Load the raw spam dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the raw CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'label' and 'message' columns
    """
    # Get absolute path relative to this file
    base_dir = Path(__file__).parent.parent
    full_path = base_dir / data_path.lstrip('../')
    
    df = pd.read_csv(full_path, encoding='latin-1')
    
    # Clean up column names
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    
    # Remove any rows with missing messages
    df = df.dropna(subset=['message'])
    
    # Remove empty messages
    df = df[df['message'].astype(str).str.strip() != '']
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} messages")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def create_train_test_split(df,
                            test_size=0.2,
                            val_size=0.0,
                            random_state=42,
                            stratify=True):
    """
    Create train/test (and optionally validation) splits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'label' column
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of data for validation set (0.0 to skip)
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Use stratified splitting based on labels
        
    Returns:
    --------
    tuple
        (train_df, test_df) or (train_df, val_df, test_df) if val_size > 0
    """
    stratify_col = df['label'] if stratify else None
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: train vs val (if requested)
    if val_size > 0:
        # Adjust val_size relative to train+val set
        val_size_adjusted = val_size / (1 - test_size)
        stratify_col = train_val_df['label'] if stratify else None
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_col
        )
        
        print(f"\nSplit sizes:")
        print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    else:
        print(f"\nSplit sizes:")
        print(f"Train: {len(train_val_df)} ({len(train_val_df)/len(df)*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_val_df, test_df


def save_processed_data(train_df, test_df, val_df=None, output_dir='../data/processed'):
    """
    Save processed data to CSV files.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    val_df : pd.DataFrame, optional
        Validation data
    output_dir : str
        Directory to save processed data
    """
    # Get absolute path
    base_dir = Path(__file__).parent.parent
    output_path = base_dir / output_dir.lstrip('../')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    train_df.to_csv(output_path / 'train.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)
    
    if val_df is not None:
        val_df.to_csv(output_path / 'val.csv', index=False)
        print(f"\nSaved processed data to {output_path}:")
        print(f"  - train.csv ({len(train_df)} rows)")
        print(f"  - val.csv ({len(val_df)} rows)")
        print(f"  - test.csv ({len(test_df)} rows)")
    else:
        print(f"\nSaved processed data to {output_path}:")
        print(f"  - train.csv ({len(train_df)} rows)")
        print(f"  - test.csv ({len(test_df)} rows)")


def main():
    """
    Main preprocessing pipeline.
    Run this to process the raw data and create train/test splits.
    """
    print("=" * 80)
    print("SPAM CLASSIFICATION - PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Load raw data
    print("\n1. Loading raw data...")
    df = load_raw_data()
    
    # Create splits
    print("\n2. Creating train/test splits...")
    train_df, test_df = create_train_test_split(
        df,
        test_size=0.2,
        val_size=0.0,  # Set to 0.15 if you want validation set
        random_state=42,
        stratify=True
    )
    
    # Save processed data
    print("\n3. Saving processed data...")
    save_processed_data(train_df, test_df)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    
    return train_df, test_df


if __name__ == '__main__':
    train_df, test_df = main()
