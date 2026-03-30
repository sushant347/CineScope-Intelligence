"""
Advanced Classical ML Models
=============================
- Random Forest
- Support Vector Machine (SVM)
- Model Comparison
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)


def train_random_forest(X_train, X_test, y_train, y_test, model_dir='models'):
    """Train Random Forest classifier."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    
    print(f"\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'random_forest.pkl'))
    print(f"Model saved to {model_dir}/random_forest.pkl")
    
    return model, metrics


def train_svm(X_train, X_test, y_train, y_test, model_dir='models'):
    """Train SVM classifier with probability calibration."""
    print("\n" + "=" * 60)
    print("SVM CLASSIFIER (LinearSVC + Calibration)")
    print("=" * 60)
    
    # LinearSVC is faster than SVC for large datasets
    base_model = LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=42,
        dual='auto'
    )
    
    # Wrap with calibration for probability estimates
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    
    print(f"\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'svm_model.pkl'))
    print(f"Model saved to {model_dir}/svm_model.pkl")
    
    return model, metrics


def compare_models(metrics_dict):
    """Print model comparison table."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison = pd.DataFrame(metrics_dict).T
    comparison.index.name = 'Model'
    comparison = comparison.round(4)
    
    print(comparison.to_string())
    print(f"\nBest model by accuracy: {comparison['accuracy'].idxmax()}")
    print(f"Best model by F1: {comparison['f1'].idxmax()}")
    
    return comparison


def train_all_advanced(df, model_dir='models'):
    """Train all advanced models and compare."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # TF-IDF (reuse same vectorizer)
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Train models
    rf_model, rf_metrics = train_random_forest(
        X_train_tfidf, X_test_tfidf, y_train, y_test, model_dir
    )
    svm_model, svm_metrics = train_svm(
        X_train_tfidf, X_test_tfidf, y_train, y_test, model_dir
    )
    
    # Compare
    comparison = compare_models({
        'Random Forest': rf_metrics,
        'SVM': svm_metrics,
    })
    
    return {
        'random_forest': {'model': rf_model, 'metrics': rf_metrics},
        'svm': {'model': svm_model, 'metrics': svm_metrics},
        'comparison': comparison,
    }


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_imdb.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_path):
        print(f"ERROR: Cleaned dataset not found at {data_path}")
        print("Run preprocessing.py first!")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    results = train_all_advanced(df, model_dir)
