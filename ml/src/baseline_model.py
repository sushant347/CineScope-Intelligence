"""
Baseline Model: TF-IDF + Logistic Regression
=============================================
Target accuracy: ~85-88%
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


def train_baseline(df, model_dir='models'):
    """
    Train TF-IDF + Logistic Regression baseline model.
    
    Args:
        df: DataFrame with 'clean_review' and 'label' columns
        model_dir: Directory to save model artifacts
    
    Returns:
        dict with model, vectorizer, and metrics
    """
    print("=" * 60)
    print("BASELINE MODEL: TF-IDF + Logistic Regression")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # TF-IDF Vectorization
    print("\nFitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    
    print(f"\n{'='*40}")
    print(f"RESULTS:")
    print(f"{'='*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Cross-validation
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    metrics['cv_accuracy'] = cv_scores.mean()
    
    # Save model and vectorizer
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'logistic_regression.pkl')
    tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(tfidf, tfidf_path)
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {tfidf_path}")
    
    return {
        'model': model,
        'vectorizer': tfidf,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }


def predict_sentiment(text, model_dir='models'):
    """
    Predict sentiment for a single review using saved baseline model.
    
    Args:
        text: Review text string
        model_dir: Directory with saved model artifacts
    
    Returns:
        dict with sentiment, confidence, and probabilities
    """
    model = joblib.load(os.path.join(model_dir, 'logistic_regression.pkl'))
    tfidf = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    
    # Preprocess (import here to avoid circular imports)
    from preprocessing import preprocess_text
    clean_text = preprocess_text(text)
    
    # Predict
    text_tfidf = tfidf.transform([clean_text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    sentiment = 'positive' if prediction == 1 else 'negative'
    confidence = float(max(probabilities))
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'positive_prob': float(probabilities[1]),
        'negative_prob': float(probabilities[0]),
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
    results = train_baseline(df, model_dir)
    
    # Test with a sample review
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    sample_reviews = [
        "This movie was absolutely amazing! Great acting and storyline.",
        "Terrible movie. Waste of time. The plot made no sense.",
        "It was okay, nothing special but not terrible either.",
    ]
    for review in sample_reviews:
        result = predict_sentiment(review, model_dir)
        print(f"\nReview: \"{review[:60]}...\"")
        print(f"  Sentiment: {result['sentiment']} ({result['confidence']:.2%})")
