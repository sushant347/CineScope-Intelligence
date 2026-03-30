"""
Text Preprocessing Pipeline for IMDB Movie Reviews
====================================================
- HTML tag removal
- Lowercasing
- Special character removal
- Stopword removal
- Lemmatization
"""

import os
import re
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'were',
}

NLTK_RESOURCES = [
    ('tokenizers/punkt', 'punkt'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
]


def _ensure_nltk_resource(resource_path, resource_name):
    """Ensure an NLTK resource exists locally; attempt download if missing."""
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        try:
            return bool(nltk.download(resource_name, quiet=True))
        except Exception as exc:
            logger.warning("Could not download NLTK resource '%s': %s", resource_name, exc)
            return False


def _init_nlp_tools():
    resources_ok = all(
        _ensure_nltk_resource(path, name)
        for path, name in NLTK_RESOURCES
    )

    # Optional resources for newer NLTK versions; non-fatal if unavailable.
    for optional in ('punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'):
        try:
            nltk.download(optional, quiet=True)
        except Exception:
            pass

    if resources_ok:
        try:
            return set(stopwords.words('english')), WordNetLemmatizer(), True
        except LookupError:
            logger.warning("NLTK stopwords unavailable; using fallback stopword set.")

    logger.warning("Some NLTK resources are unavailable; preprocessing will use safe fallbacks.")
    return set(DEFAULT_STOP_WORDS), WordNetLemmatizer(), False


stop_words, lemmatizer, nltk_ready = _init_nlp_tools()


def _safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


def _safe_lemmatize(token):
    if not nltk_ready:
        return token
    try:
        return lemmatizer.lemmatize(token)
    except LookupError:
        return token


def remove_html_tags(text):
    """Remove HTML tags from text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_special_characters(text):
    """Remove special characters, keep only alphanumeric and spaces."""
    return re.sub(r'[^a-zA-Z\s]', '', text)


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r'http\S+|www\S+', '', text)


def preprocess_text(text):
    """Full preprocessing pipeline for a single review."""
    # Remove HTML tags
    text = remove_html_tags(text)
    # Remove URLs
    text = remove_urls(text)
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = remove_special_characters(text)
    # Tokenize
    tokens = _safe_word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [
        _safe_lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return ' '.join(tokens)


def load_and_preprocess(input_path, output_path=None):
    """
    Load IMDB dataset and preprocess all reviews.
    
    Args:
        input_path: Path to raw IMDB Dataset.csv
        output_path: Path to save cleaned dataset (optional)
    
    Returns:
        DataFrame with cleaned reviews
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nClass distribution:")
    print(df['sentiment'].value_counts())
    
    # Convert sentiment to binary
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    
    # Preprocess reviews
    print("\nPreprocessing reviews...")
    tqdm.pandas(desc="Cleaning reviews")
    df['clean_review'] = df['review'].progress_apply(preprocess_text)
    
    # Add review length features
    df['review_length'] = df['review'].apply(len)
    df['clean_review_length'] = df['clean_review'].apply(len)
    df['word_count'] = df['clean_review'].apply(lambda x: len(x.split()))
    
    # Remove empty reviews after cleaning
    df = df[df['clean_review'].str.strip().str.len() > 0]
    
    print(f"\nAfter cleaning: {df.shape[0]} reviews")
    print(f"Average review length: {df['word_count'].mean():.0f} words")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned dataset to {output_path}")
    
    return df


def get_dataset_stats(df):
    """Generate basic dataset statistics."""
    stats = {
        'total_reviews': len(df),
        'positive_count': (df['label'] == 1).sum(),
        'negative_count': (df['label'] == 0).sum(),
        'avg_review_length': df['review_length'].mean(),
        'avg_clean_length': df['clean_review_length'].mean(),
        'avg_word_count': df['word_count'].mean(),
        'max_word_count': df['word_count'].max(),
        'min_word_count': df['word_count'].min(),
    }
    return stats


if __name__ == '__main__':
    import sys
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data', 'IMDB Dataset.csv')
    output_file = os.path.join(base_dir, 'data', 'cleaned_imdb.csv')
    
    if not os.path.exists(input_file):
        print(f"ERROR: Dataset not found at {input_file}")
        print("Please download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print(f"And place it at: {input_file}")
        sys.exit(1)
    
    df = load_and_preprocess(input_file, output_file)
    stats = get_dataset_stats(df)
    
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
