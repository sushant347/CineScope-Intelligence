"""
ML Training Orchestrator
=========================
Run all training pipelines in sequence.
"""

import os
import sys
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_and_preprocess
from baseline_model import train_baseline
from advanced_models import train_all_advanced


def main():
    parser = argparse.ArgumentParser(description='Train ML Models for Sentiment Analysis')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['preprocess', 'baseline', 'advanced', 'lstm', 'bert', 'all'],
                        help='Which training phase to run')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Model output directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs (for deep learning)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (for deep learning)')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, args.data_dir)
    model_dir = os.path.join(base_dir, args.model_dir)
    
    raw_path = os.path.join(data_dir, 'IMDB Dataset.csv')
    clean_path = os.path.join(data_dir, 'cleaned_imdb.csv')
    
    # Phase 1: Preprocessing
    if args.phase in ('preprocess', 'all'):
        print("\n" + "🔄" * 30)
        print("PHASE 1: TEXT PREPROCESSING")
        print("🔄" * 30)
        
        if not os.path.exists(raw_path):
            print(f"ERROR: Dataset not found at {raw_path}")
            print("Download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
            sys.exit(1)
        
        df = load_and_preprocess(raw_path, clean_path)
    
    # Load cleaned data for subsequent phases
    if args.phase != 'preprocess':
        if os.path.exists(clean_path):
            df = pd.read_csv(clean_path)
        else:
            print(f"ERROR: Cleaned dataset not found at {clean_path}")
            print("Run with --phase preprocess first!")
            sys.exit(1)
    
    # Phase 2: Baseline Model
    if args.phase in ('baseline', 'all'):
        print("\n" + "🔄" * 30)
        print("PHASE 2: BASELINE MODEL (TF-IDF + LogReg)")
        print("🔄" * 30)
        train_baseline(df, model_dir)
    
    # Phase 3: Advanced Models
    if args.phase in ('advanced', 'all'):
        print("\n" + "🔄" * 30)
        print("PHASE 3: ADVANCED MODELS (RF + SVM)")
        print("🔄" * 30)
        train_all_advanced(df, model_dir)
    
    # Phase 4: LSTM
    if args.phase in ('lstm', 'all'):
        print("\n" + "🔄" * 30)
        print("PHASE 4: LSTM MODEL")
        print("🔄" * 30)
        from lstm_model import train_lstm
        train_lstm(df, model_dir, epochs=args.epochs, batch_size=args.batch_size * 8)
    
    # Phase 5: BERT
    if args.phase in ('bert', 'all'):
        print("\n" + "🔄" * 30)
        print("PHASE 5: BERT FINE-TUNING")
        print("🔄" * 30)
        from bert_model import train_bert
        train_bert(df, model_dir, epochs=3, batch_size=args.batch_size)
    
    print("\n" + "✅" * 30)
    print("TRAINING COMPLETE!")
    print("✅" * 30)


if __name__ == '__main__':
    main()
