"""
LSTM Deep Learning Model for Sentiment Analysis
=================================================
PyTorch LSTM with embedding layer
Target accuracy: ~88-90%
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from tqdm import tqdm


# ── Dataset ──────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        # Tokenize and encode
        tokens = text.split()[:self.max_len]
        encoded = [self.vocab.get(t, self.vocab.get('<UNK>', 1)) for t in tokens]
        
        # Pad
        if len(encoded) < self.max_len:
            encoded += [0] * (self.max_len - len(encoded))
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float)


# ── LSTM Model ───────────────────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        return self.fc(context).squeeze(-1)


def build_vocab(texts, max_vocab=25000):
    """Build vocabulary from text corpus."""
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    
    # Most common words + special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    
    return vocab


def train_lstm(df, model_dir='models', epochs=5, batch_size=64, lr=0.001):
    """
    Train LSTM model on the IMDB dataset.
    
    Args:
        df: DataFrame with 'clean_review' and 'label' columns
        model_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    print("=" * 60)
    print("LSTM SENTIMENT MODEL")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(X_train)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = ReviewDataset(X_train, y_train, vocab)
    test_dataset = ReviewDataset(X_test, y_test, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model
    model = SentimentLSTM(len(vocab)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for texts, labels in pbar:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        # Evaluation
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for texts, labels in test_loader:
                texts = texts.to(device)
                outputs = model(texts)
                predicted = (outputs > 0.5).float().cpu()
                test_preds.extend(predicted.numpy())
                test_labels.extend(labels.numpy())
        
        test_acc = accuracy_score(test_labels, test_preds)
        scheduler.step(1 - test_acc)
        
        print(f"  Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, "
              f"Train Acc={correct/total:.4f}, Test Acc={test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'config': {
                    'vocab_size': len(vocab),
                    'embed_dim': 128,
                    'hidden_dim': 256,
                    'num_layers': 2,
                    'dropout': 0.3,
                }
            }, os.path.join(model_dir, 'lstm_model.pt'))
            print(f"  ✓ Best model saved (acc: {best_acc:.4f})")
    
    # Final evaluation
    print(f"\n{'='*40}")
    print(f"BEST TEST ACCURACY: {best_acc:.4f}")
    print(f"{'='*40}")
    print(classification_report(test_labels, test_preds, target_names=['Negative', 'Positive']))
    
    return model, vocab, best_acc


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_imdb.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_path):
        print(f"ERROR: Cleaned dataset not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    train_lstm(df, model_dir)
