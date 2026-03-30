"""
BERT Fine-Tuning for Sentiment Analysis
========================================
Using HuggingFace transformers with bert-base-uncased
Target accuracy: ~92-95%
Optimized for RTX 3050 (small batch sizes)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


class IMDBDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len=256):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_bert(df, model_dir='models', epochs=3, batch_size=8, lr=2e-5, max_len=256):
    """
    Fine-tune BERT for sentiment classification.
    
    Optimized for RTX 3050:
    - batch_size=8 (fits in 4GB VRAM)
    - max_len=256 (shorter sequences)
    - gradient accumulation for effective larger batch
    """
    print("=" * 60)
    print("BERT FINE-TUNING (bert-base-uncased)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    working_df = df
    accumulation_steps = 4

    # Keep the same CLI command usable on CPU-only machines.
    if device.type != 'cuda':
        cpu_sample_size = min(2000, len(df))
        if len(df) > cpu_sample_size:
            working_df, _ = train_test_split(
                df,
                train_size=cpu_sample_size,
                random_state=42,
                stratify=df['label']
            )
            working_df = working_df.reset_index(drop=True)

        epochs = min(epochs, 1)
        batch_size = max(batch_size, 16)
        max_len = min(max_len, 128)
        accumulation_steps = 1

        print("CPU mode enabled for practical runtime:")
        print(f"  Sample size: {len(working_df)}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max sequence length: {max_len}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        working_df['review'], working_df['label'],   # Use raw review for BERT (it has its own tokenizer)
        test_size=0.2, random_state=42, stratify=working_df['label']
    )
    
    # Tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Datasets
    train_dataset = IMDBDataset(X_train, y_train, tokenizer, max_len)
    test_dataset = IMDBDataset(X_test, y_test, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Effective batch size: {batch_size * accumulation_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Total steps: {total_steps}")
    
    # Training loop
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{total_loss/(step+1):.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        # Evaluation
        model.eval()
        test_preds = []
        test_labels_list = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                test_preds.extend(preds)
                test_labels_list.extend(batch['label'].numpy())
        
        test_acc = accuracy_score(test_labels_list, test_preds)
        print(f"  Epoch {epoch+1}: Test Accuracy = {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            bert_dir = os.path.join(model_dir, 'bert_sentiment')
            os.makedirs(bert_dir, exist_ok=True)
            model.save_pretrained(bert_dir)
            tokenizer.save_pretrained(bert_dir)
            print(f"  ✓ Best model saved to {bert_dir}")
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Final report
    print(f"\n{'='*40}")
    print(f"BEST TEST ACCURACY: {best_acc:.4f}")
    print(f"{'='*40}")
    print(classification_report(test_labels_list, test_preds, target_names=['Negative', 'Positive']))
    
    return model, tokenizer, best_acc


def predict_with_bert(text, model_dir='models/bert_sentiment', max_len=256):
    """Predict sentiment using fine-tuned BERT."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=max_len,
        padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    sentiment = 'positive' if probs[1] > probs[0] else 'negative'
    confidence = float(max(probs))
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'positive_prob': float(probs[1]),
        'negative_prob': float(probs[0]),
    }


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_imdb.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_path):
        print(f"ERROR: Cleaned dataset not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    train_bert(df, model_dir)
