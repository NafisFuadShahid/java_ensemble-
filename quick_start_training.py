"""
NLBSE'26 Quick Start - Train Single Model or Fast Ensemble
===========================================================
Use this for quick experimentation before full ensemble training.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ============================================
# CONFIG
# ============================================

JAVA_LABELS = ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational']
NUM_LABELS = 7
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ============================================
# DATA
# ============================================

class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_data(train_path: str, test_path: str = None):
    """Load and prepare data"""
    train_df = pd.read_parquet(train_path)
    
    texts = (train_df['comment_sentence'] + " | " + train_df['class']).tolist()
    labels = np.array(train_df['labels'].tolist())
    
    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42
    )
    
    test_texts, test_labels = None, None
    if test_path and os.path.exists(test_path):
        test_df = pd.read_parquet(test_path)
        test_texts = (test_df['comment_sentence'] + " | " + test_df['class']).tolist()
        test_labels = np.array(test_df['labels'].tolist())
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    if test_texts:
        print(f"Test: {len(test_texts)}")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# ============================================
# MODEL
# ============================================

class CodeCommentClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 7):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden = self.encoder.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(out.last_hidden_state[:, 0, :])
        
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        
        return {"loss": loss, "logits": logits}


# ============================================
# TRAINING
# ============================================

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(preds)).numpy()
    preds = (probs > 0.5).astype(int)
    
    return {
        'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
    }


def train_model(
    model_name: str,
    train_texts, train_labels,
    val_texts, val_labels,
    output_dir: str = "./model",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5
):
    """Train a single model"""
    
    print(f"\nTraining {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_ds = CommentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = CommentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    model = CodeCommentClassifier(model_name, NUM_LABELS)
    
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        save_total_limit=1,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    
    trainer.train()
    
    # Save
    trainer.save_model(os.path.join(output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best"))
    
    results = trainer.evaluate()
    print(f"Results: F1 Macro = {results['eval_f1_macro']:.4f}")
    
    return model, tokenizer, results


# ============================================
# QUICK ENSEMBLE (2 models)
# ============================================

def quick_ensemble_predict(models, tokenizers, texts, weights=None):
    """Simple ensemble prediction"""
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    all_probs = []
    
    for model, tokenizer, weight in zip(models, tokenizers, weights):
        model.eval()
        model.to(DEVICE)
        
        probs = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, 
                             max_length=MAX_LENGTH, return_tensors='pt').to(DEVICE)
            
            with torch.no_grad():
                out = model(**inputs)
                prob = torch.sigmoid(out['logits']).cpu().numpy()
                probs.append(prob)
        
        probs = np.vstack(probs) * weight
        all_probs.append(probs)
    
    return np.sum(all_probs, axis=0)


def optimize_thresholds(probs, labels):
    """Find best threshold per class"""
    thresholds = []
    
    for i in range(NUM_LABELS):
        best_t, best_f1 = 0.5, 0
        for t in np.linspace(0.2, 0.8, 30):
            preds = (probs[:, i] > t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
        print(f"  {JAVA_LABELS[i]}: t={best_t:.2f}, F1={best_f1:.4f}")
    
    return np.array(thresholds)


# ============================================
# MAIN
# ============================================

def main():
    # Paths
    train_path = "/mnt/user-data/uploads/java_train.parquet"
    test_path = "/mnt/user-data/uploads/java_test.parquet"
    
    # Load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data(
        train_path, test_path
    )
    
    # Choose models to train (pick 1 or 2 for quick results)
    # Options: "microsoft/unixcoder-base", "microsoft/codebert-base", "Salesforce/codet5p-220m"
    
    models_config = [
        ("microsoft/unixcoder-base", "./models/unixcoder", 16, 2e-5),
        ("microsoft/codebert-base", "./models/codebert", 16, 2e-5),
    ]
    
    trained = []
    for model_name, output_dir, batch_size, lr in models_config:
        model, tokenizer, results = train_model(
            model_name, train_texts, train_labels, val_texts, val_labels,
            output_dir=output_dir, epochs=5, batch_size=batch_size, lr=lr
        )
        trained.append((model, tokenizer, results['eval_f1_macro']))
    
    # Create ensemble with weights based on F1
    models = [t[0] for t in trained]
    tokenizers = [t[1] for t in trained]
    f1_scores = [t[2] for t in trained]
    
    total_f1 = sum(f1_scores)
    weights = [f / total_f1 for f in f1_scores]
    print(f"\nEnsemble weights: {weights}")
    
    # Get ensemble predictions
    print("\nOptimizing thresholds...")
    val_probs = quick_ensemble_predict(models, tokenizers, val_texts, weights)
    thresholds = optimize_thresholds(val_probs, val_labels)
    
    # Final evaluation
    val_preds = (val_probs > thresholds).astype(int)
    final_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"FINAL ENSEMBLE F1 MACRO: {final_f1:.4f}")
    print(f"{'='*50}")
    
    # Test predictions
    if test_texts is not None:
        print("\nGenerating test predictions...")
        test_probs = quick_ensemble_predict(models, tokenizers, test_texts, weights)
        test_preds = (test_probs > thresholds).astype(int)
        
        if test_labels is not None:
            test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
            print(f"Test F1 Macro: {test_f1:.4f}")
        
        # Save predictions
        np.save("test_predictions.npy", test_preds)
        np.save("test_probabilities.npy", test_probs)
        print("Saved: test_predictions.npy, test_probabilities.npy")


if __name__ == "__main__":
    main()
