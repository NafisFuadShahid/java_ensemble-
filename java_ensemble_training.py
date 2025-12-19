"""
NLBSE'26 Code Comment Classification - Java Ensemble Training Pipeline
=======================================================================
Complete training code for Java dataset using ensemble of:
- UniXcoder (best efficiency/performance ratio)
- CodeBERT (solid baseline)
- CodeT5+ (improved CodeT5)

Author: Generated for NLBSE'26 Competition
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class Config:
    """Training configuration"""
    # Paths
    train_path: str = "/mnt/user-data/uploads/java_train.parquet"
    test_path: str = "/mnt/user-data/uploads/java_test.parquet"
    output_dir: str = "./models"
    
    # Java labels (7 categories)
    labels: List[str] = None
    num_labels: int = 7
    
    # Training
    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Ensemble
    use_weighted_ensemble: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = torch.cuda.is_available()
    
    def __post_init__(self):
        self.labels = ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational']
        os.makedirs(self.output_dir, exist_ok=True)


# Model configurations
MODEL_CONFIGS = {
    "unixcoder": {
        "name": "microsoft/unixcoder-base",
        "type": "encoder",
        "batch_size": 16,
        "learning_rate": 2e-5,
    },
    "codebert": {
        "name": "microsoft/codebert-base",
        "type": "encoder",
        "batch_size": 16,
        "learning_rate": 2e-5,
    },
    "codet5plus": {
        "name": "Salesforce/codet5p-220m",
        "type": "encoder-decoder",
        "batch_size": 8,
        "learning_rate": 1e-5,
    },
}


# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

class JavaCommentDataset(Dataset):
    """Dataset class for Java code comments"""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


def load_java_data(config: Config) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load Java training and test data"""
    print("Loading Java dataset...")
    
    train_df = pd.read_parquet(config.train_path)
    print(f"  Train samples: {len(train_df)}")
    
    test_df = None
    if os.path.exists(config.test_path):
        test_df = pd.read_parquet(config.test_path)
        print(f"  Test samples: {len(test_df)}")
    else:
        print(f"  Test file not found at {config.test_path}")
    
    return train_df, test_df


def prepare_data(df: pd.DataFrame, config: Config) -> Tuple[List[str], np.ndarray]:
    """Prepare texts and labels from dataframe"""
    
    # Create input text: "comment_sentence | class_name"
    texts = (df['comment_sentence'] + " | " + df['class']).tolist()
    
    # Convert labels from list strings to numpy array
    labels = np.array(df['labels'].tolist())
    
    return texts, labels


def create_train_val_split(texts: List[str], labels: np.ndarray, val_ratio: float = 0.15, seed: int = 42):
    """Split data into training and validation sets"""
    return train_test_split(
        texts, labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=None  # Multi-label doesn't support stratify directly
    )


# ============================================
# MODEL ARCHITECTURE
# ============================================

class EncoderClassifier(nn.Module):
    """Classification head for encoder-based models (CodeBERT, UniXcoder)"""
    
    def __init__(self, model_name: str, num_labels: int = 7, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}


class EncoderDecoderClassifier(nn.Module):
    """Classification head for encoder-decoder models (CodeT5+)"""
    
    def __init__(self, model_name: str, num_labels: int = 7, dropout: float = 0.1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Get hidden size from encoder
        if hasattr(self.model.config, 'd_model'):
            hidden_size = self.model.config.d_model
        else:
            hidden_size = self.model.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, labels=None):
        # For encoder-decoder, we only use the encoder part
        if hasattr(self.model, 'encoder'):
            outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling over sequence
        hidden_states = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / sum_mask
        
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}


def create_model(model_key: str, num_labels: int = 7) -> nn.Module:
    """Factory function to create model based on type"""
    config = MODEL_CONFIGS[model_key]
    model_name = config['name']
    model_type = config['type']
    
    if model_type == 'encoder':
        return EncoderClassifier(model_name, num_labels)
    elif model_type == 'encoder-decoder':
        return EncoderDecoderClassifier(model_name, num_labels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================
# TRAINING
# ============================================

class MultiLabelTrainer(Trainer):
    """Custom trainer for multi-label classification"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute F1 scores for evaluation"""
    predictions, labels = eval_pred
    
    # Apply sigmoid and threshold
    probs = torch.sigmoid(torch.tensor(predictions)).numpy()
    preds = (probs > 0.5).astype(int)
    
    # Compute metrics
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_samples = f1_score(labels, preds, average='samples', zero_division=0)
    
    # Per-class F1
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    
    metrics = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_samples': f1_samples,
    }
    
    # Add per-class metrics
    label_names = ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational']
    for i, name in enumerate(label_names):
        metrics[f'f1_{name}'] = f1_per_class[i]
    
    return metrics


def train_single_model(
    model_key: str,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    config: Config
) -> Tuple[nn.Module, AutoTokenizer, Dict]:
    """Train a single model"""
    
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config['name']
    
    print(f"\n{'='*60}")
    print(f"Training: {model_key.upper()}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = JavaCommentDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = JavaCommentDataset(val_texts, val_labels, tokenizer, config.max_length)
    
    # Create model
    model = create_model(model_key, config.num_labels)
    
    # Training arguments
    output_dir = os.path.join(config.output_dir, model_key)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=model_config.get('learning_rate', config.learning_rate),
        per_device_train_batch_size=model_config.get('batch_size', config.train_batch_size),
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=config.fp16,
        logging_steps=50,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=2,
        gradient_accumulation_steps=2 if model_config.get('batch_size', 16) < 16 else 1,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print(f"\nStarting training...")
    trainer.train()
    
    # Save best model
    best_model_path = os.path.join(output_dir, "best")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\n{model_key.upper()} Results:")
    for key, value in eval_results.items():
        if 'f1' in key:
            print(f"  {key}: {value:.4f}")
    
    return model, tokenizer, eval_results


# ============================================
# ENSEMBLE
# ============================================

class JavaEnsemble:
    """Ensemble classifier for Java code comments"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models: Dict[str, nn.Module] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.weights: Dict[str, float] = {}
        self.thresholds: np.ndarray = np.full(config.num_labels, 0.5)
        self.device = torch.device(config.device)
    
    def add_model(self, model_key: str, model: nn.Module, tokenizer: AutoTokenizer, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[model_key] = model.to(self.device)
        self.models[model_key].eval()
        self.tokenizers[model_key] = tokenizer
        self.weights[model_key] = weight
    
    def load_model(self, model_key: str, model_path: str, weight: float = 1.0):
        """Load a trained model from path"""
        model_config = MODEL_CONFIGS[model_key]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create and load model
        model = create_model(model_key, self.config.num_labels)
        
        # Load state dict
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            # Try loading with safetensors
            from safetensors.torch import load_file
            safetensor_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensor_path):
                state_dict = load_file(safetensor_path)
                model.load_state_dict(state_dict)
        
        self.add_model(model_key, model, tokenizer, weight)
        print(f"Loaded {model_key} from {model_path}")
    
    def _get_model_predictions(self, model_key: str, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get predictions from a single model"""
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        
        all_logits = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs['logits']
                all_logits.append(logits.cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        probs = torch.sigmoid(all_logits).numpy()
        
        return probs
    
    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get ensemble probability predictions"""
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        normalized_weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Get weighted average of predictions
        ensemble_probs = np.zeros((len(texts), self.config.num_labels))
        
        for model_key in self.models:
            model_probs = self._get_model_predictions(model_key, texts, batch_size)
            ensemble_probs += model_probs * normalized_weights[model_key]
        
        return ensemble_probs
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get binary predictions using optimized thresholds"""
        probs = self.predict_proba(texts, batch_size)
        return (probs > self.thresholds).astype(int)
    
    def optimize_thresholds(self, val_texts: List[str], val_labels: np.ndarray, batch_size: int = 32):
        """Optimize per-class thresholds on validation data"""
        print("\nOptimizing thresholds...")
        
        probs = self.predict_proba(val_texts, batch_size)
        
        optimal_thresholds = []
        for i in range(self.config.num_labels):
            label_name = self.config.labels[i]
            
            def neg_f1(threshold):
                preds = (probs[:, i] > threshold).astype(int)
                return -f1_score(val_labels[:, i], preds, zero_division=0)
            
            result = minimize_scalar(neg_f1, bounds=(0.1, 0.9), method='bounded')
            optimal_t = result.x
            optimal_f1 = -result.fun
            
            optimal_thresholds.append(optimal_t)
            print(f"  {label_name}: threshold={optimal_t:.3f}, F1={optimal_f1:.4f}")
        
        self.thresholds = np.array(optimal_thresholds)
        return self.thresholds
    
    def evaluate(self, texts: List[str], labels: np.ndarray, batch_size: int = 32) -> Dict:
        """Evaluate ensemble performance"""
        preds = self.predict(texts, batch_size)
        probs = self.predict_proba(texts, batch_size)
        
        # Overall metrics
        results = {
            'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
            'precision_micro': precision_score(labels, preds, average='micro', zero_division=0),
            'recall_micro': recall_score(labels, preds, average='micro', zero_division=0),
        }
        
        # Per-class metrics
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        for i, name in enumerate(self.config.labels):
            results[f'f1_{name}'] = f1_per_class[i]
        
        return results
    
    def save(self, path: str):
        """Save ensemble configuration"""
        os.makedirs(path, exist_ok=True)
        
        config = {
            'weights': self.weights,
            'thresholds': self.thresholds.tolist(),
            'model_keys': list(self.models.keys()),
            'labels': self.config.labels,
        }
        
        with open(os.path.join(path, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved ensemble config to {path}")


# ============================================
# INFERENCE ON TEST DATA
# ============================================

def generate_predictions(ensemble: JavaEnsemble, test_df: pd.DataFrame, config: Config, output_path: str):
    """Generate predictions for test data"""
    
    test_texts, test_labels = prepare_data(test_df, config)
    
    # Get predictions
    predictions = ensemble.predict(test_texts)
    probabilities = ensemble.predict_proba(test_texts)
    
    # Create output dataframe
    output_df = test_df.copy()
    
    # Add predictions
    output_df['predicted_labels'] = predictions.tolist()
    output_df['prediction_probs'] = probabilities.tolist()
    
    # Add individual label columns
    for i, label_name in enumerate(config.labels):
        output_df[f'pred_{label_name}'] = predictions[:, i]
        output_df[f'prob_{label_name}'] = probabilities[:, i]
    
    # Save
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # If test has labels, evaluate
    if 'labels' in test_df.columns:
        results = ensemble.evaluate(test_texts, test_labels)
        print("\nTest Set Results:")
        print(f"  F1 Micro: {results['f1_micro']:.4f}")
        print(f"  F1 Macro: {results['f1_macro']:.4f}")
        print(f"  F1 Weighted: {results['f1_weighted']:.4f}")
        print("\nPer-class F1:")
        for label in config.labels:
            print(f"  {label}: {results[f'f1_{label}']:.4f}")
    
    return output_df


# ============================================
# MAIN PIPELINE
# ============================================

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("NLBSE'26 Java Code Comment Classification")
    print("Ensemble Training Pipeline")
    print("="*60)
    
    # Initialize config
    config = Config()
    print(f"\nDevice: {config.device}")
    print(f"FP16: {config.fp16}")
    print(f"Labels: {config.labels}")
    
    # Load data
    train_df, test_df = load_java_data(config)
    
    # Prepare data
    texts, labels = prepare_data(train_df, config)
    print(f"\nTotal samples: {len(texts)}")
    print(f"Label distribution:")
    for i, name in enumerate(config.labels):
        count = labels[:, i].sum()
        print(f"  {name}: {count} ({100*count/len(labels):.1f}%)")
    
    # Split into train/validation
    train_texts, val_texts, train_labels, val_labels = create_train_val_split(
        texts, labels, val_ratio=0.15, seed=42
    )
    print(f"\nTrain samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Models to train
    models_to_train = ["unixcoder", "codebert", "codet5plus"]
    
    # Train individual models
    trained_models = {}
    
    for model_key in models_to_train:
        model, tokenizer, results = train_single_model(
            model_key=model_key,
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            config=config
        )
        
        trained_models[model_key] = {
            'model': model,
            'tokenizer': tokenizer,
            'f1_macro': results.get('eval_f1_macro', 0.5)
        }
    
    # Calculate dynamic weights based on validation F1
    total_f1 = sum(m['f1_macro'] for m in trained_models.values())
    weights = {k: v['f1_macro'] / total_f1 for k, v in trained_models.items()}
    
    print("\n" + "="*60)
    print("ENSEMBLE CONFIGURATION")
    print("="*60)
    print("Dynamic weights based on validation F1:")
    for k, w in weights.items():
        print(f"  {k}: {w:.3f}")
    
    # Create ensemble
    ensemble = JavaEnsemble(config)
    for model_key, data in trained_models.items():
        ensemble.add_model(
            model_key=model_key,
            model=data['model'],
            tokenizer=data['tokenizer'],
            weight=weights[model_key]
        )
    
    # Optimize thresholds
    ensemble.optimize_thresholds(val_texts, val_labels)
    
    # Evaluate ensemble on validation
    print("\n" + "="*60)
    print("ENSEMBLE VALIDATION RESULTS")
    print("="*60)
    
    val_results = ensemble.evaluate(val_texts, val_labels)
    print(f"F1 Micro: {val_results['f1_micro']:.4f}")
    print(f"F1 Macro: {val_results['f1_macro']:.4f}")
    print(f"F1 Weighted: {val_results['f1_weighted']:.4f}")
    print("\nPer-class F1:")
    for label in config.labels:
        print(f"  {label}: {val_results[f'f1_{label}']:.4f}")
    
    # Save ensemble config
    ensemble.save(os.path.join(config.output_dir, "ensemble"))
    
    # Generate test predictions if test data exists
    if test_df is not None:
        print("\n" + "="*60)
        print("GENERATING TEST PREDICTIONS")
        print("="*60)
        
        output_path = os.path.join(config.output_dir, "test_predictions.csv")
        generate_predictions(ensemble, test_df, config, output_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Models saved to: {config.output_dir}")
    print(f"Ensemble F1 Macro: {val_results['f1_macro']:.4f}")
    
    return ensemble, val_results


if __name__ == "__main__":
    main()
