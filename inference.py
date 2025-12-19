"""
NLBSE'26 Inference Script
=========================
Load trained ensemble and generate predictions on test data.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import argparse

# ============================================
# CONFIG
# ============================================

JAVA_LABELS = ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational']
NUM_LABELS = 7
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    "unixcoder": "microsoft/unixcoder-base",
    "codebert": "microsoft/codebert-base",
    "codet5plus": "Salesforce/codet5p-220m",
}


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
# ENSEMBLE INFERENCE
# ============================================

class EnsembleInference:
    """Load and run inference with trained ensemble"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = models_dir
        self.models: Dict[str, nn.Module] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.weights: Dict[str, float] = {}
        self.thresholds: np.ndarray = np.full(NUM_LABELS, 0.5)
        self.device = torch.device(DEVICE)
        
    def load_ensemble(self, config_path: str = None):
        """Load ensemble from saved config"""
        
        # Load config if exists
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.weights = config.get('weights', {})
            self.thresholds = np.array(config.get('thresholds', [0.5] * NUM_LABELS))
            model_keys = config.get('model_keys', [])
        else:
            # Auto-detect models
            model_keys = []
            for key in MODEL_CONFIGS:
                model_path = os.path.join(self.models_dir, key, "best")
                if os.path.exists(model_path):
                    model_keys.append(key)
            self.weights = {k: 1.0/len(model_keys) for k in model_keys}
        
        # Load each model
        for key in model_keys:
            self.load_model(key)
        
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        print(f"Weights: {self.weights}")
        print(f"Thresholds: {self.thresholds}")
    
    def load_model(self, model_key: str):
        """Load a single model"""
        model_path = os.path.join(self.models_dir, model_key, "best")
        base_model_name = MODEL_CONFIGS[model_key]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create model architecture
        model = CodeCommentClassifier(base_model_name, NUM_LABELS)
        
        # Load weights
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            # Try safetensors
            from safetensors.torch import load_file
            safetensor_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensor_path):
                state_dict = load_file(safetensor_path)
                model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        
        self.models[model_key] = model
        self.tokenizers[model_key] = tokenizer
        
        if model_key not in self.weights:
            self.weights[model_key] = 1.0
    
    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get probability predictions from ensemble"""
        
        # Normalize weights
        total = sum(self.weights.values())
        norm_weights = {k: v/total for k, v in self.weights.items()}
        
        ensemble_probs = np.zeros((len(texts), NUM_LABELS))
        
        for model_key, model in self.models.items():
            tokenizer = self.tokenizers[model_key]
            weight = norm_weights[model_key]
            
            model_probs = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.sigmoid(outputs['logits']).cpu().numpy()
                    model_probs.append(probs)
            
            model_probs = np.vstack(model_probs)
            ensemble_probs += model_probs * weight
        
        return ensemble_probs
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get binary predictions"""
        probs = self.predict_proba(texts, batch_size)
        return (probs > self.thresholds).astype(int)
    
    def predict_with_labels(self, texts: List[str], batch_size: int = 32) -> List[List[str]]:
        """Get predictions as label names"""
        preds = self.predict(texts, batch_size)
        
        results = []
        for pred in preds:
            labels = [JAVA_LABELS[i] for i in range(NUM_LABELS) if pred[i] == 1]
            results.append(labels)
        
        return results


def generate_submission(
    ensemble: EnsembleInference,
    test_df: pd.DataFrame,
    output_path: str
):
    """Generate submission file"""
    
    # Prepare texts
    texts = (test_df['comment_sentence'] + " | " + test_df['class']).tolist()
    
    # Get predictions
    predictions = ensemble.predict(texts)
    probabilities = ensemble.predict_proba(texts)
    
    # Create submission dataframe
    submission = test_df.copy()
    
    # Add predictions as list (matching original format)
    submission['predicted_labels'] = predictions.tolist()
    
    # Also add individual columns for each label
    for i, label in enumerate(JAVA_LABELS):
        submission[f'pred_{label}'] = predictions[:, i]
        submission[f'prob_{label}'] = probabilities[:, i].round(4)
    
    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"  Total samples: {len(texts)}")
    for i, label in enumerate(JAVA_LABELS):
        count = predictions[:, i].sum()
        print(f"  {label}: {count} ({100*count/len(texts):.1f}%)")
    
    return submission


def main():
    parser = argparse.ArgumentParser(description='NLBSE Java Inference')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test parquet file')
    parser.add_argument('--models_dir', type=str, default='./models', help='Directory with trained models')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    
    args = parser.parse_args()
    
    print("="*50)
    print("NLBSE'26 Java Inference")
    print("="*50)
    
    # Load ensemble
    ensemble = EnsembleInference(args.models_dir)
    ensemble_config = os.path.join(args.models_dir, "ensemble", "ensemble_config.json")
    ensemble.load_ensemble(ensemble_config if os.path.exists(ensemble_config) else None)
    
    # Load test data
    print(f"\nLoading test data from: {args.test_path}")
    test_df = pd.read_parquet(args.test_path)
    print(f"Test samples: {len(test_df)}")
    
    # Generate submission
    generate_submission(ensemble, test_df, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
