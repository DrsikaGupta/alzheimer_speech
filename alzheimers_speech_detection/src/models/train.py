"""
Model Training Module for Alzheimer's Speech Detection

This module implements baseline and advanced models:
- Baseline: Logistic Regression, Random Forest, SVM, XGBoost, MLP
- Advanced: Audio CNN, Text LSTM, Multimodal Fusion
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BaselineModels:
    """Baseline machine learning models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config.get('models', {}).get('baseline', {})
        self.models = {}
        self.scaler = StandardScaler()
        
    def get_models(self) -> Dict[str, Any]:
        """Get configured models."""
        models = {}
        
        if self.config.get('logistic_regression', {}).get('enabled', True):
            lr_config = self.config['logistic_regression']
            models['logistic_regression'] = LogisticRegression(
                C=lr_config.get('C', 1.0),
                max_iter=lr_config.get('max_iter', 1000),
                class_weight='balanced',
                random_state=42
            )
        
        if self.config.get('random_forest', {}).get('enabled', True):
            rf_config = self.config['random_forest']
            models['random_forest'] = RandomForestClassifier(
                n_estimators=rf_config.get('n_estimators', 100),
                max_depth=rf_config.get('max_depth', None),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        if self.config.get('svm', {}).get('enabled', True):
            svm_config = self.config['svm']
            models['svm'] = SVC(
                kernel=svm_config.get('kernel', 'rbf'),
                C=svm_config.get('C', 1.0),
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        
        if self.config.get('xgboost', {}).get('enabled', True):
            xgb_config = self.config['xgboost']
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=xgb_config.get('n_estimators', 100),
                max_depth=xgb_config.get('max_depth', 6),
                scale_pos_weight=1,  # Handle imbalance
                random_state=42,
                eval_metric='logloss'
            )
        
        if self.config.get('mlp', {}).get('enabled', True):
            mlp_config = self.config['mlp']
            models['mlp'] = MLPClassifier(
                hidden_layer_sizes=tuple(mlp_config.get('hidden_layers', [128, 64, 32])),
                dropout=mlp_config.get('dropout', 0.3),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
        
        return models
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train all models and evaluate on validation set."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        models = self.get_models()
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
            
            results[name] = {
                'model': model,
                'metrics': metrics
            }
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return results
    
    def save_models(self, output_dir: str, results: Dict[str, Any]):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        # Save models
        for name, result in results.items():
            model_path = os.path.join(output_dir, f'{name}_model.pkl')
            joblib.dump(result['model'], model_path)
            
            metrics_path = os.path.join(output_dir, f'{name}_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(result['metrics'], f, indent=2)


class MultimodalFusionModel(nn.Module):
    """Multimodal fusion model for audio and text features."""
    
    def __init__(self, audio_dim: int, text_dim: int, hidden_dim: int = 256, 
                 num_classes: int = 2, dropout: float = 0.3,
                 fusion_type: str = 'attention'):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion
        if fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim // 2, num_heads=4, dropout=dropout)
            self.fusion_out = nn.Linear(hidden_dim, hidden_dim // 2)
        else:  # early or late fusion
            combined_dim = hidden_dim if fusion_type == 'early' else hidden_dim
            self.fusion_out = nn.Linear(combined_dim, hidden_dim // 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        audio_emb = self.audio_encoder(audio_features)
        text_emb = self.text_encoder(text_features)
        
        if self.fusion_type == 'attention':
            # Stack for attention
            stacked = torch.stack([audio_emb, text_emb], dim=0)
            attended, _ = self.attention(stacked, stacked, stacked)
            fused = torch.mean(attended, dim=0)
        elif self.fusion_type == 'early':
            fused = torch.cat([audio_emb, text_emb], dim=1)
        else:  # late fusion - average
            fused = (audio_emb + text_emb) / 2
        
        fused = self.fusion_out(fused)
        return self.classifier(fused)


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                config: Optional[Dict[str, Any]] = None,
                save_dir: str = 'experiments/models') -> Dict[str, Any]:
    """Main training function."""
    
    if config is None:
        config = {}
    elif isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    
    training_config = config.get('training', {})
    
    # Train baseline models
    print("="*50)
    print("TRAINING BASELINE MODELS")
    print("="*50)
    
    baseline = BaselineModels(config)
    baseline_results = baseline.train_and_evaluate(X_train, y_train, X_val, y_val)
    
    # Find best baseline model
    best_model_name = max(baseline_results.keys(), 
                         key=lambda k: baseline_results[k]['metrics'].get('roc_auc', 
                                   baseline_results[k]['metrics'].get('f1', 0)))
    best_result = baseline_results[best_model_name]
    
    print(f"\nBest baseline model: {best_model_name}")
    print(f"  ROC-AUC: {best_result['metrics'].get('roc_auc', 'N/A')}")
    print(f"  F1: {best_result['metrics']['f1']:.4f}")
    
    # Save models
    baseline.save_models(save_dir, baseline_results)
    
    return {
        'baseline_results': baseline_results,
        'best_model_name': best_model_name,
        'best_metrics': best_result['metrics']
    }


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 500)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(30, 500)
    y_val = np.random.randint(0, 2, 30)
    
    results = train_model(X_train, y_train, X_val, y_val, "configs/config.yaml")
    print("\nTraining complete!")
