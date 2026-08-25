"""
Evaluation Module for Alzheimer's Speech Detection

Comprehensive evaluation including:
- Standard metrics (accuracy, precision, recall, F1)
- ROC-AUC and PR-AUC
- Sensitivity and specificity
- Confusion matrices
- Calibration curves
- Cross-validation
- Ablation studies
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    confusion_matrix, classification_report, calibration_curve,
    brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


class Evaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config.get('evaluation', {})
        self.metrics_list = self.config.get('metrics', [
            'accuracy', 'precision', 'recall', 'f1', 
            'roc_auc', 'pr_auc', 'sensitivity', 'specificity'
        ])
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC-AUC
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # PR-AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Control', 'AD/MCI'],
                   yticklabels=['Control', 'AD/MCI'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None) -> Tuple[plt.Figure, float]:
        """Plot ROC curve and compute AUC."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic Curve')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, roc_auc
    
    def plot_pr_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                     save_path: Optional[str] = None) -> Tuple[plt.Figure, float]:
        """Plot Precision-Recall curve and compute AUC."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, pr_auc
    
    def plot_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                        save_path: Optional[str] = None) -> Tuple[plt.Figure, float]:
        """Plot calibration curve."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'Calibration Curve (Brier Score = {brier_score:.3f})')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, brier_score
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                      n_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        cv_results = {}
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric == 'roc_auc':
                scoring = 'roc_auc'
            elif metric == 'f1':
                scoring = 'f1'
            else:
                scoring = metric
            
            scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
            cv_results[metric] = scores.tolist()
        
        return cv_results
    
    def ablation_study(self, feature_sets: Dict[str, np.ndarray], y_train: np.ndarray,
                      y_val: np.ndarray, model_class, **model_kwargs) -> Dict[str, Any]:
        """Perform ablation study on different feature sets."""
        results = {}
        
        for name, X in feature_sets.items():
            # Split if needed
            if isinstance(X, dict):
                X_train, X_test = X['train'], X['val']
            else:
                # Assume sequential split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train_subset = y_train[:split_idx]
                y_val_subset = y_val if len(y_val) == len(X_test) else y_train[split_idx:]
            
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train_subset)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = self.compute_metrics(y_val_subset, y_pred, y_pred_proba)
            results[name] = metrics
        
        return results
    
    def full_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_pred_proba: np.ndarray, output_dir: str = 'experiments/results') -> Dict[str, Any]:
        """Run full evaluation and save results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_pred_proba)
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred, 
                                  os.path.join(output_dir, 'confusion_matrix.png'))
        self.plot_roc_curve(y_true, y_pred_proba,
                           os.path.join(output_dir, 'roc_curve.png'))
        self.plot_pr_curve(y_true, y_pred_proba,
                          os.path.join(output_dir, 'pr_curve.png'))
        self.plot_calibration(y_true, y_pred_proba,
                             os.path.join(output_dir, 'calibration_curve.png'))
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                      target_names=['Control', 'AD/MCI'],
                                      output_dict=True)
        with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        print("="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)
        
        return metrics


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    
    evaluator = Evaluator("configs/config.yaml")
    metrics = evaluator.full_evaluation(y_true, y_pred, y_pred_proba)
