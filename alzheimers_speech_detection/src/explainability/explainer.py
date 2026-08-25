"""
Explainability Module for Alzheimer's Speech Detection

This module provides model interpretability using:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Permutation Importance
- Feature Importance
- Attention Visualization
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


class ModelExplainer:
    """Comprehensive model explainability."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config.get('explainability', {})
        self.feature_names = None
    
    def set_feature_names(self, feature_names: List[str]):
        """Set feature names for interpretation."""
        self.feature_names = feature_names
    
    def compute_permutation_importance(self, model, X: np.ndarray, y: np.ndarray,
                                       n_repeats: int = 10, random_state: int = 42) -> Dict[str, np.ndarray]:
        """Compute permutation importance."""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X, y, 
            n_repeats=n_repeats, 
            random_state=random_state,
            n_jobs=-1
        )
        
        importance = {
            'mean': result.importances_mean,
            'std': result.importances_std,
            'importances': result.importances
        }
        
        return importance
    
    def plot_permutation_importance(self, importance: Dict[str, np.ndarray],
                                   top_n: int = 20, save_path: Optional[str] = None) -> plt.Figure:
        """Plot permutation importance."""
        mean_imp = importance['mean']
        std_imp = importance['std']
        
        # Get top features
        if self.feature_names is not None:
            indices = np.argsort(np.abs(mean_imp))[-top_n:][::-1]
            feature_labels = [self.feature_names[i] for i in indices]
        else:
            indices = np.argsort(np.abs(mean_imp))[-top_n:][::-1]
            feature_labels = [f'Feature {i}' for i in indices]
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        ax.barh(range(top_n), mean_imp[indices], xerr=std_imp[indices])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Top Feature Importances (Permutation)')
        ax.invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compute_shap_values(self, model, X_sample: np.ndarray, 
                           background: Optional[np.ndarray] = None,
                           max_samples: int = 100) -> np.ndarray:
        """Compute SHAP values."""
        try:
            import shap
            
            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.KernelExplainer(model.predict_proba, background or X_sample[:max_samples])
                shap_values = explainer.shap_values(X_sample[:max_samples])
                
                # For binary classification, get positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                explainer = shap.KernelExplainer(model.predict, background or X_sample[:max_samples])
                shap_values = explainer.shap_values(X_sample[:max_samples])
            
            return shap_values
            
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return np.array([])
    
    def plot_shap_summary(self, shap_values: np.ndarray, X: np.ndarray,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot SHAP summary."""
        try:
            import shap
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"Error plotting SHAP summary: {e}")
            return None
    
    def compute_lime_explanation(self, model, X_sample: np.ndarray, y_sample: np.ndarray,
                                instance_idx: int = 0, num_features: int = 10) -> Any:
        """Compute LIME explanation for a single instance."""
        try:
            import lime
            import lime.lime_tabular
            
            # Create explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_sample,
                feature_names=self.feature_names,
                class_names=['Control', 'AD/MCI'],
                mode='classification'
            )
            
            # Explain instance
            exp = explainer.explain_instance(
                X_sample[instance_idx],
                model.predict_proba,
                num_features=num_features,
                top_labels=1
            )
            
            return exp
            
        except ImportError:
            print("LIME not installed. Install with: pip install lime")
            return None
    
    def plot_lime_explanation(self, exp: Any, save_path: Optional[str] = None) -> plt.Figure:
        """Plot LIME explanation."""
        try:
            fig = exp.as_pyplot_figure()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"Error plotting LIME explanation: {e}")
            return None
    
    def get_feature_importance_from_model(self, model) -> Optional[np.ndarray]:
        """Get feature importance from tree-based models."""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        return None
    
    def plot_feature_importance(self, model, top_n: int = 20,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance from model."""
        importance = self.get_feature_importance_from_model(model)
        
        if importance is None:
            print("Model does not support feature importance")
            return None
        
        # Get top features
        if self.feature_names is not None:
            indices = np.argsort(importance)[-top_n:][::-1]
            feature_labels = [self.feature_names[i] for i in indices]
        else:
            indices = np.argsort(importance)[-top_n:][::-1]
            feature_labels = [f'Feature {i}' for i in indices]
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        ax.barh(range(top_n), importance[indices])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        ax.invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def full_explanation(self, model, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, output_dir: str = 'experiments/explainability') -> Dict[str, Any]:
        """Generate comprehensive explanations."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Set feature names if available
        if X_train.shape[1] == len(self.feature_names or []):
            pass  # Already set
        else:
            self.feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Permutation importance
        print("Computing permutation importance...")
        perm_imp = self.compute_permutation_importance(model, X_test, y_train[:len(X_test)])
        results['permutation_importance'] = {
            'mean': perm_imp['mean'].tolist(),
            'std': perm_imp['std'].tolist()
        }
        self.plot_permutation_importance(perm_imp, save_path=os.path.join(output_dir, 'permutation_importance.png'))
        
        # Model feature importance
        print("Extracting model feature importance...")
        self.plot_feature_importance(model, save_path=os.path.join(output_dir, 'model_importance.png'))
        
        # SHAP (sample)
        print("Computing SHAP values (sample)...")
        sample_size = min(50, len(X_test))
        shap_vals = self.compute_shap_values(model, X_test[:sample_size], X_train[:100])
        if len(shap_vals) > 0:
            self.plot_shap_summary(shap_vals, X_test[:sample_size], 
                                  save_path=os.path.join(output_dir, 'shap_summary.png'))
            results['shap_available'] = True
        
        # LIME (single instance)
        print("Computing LIME explanation...")
        lime_exp = self.compute_lime_explanation(model, X_train, y_train, instance_idx=0)
        if lime_exp is not None:
            self.plot_lime_explanation(lime_exp, save_path=os.path.join(output_dir, 'lime_explanation.png'))
            results['lime_available'] = True
        
        # Save results
        with open(os.path.join(output_dir, 'explanation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Explanations saved to {output_dir}")
        
        return results


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    X_train = np.random.randn(100, 50)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(30, 50)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    explainer = ModelExplainer()
    results = explainer.full_explanation(model, X_train, y_train, X_test)
    print("Explanation complete!")
