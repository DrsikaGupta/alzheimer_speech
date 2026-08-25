"""
Unit Tests for Alzheimer's Speech Detection Pipeline

Run with: pytest tests/ -v
"""

import os
import sys
import numpy as np
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAudioPreprocessing:
    """Tests for audio preprocessing module."""
    
    def test_audio_preprocessor_init(self):
        from src.preprocessing.audio_preprocessing import AudioPreprocessor
        
        preprocessor = AudioPreprocessor()
        assert preprocessor.sample_rate == 16000
    
    def test_audio_preprocessor_with_config(self):
        from src.preprocessing.audio_preprocessing import AudioPreprocessor
        
        config = {"audio": {"sample_rate": 22050}}
        preprocessor = AudioPreprocessor(config)
        assert preprocessor.sample_rate == 22050


class TestTextPreprocessing:
    """Tests for text preprocessing module."""
    
    def test_text_preprocessor_init(self):
        from src.preprocessing.text_preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
    
    def test_clean_text(self):
        from src.preprocessing.text_preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        text = "Hello [laughter] this is a <test>."
        cleaned = preprocessor.clean_text(text)
        assert "[laughter]" not in cleaned
        assert "<test>" not in cleaned


class TestAcousticFeatures:
    """Tests for acoustic feature extraction."""
    
    def test_extractor_init(self):
        from src.features.acoustic_features import AcousticFeatureExtractor
        
        extractor = AcousticFeatureExtractor()
        assert extractor is not None
    
    def test_extract_mfcc(self):
        from src.features.acoustic_features import AcousticFeatureExtractor
        
        extractor = AcousticFeatureExtractor()
        audio = np.random.randn(16000)  # 1 second at 16kHz
        mfcc = extractor.extract_mfcc(audio, 16000)
        assert len(mfcc) > 0


class TestLinguisticFeatures:
    """Tests for linguistic feature extraction."""
    
    def test_extractor_init(self):
        from src.features.linguistic_features import LinguisticFeatureExtractor
        
        extractor = LinguisticFeatureExtractor()
        assert extractor is not None
    
    def test_extract_lexical_features(self):
        from src.features.linguistic_features import LinguisticFeatureExtractor
        
        extractor = LinguisticFeatureExtractor()
        text = "The patient described their daily routine with difficulty."
        tokens = text.split()
        features = extractor.extract_lexical_features(text, tokens)
        assert 'ttr' in features or len(features) > 0


class TestDataPreparation:
    """Tests for data preparation."""
    
    def test_preparator_init(self):
        from src.preprocessing.data_preparation import DataPreparator
        
        preparator = DataPreparator()
        assert preparator is not None
    
    def test_subject_level_split(self):
        import pandas as pd
        from src.preprocessing.data_preparation import DataPreparator
        
        preparator = DataPreparator()
        
        # Create test data
        df = pd.DataFrame({
            'subject_id': ['s1', 's1', 's2', 's2', 's3', 's4', 's5', 's6'],
            'label': [0, 0, 1, 1, 0, 1, 0, 1]
        })
        
        train_df, val_df, test_df = preparator.subject_level_split(df)
        
        # Check no subject overlap
        train_subjects = set(train_df['subject_id'])
        val_subjects = set(val_df['subject_id'])
        test_subjects = set(test_df['subject_id'])
        
        assert len(train_subjects & val_subjects) == 0
        assert len(train_subjects & test_subjects) == 0
        assert len(val_subjects & test_subjects) == 0


class TestBaselineModels:
    """Tests for baseline models."""
    
    def test_baseline_models_init(self):
        from src.models.train import BaselineModels
        
        models = BaselineModels()
        model_dict = models.get_models()
        assert len(model_dict) > 0
    
    def test_model_training(self):
        from src.models.train import BaselineModels
        
        models = BaselineModels()
        
        X_train = np.random.randn(50, 100)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(20, 100)
        y_val = np.random.randint(0, 2, 20)
        
        results = models.train_and_evaluate(X_train, y_train, X_val, y_val)
        
        assert len(results) > 0
        for name, result in results.items():
            assert 'accuracy' in result['metrics']


class TestEvaluation:
    """Tests for evaluation module."""
    
    def test_evaluator_init(self):
        from src.evaluation.evaluate import Evaluator
        
        evaluator = Evaluator()
        assert evaluator is not None
    
    def test_compute_metrics(self):
        from src.evaluation.evaluate import Evaluator
        
        evaluator = Evaluator()
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
        
        metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics


class TestExplainer:
    """Tests for explainability module."""
    
    def test_explainer_init(self):
        from src.explainability.explainer import ModelExplainer
        
        explainer = ModelExplainer()
        assert explainer is not None
    
    def test_permutation_importance(self):
        from sklearn.ensemble import RandomForestClassifier
        from src.explainability.explainer import ModelExplainer
        
        explainer = ModelExplainer()
        
        X = np.random.randn(50, 20)
        y = np.random.randint(0, 2, 50)
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        importance = explainer.compute_permutation_importance(model, X, y, n_repeats=3)
        
        assert 'mean' in importance
        assert len(importance['mean']) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
