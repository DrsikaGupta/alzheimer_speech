"""
Linguistic Feature Extraction Module for Alzheimer's Speech Detection
"""

import os
import re
import numpy as np
import yaml
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter


class LinguisticFeatureExtractor:
    """Extract linguistic features from text transcripts."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, spacy_model: str = "en_core_web_sm"):
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config.get('linguistic_features', {})
        
        try:
            import spacy
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
            import spacy
            self.nlp = spacy.load(spacy_model)
        
        self.filler_words = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'sort of', 'kind of',
                            'basically', 'actually', 'literally', 'well', 'so', 'anyway'}
        self.pronouns = {'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 
                        'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 
                        'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs'}
    
    def extract_lexical_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        lexical_config = self.config.get('lexical', {})
        if not lexical_config.get('enabled', True):
            return {}
        
        features = {}
        num_tokens = len(tokens)
        unique_tokens = set(tokens)
        num_unique = len(unique_tokens)
        
        features['num_tokens'] = num_tokens
        features['num_unique_tokens'] = num_unique
        
        if lexical_config.get('ttr', True):
            features['ttr'] = num_unique / num_tokens if num_tokens > 0 else 0
        
        features['cttr'] = num_unique / np.sqrt(2 * num_tokens) if num_tokens > 0 else 0
        features['root_ttr'] = num_unique / np.sqrt(num_tokens) if num_tokens > 0 else 0
        
        if lexical_config.get('mtld', True):
            features['mtld'] = self._calculate_mtld(tokens)
        
        return features
    
    def _calculate_mtld(self, tokens: List[str], threshold: float = 0.72) -> float:
        if len(tokens) == 0:
            return 0
        
        segments_forward = []
        current_segment = []
        for token in tokens:
            current_segment.append(token)
            ttr = len(set(current_segment)) / len(current_segment) if current_segment else 0
            if ttr <= threshold:
                segments_forward.append(len(current_segment))
                current_segment = []
        
        if current_segment:
            segments_forward.append(len(current_segment))
        
        mtld_forward = np.mean(segments_forward) if segments_forward else 0
        return mtld_forward
    
    def extract_syntactic_features(self, text: str) -> Dict[str, float]:
        syntactic_config = self.config.get('syntactic', {})
        if not syntactic_config.get('enabled', True):
            return {}
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        features = {}
        num_sentences = len(sentences)
        features['num_sentences'] = num_sentences
        
        if num_sentences == 0:
            return features
        
        sent_lengths = [len(sent) for sent in sentences]
        features['avg_sentence_length'] = np.mean(sent_lengths)
        features['std_sentence_length'] = np.std(sent_lengths) if num_sentences > 1 else 0
        
        return features
    
    def extract_semantic_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        semantic_config = self.config.get('semantic', {})
        if not semantic_config.get('enabled', True):
            return {}
        
        features = {}
        word_freqs = Counter(tokens)
        hapax = sum(1 for count in word_freqs.values() if count == 1)
        features['hapax_legomena'] = hapax
        features['hapax_ratio'] = hapax / len(tokens) if len(tokens) > 0 else 0
        
        pronoun_count = sum(1 for token in tokens if token.lower() in self.pronouns)
        features['pronoun_count'] = pronoun_count
        features['pronoun_ratio'] = pronoun_count / len(tokens) if len(tokens) > 0 else 0
        
        return features
    
    def extract_discourse_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        discourse_config = self.config.get('discourse', {})
        if not discourse_config.get('enabled', True):
            return {}
        
        features = {}
        text_lower = text.lower()
        filler_count = sum(1 for filler in self.filler_words if filler in text_lower)
        features['filler_word_count'] = filler_count
        features['filler_ratio'] = filler_count / len(tokens) if len(tokens) > 0 else 0
        
        if discourse_config.get('repetitions', True):
            immediate_reps = sum(1 for i in range(len(tokens) - 1) if tokens[i].lower() == tokens[i+1].lower())
            features['immediate_repetitions'] = immediate_reps
        
        return features
    
    def extract_all_features(self, text: str, tokens: Optional[List[str]] = None) -> Dict[str, float]:
        if tokens is None:
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
        
        features = {}
        features.update(self.extract_lexical_features(text, tokens))
        features.update(self.extract_syntactic_features(text))
        features.update(self.extract_semantic_features(text, tokens))
        features.update(self.extract_discourse_features(text, tokens))
        
        return features
    
    def extract_feature_vector(self, text: str, tokens: Optional[List[str]] = None) -> np.ndarray:
        features = self.extract_all_features(text, tokens)
        if not features:
            return np.array([])
        feature_values = [features[k] for k in sorted(features.keys())]
        return np.array(feature_values)
    
    def extract_from_file(self, transcript_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'<.*?>', '', text)
            text = text.lower()
            
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            
            features = self.extract_feature_vector(text, tokens)
            
            metadata = {
                'transcript_path': transcript_path,
                'text_length': len(text),
                'num_tokens': len(tokens),
                'num_sentences': len(list(doc.sents)),
                'feature_dim': len(features)
            }
            
            return features, metadata
            
        except Exception as e:
            raise IOError(f"Failed to extract features from {transcript_path}: {str(e)}")


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    extractor = LinguisticFeatureExtractor(config_path)
    sample_text = "The patient described their daily routine. They mentioned having difficulty."
    features = extractor.extract_feature_vector(sample_text)
    print(f"Feature vector shape: {features.shape}")
