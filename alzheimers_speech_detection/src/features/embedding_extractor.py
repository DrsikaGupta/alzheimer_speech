"""
Pretrained Embedding Extraction Module for Alzheimer's Speech Detection

This module extracts embeddings from pretrained models:
- Audio: wav2vec 2.0, HuBERT, Whisper
- Text: BERT, Sentence Transformers
"""

import os
import numpy as np
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F


class AudioEmbeddingExtractor:
    """
    Extract embeddings from pretrained audio models.
    
    Supports:
    - wav2vec 2.0
    - HuBERT
    - Whisper (optional)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the audio embedding extractor.
        
        Args:
            config: Configuration dictionary or path to YAML file
        """
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config.get('embeddings', {}).get('audio', {})
        self.audio_config = config.get('audio', {})
        self.sample_rate = self.audio_config.get('sample_rate', 16000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load pretrained models."""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            
            # wav2vec 2.0
            if self.config.get('wav2vec2', {}).get('enabled', True):
                model_name = self.config['wav2vec2'].get('model_name', 'facebook/wav2vec2-base-960h')
                print(f"Loading wav2vec 2.0: {model_name}...")
                self.processor_wav2vec = Wav2Vec2Processor.from_pretrained(model_name)
                self.models['wav2vec2'] = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
                self.models['wav2vec2'].eval()
            
            # HuBERT
            if self.config.get('hubert', {}).get('enabled', True):
                from transformers import HubertModel
                model_name = self.config['hubert'].get('model_name', 'facebook/hubert-base-ls960')
                print(f"Loading HuBERT: {model_name}...")
                self.processor_hubert = Wav2Vec2Processor.from_pretrained(model_name)
                self.models['hubert'] = HubertModel.from_pretrained(model_name).to(self.device)
                self.models['hubert'].eval()
                
        except Exception as e:
            print(f"Warning: Could not load audio embedding models: {e}")
    
    def extract_wav2vec2(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract wav2vec 2.0 embeddings.
        
        Args:
            audio: Audio signal
            
        Returns:
            Embedding vector
        """
        if 'wav2vec2' not in self.models:
            return np.array([])
        
        pooling = self.config['wav2vec2'].get('pooling', 'mean')
        
        # Process audio
        inputs = self.processor_wav2vec(
            audio, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.models['wav2vec2'](**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Pooling
        if pooling == 'mean':
            embedding = torch.mean(hidden_states, dim=1)
        elif pooling == 'max':
            embedding, _ = torch.max(hidden_states, dim=1)
        elif pooling == 'cls':
            embedding = hidden_states[:, 0, :]
        else:
            embedding = torch.mean(hidden_states, dim=1)
        
        return embedding.cpu().numpy().flatten()
    
    def extract_hubert(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract HuBERT embeddings.
        
        Args:
            audio: Audio signal
            
        Returns:
            Embedding vector
        """
        if 'hubert' not in self.models:
            return np.array([])
        
        pooling = self.config['hubert'].get('pooling', 'mean')
        
        # Process audio
        inputs = self.processor_hubert(
            audio, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.models['hubert'](**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Pooling
        if pooling == 'mean':
            embedding = torch.mean(hidden_states, dim=1)
        elif pooling == 'max':
            embedding, _ = torch.max(hidden_states, dim=1)
        else:
            embedding = torch.mean(hidden_states, dim=1)
        
        return embedding.cpu().numpy().flatten()
    
    def extract_all(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all configured audio embeddings.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of embeddings
        """
        embeddings = {}
        
        if self.config.get('wav2vec2', {}).get('enabled', True):
            embeddings['wav2vec2'] = self.extract_wav2vec2(audio)
        
        if self.config.get('hubert', {}).get('enabled', True):
            embeddings['hubert'] = self.extract_hubert(audio)
        
        return embeddings
    
    def extract_from_file(self, audio_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract embeddings from an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (concatenated embeddings, metadata)
        """
        import librosa
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            embeddings = self.extract_all(audio)
            
            # Concatenate all embeddings
            embedding_list = [e for e in embeddings.values() if len(e) > 0]
            concatenated = np.concatenate(embedding_list) if embedding_list else np.array([])
            
            metadata = {
                'audio_path': audio_path,
                'embedding_dims': {k: v.shape[0] for k, v in embeddings.items()},
                'total_dim': concatenated.shape[0]
            }
            
            return concatenated, metadata
            
        except Exception as e:
            raise IOError(f"Failed to extract audio embeddings from {audio_path}: {str(e)}")


class TextEmbeddingExtractor:
    """
    Extract embeddings from pretrained text models.
    
    Supports:
    - BERT
    - Sentence Transformers
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text embedding extractor.
        
        Args:
            config: Configuration dictionary or path to YAML file
        """
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config.get('embeddings', {}).get('text', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load pretrained models."""
        try:
            # BERT
            if self.config.get('bert', {}).get('enabled', True):
                from transformers import BertTokenizer, BertModel
                model_name = self.config['bert'].get('model_name', 'bert-base-uncased')
                print(f"Loading BERT: {model_name}...")
                self.tokenizer_bert = BertTokenizer.from_pretrained(model_name)
                self.models['bert'] = BertModel.from_pretrained(model_name).to(self.device)
                self.models['bert'].eval()
            
            # Sentence Transformer
            if self.config.get('sentence_transformer', {}).get('enabled', True):
                from sentence_transformers import SentenceTransformer
                model_name = self.config['sentence_transformer'].get('model_name', 'all-MiniLM-L6-v2')
                print(f"Loading Sentence Transformer: {model_name}...")
                self.models['sentence_transformer'] = SentenceTransformer(model_name, device=str(self.device))
                
        except Exception as e:
            print(f"Warning: Could not load text embedding models: {e}")
    
    def extract_bert(self, text: str) -> np.ndarray:
        """
        Extract BERT embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if 'bert' not in self.models:
            return np.array([])
        
        pooling = self.config['bert'].get('pooling', 'cls')
        
        # Tokenize
        inputs = self.tokenizer_bert(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.models['bert'](**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Pooling
        if pooling == 'cls':
            embedding = hidden_states[:, 0, :]
        elif pooling == 'mean':
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            embedding = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:
            embedding = hidden_states[:, 0, :]
        
        return embedding.cpu().numpy().flatten()
    
    def extract_sentence_transformer(self, text: str) -> np.ndarray:
        """
        Extract Sentence Transformer embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if 'sentence_transformer' not in self.models:
            return np.array([])
        
        embedding = self.models['sentence_transformer'].encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding.flatten()
    
    def extract_all(self, text: str) -> Dict[str, np.ndarray]:
        """
        Extract all configured text embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of embeddings
        """
        embeddings = {}
        
        if self.config.get('bert', {}).get('enabled', True):
            embeddings['bert'] = self.extract_bert(text)
        
        if self.config.get('sentence_transformer', {}).get('enabled', True):
            embeddings['sentence_transformer'] = self.extract_sentence_transformer(text)
        
        return embeddings
    
    def extract_batch(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extract embeddings from multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Tuple of (embedding matrix, metadata list)
        """
        all_embeddings = []
        all_metadata = []
        
        for i, text in enumerate(tqdm(texts, desc="Extracting text embeddings")):
            try:
                embeddings = self.extract_all(text)
                
                embedding_list = [e for e in embeddings.values() if len(e) > 0]
                concatenated = np.concatenate(embedding_list) if embedding_list else np.array([])
                
                if len(concatenated) > 0:
                    all_embeddings.append(concatenated)
                    all_metadata.append({
                        'index': i,
                        'embedding_dims': {k: v.shape[0] for k, v in embeddings.items()},
                        'total_dim': concatenated.shape[0]
                    })
                    
            except Exception as e:
                print(f"Error processing text {i}: {str(e)}")
        
        embedding_matrix = np.array(all_embeddings) if all_embeddings else np.array([])
        
        return embedding_matrix, all_metadata


if __name__ == "__main__":
    # Example usage
    config_path = "configs/config.yaml"
    
    print("Testing Audio Embedding Extractor...")
    audio_extractor = AudioEmbeddingExtractor(config_path)
    
    print("\nTesting Text Embedding Extractor...")
    text_extractor = TextEmbeddingExtractor(config_path)
    
    sample_text = "The patient described their daily routine with some difficulty."
    embeddings = text_extractor.extract_all(sample_text)
    
    for name, emb in embeddings.items():
        print(f"{name}: shape = {emb.shape}")
