"""
Text Preprocessing Module for Alzheimer's Speech Detection

This module handles transcript cleaning, tokenization, and text normalization
with support for various linguistic analyses.
"""

import os
import re
import string
from typing import List, Dict, Any, Optional, Tuple
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class TextPreprocessor:
    """
    Preprocess text transcripts for feature extraction.
    
    Handles:
    - Loading transcript files
    - Text cleaning and normalization
    - Tokenization
    - Lemmatization
    - POS tagging (optional)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the text preprocessor.
        
        Args:
            config: Configuration dictionary or path to YAML file
            spacy_model: SpaCy model name for NLP processing
        """
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config.get('text', {})
        self.lowercase = self.config.get('lowercase', True)
        self.remove_punctuation = self.config.get('remove_punctuation', True)
        self.remove_numbers = self.config.get('remove_numbers', True)
        self.lemmatize = self.config.get('lemmatize', True)
        self.remove_stopwords = self.config.get('remove_stopwords', False)
        self.min_length = self.config.get('min_length', 10)
        
        # Load spaCy model
        try:
            import spacy
            self.nlp = spacy.load(spacy_model)
        except OSError:
            # Download model if not available
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
            import spacy
            self.nlp = spacy.load(spacy_model)
        
        # Stopwords
        if self.remove_stopwords:
            self.stopwords = set(self.nlp.Defaults.stop_words)
    
    def load_transcript(self, transcript_path: str) -> str:
        """
        Load a transcript from file.
        
        Args:
            transcript_path: Path to the transcript file
            
        Returns:
            Transcript text
        """
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except Exception as e:
            raise IOError(f"Failed to load transcript {transcript_path}: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove special markers often found in transcripts
        # e.g., [laughter], <pause>, etc.
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            # Keep some punctuation that might be meaningful
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Cleaned text
            
        Returns:
            List of tokens
        """
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        return tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.lemmatize:
            return tokens
        
        doc = self.nlp(' '.join(tokens))
        lemmas = [token.lemma_ for token in doc]
        return lemmas
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with stopwords removed
        """
        if not self.remove_stopwords:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def preprocess(self, transcript_path: str, return_info: bool = False) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Full preprocessing pipeline for a transcript.
        
        Args:
            transcript_path: Path to the transcript file
            return_info: Whether to return metadata about the processing
            
        Returns:
            Tuple of (cleaned text, tokens, info dict if requested)
        """
        info = {
            'original_path': transcript_path,
            'original_length': None,
            'cleaned_length': None,
            'num_tokens': None,
            'num_sentences': None,
            'steps_applied': []
        }
        
        # Load transcript
        text = self.load_transcript(transcript_path)
        info['original_length'] = len(text)
        
        # Clean text
        cleaned_text = self.clean_text(text)
        info['cleaned_length'] = len(cleaned_text)
        info['steps_applied'].append('clean')
        
        # Check minimum length
        if len(cleaned_text) < self.min_length:
            info['warning'] = f'Text too short: {len(cleaned_text)} < {self.min_length}'
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        info['num_tokens'] = len(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
            info['steps_applied'].append('lemmatize')
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
            info['steps_applied'].append('remove_stopwords')
        
        # Count sentences
        doc = self.nlp(cleaned_text)
        info['num_sentences'] = len(list(doc.sents))
        
        if return_info:
            return cleaned_text, tokens, info
        
        return cleaned_text, tokens, {}
    
    def preprocess_batch(self, transcript_paths: list, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Preprocess multiple transcripts.
        
        Args:
            transcript_paths: List of paths to transcript files
            output_dir: Optional directory to save preprocessed files
            
        Returns:
            Dictionary with preprocessed texts, tokens, and metadata
        """
        results = {
            'texts': {},
            'tokens': {},
            'metadata': {},
            'errors': {}
        }
        
        for transcript_path in tqdm(transcript_paths, desc="Preprocessing text"):
            try:
                cleaned_text, tokens, info = self.preprocess(transcript_path, return_info=True)
                
                # Store in memory
                results['texts'][transcript_path] = cleaned_text
                results['tokens'][transcript_path] = tokens
                results['metadata'][transcript_path] = info
                
                # Optionally save to disk
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save cleaned text
                    text_filename = Path(transcript_path).stem + '_cleaned.txt'
                    text_output_path = os.path.join(output_dir, text_filename)
                    with open(text_output_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                    
                    # Save tokens
                    tokens_filename = Path(transcript_path).stem + '_tokens.json'
                    tokens_output_path = os.path.join(output_dir, tokens_filename)
                    import json
                    with open(tokens_output_path, 'w', encoding='utf-8') as f:
                        json.dump(tokens, f, ensure_ascii=False)
                    
                    results['metadata'][transcript_path]['output_paths'] = {
                        'text': text_output_path,
                        'tokens': tokens_output_path
                    }
                    
            except Exception as e:
                results['errors'][transcript_path] = str(e)
        
        return results
    
    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """
        Get part-of-speech tags for text.
        
        Args:
            text: Input text
            
        Returns:
            List of (token, POS tag) tuples
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def get_dependency_parse(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Get dependency parse for text.
        
        Args:
            text: Input text
            
        Returns:
            List of (token, dependency, head) tuples
        """
        doc = self.nlp(text)
        return [(token.text, token.dep_, token.head.text) for token in doc]


def validate_transcript_file(transcript_path: str) -> bool:
    """
    Validate that a transcript file exists and can be loaded.
    
    Args:
        transcript_path: Path to the transcript file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(transcript_path):
        return False
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return len(text.strip()) > 0
    except:
        return False


if __name__ == "__main__":
    # Example usage
    config_path = "configs/config.yaml"
    preprocessor = TextPreprocessor(config_path)
    
    # Test with a sample file
    test_transcript = "data/raw/sample_transcript.txt"
    if os.path.exists(test_transcript):
        cleaned_text, tokens, info = preprocessor.preprocess(test_transcript, return_info=True)
        print(f"Cleaned text length: {len(cleaned_text)}")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Info: {info}")
    else:
        print(f"Test file {test_transcript} not found. Please add transcript files to data/raw/")
