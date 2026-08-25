"""
Audio Preprocessing Module for Alzheimer's Speech Detection

This module handles audio file loading, cleaning, and preprocessing
with support for various audio formats and quality improvements.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import yaml
from tqdm import tqdm


class AudioPreprocessor:
    """
    Preprocess audio files for feature extraction.
    
    Handles:
    - Loading audio files in various formats
    - Resampling to target sample rate
    - Silence trimming
    - Noise reduction (optional)
    - Normalization
    - Duration limiting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the audio preprocessor.
        
        Args:
            config: Configuration dictionary or path to YAML file
        """
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config.get('audio', {})
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.duration_max = self.config.get('duration_max', 30)
        self.silence_trim = self.config.get('silence_trim', True)
        self.noise_reduction = self.config.get('noise_reduction', False)
        self.normalize = self.config.get('normalize', True)
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio signal, sample rate)
        """
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            return audio, sr
        except Exception as e:
            raise IOError(f"Failed to load audio file {audio_path}: {str(e)}")
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio signal
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio signal
        """
        if orig_sr == target_sr:
            return audio
        
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def trim_silence(self, audio: np.ndarray, top_db: float = 20) -> np.ndarray:
        """
        Trim leading and trailing silence.
        
        Args:
            audio: Audio signal
            top_db: Threshold in dB below which audio is considered silent
            
        Returns:
            Audio with silence trimmed
        """
        if not self.silence_trim:
            return audio
        
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude.
        
        Args:
            audio: Audio signal
            
        Returns:
            Normalized audio signal
        """
        if not self.normalize:
            return audio
        
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            audio = audio / max_amplitude
        
        return audio
    
    def limit_duration(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Limit audio duration to maximum specified.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Audio limited to maximum duration
        """
        max_samples = int(self.duration_max * sr)
        
        if len(audio) > max_samples:
            # Take the first portion (or could use center portion)
            audio = audio[:max_samples]
        
        return audio
    
    def pad_audio(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad audio to target length if shorter.
        
        Args:
            audio: Audio signal
            target_length: Target number of samples
            
        Returns:
            Padded audio signal
        """
        if len(audio) >= target_length:
            return audio
        
        padding = target_length - len(audio)
        padded = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        
        return padded
    
    def preprocess(self, audio_path: str, return_info: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full preprocessing pipeline for an audio file.
        
        Args:
            audio_path: Path to the audio file
            return_info: Whether to return metadata about the processing
            
        Returns:
            Tuple of (preprocessed audio, info dict if requested)
        """
        info = {
            'original_path': audio_path,
            'original_sr': None,
            'target_sr': self.sample_rate,
            'original_duration': None,
            'final_duration': None,
            'steps_applied': []
        }
        
        # Load audio
        audio, orig_sr = self.load_audio(audio_path)
        info['original_sr'] = orig_sr
        info['original_duration'] = len(audio) / orig_sr
        
        # Resample
        if orig_sr != self.sample_rate:
            audio = self.resample(audio, orig_sr, self.sample_rate)
            info['steps_applied'].append('resample')
        
        # Trim silence
        if self.silence_trim:
            original_len = len(audio)
            audio = self.trim_silence(audio)
            if len(audio) < original_len:
                info['steps_applied'].append('trim_silence')
        
        # Normalize
        if self.normalize:
            audio = self.normalize_audio(audio)
            info['steps_applied'].append('normalize')
        
        # Limit duration
        audio = self.limit_duration(audio, self.sample_rate)
        info['final_duration'] = len(audio) / self.sample_rate
        
        if return_info:
            return audio, info
        
        return audio
    
    def preprocess_batch(self, audio_paths: list, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Preprocess multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            output_dir: Optional directory to save preprocessed files
            
        Returns:
            Dictionary with preprocessed audio and metadata
        """
        results = {
            'audio': {},
            'metadata': {},
            'errors': {}
        }
        
        for audio_path in tqdm(audio_paths, desc="Preprocessing audio"):
            try:
                audio, info = self.preprocess(audio_path, return_info=True)
                
                # Store in memory
                results['audio'][audio_path] = audio
                results['metadata'][audio_path] = info
                
                # Optionally save to disk
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = Path(audio_path).stem + '_processed.wav'
                    output_path = os.path.join(output_dir, filename)
                    sf.write(output_path, audio, self.sample_rate)
                    results['metadata'][audio_path]['output_path'] = output_path
                    
            except Exception as e:
                results['errors'][audio_path] = str(e)
        
        return results


def validate_audio_file(audio_path: str) -> bool:
    """
    Validate that an audio file exists and can be loaded.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(audio_path):
        return False
    
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        return len(audio) > 0
    except:
        return False


if __name__ == "__main__":
    # Example usage
    config_path = "configs/config.yaml"
    preprocessor = AudioPreprocessor(config_path)
    
    # Test with a sample file
    test_audio = "data/raw/sample.wav"
    if os.path.exists(test_audio):
        audio, info = preprocessor.preprocess(test_audio, return_info=True)
        print(f"Processed audio shape: {audio.shape}")
        print(f"Info: {info}")
    else:
        print(f"Test file {test_audio} not found. Please add audio files to data/raw/")
