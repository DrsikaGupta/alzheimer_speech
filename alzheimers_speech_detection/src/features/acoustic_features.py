"""
Acoustic Feature Extraction Module for Alzheimer's Speech Detection

This module extracts acoustic features from audio signals including:
- MFCCs and derivatives
- Spectrogram features
- Chroma features
- Spectral contrast
- Zero-crossing rate
- Pitch and prosody features
- Pause and silence features
"""

import numpy as np
import librosa
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class AcousticFeatureExtractor:
    """
    Extract acoustic features from audio signals.
    
    Features include:
    - MFCCs (Mel-frequency cepstral coefficients)
    - Spectrogram statistics
    - Chroma features
    - Spectral contrast
    - Tonnetz
    - Zero-crossing rate
    - Tempo and beat features
    - Pitch features
    - Jitter and shimmer
    - Pause/silence features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the acoustic feature extractor.
        
        Args:
            config: Configuration dictionary or path to YAML file
        """
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config.get('acoustic_features', {})
        self.audio_config = config.get('audio', {})
        self.sample_rate = self.audio_config.get('sample_rate', 16000)
        
    def extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            MFCC features (mean and std for each coefficient)
        """
        mfcc_config = self.config.get('mfcc', {})
        if not mfcc_config.get('enabled', True):
            return np.array([])
        
        n_mfcc = mfcc_config.get('n_mfcc', 13)
        n_fft = mfcc_config.get('n_fft', 512)
        hop_length = mfcc_config.get('hop_length', 256)
        
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Compute statistics
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)
        
        return np.concatenate([mfcc_mean, mfcc_std, mfcc_delta_mean, mfcc_delta_std])
    
    def extract_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectrogram features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Spectrogram statistics
        """
        spec_config = self.config.get('spectrogram', {})
        if not spec_config.get('enabled', True):
            return np.array([])
        
        n_fft = spec_config.get('n_fft', 512)
        hop_length = spec_config.get('hop_length', 256)
        
        spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        
        # Statistics
        spec_mean = np.mean(spectrogram)
        spec_std = np.std(spectrogram)
        spec_skew = np.mean(((spectrogram - spec_mean) / spec_std) ** 3) if spec_std > 0 else 0
        spec_kurtosis = np.mean(((spectrogram - spec_mean) / spec_std) ** 4) - 3 if spec_std > 0 else 0
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        centroid_mean = np.mean(spectral_centroid)
        centroid_std = np.std(spectral_centroid)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        bandwidth_mean = np.mean(spectral_bandwidth)
        bandwidth_std = np.std(spectral_bandwidth)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        rolloff_mean = np.mean(spectral_rolloff)
        rolloff_std = np.std(spectral_rolloff)
        
        return np.array([
            spec_mean, spec_std, spec_skew, spec_kurtosis,
            centroid_mean, centroid_std,
            bandwidth_mean, bandwidth_std,
            rolloff_mean, rolloff_std
        ])
    
    def extract_chroma(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract chroma features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Chroma feature statistics
        """
        chroma_config = self.config.get('chroma', {})
        if not chroma_config.get('enabled', True):
            return np.array([])
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        return np.concatenate([chroma_mean, chroma_std])
    
    def extract_spectral_contrast(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectral contrast features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Spectral contrast statistics
        """
        contrast_config = self.config.get('spectral_contrast', {})
        if not contrast_config.get('enabled', True):
            return np.array([])
        
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        return np.concatenate([contrast_mean, contrast_std])
    
    def extract_tonnetz(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract tonal centroid (tonnetz) features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Tonnetz feature statistics
        """
        tonnetz_config = self.config.get('tonnetz', {})
        if not tonnetz_config.get('enabled', True):
            return np.array([])
        
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
        
        return np.concatenate([tonnetz_mean, tonnetz_std])
    
    def extract_zero_crossing_rate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract zero-crossing rate features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            ZCR statistics
        """
        zcr_config = self.config.get('zero_crossing_rate', {})
        if not zcr_config.get('enabled', True):
            return np.array([])
        
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        return np.array([np.mean(zcr), np.std(zcr)])
    
    def extract_tempo(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract tempo features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Tempo features
        """
        tempo_config = self.config.get('tempo', {})
        if not tempo_config.get('enabled', True):
            return np.array([])
        
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Beat tracking features
        if len(beats) > 1:
            beat_intervals = np.diff(beats) / sr
            beat_interval_mean = np.mean(beat_intervals)
            beat_interval_std = np.std(beat_intervals)
        else:
            beat_interval_mean = 0
            beat_interval_std = 0
        
        return np.array([tempo if isinstance(tempo, float) else tempo[0], beat_interval_mean, beat_interval_std])
    
    def extract_pitch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract pitch features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Pitch statistics
        """
        pitch_config = self.config.get('pitch', {})
        if not pitch_config.get('enabled', True):
            return np.array([])
        
        # Extract fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Handle NaN values (unvoiced regions)
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            f0_mean = np.mean(f0_voiced)
            f0_std = np.std(f0_voiced)
            f0_min = np.min(f0_voiced)
            f0_max = np.max(f0_voiced)
            f0_range = f0_max - f0_min
            voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_range = voiced_ratio = 0
        
        return np.array([f0_mean, f0_std, f0_min, f0_max, f0_range, voiced_ratio])
    
    def extract_jitter_shimmer(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract jitter and shimmer features (voice quality measures).
        
        Note: These are simplified approximations. For clinical applications,
        consider using specialized tools like Praat.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Jitter and shimmer estimates
        """
        js_config = self.config.get('jitter_shimmer', {})
        if not js_config.get('enabled', True):
            return np.array([])
        
        # Simplified jitter (frequency variation) and shimmer (amplitude variation)
        # Using pitch extraction as proxy
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 1:
            # Jitter: relative variation in F0
            f0_diff = np.diff(f0_voiced)
            jitter = np.mean(np.abs(f0_diff)) / np.mean(f0_voiced) if np.mean(f0_voiced) > 0 else 0
            
            # Shimmer: approximate from amplitude envelope
            envelope = np.abs(librosa.hilbert(audio))
            envelope_voiced = envelope[voiced_flag[:len(envelope)]]
            if len(envelope_voiced) > 1:
                env_diff = np.diff(envelope_voiced)
                shimmer = np.mean(np.abs(env_diff)) / np.mean(envelope_voiced) if np.mean(envelope_voiced) > 0 else 0
            else:
                shimmer = 0
        else:
            jitter = shimmer = 0
        
        return np.array([jitter, shimmer])
    
    def extract_pause_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract pause and silence features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Pause-related features
        """
        pause_config = self.config.get('pause_features', {})
        if not pause_config.get('enabled', True):
            return np.array([])
        
        # Detect silent regions
        intervals = librosa.effects.split(audio, top_db=20)
        
        total_duration = len(audio) / sr
        speech_duration = np.sum(np.diff(intervals, axis=1)) / sr
        pause_duration = total_duration - speech_duration
        
        num_pauses = len(intervals) - 1 if len(intervals) > 1 else 0
        
        if num_pauses > 0:
            pause_lengths = np.diff(intervals, axis=1).flatten() / sr
            mean_pause_length = np.mean(pause_lengths)
            std_pause_length = np.std(pause_lengths)
            max_pause_length = np.max(pause_lengths)
        else:
            mean_pause_length = std_pause_length = max_pause_length = 0
        
        speech_rate = len(audio) / (speech_duration * sr) if speech_duration > 0 else 0
        
        return np.array([
            pause_duration,
            num_pauses,
            mean_pause_length,
            std_pause_length,
            max_pause_length,
            pause_duration / total_duration if total_duration > 0 else 0,  # pause ratio
            speech_rate
        ])
    
    def extract_all_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract all configured acoustic features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of feature arrays
        """
        features = {}
        
        features['mfcc'] = self.extract_mfcc(audio, sr)
        features['spectrogram'] = self.extract_spectrogram(audio, sr)
        features['chroma'] = self.extract_chroma(audio, sr)
        features['spectral_contrast'] = self.extract_spectral_contrast(audio, sr)
        features['tonnetz'] = self.extract_tonnetz(audio, sr)
        features['zero_crossing_rate'] = self.extract_zero_crossing_rate(audio, sr)
        features['tempo'] = self.extract_tempo(audio, sr)
        features['pitch'] = self.extract_pitch(audio, sr)
        features['jitter_shimmer'] = self.extract_jitter_shimmer(audio, sr)
        features['pause_features'] = self.extract_pause_features(audio, sr)
        
        return features
    
    def extract_feature_vector(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract and concatenate all features into a single vector.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Concatenated feature vector
        """
        features = self.extract_all_features(audio, sr)
        
        # Concatenate all non-empty feature arrays
        feature_list = [f for f in features.values() if len(f) > 0]
        
        if len(feature_list) == 0:
            return np.array([])
        
        return np.concatenate(feature_list)
    
    def extract_from_file(self, audio_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract features from an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (feature vector, metadata)
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            features = self.extract_feature_vector(audio, sr)
            
            metadata = {
                'audio_path': audio_path,
                'original_sr': sr,
                'resampled_sr': self.sample_rate,
                'duration': len(audio) / self.sample_rate,
                'num_samples': len(audio),
                'feature_dim': len(features)
            }
            
            return features, metadata
            
        except Exception as e:
            raise IOError(f"Failed to extract features from {audio_path}: {str(e)}")
    
    def extract_batch(self, audio_paths: list) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            Tuple of (feature matrix, metadata DataFrame)
        """
        all_features = []
        all_metadata = []
        errors = []
        
        for audio_path in tqdm(audio_paths, desc="Extracting acoustic features"):
            try:
                features, metadata = self.extract_from_file(audio_path)
                
                if len(features) > 0:
                    all_features.append(features)
                    all_metadata.append(metadata)
                else:
                    errors.append({'path': audio_path, 'error': 'Empty feature vector'})
                    
            except Exception as e:
                errors.append({'path': audio_path, 'error': str(e)})
        
        feature_matrix = np.array(all_features) if all_features else np.array([])
        metadata_df = pd.DataFrame(all_metadata)
        
        if errors:
            print(f"Warning: {len(errors)} files had errors during feature extraction")
        
        return feature_matrix, metadata_df


if __name__ == "__main__":
    # Example usage
    config_path = "configs/config.yaml"
    extractor = AcousticFeatureExtractor(config_path)
    
    # Test with a sample file
    test_audio = "data/raw/sample.wav"
    if os.path.exists(test_audio):
        features, metadata = extractor.extract_from_file(test_audio)
        print(f"Feature vector shape: {features.shape}")
        print(f"Metadata: {metadata}")
    else:
        print(f"Test file {test_audio} not found.")
