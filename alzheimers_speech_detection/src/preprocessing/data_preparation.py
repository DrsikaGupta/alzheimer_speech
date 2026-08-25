"""
Data Preparation Module for Alzheimer's Speech Detection

This module handles dataset loading, subject-level splitting, and data preparation
with strict prevention of data leakage across train/validation/test sets.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataPreparator:
    """
    Prepare datasets with subject-level train/val/test splits.
    
    Ensures:
    - No data leakage (all samples from same subject in one split)
    - Stratified splitting by label
    - Support for multiple dataset formats (ADReSS, ADReSSo, DementiaBank)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data preparator.
        
        Args:
            config: Configuration dictionary or path to YAML file
        """
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config
        self.dataset_config = config.get('dataset', {})
        self.split_config = config.get('split', {})
        
        self.subject_id_col = self.dataset_config.get('subject_id_column', 'subject_id')
        self.label_col = self.dataset_config.get('label_column', 'label')
        self.audio_col = self.dataset_config.get('audio_column', 'audio_path')
        self.transcript_col = self.dataset_config.get('transcript_column', 'transcript_path')
        
        self.train_ratio = self.split_config.get('train_ratio', 0.7)
        self.val_ratio = self.split_config.get('val_ratio', 0.15)
        self.test_ratio = self.split_config.get('test_ratio', 0.15)
        self.random_state = self.split_config.get('random_state', 42)
        self.stratify = self.split_config.get('stratify', True)
    
    def load_adress_dataset(self, data_path: str) -> pd.DataFrame:
        """
        Load ADReSS dataset format.
        
        Args:
            data_path: Path to ADReSS dataset directory
            
        Returns:
            DataFrame with metadata
        """
        records = []
        
        # ADReSS structure: train/ and test/ directories with audio/ and transcript/
        for split_type in ['train', 'test']:
            split_dir = os.path.join(data_path, split_type)
            if not os.path.exists(split_dir):
                continue
            
            audio_dir = os.path.join(split_dir, 'audio')
            transcript_dir = os.path.join(split_dir, 'transcript')
            
            # Get labels from directory names or metadata files
            label_map = {}
            label_file = os.path.join(split_dir, 'labels.csv')
            if os.path.exists(label_file):
                label_df = pd.read_csv(label_file)
                label_map = dict(zip(label_df['filename'], label_df['label']))
            
            # Process audio files
            if os.path.exists(audio_dir):
                for audio_file in os.listdir(audio_dir):
                    if audio_file.endswith(('.wav', '.flac', '.mp3')):
                        subject_id = Path(audio_file).stem
                        audio_path = os.path.join(audio_dir, audio_file)
                        
                        # Find corresponding transcript
                        transcript_file = None
                        for ext in ['.txt', '.cha', '.json']:
                            potential_transcript = os.path.join(
                                transcript_dir, 
                                Path(audio_file).stem + ext
                            )
                            if os.path.exists(potential_transcript):
                                transcript_file = potential_transcript
                                break
                        
                        # Get label
                        label = label_map.get(audio_file, None)
                        if label is None:
                            # Try to infer from directory structure
                            if 'dementia' in split_dir.lower() or 'ad' in split_dir.lower():
                                label = 1
                            elif 'control' in split_dir.lower():
                                label = 0
                        
                        records.append({
                            'subject_id': subject_id,
                            'audio_path': audio_path,
                            'transcript_path': transcript_file,
                            'label': label,
                            'dataset': 'ADReSS',
                            'split_original': split_type
                        })
        
        return pd.DataFrame(records)
    
    def load_dementiabank_dataset(self, data_path: str) -> pd.DataFrame:
        """
        Load DementiaBank (Pitt Corpus) dataset format.
        
        Args:
            data_path: Path to DementiaBank dataset directory
            
        Returns:
            DataFrame with metadata
        """
        records = []
        
        # DementiaBank typically has .wav and .cha files
        audio_files = list(Path(data_path).rglob('*.wav'))
        
        for audio_path in audio_files:
            subject_id = audio_path.stem
            transcript_path = audio_path.with_suffix('.cha')
            
            if not transcript_path.exists():
                transcript_path = None
            
            # Extract label from filename or metadata
            label = None
            if 'dementia' in str(audio_path).lower() or 'ad' in str(audio_path).lower():
                label = 1
            elif 'control' in str(audio_path).lower():
                label = 0
            
            records.append({
                'subject_id': subject_id,
                'audio_path': str(audio_path),
                'transcript_path': str(transcript_path) if transcript_path else None,
                'label': label,
                'dataset': 'DementiaBank'
            })
        
        return pd.DataFrame(records)
    
    def load_custom_dataset(self, metadata_path: str) -> pd.DataFrame:
        """
        Load custom dataset from metadata file.
        
        Args:
            metadata_path: Path to CSV/JSON metadata file
            
        Returns:
            DataFrame with metadata
        """
        if metadata_path.endswith('.csv'):
            df = pd.read_csv(metadata_path)
        elif metadata_path.endswith('.json'):
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path}")
        
        # Ensure required columns exist
        required_cols = [self.subject_id_col, self.audio_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add dataset identifier
        df['dataset'] = 'Custom'
        
        return df
    
    def load_dataset(self, dataset_name: str, data_path: str) -> pd.DataFrame:
        """
        Load dataset based on name.
        
        Args:
            dataset_name: Name of dataset (ADReSS, ADReSSo, DementiaBank, Custom)
            data_path: Path to dataset
            
        Returns:
            DataFrame with metadata
        """
        if dataset_name.lower() == 'adress':
            return self.load_adress_dataset(data_path)
        elif dataset_name.lower() == 'adresso':
            return self.load_adress_dataset(data_path)
        elif dataset_name.lower() == 'dementiabank':
            return self.load_dementiabank_dataset(data_path)
        elif dataset_name.lower() == 'custom':
            return self.load_custom_dataset(data_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def subject_level_split(
        self, 
        df: pd.DataFrame, 
        return_indices: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data at subject level to prevent data leakage.
        
        Args:
            df: DataFrame with subject_id and label columns
            return_indices: Whether to return indices instead of DataFrames
            
        Returns:
            Tuple of (train_df, val_df, test_df) or (train_idx, val_idx, test_idx)
        """
        # Get unique subjects
        unique_subjects = df[self.subject_id_col].unique()
        
        # Get labels for each subject (use first occurrence if multiple)
        subject_labels = df.groupby(self.subject_id_col)[self.label_col].first()
        
        # Split subjects (not individual samples)
        train_subjects, temp_subjects = train_test_split(
            unique_subjects,
            train_size=self.train_ratio,
            stratify=subject_labels if self.stratify else None,
            random_state=self.random_state
        )
        
        # Calculate adjusted ratios for val/test split
        adjusted_val_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
        
        val_subjects, test_subjects = train_test_split(
            temp_subjects,
            train_size=adjusted_val_ratio,
            stratify=subject_labels[temp_subjects] if self.stratify else None,
            random_state=self.random_state
        )
        
        # Create splits
        train_df = df[df[self.subject_id_col].isin(train_subjects)].reset_index(drop=True)
        val_df = df[df[self.subject_id_col].isin(val_subjects)].reset_index(drop=True)
        test_df = df[df[self.subject_id_col].isin(test_subjects)].reset_index(drop=True)
        
        # Verify no subject overlap
        train_subjects_set = set(train_df[self.subject_id_col])
        val_subjects_set = set(val_df[self.subject_id_col])
        test_subjects_set = set(test_df[self.subject_id_col])
        
        assert len(train_subjects_set & val_subjects_set) == 0, "Subject overlap between train and val!"
        assert len(train_subjects_set & test_subjects_set) == 0, "Subject overlap between train and test!"
        assert len(val_subjects_set & test_subjects_set) == 0, "Subject overlap between val and test!"
        
        if return_indices:
            train_idx = train_df.index.tolist()
            val_idx = val_df.index.tolist()
            test_idx = test_df.index.tolist()
            return train_idx, val_idx, test_idx
        
        return train_df, val_df, test_df
    
    def prepare_and_save(
        self, 
        dataset_name: str, 
        data_path: str, 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Load dataset, split it, and save the splits.
        
        Args:
            dataset_name: Name of dataset
            data_path: Path to raw dataset
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary with split statistics
        """
        # Load dataset
        print(f"Loading {dataset_name} dataset from {data_path}...")
        df = self.load_dataset(dataset_name, data_path)
        
        print(f"Loaded {len(df)} samples from {df[self.subject_id_col].nunique()} subjects")
        
        # Check for missing labels
        missing_labels = df[self.label_col].isna().sum()
        if missing_labels > 0:
            print(f"Warning: {missing_labels} samples have missing labels")
        
        # Split data
        print("Performing subject-level split...")
        train_df, val_df, test_df = self.subject_level_split(df)
        
        # Save splits
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, 'train.csv')
        val_path = os.path.join(output_dir, 'val.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Compute statistics
        stats = {
            'dataset': dataset_name,
            'total_samples': len(df),
            'total_subjects': df[self.subject_id_col].nunique(),
            'train': {
                'samples': len(train_df),
                'subjects': train_df[self.subject_id_col].nunique(),
                'path': train_path
            },
            'val': {
                'samples': len(val_df),
                'subjects': val_df[self.subject_id_col].nunique(),
                'path': val_path
            },
            'test': {
                'samples': len(test_df),
                'subjects': test_df[self.subject_id_col].nunique(),
                'path': test_path
            },
            'label_distribution': df[self.label_col].value_counts().to_dict(),
            'train_label_dist': train_df[self.label_col].value_counts().to_dict(),
            'val_label_dist': val_df[self.label_col].value_counts().to_dict(),
            'test_label_dist': test_df[self.label_col].value_counts().to_dict()
        }
        
        # Save statistics
        stats_path = os.path.join(output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Total subjects: {stats['total_subjects']}")
        print(f"\nTrain: {stats['train']['samples']} samples, {stats['train']['subjects']} subjects")
        print(f"Val:   {stats['val']['samples']} samples, {stats['val']['subjects']} subjects")
        print(f"Test:  {stats['test']['samples']} samples, {stats['test']['subjects']} subjects")
        print(f"\nLabel distribution: {stats['label_distribution']}")
        print(f"Splits saved to: {output_dir}")
        print("="*50)
        
        return stats


if __name__ == "__main__":
    # Example usage
    config_path = "configs/config.yaml"
    preparator = DataPreparator(config_path)
    
    # Prepare dataset
    dataset_name = preparator.dataset_config.get('name', 'ADReSS')
    data_path = preparator.dataset_config.get('data_path', 'data/raw')
    output_path = preparator.dataset_config.get('processed_path', 'data/processed')
    
    if os.path.exists(data_path):
        stats = preparator.prepare_and_save(dataset_name, data_path, output_path)
    else:
        print(f"Data path {data_path} does not exist.")
        print("Please download and place your dataset in the data/raw/ directory.")
        print("\nSupported datasets:")
        print("- ADReSS: https://dementia.talkbank.org/access/")
        print("- ADReSSo: https://dementia.talkbank.org/access/")
        print("- DementiaBank: https://dementia.talkbank.org/access/")
