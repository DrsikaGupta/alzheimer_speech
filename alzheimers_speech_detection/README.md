# Speech-Based Early Detection of Alzheimer's Disease Using Machine Learning

## Project Overview

This project implements an end-to-end system for analyzing speech recordings and transcripts to identify patterns associated with Alzheimer's disease (AD) and Mild Cognitive Impairment (MCI). The system uses strict subject-level train/validation/test separation to prevent data leakage and provides explainable predictions through a web interface.

## Key Features

- **Multi-modal Analysis**: Combines acoustic features from audio and linguistic features from transcripts
- **Subject-Level Separation**: Ensures no data leakage by keeping all samples from the same subject in a single split
- **Pretrained Embeddings**: Leverages wav2vec 2.0, HuBERT, Whisper for audio; BERT, Sentence Transformers for text
- **Baseline & Advanced Models**: From Logistic Regression to multimodal fusion architectures
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Sensitivity, Specificity, Calibration
- **Explainability**: SHAP, LIME, Permutation Importance, Attention Visualization
- **Web Interface**: Streamlit/Gradio-based UI for real-time predictions
- **REST API**: FastAPI backend for integration
- **Docker Deployment**: Containerized deployment ready
- **Ethical Safeguards**: Privacy protection, fairness considerations, clear disclaimers

## Directory Structure

```
alzheimers_speech_detection/
├── data/                    # Dataset storage (ADReSS, ADReSSo, DementiaBank)
├── notebooks/               # Jupyter notebooks for exploration and experiments
├── src/
│   ├── preprocessing/       # Audio and transcript preprocessing
│   ├── features/            # Feature extraction (acoustic, linguistic, embeddings)
│   ├── models/              # Model definitions and training
│   ├── evaluation/          # Metrics and evaluation pipelines
│   ├── explainability/      # SHAP, LIME, feature importance
│   └── api/                 # FastAPI and Streamlit applications
├── tests/                   # Unit and integration tests
├── docs/                    # Documentation
├── configs/                 # Configuration files
├── experiments/             # Experiment logs and results
└── deployment/              # Docker and deployment scripts
```

## Datasets

This project supports:
- **ADReSS** (Alzheimer's Dementia Recognition through Spontaneous Speech)
- **ADReSSo** (ADReSS Out-of-domain)
- **DementiaBank** (Pitt Corpus)
- **Custom datasets** with proper formatting

*Note: Due to licensing restrictions, actual dataset files are not included. Users must obtain datasets from:*
- ADReSS/ADReSSo: https://dementia.talkbank.org/access/
- DementiaBank: https://dementia.talkbank.org/access/

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation
```bash
python src/preprocessing/prepare_data.py --dataset ADReSS --data_path data/raw
```

### 2. Feature Extraction
```bash
python src/features/extract_features.py --config configs/features.yaml
```

### 3. Model Training
```bash
python src/models/train.py --config configs/model_config.yaml
```

### 4. Evaluation
```bash
python src/evaluation/evaluate.py --model_path experiments/best_model.pkl
```

### 5. Web Interface
```bash
streamlit run src/api/app.py
```

### 6. API Server
```bash
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

## Performance Targets

- **Target Accuracy**: >90%
- **Target AUC**: >0.95

*Important: These targets are aspirational. Actual performance depends on dataset quality, size, and characteristics. All results are reported with strict subject-level separation.*

## Ethical Considerations

⚠️ **DISCLAIMER**: This system is a **research screening/decision-support tool** and **NOT** a medical diagnostic system.

- Requires informed consent for any clinical use
- Results must be interpreted by qualified healthcare professionals
- Privacy protection mechanisms are implemented
- Fairness across demographics is monitored
- Not intended for standalone diagnosis

## License

MIT License - See LICENSE file for details.

## Citation

If you use this project in your research, please cite appropriately based on the datasets and methods used.

## Contact

For questions and collaborations, please open an issue on GitHub.
