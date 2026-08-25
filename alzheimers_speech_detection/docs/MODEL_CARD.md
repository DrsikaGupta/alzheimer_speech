# Model Card: Alzheimer's Speech Detection System

## Model Details

### Model Description
This system uses machine learning to analyze speech patterns for early detection of Alzheimer's disease and Mild Cognitive Impairment (MCI). It processes both audio recordings and text transcripts to extract acoustic and linguistic features.

### Model Type
- **Architecture**: Ensemble of baseline ML models (Random Forest, XGBoost, SVM, Logistic Regression) with optional deep learning fusion
- **Input**: Audio files (.wav, .mp3) and/or text transcripts
- **Output**: Binary classification (Control vs. AD/MCI) with probability scores

### Model Versions
- Version: 1.0.0
- Release Date: 2024

## Intended Use

### Primary Use Cases
- Research screening tool for cognitive impairment studies
- Decision support for healthcare professionals
- Early detection research in clinical settings

### Out-of-Scope Uses
⚠️ **This model is NOT intended for:**
- Standalone medical diagnosis
- Self-diagnosis by patients
- Treatment decisions without professional oversight
- Legal or insurance determinations

## Data

### Training Data
- **Datasets**: ADReSS, ADReSSo, DementiaBank (Pitt Corpus)
- **Subject-Level Split**: Strict separation to prevent data leakage
- **Demographics**: Varies by dataset; see individual dataset documentation

### Evaluation Data
- Held-out test set with complete subject separation
- Cross-validation with stratified k-fold

## Performance Metrics

### Target Metrics
- Accuracy: >90% (target)
- ROC-AUC: >0.95 (target)

### Actual Performance
*Note: Actual performance depends on dataset characteristics and must be evaluated under rigorous subject-independent conditions.*

**Important**: If targets cannot be achieved under strict evaluation, actual performance will be reported with full transparency about limitations.

### Metrics Reported
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Sensitivity, Specificity
- Calibration metrics
- Confusion matrix analysis

## Limitations

### Known Limitations
1. **Dataset Bias**: Performance may vary across demographics not well-represented in training data
2. **Language**: Primarily trained on English speech
3. **Recording Quality**: Performance degrades with poor audio quality
4. **Comorbidities**: May not distinguish AD from other cognitive disorders
5. **Early Stage**: Detection of very early MCI may be less reliable

### Ethical Considerations
- Requires informed consent for any clinical application
- Results must be interpreted by qualified healthcare professionals
- Privacy protection mechanisms must be implemented
- Potential for bias across demographic groups requires monitoring

## Fairness

### Demographic Considerations
- Age: Performance may vary across age groups
- Gender: Balanced representation recommended
- Education: Educational background affects linguistic features
- Native Language: Non-native speakers may show different patterns

### Mitigation Strategies
- Stratified sampling during training
- Class weighting for imbalanced datasets
- Regular fairness audits recommended
- Continuous monitoring across demographic groups

## Explainability

### Available Explanations
- Feature importance (permutation-based)
- SHAP values (sample-based)
- LIME explanations (individual predictions)
- Attention visualization (for deep learning models)

### Key Features
Acoustic features often include:
- MFCCs and spectral features
- Pause patterns and speech rate
- Pitch variability
- Voice quality measures (jitter, shimmer)

Linguistic features often include:
- Lexical diversity measures
- Syntactic complexity
- Repetition patterns
- Filler word usage

## Usage Guidelines

### Input Requirements
- Audio: 16kHz sample rate recommended, <30 seconds
- Text: Minimum 50 words recommended
- Format: WAV, MP3 for audio; plain text for transcripts

### Output Interpretation
- Probability scores indicate model confidence
- Higher probability = higher likelihood of AD/MCI
- Results should be considered alongside other clinical assessments
- False positives and negatives are possible

### Recommended Workflow
1. Obtain informed consent
2. Collect speech sample following standardized protocol
3. Run analysis through the system
4. Review results with explanations
5. Integrate with other clinical assessments
6. Document decision-making process

## Regulatory Status

⚠️ **Research Use Only**
- Not FDA approved
- Not CE marked
- Not intended for diagnostic use
- Requires IRB approval for research studies

## Citation

When using this system in research, please cite:
- Original datasets (ADReSS, DementiaBank)
- Methods used (feature extraction, models)
- This implementation appropriately

## Contact & Support

- Documentation: See project README
- Issues: Report via GitHub issues
- Updates: Check repository for latest version

## License

MIT License - See LICENSE file for details.

---

**Last Updated**: 2024
**Version**: 1.0.0
