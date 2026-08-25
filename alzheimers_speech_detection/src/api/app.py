"""
Streamlit Web Interface for Alzheimer's Speech Detection

This module provides an interactive web interface for:
- Uploading audio files and transcripts
- Viewing predictions and confidence scores
- Understanding model explanations
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import tempfile
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    st.set_page_config(
        page_title="Alzheimer's Speech Detection",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Speech-Based Early Detection of Alzheimer's Disease")
    st.markdown("""
    This system analyzes speech recordings and transcripts to identify patterns 
    associated with Alzheimer's disease or Mild Cognitive Impairment.
    
    **⚠️ DISCLAIMER**: This is a **research screening/decision-support tool** and 
    **NOT** a medical diagnostic system. Results must be interpreted by qualified 
    healthcare professionals.
    """)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost", "Logistic Regression", "SVM"]
    )
    
    # Show feature importance
    show_explanations = st.sidebar.checkbox("Show Explanations", value=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎤 Audio Analysis", "📝 Transcript Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Audio Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload audio file (.wav, .mp3)",
            type=["wav", "mp3", "flac"]
        )
        
        if uploaded_file is not None:
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(uploaded_file.getvalue())
                temp_path = f.name
            
            st.audio(uploaded_file)
            
            if st.button("Analyze Audio"):
                with st.spinner("Processing audio..."):
                    try:
                        # Load preprocessor and extract features
                        from src.preprocessing.audio_preprocessing import AudioPreprocessor
                        from src.features.acoustic_features import AcousticFeatureExtractor
                        
                        config_path = "configs/config.yaml"
                        
                        # Preprocess
                        preprocessor = AudioPreprocessor(config_path)
                        audio, info = preprocessor.preprocess(temp_path, return_info=True)
                        
                        # Extract features
                        extractor = AcousticFeatureExtractor(config_path)
                        features = extractor.extract_feature_vector(audio, 16000)
                        
                        # Load model and predict
                        model_path = f"experiments/models/{model_type.lower().replace(' ', '_')}_model.pkl"
                        scaler_path = "experiments/models/scaler.pkl"
                        
                        if os.path.exists(model_path) and os.path.exists(scaler_path):
                            scaler = joblib.load(scaler_path)
                            model = joblib.load(model_path)
                            
                            features_scaled = scaler.transform([features])
                            prediction = model.predict(features_scaled)[0]
                            probability = model.predict_proba(features_scaled)[0]
                            
                            # Display results
                            st.subheader("Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Prediction",
                                    "AD/MCI Detected" if prediction == 1 else "Control",
                                    delta=None
                                )
                            
                            with col2:
                                confidence = max(probability) * 100
                                st.metric(
                                    "Confidence",
                                    f"{confidence:.1f}%"
                                )
                            
                            # Probability breakdown
                            st.progress(float(probability[1]))
                            st.caption(f"Probability of AD/MCI: {probability[1]*100:.1f}%")
                            
                            # Show explanations
                            if show_explanations:
                                st.subheader("Feature Analysis")
                                
                                # Get top features
                                if hasattr(model, 'feature_importances_'):
                                    importance = model.feature_importances_
                                    top_indices = np.argsort(importance)[-10:][::-1]
                                    
                                    feature_names = [f"Feature_{i}" for i in range(len(importance))]
                                    
                                    df_features = pd.DataFrame({
                                        'Feature': [feature_names[i] for i in top_indices],
                                        'Importance': importance[top_indices]
                                    })
                                    
                                    st.bar_chart(df_features.set_index('Feature'))
                        else:
                            st.warning("Model not found. Please train models first.")
                            
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                    finally:
                        os.unlink(temp_path)
    
    with tab2:
        st.header("Transcript Analysis")
        
        transcript_text = st.text_area(
            "Enter or paste transcript text",
            height=200,
            placeholder="The patient described their daily routine..."
        )
        
        if transcript_text and st.button("Analyze Transcript"):
            with st.spinner("Analyzing transcript..."):
                try:
                    from src.features.linguistic_features import LinguisticFeatureExtractor
                    
                    config_path = "configs/config.yaml"
                    extractor = LinguisticFeatureExtractor(config_path)
                    
                    # Extract features
                    features = extractor.extract_feature_vector(transcript_text)
                    
                    st.subheader("Linguistic Features Extracted")
                    st.write(f"Number of features: {len(features)}")
                    
                    # Sample analysis
                    doc_length = len(transcript_text.split())
                    st.metric("Word Count", doc_length)
                    
                    # Note: Full prediction would require trained model
                    st.info("Full prediction requires a trained text model.")
                    
                except Exception as e:
                    st.error(f"Error analyzing transcript: {str(e)}")
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### Purpose
        This system is designed to assist researchers and clinicians in early 
        detection of Alzheimer's disease through speech analysis.
        
        ### How It Works
        1. **Audio Processing**: Extracts acoustic features including MFCCs, 
           pitch, pause patterns, and voice quality measures
        2. **Text Analysis**: Analyzes linguistic features including lexical 
           diversity, syntactic complexity, and repetition patterns
        3. **Machine Learning**: Uses trained models to classify speech patterns
        
        ### Supported Datasets
        - ADReSS (Alzheimer's Dementia Recognition through Spontaneous Speech)
        - ADReSSo (Out-of-domain)
        - DementiaBank (Pitt Corpus)
        
        ### Performance Targets
        - Target Accuracy: >90%
        - Target AUC: >0.95
        
        *Note: Actual performance depends on dataset characteristics and may vary.*
        
        ### Ethical Considerations
        - Requires informed consent for clinical use
        - Results must be interpreted by qualified professionals
        - Not intended for standalone diagnosis
        - Privacy protection mechanisms implemented
        
        ### Citation
        If using this system in research, please cite appropriately based on 
        datasets and methods used.
        """)
        
        st.subheader("System Information")
        st.json({
            "Version": "1.0.0",
            "Framework": "Streamlit",
            "Models": ["Random Forest", "XGBoost", "Logistic Regression", "SVM"],
            "Features": ["Acoustic", "Linguistic", "Pretrained Embeddings"]
        })


if __name__ == "__main__":
    main()
