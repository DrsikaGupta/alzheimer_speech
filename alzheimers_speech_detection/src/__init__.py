cat > src/features/__init__.py << 'EOF'
from .acoustic_features import AcousticFeatureExtractor
from .linguistic_features import LinguisticFeatureExtractor
from .embedding_extractor import AudioEmbeddingExtractor, TextEmbeddingExtractor

__all__ = ['AcousticFeatureExtractor', 'LinguisticFeatureExtractor', 
           'AudioEmbeddingExtractor', 'TextEmbeddingExtractor']
EOF
cat > src/models/__init__.py << 'EOF'
from .train import BaselineModels, MultimodalFusionModel, train_model

__all__ = ['BaselineModels', 'MultimodalFusionModel', 'train_model']
EOF
cat > src/evaluation/__init__.py << 'EOF'
from .evaluate import Evaluator

__all__ = ['Evaluator']
EOF
cat > src/explainability/__init__.py << 'EOF'
from .explainer import ModelExplainer

__all__ = ['ModelExplainer']
EOF
echo "Init files created"
