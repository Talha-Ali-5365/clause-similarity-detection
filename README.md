# Legal Clause Similarity Detection

Deep Learning Assignment 2 - NUCES FAST

## Overview

This project implements two baseline architectures for semantic similarity detection between legal clauses:

1. **BiLSTM Siamese Network**: Uses bidirectional LSTM layers to encode clause pairs and compute similarity
2. **Attention-based Encoder Network**: Uses attention mechanisms to focus on important parts of clauses

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

**Note**: The code automatically detects and uses GPU if available (CUDA-compatible GPU with TensorFlow-GPU installed). Training will fall back to CPU if no GPU is detected.

## Dataset Structure

The dataset consists of 395 CSV files in the `archive/` folder, containing 150,881 legal clauses across 395 categories. Each CSV file contains:
- `clause_text`: The legal clause text
- `clause_type`: The category/type of the clause (same as filename)

## Usage

Run the main script:

```bash
python legal_clause_similarity.py
```

The script will:
1. **Detect GPU**: Automatically configure GPU settings if available
2. **Load Data**: Load all CSV files from the `archive/` folder
3. **Create Pairs**: Generate positive pairs (same category) and negative pairs (different categories)
4. **Preprocess**: Clean text, tokenize, and create sequences
5. **Train Models**: Train both models on the dataset
6. **Evaluate**: Compute comprehensive metrics and generate visualizations
7. **Display Results**: Show qualitative examples of predictions

## Results

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|--------|---------------|
| **BiLSTM Siamese Network** | 99.85% | 99.69% | 100.00% | 99.85% | 99.98% | 99.97% | ~8.3 min |
| **Attention-based Encoder** | 98.66% | 98.03% | 99.28% | 98.65% | 99.60% | 99.32% | ~9.3 min |

### Key Findings

- **BiLSTM Model**: Achieved superior performance with 99.85% accuracy and perfect recall (100%)
- **Attention Model**: Slightly lower performance but still excellent at 98.66% accuracy
- **Training Efficiency**: Both models converged quickly with early stopping (16 epochs for BiLSTM, 10 epochs for Attention)
- **GPU Acceleration**: Successfully utilized Tesla T4 GPU, significantly speeding up training

## Output Files

The script generates the following visualization files:

- `bilstm_training_history.png`: Training/validation loss and accuracy curves for BiLSTM model
- `attention_training_history.png`: Training/validation loss and accuracy curves for Attention model
- `model_comparison_curves.png`: ROC and Precision-Recall curves comparison between both models

## Model Architectures

### BiLSTM Siamese Network
- **Input**: Two clause texts (max 200 tokens each)
- **Embedding Layer**: 128-dimensional embeddings (vocab size: 10,000)
- **Encoder**: Bidirectional LSTM (128 units each direction = 256 total)
- **Feature Combination**: Concatenation, element-wise difference, and product
- **Classifier**: Two dense layers (64 → 32 units) with dropout (0.3)
- **Output**: Binary classification (similar/dissimilar)
- **Total Parameters**: 1,610,881 (6.15 MB)

### Attention-based Encoder Network
- **Input**: Two clause texts (max 200 tokens each)
- **Embedding Layer**: 128-dimensional embeddings (vocab size: 10,000)
- **Encoder**: Bidirectional LSTM with sequence output (128 units each direction)
- **Attention**: Multi-head attention mechanism (4 heads)
- **Pooling**: Global average pooling
- **Feature Combination**: Concatenation, element-wise difference, and product
- **Classifier**: Two dense layers (64 → 32 units) with dropout (0.3)
- **Output**: Binary classification (similar/dissimilar)
- **Total Parameters**: 2,137,217 (8.15 MB)

## Configuration

You can modify parameters in the `Config` class:

```python
# Text preprocessing
MAX_VOCAB_SIZE = 10000          # Maximum vocabulary size
MAX_SEQUENCE_LENGTH = 200        # Maximum sequence length
EMBEDDING_DIM = 128              # Embedding dimension

# Model architecture
LSTM_UNITS = 128                 # Number of LSTM units
DENSE_UNITS = 64                 # Dense layer units
DROPOUT_RATE = 0.3               # Dropout rate

# Training
BATCH_SIZE = 32                  # Training batch size
EPOCHS = 20                      # Maximum training epochs
VALIDATION_SPLIT = 0.2           # Validation set proportion
TEST_SPLIT = 0.1                 # Test set proportion

# Pair generation
POSITIVE_PAIRS_PER_CATEGORY = 50 # Positive pairs per category
NEGATIVE_PAIRS_PER_CATEGORY = 50 # Negative pairs per category
```

## Evaluation Metrics

The models are evaluated using comprehensive metrics:

- **Accuracy**: Overall classification accuracy (99.85% BiLSTM, 98.66% Attention)
- **Precision**: Precision for similar pairs (99.69% BiLSTM, 98.03% Attention)
- **Recall**: Recall for similar pairs (100% BiLSTM, 99.28% Attention)
- **F1-Score**: Harmonic mean of precision and recall (99.85% BiLSTM, 98.65% Attention)
- **ROC-AUC**: Area under ROC curve (99.98% BiLSTM, 99.60% Attention)
- **PR-AUC**: Area under Precision-Recall curve (99.97% BiLSTM, 99.32% Attention)

## Dataset Statistics

- **Total CSV Files**: 395
- **Total Clauses**: 150,881
- **Categories**: 395 distinct clause types
- **Training Pairs**: 27,649 (70%)
- **Validation Pairs**: 7,900 (20%)
- **Test Pairs**: 3,951 (10%)
- **Positive/Negative Balance**: 50/50 (balanced dataset)

## GPU Support

The code automatically detects and configures GPU usage:

- **GPU Detection**: Automatically lists available GPUs at startup
- **Memory Growth**: Enables dynamic memory allocation to prevent OOM errors
- **Device Placement**: Explicitly places models on GPU for optimal performance
- **Fallback**: Automatically falls back to CPU if no GPU is detected

Example GPU output:
```
✓ Found 1 GPU(s):
  GPU 0: /physical_device:GPU:0
    Memory growth enabled
✓ GPU will be used for training
  TensorFlow version: 2.19.0
  GPU Details: {'compute_capability': (7, 5), 'device_name': 'Tesla T4'}
```

## Qualitative Results

The script provides examples of:
- **Correctly Predicted Pairs**: Shows cases where models correctly identify similar/dissimilar clauses
- **Incorrectly Predicted Pairs**: Shows edge cases where models make mistakes (useful for analysis)

Examples demonstrate that models successfully capture semantic similarity even when clauses use different wording but express the same legal principle.

## Technical Details

- **Framework**: TensorFlow 2.x / Keras
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Binary cross-entropy
- **Regularization**: Dropout (0.3) and early stopping
- **Callbacks**: Early stopping (patience=5) and learning rate reduction (patience=3)

## Notes

- The high accuracy (99%+) suggests that legal clauses within categories are semantically very similar, while clauses across categories are distinctly different
- Both models demonstrate excellent performance, with BiLSTM achieving slightly better results
- The attention mechanism adds computational overhead but provides interpretability benefits
- Training converges quickly due to the clear separation between similar and dissimilar pairs

