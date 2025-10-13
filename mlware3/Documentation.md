# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** MLware
**Team Members:** [Kumarswami Kallimath, Jishnu Khargharia, Shreeya H M, Aviral Sharma]  
**Submission Date:** October 13, 2025

---

## 1. Executive Summary

This solution implements a transformer-based text regression model for product price prediction using catalog descriptions. We fine-tuned the sentence-transformers/all-MiniLM-L6-v2 model with a custom regression head on the training data, achieving competitive SMAPE scores through log-space transformation and careful regularization strategies.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

The challenge involves predicting product prices from catalog text descriptions. Key insights from our exploratory data analysis:

**Key Observations:**
- Product prices have a wide distribution requiring log transformation for stable training
- Text descriptions vary in length and quality, requiring robust preprocessing
- Missing catalog content values were filled with default text to maintain data integrity
- A 90-10 train-validation split was used to monitor model performance

### 2.2 Solution Strategy

**Approach Type:** Single Model (Transformer-based Regression)  
**Core Innovation:** Fine-tuned sentence transformer with custom multi-layer regression head, trained on log-transformed prices for improved stability and convergence.

The solution leverages transfer learning from a pre-trained sentence embedding model, adding a specialized regression architecture optimized for price prediction tasks.

---

## 3. Model Architecture

### 3.1 Architecture Overview

```
Input Text (catalog_content)
    ↓
Text Preprocessing & Cleaning
    ↓
Tokenization (max_length=256)
    ↓
Transformer Encoder (sentence-transformers/all-MiniLM-L6-v2)
    ↓
[CLS] Token Pooling
    ↓
Dropout Layer (0.3)
    ↓
Multi-Layer Regression Head
    ├── Linear(hidden_size → 512) + ReLU + Dropout(0.3)
    ├── Linear(512 → 256) + ReLU + Dropout(0.3)
    └── Linear(256 → 1)
    ↓
Log-Price Prediction
    ↓
Exponentiation (np.expm1) → Final Price
```

### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing steps:
  - Fill missing values with "No description available"
  - Remove newlines and carriage returns
  - Normalize whitespace
  - Apply log1p transformation to target prices
- Model type: `sentence-transformers/all-MiniLM-L6-v2` (AutoModel from Hugging Face)
- Key parameters:
  - Max sequence length: 256 tokens
  - Tokenization: Padding + truncation enabled
  - Batch size: 32

**Regression Head Architecture:**
- Three-layer feed-forward network
- Hidden dimensions: hidden_size → 512 → 256 → 1
- Activation: ReLU
- Regularization: Dropout (0.3) after each layer
- Output: Single continuous value (log-price)

**Training Configuration:**
- Loss function: MSE (Mean Squared Error) on log-transformed prices
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Scheduler: Linear warmup (500 steps) + linear decay
- Training epochs: 15
- Gradient clipping: Max norm 1.0
- Random seed: 42 (for reproducibility)

---

## 4. Data Processing

### 4.1 Input Data Requirements

**Training Data (train.csv):**
- Required columns: `catalog_content`, `price`
- Data split: 90% training, 10% validation
- Missing value handling: Fillna with default text

**Test Data (test.csv):**
- Required columns: `sample_id`, `catalog_content`
- Output format: CSV with `sample_id` and `price` columns

### 4.2 Preprocessing Pipeline

1. **Text Cleaning:**
   - Convert all text to string type
   - Replace newlines and carriage returns with spaces
   - Normalize multiple whitespaces to single space

2. **Target Transformation:**
   - Apply `log1p(price)` transformation during training
   - Apply `expm1(predicted_log_price)` during inference
   - Clip negative predictions to minimum value of 0.01

3. **Tokenization:**
   - AutoTokenizer from pre-trained model
   - Max length: 256 tokens
   - Padding: 'max_length'
   - Truncation: Enabled
   - Return format: PyTorch tensors

---

## 5. Model Performance

### 5.1 Validation Results

- **SMAPE Score:** 48.86%
- **Validation Loss:** 0.4534
- **Training Loss:** 0.3280
- **Other Metrics:**
  - MSE Loss on log-transformed prices
  - Model converges over 15 epochs with learning rate warmup
  - Training throughput: ~14.05 iterations/second
  - Validation throughput: ~44.28 iterations/second

**Best Model Training Results:**
```
Training: 100% |████████████████████| 2110/2110 [02:30<00:00, 14.05it/s, loss=0.4928]
Validation: 100% |██████████████████| 235/235 [00:05<00:00, 44.28it/s]

Results:
  Train Loss: 0.3280
  Val Loss: 0.4534
  Val SMAPE: 48.86%

  ✓ New best model! Saving...
```

### 5.2 Model Checkpointing

The training script saves:
- `best_model.pt`: Complete checkpoint including:
  - Model state dict
  - Optimizer state dict
  - Scheduler state dict
  - Best validation loss and SMAPE
  - Model configuration (model_name, max_length, dropout)
- Tokenizer files for inference
- `training_history.json`: Per-epoch metrics for analysis

### 5.3 Inference Validation

The prediction script performs six validation checks:
1. ✓ Number of predictions matches test data
2. ✓ No missing values in output
3. ✓ All prices are positive
4. ✓ All sample_ids are unique
5. ✓ Sample_ids match test data exactly
6. ✓ Price column has correct data type (float)

---

## 6. Implementation Details

### 6.1 Files Structure

- `script1.py`: Training and fine-tuning script
  - Loads train.csv
  - Preprocesses data and applies transformations
  - Fine-tunes model and saves best checkpoint
  - Tracks training metrics

- `script2.py`: Inference script
  - Loads test.csv
  - Loads fine-tuned model from checkpoint
  - Generates predictions
  - Validates and exports output.csv

- `finetuned_price_model/`: Model artifacts directory
  - `best_model.pt`: Model checkpoint
  - Tokenizer configuration files
  - `training_history.json`: Training metrics

### 6.2 Dependencies

Required Python packages:
```
transformers
datasets
torch
pandas
numpy
scikit-learn
sentencepiece
accelerate
tqdm
```

### 6.3 Hardware Requirements

- **Recommended:** CUDA-enabled GPU for faster training
- **Fallback:** CPU training supported (slower)
- Both scripts automatically detect and use available hardware

---

## 7. Usage Instructions

### 7.1 Training the Model

```bash
# Ensure train.csv is in the working directory
python script1.py
```

**Expected outputs:**
- Model directory: `finetuned_price_model/`
- Training logs with per-epoch metrics
- Best validation SMAPE reported at completion

### 7.2 Generating Predictions

```bash
# Ensure test.csv is in the working directory
# Ensure finetuned_price_model/ exists from training
python script2.py
```

**Expected outputs:**
- `output.csv`: Predictions with columns [sample_id, price]
- Prediction statistics and validation check results
- Ready for competition submission

### 7.3 Configuration Customization

Modify the `Config` class in each script to adjust:
- `model_name`: Change base transformer model
- `batch_size`: Adjust based on memory constraints
- `learning_rate`: Fine-tune learning dynamics
- `num_epochs`: Extend training duration
- `max_length`: Modify sequence length
- `dropout`: Adjust regularization strength

---

## 8. Results Analysis

### 8.1 Training Insights

- Model uses log-space transformation for price stability
- Dropout (0.3) provides effective regularization
- Linear warmup scheduler improves early training stability
- Best model selected based on validation loss
- SMAPE metric calculated on original price scale

### 8.2 Prediction Statistics

The inference script provides comprehensive statistics:
- Price distribution across ranges (<$10, $10-50, $50-100, $100-500, ≥$500)
- Mean, median, min, max, and standard deviation
- Validation checks ensure submission quality

---

## 9. Conclusion

This solution demonstrates effective transfer learning for price prediction from text descriptions. By leveraging a pre-trained sentence transformer with a custom regression architecture, we achieved stable training through log-space transformations and careful regularization. The pipeline includes comprehensive validation checks and produces submission-ready outputs. Key learnings include the importance of price transformation, appropriate model capacity, and robust text preprocessing for handling diverse catalog descriptions.

---

## Appendix

### A. Code Artifacts

**Script 1 (Training):**
- 14 step pipeline from data loading to model checkpointing
- Implements custom Dataset and Model classes
- Includes SMAPE calculation and training history tracking

**Script 2 (Inference):**
- 13 step pipeline from data loading to output validation
- Recreates identical model architecture
- Comprehensive validation checks before submission

### B. Model Architecture Details

**Total Parameters:** ~23M (varies with base model)
- Pre-trained transformer encoder: ~22M parameters (all trainable)
- Custom regression head: ~1M parameters
- All parameters fine-tuned during training

**Computational Requirements:**
- Training time: ~15-30 minutes per epoch (GPU-dependent)
- Inference time: Milliseconds per sample
- Memory: ~2GB GPU memory for batch_size=32

### C. Additional Results

Training history saved in `training_history.json` includes:
- `train_loss`: Per-epoch training loss values
- `val_loss`: Per-epoch validation loss values
- `val_smape`: Per-epoch SMAPE on original price scale

Use this data to:
- Plot learning curves
- Diagnose overfitting/underfitting
- Optimize hyperparameters
- Compare different model configurations

---

**Note:** This documentation covers the complete implementation of a transformer-based price prediction solution. The modular design allows for easy experimentation with different base models, hyperparameters, and preprocessing strategies.
