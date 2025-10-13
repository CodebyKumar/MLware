# MLware Pricing Model Documentation

## Project Overview

This repository contains an implementation for product price prediction using catalog text data. The solution leverages transformer-based text encoding combined with engineered statistical features and advanced training techniques (SWA, AMP, gradient accumulation) to predict product prices (log-transformed during training).

The workspace contains:
- `script1.py` — training pipeline with extensive feature engineering, model definition (DeBERTa-based encoder + regressor), SWA training, and SMAPE evaluation.
- `script2.py` — inference pipeline that loads the saved model and tokenizer, extracts features, scales them, and produces CSV predictions.
- `Documentation_template.md` — original template used as a starting point.

## Quick Start

Prerequisites:
- Python 3.8+
- CUDA-capable GPU for training (optional but recommended)

Install dependencies (recommended in a virtualenv):

```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install core packages:

```bash
pip install transformers datasets torch pandas numpy scikit-learn sentencepiece accelerate -q
```

Training
1. Place your training CSV as `train.csv` in the repository root. The training script expects columns: `sample_id`, `catalog_content`, `price`, and `image_link` (image_link optional).
2. Run training:

```bash
python script1.py
```

Model checkpoints and artifacts will be saved to `a10g_max_perf_price_model/`.

Inference
1. Prepare `test.csv` with columns: `sample_id`, `catalog_content`.
2. Run inference:

```bash
python script2.py
```

The script will write `output.csv` with `sample_id` and predicted `price`.

## Data Format

Training CSV (`train.csv`) required columns:
- `sample_id` — Unique identifier
- `catalog_content` — Product description text
- `price` — Numeric price (used to compute log1p during training)
- `image_link` — Optional image URL

Test CSV (`test.csv`) required columns:
- `sample_id`
- `catalog_content`

## Feature Engineering

The project extracts a wide range of textual and statistical features from `catalog_content`. Key features include:
- text_length, word_count
- numeric extractions: max_number, min_number, avg_number, number_count
- unit/pack indicators: has_ounce, has_pound, has_pack, pack_size
- brand indicator: has_brand
- linguistic features: bullet_points, capital_ratio, digit_ratio, punctuation_count, unique_words_ratio, avg_word_length, sentence_count, avg_sentence_length

Features are normalized using `StandardScaler` during training and the scaler is saved to `a10g_max_perf_price_model/scaler.pkl` for inference.

## Model Architecture

Text encoder:
- `microsoft/deberta-v3-base` (AutoModel via HuggingFace)

Regressor:
- Multi-layer fully connected network combining the [CLS] token features with statistical features. Multiple linear layers with LayerNorm, ReLU, Dropout, ending with a single output neuron for log-price prediction.

Training techniques:
- Mixed precision training (AMP)
- Gradient accumulation to achieve effective large batch sizes
- Stochastic Weight Averaging (SWA) for improved generalization
- Cosine learning rate scheduler with warmup
- Huber loss used as training criterion

## Training Configuration (defaults in `script1.py` Config class)
- text_model: `microsoft/deberta-v3-base`
- max_length: 384
- physical_batch_size: 12
- accumulation_steps: 4 (effective batch size 48)
- learning_rate: scaled according to effective batch size
- num_epochs: 10
- swa_start_epoch: 6
- swa_lr: 1e-6
- warmup_ratio: 0.1
- dropout: 0.2
- weight_decay: 0.01

## Metrics
- Validation SMAPE (Symmetric Mean Absolute Percentage Error) is the primary metric. The training process saves the best model by SMAPE.

## Saved Artifacts
- `a10g_max_perf_price_model/best_model.pt` — checkpoint with model state dict and training config
- `a10g_max_perf_price_model/tokenizer` — HuggingFace tokenizer files
- `a10g_max_perf_price_model/scaler.pkl` — StandardScaler used for statistical features
- `a10g_max_perf_price_model/training_history.json` — training history

## Notes & Recommendations
- The training script expects `train.csv` in the current directory. If your dataset has different column names, adjust preprocessing accordingly.
- SWA is activated after `swa_start_epoch` and the script updates batch norm statistics before saving.
- For large datasets, increase `num_workers` in DataLoader and consider distributed training.
- If GPU memory is insufficient, reduce `physical_batch_size` and/or `max_length`.

## Troubleshooting
- FileNotFoundError for `train.csv` or `test.csv`: ensure file present in repo root.
- CUDA OOM: lower `physical_batch_size` or use gradient checkpointing.
- Tokenizer/model download failures: ensure connection to HuggingFace or pre-download models.

## Appendix: Key Functions and Files
- `script1.py` — training pipeline and model definitions
- `script2.py` — inference pipeline


---

Generated on October 13, 2025
