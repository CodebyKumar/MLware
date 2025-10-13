"""
SCRIPT 3: PREDICT ON FINAL TEST DATA
This script loads the fine-tuned model and generates predictions
on test.csv for final submission
"""

# ============================================================================
# STEP 1: Import Libraries
# ============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm
import os

# ============================================================================
# STEP 2: Configuration
# ============================================================================
class Config:
    # Paths
    test_path = "test.csv"
    model_load_path = "finetuned_price_model"
    output_path = "output.csv"

    # Model configuration
    batch_size = 32

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
print(f"Using device: {config.device}")

# ============================================================================
# STEP 3: Recreate Model Architecture
# ============================================================================
class PricePredictionModel(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super(PricePredictionModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.transformer.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        price = self.regressor(pooled_output)
        return price

# ============================================================================
# STEP 4: Create Dataset Class
# ============================================================================
class PriceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# ============================================================================
# STEP 5: Load Test Data
# ============================================================================
print("\n" + "="*70)
print("LOADING TEST DATA")
print("="*70)

test_df = pd.read_csv(config.test_path)

print(f"Test data shape: {test_df.shape}")
print(f"Test columns: {test_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(test_df.head(3))

# Check for sample_id column
if 'sample_id' not in test_df.columns:
    raise ValueError("Test data must contain 'sample_id' column!")

print(f"\nTotal test samples: {len(test_df)}")
print(f"Unique sample_ids: {test_df['sample_id'].nunique()}")

# Handle missing values
test_df['catalog_content'] = test_df['catalog_content'].fillna("No description available")

# Clean text
def clean_text(text):
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text

test_df['catalog_content'] = test_df['catalog_content'].apply(clean_text)

print(f"\n✓ Data preprocessing completed")

# ============================================================================
# STEP 6: Load Fine-tuned Model
# ============================================================================
print("\n" + "="*70)
print("LOADING FINE-TUNED MODEL")
print("="*70)

# Check if model exists
if not os.path.exists(f"{config.model_load_path}/best_model.pt"):
    raise FileNotFoundError(
        f"Model not found at {config.model_load_path}/best_model.pt\n"
        f"Please run Script 1 to train the model first!"
    )

# Load checkpoint (weights_only=False for compatibility with saved optimizers and config)
checkpoint = torch.load(f"{config.model_load_path}/best_model.pt", map_location=config.device, weights_only=False)
model_config = checkpoint['config']

print(f"Model: {model_config['model_name']}")
print(f"Max length: {model_config['max_length']}")
print(f"Dropout: {model_config['dropout']}")
print(f"\nModel Training Info:")
print(f"  Trained epochs: {checkpoint['epoch'] + 1}")
print(f"  Best validation loss: {checkpoint['val_loss']:.4f}")
print(f"  Best validation SMAPE: {checkpoint['val_smape']:.2f}%")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.model_load_path)

# Initialize model and load weights
print("Loading model weights...")
model = PricePredictionModel(model_config['model_name'], dropout=model_config['dropout'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(config.device)
model.eval()

print("\n✓ Model loaded successfully!")

# ============================================================================
# STEP 7: Create Data Loader
# ============================================================================
print("\n" + "="*70)
print("PREPARING DATA LOADER")
print("="*70)

test_dataset = PriceDataset(
    test_df['catalog_content'].values,
    tokenizer,
    model_config['max_length']
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=2
)

print(f"Total samples: {len(test_dataset)}")
print(f"Total batches: {len(test_loader)}")
print(f"Batch size: {config.batch_size}")

# ============================================================================
# STEP 8: Generate Predictions
# ============================================================================
print("\n" + "="*70)
print("GENERATING PREDICTIONS ON TEST DATA")
print("="*70)

predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)

        preds = model(input_ids, attention_mask)
        predictions.extend(preds.cpu().numpy())

# Convert from log space back to original price
predictions = np.expm1(np.array(predictions).flatten())

# Ensure all predictions are positive
predictions = np.maximum(predictions, 0.01)

print(f"\n✓ Predictions generated: {len(predictions)}")

# ============================================================================
# STEP 9: Statistics and Validation
# ============================================================================
print("\n" + "="*70)
print("PREDICTION STATISTICS")
print("="*70)

print(f"\nPrice Statistics:")
print(f"  Count: {len(predictions)}")
print(f"  Min: ${predictions.min():.2f}")
print(f"  Max: ${predictions.max():.2f}")
print(f"  Mean: ${predictions.mean():.2f}")
print(f"  Median: ${np.median(predictions):.2f}")
print(f"  Std Dev: ${predictions.std():.2f}")

print(f"\nPrice Distribution:")
print(f"  < $10: {(predictions < 10).sum()} ({(predictions < 10).sum()/len(predictions)*100:.1f}%)")
print(f"  $10-$50: {((predictions >= 10) & (predictions < 50)).sum()} ({((predictions >= 10) & (predictions < 50)).sum()/len(predictions)*100:.1f}%)")
print(f"  $50-$100: {((predictions >= 50) & (predictions < 100)).sum()} ({((predictions >= 50) & (predictions < 100)).sum()/len(predictions)*100:.1f}%)")
print(f"  $100-$500: {((predictions >= 100) & (predictions < 500)).sum()} ({((predictions >= 100) & (predictions < 500)).sum()/len(predictions)*100:.1f}%)")
print(f"  >= $500: {(predictions >= 500).sum()} ({(predictions >= 500).sum()/len(predictions)*100:.1f}%)")

# ============================================================================
# STEP 10: Create Output DataFrame
# ============================================================================
print("\n" + "="*70)
print("CREATING OUTPUT FILE")
print("="*70)

output_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': predictions
})

# Verify output format
print(f"\nOutput DataFrame:")
print(f"  Shape: {output_df.shape}")
print(f"  Columns: {output_df.columns.tolist()}")
print(f"\nFirst 10 predictions:")
print(output_df.head(10))

# ============================================================================
# STEP 11: Validation Checks
# ============================================================================
print("\n" + "="*70)
print("VALIDATION CHECKS")
print("="*70)

checks_passed = True

# Check 1: Number of predictions
if len(output_df) == len(test_df):
    print(f"✓ Check 1: Number of predictions matches test data ({len(output_df)})")
else:
    print(f"✗ Check 1: FAILED - Predictions: {len(output_df)}, Test data: {len(test_df)}")
    checks_passed = False

# Check 2: No missing values
if output_df.isna().sum().sum() == 0:
    print(f"✓ Check 2: No missing values")
else:
    print(f"✗ Check 2: FAILED - Found {output_df.isna().sum().sum()} missing values")
    checks_passed = False

# Check 3: All prices are positive
if (output_df['price'] > 0).all():
    print(f"✓ Check 3: All prices are positive")
else:
    print(f"✗ Check 3: FAILED - Found {(output_df['price'] <= 0).sum()} non-positive prices")
    checks_passed = False

# Check 4: All sample_ids are unique
if output_df['sample_id'].nunique() == len(output_df):
    print(f"✓ Check 4: All sample_ids are unique")
else:
    print(f"✗ Check 4: FAILED - Duplicate sample_ids found")
    checks_passed = False

# Check 5: Sample_ids match test data
if output_df['sample_id'].equals(test_df['sample_id']):
    print(f"✓ Check 5: Sample_ids match test data exactly")
else:
    print(f"✗ Check 5: FAILED - Sample_ids don't match test data")
    checks_passed = False

# Check 6: Correct data types
if output_df['price'].dtype in [np.float64, np.float32]:
    print(f"✓ Check 6: Price column has correct data type (float)")
else:
    print(f"✗ Check 6: FAILED - Price column type is {output_df['price'].dtype}")
    checks_passed = False

if checks_passed:
    print(f"\n{'='*70}")
    print("ALL VALIDATION CHECKS PASSED ✓")
    print(f"{'='*70}")
else:
    print(f"\n{'='*70}")
    print("SOME VALIDATION CHECKS FAILED ✗")
    print(f"{'='*70}")

# ============================================================================
# STEP 12: Save Output
# ============================================================================
print(f"\n" + "="*70)
print("SAVING OUTPUT")
print("="*70)

output_df.to_csv(config.output_path, index=False)

print(f"\n✓ Predictions saved to: {config.output_path}")
print(f"  File size: {os.path.getsize(config.output_path) / 1024:.2f} KB")

# ============================================================================
# STEP 13: Final Summary
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\n✓ Process Completed Successfully!")
print(f"\nOutput Details:")
print(f"  File: {config.output_path}")
print(f"  Total predictions: {len(output_df)}")
print(f"  Format: CSV with columns ['sample_id', 'price']")

print(f"\nPrediction Summary:")
print(f"  Mean price: ${predictions.mean():.2f}")
print(f"  Median price: ${np.median(predictions):.2f}")
print(f"  Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")

print(f"\n✓ Ready for submission!")
print(f"  Submit {config.output_path} to the competition platform")

print("\n" + "="*70)
print("PREDICTION PIPELINE COMPLETE")
print("="*70)
