import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Install required libraries silently if running in a notebook environment
try:
    import transformers, datasets, torch, pandas, numpy, sklearn, sentencepiece, accelerate
except ImportError:
    print("Installing required libraries...")
    os.system("pip install transformers datasets torch pandas numpy scikit-learn sentencepiece accelerate -q")
    import transformers, datasets, torch, pandas, numpy, sklearn, sentencepiece, accelerate

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import warnings
from tqdm.auto import tqdm
import gc
import json
import re
from torch.cuda.amp import GradScaler, autocast
import math
from sklearn.preprocessing import StandardScaler
import pickle
from collections import OrderedDict
from torch.optim.swa_utils import AveragedModel, SWALR 

# --- FIX 1: Add environment variable to reduce CUDA fragmentation ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 2: Set Random Seeds
# ============================================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch.cuda.current_device())

set_seed(42)

# ============================================================================
# STEP 3: Configuration (NVIDIA A10G MAX UTILIZATION)
# ============================================================================
class Config:
    # Model configuration
    text_model = "microsoft/deberta-v3-base"
    max_length = 384
    use_statistical_features = True

    # --- A10G MAX UTILIZATION FIXES (23GB VRAM) ---
    # Increased Physical Batch Size to fully utilize VRAM
    physical_batch_size = 12         # Doubled from 6 to 12
    accumulation_steps = 4
    effective_batch_size = physical_batch_size * accumulation_steps # Effective Batch Size: 48

    base_lr = 1e-5
    # Scaled LR: 1e-5 * (48/8) = 6e-5
    learning_rate = base_lr * (effective_batch_size / 8) # Scaled LR for effective batch size 48

    # --- ADVANCED TRAINING TECHNIQUE: SWA & Epochs Configuration ---
    num_epochs = 10                  
    swa_start_epoch = 6              
    swa_lr = 1e-6                    
    clip_grad_norm = 1.0

    warmup_ratio = 0.1
    dropout = 0.2
    weight_decay = 0.01

    # Paths
    train_path = "train.csv"
    model_save_path = "a10g_max_perf_price_model"

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
print(f"Using device: {config.device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} (23GB VRAM)")
    print(f"Physical Batch Size: {config.physical_batch_size} | Accumulation Steps: {config.accumulation_steps}")
    print(f"Effective Batch Size: {config.effective_batch_size} (Increased for performance! 🚀)")
    print(f"Scaled Learning Rate: {config.learning_rate:.2e} (Increased for effective batch size!)")

# ============================================================================
# STEP 4: Enhanced Feature Engineering (More Features - Unchanged)
# ============================================================================
def extract_features(text):
    """Extracts a richer set of statistical and linguistic features."""
    text = str(text)
    features = {}
    text_lower = text.lower()

    # Basic features
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())

    # Numeric features
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        nums = [float(x) for x in numbers]
        features['max_number'] = max(nums)
        features['min_number'] = min(nums)
        features['avg_number'] = np.mean(nums)
        features['number_count'] = len(nums)
    else:
        features['max_number'] = 0
        features['min_number'] = 0
        features['avg_number'] = 0
        features['number_count'] = 0

    # Unit/Packaging features
    features['has_ounce'] = 1 if 'ounce' in text_lower or 'oz' in text_lower else 0
    features['has_pound'] = 1 if 'pound' in text_lower or 'lb' in text_lower else 0
    features['has_pack'] = 1 if 'pack' in text_lower else 0
    features['has_count'] = 1 if 'count' in text_lower else 0
    pack_match = re.search(r'pack of (\d+)', text_lower)
    features['pack_size'] = int(pack_match.group(1)) if pack_match else 1
    features['has_brand'] = 1 if any(x in text_lower for x in ['brand', 'trademark', '®', '™']) else 0

    # --- MORE FEATURES: Linguistic and Structural ---
    features['bullet_points'] = text_lower.count('*') + text_lower.count('-') + text_lower.count('•')
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / (len(text) + 1e-6)
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / (len(text) + 1e-6)
    features['punctuation_count'] = sum(1 for c in text if c in '.,;!?')
    words = text_lower.split()
    features['unique_words_ratio'] = len(set(words)) / (features['word_count'] + 1e-6)
    features['has_percent'] = 1 if '%' in text else 0
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    sentences = re.split(r'[.!?]', text)
    features['sentence_count'] = len(sentences)
    features['avg_sentence_length'] = features['text_length'] / (features['sentence_count'] + 1e-6)

    return features

# ============================================================================
# STEP 5: Load and Preprocess Data (Unchanged logic, stat_features_normalized calculated here)
# ============================================================================
print("\n" + "="*70)
print("LOADING AND PREPROCESSING DATA")
print("="*70)
try:
    train_df = pd.read_csv(config.train_path)
except FileNotFoundError:
    print(f"Error: Could not find {config.train_path}. Please ensure the file is in the environment.")
    exit()

train_df['catalog_content'] = train_df['catalog_content'].fillna("No description available")
train_df['image_link'] = train_df['image_link'].fillna("")

def clean_text(text):
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text

train_df['catalog_content'] = train_df['catalog_content'].apply(clean_text)
stat_features = train_df['catalog_content'].apply(extract_features)
stat_features_df = pd.DataFrame(stat_features.tolist())

scaler = StandardScaler()
stat_features_normalized = scaler.fit_transform(stat_features_df)
os.makedirs(config.model_save_path, exist_ok=True)
with open(f"{config.model_save_path}/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

STAT_FEATURE_DIM = stat_features_normalized.shape[1]
train_df['log_price'] = np.log1p(train_df['price'])

train_texts, val_texts, train_prices, val_prices, train_stats, val_stats = train_test_split(
    train_df['catalog_content'].values,
    train_df['log_price'].values,
    stat_features_normalized,
    test_size=0.1,
    random_state=42
)

# ============================================================================
# STEP 6: Text-Only Dataset Class (Unchanged)
# ============================================================================
class TextOnlyPriceDataset(Dataset):
    def __init__(self, texts, prices, stat_features, text_tokenizer, max_length):
        self.texts = texts
        self.prices = prices
        self.stat_features = stat_features
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        price = self.prices[idx]
        stat_feat = self.stat_features[idx]

        text_encoding = self.text_tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'stat_features': torch.tensor(stat_feat, dtype=torch.float),
            'price': torch.tensor(price, dtype=torch.float)
        }

# ============================================================================
# STEP 7: Text-Only Model Architecture (Unchanged)
# ============================================================================
class TextOnlyPriceModel(nn.Module):
    def __init__(self, text_model_name, stat_feature_dim, dropout=0.2):
        super(TextOnlyPriceModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size

        fusion_input_size = text_hidden_size + stat_feature_dim

        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_size, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, stat_features):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        combined_features = torch.cat([
            text_features, stat_features
        ], dim=1)

        combined_features = self.dropout(combined_features)
        price = self.regressor(combined_features)
        return price

# ============================================================================
# STEP 8: Initialize Models and SWA
# ============================================================================
print("\n" + "="*70)
print("INITIALIZING MODELS")
print("="*70)
text_tokenizer = AutoTokenizer.from_pretrained(config.text_model)

model = TextOnlyPriceModel(
    config.text_model,
    stat_feature_dim=STAT_FEATURE_DIM,
    dropout=config.dropout
)
model.to(config.device)
swa_model = AveragedModel(model) # Advanced Training: SWA

# ============================================================================
# STEP 9: Create Data Loaders (Using new physical_batch_size = 12)
# ============================================================================
print("\nCreating data loaders...")
train_dataset = TextOnlyPriceDataset(
    train_texts, train_prices, train_stats,
    text_tokenizer, config.max_length
)
val_dataset = TextOnlyPriceDataset(
    val_texts, val_prices, val_stats,
    text_tokenizer, config.max_length
)
# Use the large physical_batch_size = 12 for the DataLoader
train_loader = DataLoader(train_dataset, batch_size=config.physical_batch_size, shuffle=True, num_workers=4) # Increased num_workers
val_loader = DataLoader(val_dataset, batch_size=config.physical_batch_size, shuffle=False, num_workers=4) # Increased num_workers
print(f"Total batches per epoch: {len(train_loader)}")

# ============================================================================
# STEP 10: Training Setup
# ============================================================================
criterion = nn.HuberLoss(delta=1.0)

# Optimizer: Layer-wise Learning Rates
optimizer = AdamW([
    # Lower LR for the pre-trained encoder (0.1x of base)
    {'params': model.text_encoder.parameters(), 'lr': config.learning_rate * 0.1},
    # Higher LR for the new regressor
    {'params': model.regressor.parameters(), 'lr': config.learning_rate},
], weight_decay=config.weight_decay)

total_steps = math.ceil(len(train_loader) / config.accumulation_steps) * config.num_epochs
warmup_steps = int(total_steps * config.warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr)
scaler = GradScaler()

# ============================================================================
# STEP 11: Training Functions (Modified for SWA)
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, scheduler, swa_scheduler, device, scaler, accumulation_steps, epoch, swa_start_epoch, swa_model):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training")

    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        stat_features = batch['stat_features'].to(device).to(torch.float32)
        prices = batch['price'].to(device).unsqueeze(1)

        with autocast():
            predictions = model(input_ids, attention_mask, stat_features)
            loss = criterion(predictions, prices) / accumulation_steps

        scaler.scale(loss).backward()

        total_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # Two-Stage Scheduling: Switch to SWA
            if epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            optimizer.zero_grad()

        progress_bar.set_postfix({'loss': f'{(total_loss / (step + 1) * accumulation_steps):.4f}'})

    return (total_loss / len(loader)) * accumulation_steps

def validate(model, loader, criterion, device):
    """Validation function for both regular and SWA models."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            stat_features = batch['stat_features'].to(device).to(torch.float32)
            prices = batch['price'].to(device).unsqueeze(1)

            with autocast():
                predictions = model(input_ids, attention_mask, stat_features)
                loss = criterion(predictions, prices)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_actuals.extend(prices.cpu().numpy())

    return total_loss / len(loader), np.array(all_predictions), np.array(all_actuals)

def calculate_smape(actual, predicted):
    """Calculates the SMAPE metric on the original (un-logged) scale."""
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    smape = np.zeros_like(numerator)
    smape[mask] = numerator[mask] / denominator[mask]
    return np.mean(smape) * 100

# ============================================================================
# STEP 12: Train the Model (SMAPE-Optimized, Reward-Based)
# ============================================================================
print("\n" + "="*70)
print("STARTING ADVANCED TRAINING WITH SWA (SMAPE-Optimized)")
print("======================================================================\n")
best_val_smape = float('inf')
training_history = {'train_loss': [], 'val_loss': [], 'val_smape': []}

for epoch in range(config.num_epochs):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch + 1}/{config.num_epochs} (SWA: {epoch >= config.swa_start_epoch})")
    print(f"{'='*70}")

    train_loss = train_epoch(
        model, train_loader, optimizer, criterion, scheduler, swa_scheduler,
        config.device, scaler, config.accumulation_steps, epoch, config.swa_start_epoch, swa_model
    )

    eval_model = swa_model if epoch >= config.swa_start_epoch else model

    val_loss, val_preds, val_actuals = validate(eval_model, val_loader, criterion, config.device)

    val_preds_original = np.expm1(val_preds)
    val_actuals_original = np.expm1(val_actuals)
    val_preds_original = np.maximum(val_preds_original, 0.01)

    smape = calculate_smape(val_actuals_original, val_preds_original)

    training_history['train_loss'].append(float(train_loss))
    training_history['val_loss'].append(float(val_loss))
    training_history['val_smape'].append(float(smape))

    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val SMAPE: {smape:.2f}%") # This is the primary 'reward' signal

    if smape < best_val_smape:
        best_val_smape = smape
        print(f"\n  ✓ New best SMAPE! Saving...")
        save_model = eval_model

        if isinstance(save_model, AveragedModel):
            # Update Batch Norm before saving the SWA model for inference
            torch.optim.swa_utils.update_bn(train_loader, save_model, device=config.device)
            save_state_dict = save_model.module.state_dict()
        else:
            save_state_dict = save_model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': save_state_dict,
            'best_smape': smape,
            'config': {
                'text_model': config.text_model,
                'max_length': config.max_length,
                'dropout': config.dropout,
                'stat_feature_dim': STAT_FEATURE_DIM
            }
        }, f"{config.model_save_path}/best_model.pt")
        text_tokenizer.save_pretrained(config.model_save_path)
        with open(f"{config.model_save_path}/training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)

    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)
print(f"\nBest Validation SMAPE: {best_val_smape:.2f}%")
print(f"Model saved at: {config.model_save_path}/")