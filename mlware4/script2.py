import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import pickle
import re
from tqdm.auto import tqdm
import gc
from torch.cuda.amp import autocast
import warnings

warnings.filterwarnings('ignore')

# Set random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Feature extraction function (same as training)
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

# Text cleaning function (same as training)
def clean_text(text):
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text

# Dataset for inference (without prices)
class TextOnlyPriceDataset(Dataset):
    def __init__(self, texts, stat_features, text_tokenizer, max_length):
        self.texts = texts
        self.stat_features = stat_features
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        stat_feat = self.stat_features[idx]

        text_encoding = self.text_tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'stat_features': torch.tensor(stat_feat, dtype=torch.float)
        }

# Model architecture (same as training)
class TextOnlyPriceModel(torch.nn.Module):
    def __init__(self, text_model_name, stat_feature_dim, dropout=0.2):
        super(TextOnlyPriceModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size

        fusion_input_size = text_hidden_size + stat_feature_dim

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(fusion_input_size, 1024), torch.nn.LayerNorm(1024), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, 512), torch.nn.LayerNorm(512), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256), torch.nn.LayerNorm(256), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 1)
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, stat_features):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        combined_features = torch.cat([
            text_features, stat_features
        ], dim=1)

        combined_features = self.dropout(combined_features)
        price = self.regressor(combined_features)
        return price

# Load model and components
model_path = "a10g_max_perf_price_model/"
checkpoint = torch.load(os.path.join(model_path, "best_model.pt"), map_location=device, weights_only=False)

# Load config from checkpoint
cfg = checkpoint['config']
text_model_name = cfg['text_model']
max_length = cfg['max_length']
dropout = cfg['dropout']
stat_feature_dim = cfg['stat_feature_dim']

# Initialize model
model = TextOnlyPriceModel(text_model_name, stat_feature_dim, dropout)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load tokenizer
text_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load scaler
with open(os.path.join(model_path, "scaler.pkl"), 'rb') as f:
    scaler = pickle.load(f)

# Load test data
test_path = "test.csv"
try:
    test_df = pd.read_csv(test_path)
except FileNotFoundError:
    print(f"Error: Could not find {test_path}. Please ensure the file is in the environment.")
    exit()

# Preprocess test data
test_df['catalog_content'] = test_df['catalog_content'].fillna("No description available")
test_df['catalog_content'] = test_df['catalog_content'].apply(clean_text)
stat_features = test_df['catalog_content'].apply(extract_features)
stat_features_df = pd.DataFrame(stat_features.tolist())
stat_features_normalized = scaler.transform(stat_features_df)

# Sample IDs
sample_ids = test_df['sample_id'].values

# Create dataset and dataloader
test_dataset = TextOnlyPriceDataset(
    test_df['catalog_content'].values,
    stat_features_normalized,
    text_tokenizer,
    max_length
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)  # Larger batch for inference

# Inference
all_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        stat_features = batch['stat_features'].to(device).to(torch.float32)

        with autocast():
            predictions = model(input_ids, attention_mask, stat_features)

        all_predictions.extend(predictions.cpu().numpy())

# Post-process predictions (back to original scale)
predictions = np.expm1(np.array(all_predictions))
predictions = np.maximum(predictions, 0.01).flatten()

# Create output DataFrame
output_df = pd.DataFrame({
    'sample_id': sample_ids,
    'price': predictions
})

# Save to CSV
output_path = "output.csv"
output_df.to_csv(output_path, index=False)
print(f"Output saved to {output_path}")

# Cleanup
gc.collect()
torch.cuda.empty_cache()