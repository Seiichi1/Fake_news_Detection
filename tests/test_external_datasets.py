import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

# --- Configuration ---
MODEL_PATH = 'best_model_state_finetuned.bin' # Or 'best_model_state.bin' if finetuned not found
LIAR_PATH = 'data/LIAR dataset/test.tsv'
SNOPES_PATH = 'data/snopeswithsum.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128
BATCH_SIZE = 16

# --- 1. Model Definition (Must match training) ---
class HybridBertBiLSTM(nn.Module):
    def __init__(self, n_classes):
        super(HybridBertBiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(128 * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_state)
        out = torch.mean(lstm_out, dim=1)
        out = self.drop(out)
        return self.out(out)

# --- 2. Preprocessing ---
import re
import string
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# --- 3. Data Loading Functions ---

def load_liar_test(path, n_samples=100):
    print(f"Loading LIAR dataset from {path}...")
    try:
        # LIAR format: ID, label, statement, ...
        df = pd.read_csv(path, sep='\t', header=None, usecols=[1, 2], names=['label_text', 'text'])
        
        # Mapping: 0=Real, 1=Fake
        # Fake: false, pants-fire, barely-true
        # Real: true, mostly-true, half-true
        label_map = {
            'false': 1, 'pants-fire': 1, 'barely-true': 1, 
            'true': 0, 'mostly-true': 0, 'half-true': 0
        }
        df['label'] = df['label_text'].map(label_map)
        df = df.dropna(subset=['label']) # Drop unknown labels if any
        df['label'] = df['label'].astype(int)
        
        # Sample
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
        
        print(f"✅ Loaded {len(df)} LIAR samples.")
        return df
    except Exception as e:
        print(f"❌ Error loading LIAR: {e}")
        return None

def load_snopes_test(path, n_samples=100):
    print(f"Loading Snopes dataset from {path}...")
    try:
        df = pd.read_csv(path)
        
        # 1. DROP Forbidden Columns (Leakage)
        drop_cols = ['what\'s true', 'what\'s false', 'what\'s unknown', 'summary', 'origin', 'question', 'comment']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # 2. Process Target 'rate'
        # Mapping based on inspection
        # Real (0): 'True', 'Mostly True'
        # Fake (1): 'False', 'Mostly False', 'Miscaptioned', 'Unproven', 'Mixture', 'Legend'
        # We will filter for clear labels to be fair, or map all. Let's map common ones.
        
        label_map = {
            'True': 0, 'Mostly True': 0,
            'False': 1, 'Mostly False': 1, 'Miscaptioned': 1, 'Unproven': 1, 'Mixture': 1, 'Legend': 1,
            'Correct Attribution': 0, 'Misattributed': 1, 'Scam': 1
        }
        
        df['label'] = df['rate'].map(label_map)
        
        # Drop rows where label is NaN (e.g. 'Research In Progress' or unknown types)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        # 3. Rename 'claim' to 'text' for consistency
        df = df.rename(columns={'claim': 'text'})
        
        # Sample
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
            
        print(f"✅ Loaded {len(df)} Snopes samples.")
        return df[['text', 'label']]
    except Exception as e:
        print(f"❌ Error loading Snopes: {e}")
        return None

# --- 4. Evaluation Function ---
def evaluate_model(model, tokenizer, df, dataset_name):
    if df is None or len(df) == 0:
        print(f"Skipping {dataset_name} (No data).")
        return

    print(f"\n--- Evaluating on {dataset_name} ({len(df)} samples) ---")
    
    texts = df['text'].apply(clean_text).tolist()
    labels = df['label'].tolist()
    
    preds = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            
            encoded = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(DEVICE)
            attention_mask = encoded['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask)
            _, batch_preds = torch.max(outputs, dim=1)
            
            preds.extend(batch_preds.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=['Real', 'Fake']))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Model
    print("Initializing Model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = HybridBertBiLSTM(n_classes=2).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    elif os.path.exists('best_model_state.bin'):
        print(f"⚠️ {MODEL_PATH} not found. Falling back to 'best_model_state.bin'...")
        model.load_state_dict(torch.load('best_model_state.bin', map_location=DEVICE))
    else:
        print("❌ No model weights found! Please train the model first.")
        exit()

    # 2. Load Data
    df_liar = load_liar_test(LIAR_PATH, n_samples=100)
    df_snopes = load_snopes_test(SNOPES_PATH, n_samples=100)
    
    # 3. Evaluate
    evaluate_model(model, tokenizer, df_liar, "LIAR Dataset")
    evaluate_model(model, tokenizer, df_snopes, "Snopes Dataset")
