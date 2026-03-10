import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import os

# --- Config ---
# Get project root (parent of src)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, 'best_model_state.bin')
DATA_PATH = os.path.join(DATA_DIR, 'synthetic_fake_news.csv')
MAIN_DATA_PATH = os.path.join(DATA_DIR, 'WELFake_Dataset.csv')

EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Model Architecture (Must match) ---
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

# --- Dataset ---
class SyntheticDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def fine_tune():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print(f"Loading synthetic data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # IMPORTANT: We need to mix in some REAL news so the model doesn't forget what Real news looks like!
    # Let's load a few real samples from the main dataset
    try:
        df_main = pd.read_csv(MAIN_DATA_PATH)
        df_real = df_main[df_main['label'] == 1].sample(n=len(df)) # Balance it
        df_combined = pd.concat([df, df_real[['text', 'label']]])
        df_combined = df_combined.sample(frac=1).reset_index(drop=True) # Shuffle
        print(f"Combined with real news. Total samples: {len(df_combined)}")
    except Exception as e:
        print(f"Warning: Could not load main dataset at {MAIN_DATA_PATH} for balancing. Training only on fake news (risky!). Error: {e}")
        df_combined = df

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = SyntheticDataset(
        df_combined['text'].to_numpy(),
        df_combined['label'].to_numpy(),
        tokenizer
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = HybridBertBiLSTM(n_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.train()

    # 3. Train
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    print("Starting fine-tuning...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0
        n_examples = len(df_combined)
        
        for i, d in enumerate(data_loader):
            input_ids = d['input_ids'].to(DEVICE)
            attention_mask = d['attention_mask'].to(DEVICE)
            targets = d['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            total_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i}/{len(data_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        acc = correct_predictions.double() / n_examples
        print(f"Epoch {epoch+1} completed. Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # 4. Save
    new_model_path = os.path.join(MODELS_DIR, 'best_model_state_finetuned.bin')
    torch.save(model.state_dict(), new_model_path)
    print(f"Fine-tuned model saved to {new_model_path}")
    print("Update your app to use this new model file!")

if __name__ == "__main__":
    fine_tune()
