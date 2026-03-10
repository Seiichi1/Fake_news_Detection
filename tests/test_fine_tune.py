import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np

# --- Config ---
MODEL_PATH = 'best_model_state.bin'
DATA_PATH = 'data/synthetic_fake_news.csv'
EPOCHS = 1
BATCH_SIZE = 2
LR = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Architecture ---
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

class SyntheticDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128): # Shorter max_len for speed
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

def fine_tune_test():
    print(f"Using device: {DEVICE}")
    print("Running QUICK TEST with 10 samples...")
    
    df = pd.read_csv(DATA_PATH).head(10)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = SyntheticDataset(df['text'].to_numpy(), df['label'].to_numpy(), tokenizer)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HybridBertBiLSTM(n_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    for i, d in enumerate(data_loader):
        input_ids = d['input_ids'].to(DEVICE)
        attention_mask = d['attention_mask'].to(DEVICE)
        targets = d['labels'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Batch {i} processed. Loss: {loss.item():.4f}")

    print("Test passed! The training script works.")

if __name__ == "__main__":
    fine_tune_test()
