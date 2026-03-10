import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import re
import string
import argparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Architecture (Must match training) ---
class HybridBertBiLSTM(nn.Module):
    def __init__(self, n_classes):
        super(HybridBertBiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(128 * 2, n_classes) # *2 for bidirectional

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_state)
        out = torch.mean(lstm_out, dim=1)
        out = self.drop(out)
        return self.out(out)

# --- Preprocessing ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

import os

# --- Prediction Class ---
class FakeNewsPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            # Check for models in models/ directory relative to root
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(root_dir, 'models')
            
            final_path = os.path.join(models_dir, 'best_model_state_final.bin')
            v2_path = os.path.join(models_dir, 'best_model_state_v2.bin')
            base_path = os.path.join(models_dir, 'best_model_state.bin')

            if os.path.exists(final_path):
                model_path = final_path
                print(f"Loading final model from {model_path}")
            elif os.path.exists(v2_path):
                model_path = v2_path
                print(f"Loading v2 model from {model_path}")
            elif os.path.exists(base_path):
                model_path = base_path
                print(f"Loading base model from {model_path}")
            else:
                # Fallback to local if running from src
                if os.path.exists('best_model_state_final.bin'):
                    model_path = 'best_model_state_final.bin'
                else:
                    raise FileNotFoundError("Could not find any model binary in models/ or current directory.")

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = HybridBertBiLSTM(n_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.max_len = 512

    def predict(self, text):
        cleaned_text = clean_text(text)
        
        encoding = self.tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
        confidence = torch.max(probs).item()
        prediction = "Real" if preds.item() == 0 else "Fake"
        
        return prediction, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fake News Detection Inference')
    parser.add_argument('--text', type=str, help='News text to classify', required=True)
    args = parser.parse_args()

    predictor = FakeNewsPredictor()
    pred, conf = predictor.predict(args.text)
    
    print(f"\nPrediction: {pred}")
    print(f"Confidence: {conf:.4f}")
