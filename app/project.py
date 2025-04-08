import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch.amp as amp
import joblib
import os

# Enable optimized GPU performance
torch.backends.cudnn.benchmark = True

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
fake_df = pd.read_csv(r'C:\Users\ASUS\Desktop\fake news\data\Cleaned_Fake (2).csv')
real_df = pd.read_csv(r'C:\Users\ASUS\Desktop\fake news\data\Cleaned_True (2).csv')

# Check if required column exists
required_columns = {'text'}
if not required_columns.issubset(fake_df.columns) or not required_columns.issubset(real_df.columns):
    raise ValueError(f"Dataset must contain the columns: {required_columns}")

# Assign labels: Fake = 0, Real = 1
fake_df['label'] = 0
real_df['label'] = 1

# Combine and shuffle data
data = pd.concat([fake_df, real_df], axis=0).dropna().sample(frac=1, random_state=42).reset_index(drop=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
)

# ================= XGBOOST PART =================
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train_tfidf, y_train)
xgb_preds = xgb_model.predict(X_test_tfidf)

# Evaluation
print("\n=== XGBoost Model Evaluation ===")
print(f'XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.4f}')
print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_preds))

# Confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("XGBoost Confusion Matrix")
plt.savefig("xgboost_confusion_matrix.png")
plt.show()

# Save XGBoost model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/xgboost_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
print("✅ XGBoost model and vectorizer saved!")

# ================= BERT PART =================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.astype(str).tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=256, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

train_loader = DataLoader(NewsDataset(X_train, y_train), batch_size=16, shuffle=True, pin_memory=True)
test_loader = DataLoader(NewsDataset(X_test, y_test), batch_size=16, shuffle=False, pin_memory=True)

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
optimizer = AdamW(bert_model.parameters(), lr=2e-5)
scaler = amp.GradScaler()
gradient_accumulation_steps = 2

num_epochs = 5
for epoch in range(num_epochs):
    bert_model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        with amp.autocast(device_type='cuda'):
            inputs = {key: batch[key].to(device) for key in ['input_ids', 'attention_mask']}
            labels = batch['label'].to(device)
            outputs = bert_model(**inputs, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {total_loss/len(train_loader):.4f}')

# BERT Evaluation
bert_model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        inputs = {key: batch[key].to(device) for key in ['input_ids', 'attention_mask']}
        labels = batch['label'].to(device)
        outputs = bert_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

print("\n=== BERT Model Evaluation ===")
print(f'BERT Accuracy: {accuracy_score(all_labels, all_preds):.4f}')
print("\nBERT Classification Report:\n", classification_report(all_labels, all_preds))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt="d", cmap="Greens",
            xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("BERT Confusion Matrix")
plt.savefig("bert_confusion_matrix.png")
plt.show()

# Save BERT model and tokenizer
os.makedirs("models/bert_model", exist_ok=True)
bert_model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")
print("✅ BERT model and tokenizer saved!")

# ================= PREDICTION FUNCTIONS =================
def predict_news_xgboost(news_text):
    prediction = xgb_model.predict(vectorizer.transform([news_text]))[0]
    return prediction  # 0 = Fake, 1 = Real

def predict_news_bert(news_text):
    encoding = tokenizer(news_text, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        prediction = torch.argmax(bert_model(**encoding).logits, dim=1).item()
    return prediction  # 0 = Fake, 1 = Real

def combined_prediction(news_text):
    xgb_result = predict_news_xgboost(news_text)
    bert_result = predict_news_bert(news_text)
    
    if xgb_result == bert_result:
        return "Real News" if xgb_result == 1 else "Fake News"
    else:
        return "Uncertain (Disagreement between models)"

# ================= INTERACTIVE PREDICTION =================
if __name__ == "__main__":
    while True:
        news_input = input("\n📰 Enter a news article (or type 'exit' to quit): ").strip()
        
        if news_input.lower() in ["exit", "quit"]:
            print("👋 Exiting the program. Goodbye!")
            break

        if news_input:
            print("🧠 Final Prediction:", combined_prediction(news_input))
        else:
            print("⚠️ Invalid input. Please enter a valid news article.")


