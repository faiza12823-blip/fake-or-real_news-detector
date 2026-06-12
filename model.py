import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv("fake_or_real_news.csv")
print(f"Dataset loaded: {len(df)} rows")

x = df['text'].tolist()
y = df['label'].map({'REAL': 1, 'FAKE': 0}).tolist()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────
# 2. Load BERT Tokenizer
# ─────────────────────────────────────────
print("Loading BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ─────────────────────────────────────────
# 3. Dataset Class
# ─────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = NewsDataset(x_train, y_train, tokenizer)
test_dataset  = NewsDataset(x_test,  y_test,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8)

# ─────────────────────────────────────────
# 4. Load BERT Model
# ─────────────────────────────────────────
print("Loading BERT Model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
model.to(device)

# ─────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
EPOCHS = 3
print("\nStarting Training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_num, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_num % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_num} | Loss: {loss.item():.4f}")

    print(f"\n  Epoch {epoch+1} Complete | Avg Loss: {total_loss/len(train_loader):.4f}\n")

# ─────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────
print("Evaluating Model...")
model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

score = accuracy_score(y_test, all_preds)
print(f"\n✅ BERT Accuracy: {score * 100:.2f}%")

# ─────────────────────────────────────────
# 7. Save
# ─────────────────────────────────────────
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_tokenizer")
print("✅ Model & Tokenizer Saved!")
