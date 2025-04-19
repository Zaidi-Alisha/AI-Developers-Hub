import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from tqdm import tqdm

df = pd.read_csv("goemotions_dataset.csv")

#filter
df = df[df['example_very_unclear'] == False].reset_index(drop=True)

text_data = df['text'].tolist()
label_cols = df.columns[3:]
labels = df[label_cols].values.tolist()

#split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    text_data, labels, test_size=0.1, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class GoEmotionsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = GoEmotionsDataset(train_encodings, train_labels)
val_dataset = GoEmotionsDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=len(label_cols), problem_type="multi_label_classification")

#optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_training_steps = epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.train()
for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

model.eval()
preds, true = [], []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.sigmoid(logits) > 0.5
        preds.extend(predictions.cpu().numpy())
        true.extend(batch['labels'].cpu().numpy())

f1 = f1_score(true, preds, average='micro')
hl = hamming_loss(true, preds)
print(f"F1 Score: {f1:.4f}")
print(f"Hamming Loss: {hl:.4f}")
