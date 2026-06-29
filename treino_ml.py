import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# CONFIG
# =====================================================

RANDOM_STATE = 42
BATCH_SIZE = 1024
EMBEDDING_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 10

DATA_PATH = "data/instacart_orders.parquet"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_parquet(DATA_PATH)

interactions = df[["user_id", "product_id"]].drop_duplicates()

# =====================================================
# BUILD USER HISTORIES
# =====================================================

user_history = interactions.groupby("user_id")["product_id"].apply(list).to_dict()
all_items = interactions["product_id"].unique()

# =====================================================
# CREATE TRAINING SAMPLES (SEQUENCIAL STYLE)
# =====================================================

data = []

for user, items in user_history.items():
    if len(items) < 2:
        continue

    for i in range(1, len(items)):
        history = items[:i]
        positive = items[i]

        data.append((history, positive, 1))

        # negative sample
        neg = np.random.choice(all_items)
        while neg in items:
            neg = np.random.choice(all_items)

        data.append((history, neg, 0))

# =====================================================
# ENCODE ITEMS
# =====================================================

from sklearn.preprocessing import LabelEncoder

item_encoder = LabelEncoder()
all_items_list = list(all_items)
item_encoder.fit(all_items_list)

def encode_list(lst):
    return item_encoder.transform(lst)

def encode_item(x):
    return item_encoder.transform([x])[0]

# =====================================================
# DATASET CLASS
# =====================================================

class RecDataset(Dataset):
    def __init__(self, data):
        self.histories = []
        self.candidates = []
        self.labels = []

        for h, c, y in data:
            self.histories.append(torch.tensor(encode_list(h), dtype=torch.long))
            self.candidates.append(torch.tensor(encode_item(c), dtype=torch.long))
            self.labels.append(torch.tensor(y, dtype=torch.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.histories[idx], self.candidates[idx], self.labels[idx]

dataset = RecDataset(data)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# =====================================================
# MODEL
# =====================================================

class RecommenderNet(nn.Module):
    def __init__(self, n_items, emb_dim):
        super().__init__()

        self.item_embedding = nn.Embedding(n_items, emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, history, item):

        hist_emb = self.item_embedding(history).mean(dim=1)
        item_emb = self.item_embedding(item)

        x = torch.cat([hist_emb, item_emb], dim=1)

        return self.fc(x).squeeze()

model = RecommenderNet(len(item_encoder.classes_), EMBEDDING_DIM).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =====================================================
# TRAIN
# =====================================================

best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for hist, item, label in train_loader:

        hist = hist.to(device)
        item = item.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(hist, item)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pth")

joblib.dump(item_encoder, f"{MODEL_DIR}/item_encoder.pkl")
print("Model saved!")