import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# =====================================================
# CONFIG
# =====================================================

RANDOM_STATE = 42
N_USERS_SAMPLE = 10000
BATCH_SIZE = 1024
EMBEDDING_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 10

DATA_PATH = "data/instacart_orders.parquet"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")

df = pd.read_parquet(DATA_PATH)

interactions = (
    df[["user_id", "product_id"]]
    .drop_duplicates()
    .copy()
)

interactions["interaction"] = 1

print(f"Original interactions: {interactions.shape}")


# =====================================================
# SAMPLE USERS
# =====================================================

sample_users = (
    interactions["user_id"]
    .drop_duplicates()
    .sample(
        N_USERS_SAMPLE,
        random_state=RANDOM_STATE
    )
)

interactions = interactions[
    interactions["user_id"].isin(sample_users)
].copy()

print(f"Sampled interactions: {interactions.shape}")


# =====================================================
# ENCODERS
# =====================================================

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

interactions["user_idx"] = user_encoder.fit_transform(
    interactions["user_id"]
)

interactions["item_idx"] = item_encoder.fit_transform(
    interactions["product_id"]
)


# =====================================================
# NEGATIVE SAMPLING
# =====================================================

print("Generating negative samples...")

user_items = (
    interactions
    .groupby("user_idx")["item_idx"]
    .apply(set)
    .to_dict()
)

all_items = interactions["item_idx"].unique()

negative_samples = []

for user, bought_items in user_items.items():

    n_negatives = len(bought_items)

    generated = 0

    while generated < n_negatives:

        candidate = np.random.choice(all_items)

        if candidate not in bought_items:

            negative_samples.append(
                (user, candidate, 0)
            )

            generated += 1

negative_df = pd.DataFrame(
    negative_samples,
    columns=[
        "user_idx",
        "item_idx",
        "interaction"
    ]
)

print(f"Negative samples: {negative_df.shape}")


# =====================================================
# FINAL DATASET
# =====================================================

positive_df = interactions[
    [
        "user_idx",
        "item_idx",
        "interaction"
    ]
]

dataset = pd.concat(
    [positive_df, negative_df],
    ignore_index=True
)

dataset = dataset.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print(f"Final dataset: {dataset.shape}")


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

train_df, test_df = train_test_split(
    dataset,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=dataset["interaction"]
)


# =====================================================
# PYTORCH DATASET
# =====================================================

class InteractionDataset(Dataset):

    def __init__(self, df):

        self.users = torch.tensor(
            df["user_idx"].values,
            dtype=torch.long
        )

        self.items = torch.tensor(
            df["item_idx"].values,
            dtype=torch.long
        )

        self.labels = torch.tensor(
            df["interaction"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return (
            self.users[idx],
            self.items[idx],
            self.labels[idx]
        )


train_dataset = InteractionDataset(train_df)
test_dataset = InteractionDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)


# =====================================================
# MODEL
# =====================================================

class RecommenderNet(nn.Module):

    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim
    ):

        super().__init__()

        self.user_embedding = nn.Embedding(
            n_users,
            embedding_dim
        )

        self.item_embedding = nn.Embedding(
            n_items,
            embedding_dim
        )

        self.layers = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1
            )
        )

    def forward(
        self,
        users,
        items
    ):

        user_vec = self.user_embedding(users)

        item_vec = self.item_embedding(items)

        x = torch.cat(
            [user_vec, item_vec],
            dim=1
        )

        return self.layers(x).squeeze()


n_users = dataset["user_idx"].nunique()
n_items = dataset["item_idx"].nunique()

model = RecommenderNet(
    n_users=n_users,
    n_items=n_items,
    embedding_dim=EMBEDDING_DIM
).to(device)


# =====================================================
# TRAINING SETUP
# =====================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================================
# TRAINING
# =====================================================

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for users, items, labels in train_loader:

        users = users.to(device)
        items = items.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            users,
            items
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={avg_loss:.4f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            f"{MODEL_DIR}/best_model.pth"
        )


# =====================================================
# EVALUATION
# =====================================================

print("\nEvaluating model...")

model.eval()

preds = []
targets = []

with torch.no_grad():

    for users, items, labels in test_loader:

        users = users.to(device)
        items = items.to(device)

        outputs = model(
            users,
            items
        )

        probs = torch.sigmoid(
            outputs
        )

        preds.extend(
            probs.cpu().numpy()
        )

        targets.extend(
            labels.numpy()
        )

pred_binary = (
    np.array(preds) >= 0.5
).astype(int)

auc = roc_auc_score(
    targets,
    preds
)

accuracy = accuracy_score(
    targets,
    pred_binary
)

precision = precision_score(
    targets,
    pred_binary
)

recall = recall_score(
    targets,
    pred_binary
)

f1 = f1_score(
    targets,
    pred_binary
)

print("\nRESULTS")
print("-" * 30)
print(f"AUC       : {auc:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# =====================================================
# SAVE ARTIFACTS
# =====================================================

joblib.dump(
    user_encoder,
    f"{MODEL_DIR}/user_encoder.pkl"
)

joblib.dump(
    item_encoder,
    f"{MODEL_DIR}/item_encoder.pkl"
)

print("\nArtifacts saved successfully.")