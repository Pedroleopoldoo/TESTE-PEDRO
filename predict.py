import torch
import torch.nn as nn
import joblib
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

item_encoder = joblib.load("models/item_encoder.pkl")

class RecommenderNet(nn.Module):
    def __init__(self, n_items, emb_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, history, item):
        hist_emb = self.item_embedding(history).mean(dim=1)
        item_emb = self.item_embedding(item)
        x = torch.cat([hist_emb, item_emb], dim=1)
        return self.fc(x).squeeze()

model = RecommenderNet(len(item_encoder.classes_), 64)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

def recommend(history_items, top_k=10):

    history = item_encoder.transform(history_items)
    history = torch.tensor(history).unsqueeze(0)

    scores = []

    for item_id in range(len(item_encoder.classes_)):
        item = torch.tensor([item_id]).unsqueeze(0)

        with torch.no_grad():
            score = torch.sigmoid(model(history, item)).item()

        scores.append((item_id, score))

    top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    return [
        (item_encoder.inverse_transform([i])[0], score)
        for i, score in top
    ]