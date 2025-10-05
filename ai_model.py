import os, math, joblib, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerEncoder(nn.Module):
    def __init__(self, feature_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.transformer(x)
        # take mean pooling over sequence
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.out(x)
        return x.squeeze(-1)

class ExperienceReplay:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    def push(self, x, y):
        self.buffer.append((x, y))
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        xs, ys = zip(*[self.buffer[i] for i in idx])
        return np.stack(xs), np.stack(ys)
    def __len__(self):
        return len(self.buffer)

class SelfLearningTransformer:
    def __init__(self, n_features, seq_len=30, model_path='models/transformer.pt'):
        self.n_features = n_features
        self.seq_len = seq_len
        self.model = TransformerEncoder(n_features).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)
        self.criterion = nn.BCEWithLogitsLoss()
        self.replay = ExperienceReplay(capacity=20000)
        self.batch_size = 64
        self.min_replay = 128
        self.model_path = model_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.steps = 0
        # try load existing
        self.load()

    def predict_proba(self, X, mc_runs=6):
        # X: (batch, seq_len, features)
        self.model.train()  # enable dropout for MC
        xs = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        preds = []
        for _ in range(mc_runs):
            with torch.no_grad():
                logits = self.model(xs)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
        preds = np.stack(preds, axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    def store_label(self, X_seq, label):
        self.replay.push(X_seq.astype('float32'), np.array([label], dtype='float32'))
        self.steps += 1
        if len(self.replay) >= self.min_replay and (self.steps % 8 == 0):
            self._update_from_replay()

    def _update_from_replay(self):
        self.model.train()
        xs, ys = self.replay.sample(self.batch_size)
        xs_t = torch.tensor(xs, dtype=torch.float32, device=DEVICE)
        ys_t = torch.tensor(ys, dtype=torch.float32, device=DEVICE).squeeze(-1)
        logits = self.model(xs_t)
        loss = self.criterion(logits, ys_t)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def save(self):
        torch.save({'state': self.model.state_dict(), 'opt': self.optimizer.state_dict()}, self.model_path)
        joblib.dump(self.replay, self.model_path + '.replay')

    def load(self):
        try:
            if os.path.exists(self.model_path):
                ckpt = torch.load(self.model_path, map_location=DEVICE)
                self.model.load_state_dict(ckpt['state'])
                self.optimizer.load_state_dict(ckpt['opt'])
            if os.path.exists(self.model_path + '.replay'):
                self.replay = joblib.load(self.model_path + '.replay')
        except Exception:
            pass

    def warm_start(self, X, y, epochs=3):
        # X: (n, seq, features)
        self.model.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        opt = optim.Adam(self.model.parameters(), lr=1e-4)
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE).squeeze(-1)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
