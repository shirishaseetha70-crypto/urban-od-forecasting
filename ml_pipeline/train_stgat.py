# ml_pipeline/train_stgat.py
import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ── Path fix ──────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'ml_pipeline'))
# ──────────────────────────────────────────────────────────────────────────────

from stgat_model import STGAT

# ── Moderate Config ───────────────────────────────────────────────────────────
SEQ_LEN    = 12      # keep original
PRED_LEN   = 6       # keep original
BATCH_SIZE = 128     # increased from 32 → faster iteration
EPOCHS     = 50      # reduced from 100 → still converges well
LR         = 0.002   # slightly higher LR → faster convergence
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")
# ──────────────────────────────────────────────────────────────────────────────

cleaned_path = os.path.join(DATASET_DIR, "od_cleaned.csv")
graph_path   = os.path.join(DATASET_DIR, "road_graph.pkl")

for p in [cleaned_path, graph_path]:
    if not os.path.exists(p):
        print(f"\nERROR: {p} not found!")
        print("Run from project ROOT in this order:")
        print("  1. python generate_dataset.py")
        print("  2. python ml_pipeline/dtw_cleaner.py")
        print("  3. python ml_pipeline/graph_builder.py")
        print("  4. python ml_pipeline/train_stgat.py")
        sys.exit(1)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(cleaned_path)
with open(graph_path, "rb") as f:
    G = pickle.load(f)

zones       = sorted(list(G.nodes()))
zone_to_idx = {z: i for i, z in enumerate(zones)}
NUM_NODES   = len(zones)
print(f"Zones: {NUM_NODES} | Graph edges: {G.number_of_edges()} | Records: {len(df)}")

edges = list(G.edges())
edge_index = torch.tensor(
    [[zone_to_idx[u], zone_to_idx[v]] for u, v in edges],
    dtype=torch.long
).t().contiguous()

# ── Build feature matrices (vectorized — no row loop) ─────────────────────────
print("Building feature matrices (fast vectorized)...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Map zones to indices directly in dataframe
df['oi'] = df['origin_zone'].map(zone_to_idx)
df['di'] = df['dest_zone'].map(zone_to_idx)
df = df.dropna(subset=['oi', 'di'])
df['oi'] = df['oi'].astype(int)
df['di'] = df['di'].astype(int)

time_steps  = sorted(df['timestamp'].unique())
T           = len(time_steps)
ts_to_idx   = {ts: i for i, ts in enumerate(time_steps)}
df['t_idx'] = df['timestamp'].map(ts_to_idx)

# Build full OD tensor in one shot using numpy indexing
od_matrices = np.zeros((T, NUM_NODES, NUM_NODES), dtype=np.float32)
od_matrices[df['t_idx'].values, df['oi'].values, df['di'].values] = df['od_flow'].values

print(f"OD tensor shape: {od_matrices.shape}")

# Features: outflow + inflow per node → [T, N, 2]
features = np.stack([
    od_matrices.sum(axis=2),   # outflow
    od_matrices.sum(axis=1),   # inflow
], axis=2)

# Normalize
scaler   = MinMaxScaler()
features = scaler.fit_transform(features.reshape(-1, 2)).reshape(T, NUM_NODES, 2)

with open(os.path.join(DATASET_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved.")

# ── Sequences ─────────────────────────────────────────────────────────────────
print("Creating sequences...")
X_arr = np.array([features[i:i + SEQ_LEN]               for i in range(T - SEQ_LEN - PRED_LEN)])
y_arr = np.array([features[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN, :, 0] for i in range(T - SEQ_LEN - PRED_LEN)])

# [samples, N, SEQ_LEN, 2]  and  [samples, N, PRED_LEN]
X = torch.tensor(X_arr, dtype=torch.float32).permute(0, 2, 1, 3)
y = torch.tensor(y_arr, dtype=torch.float32).permute(0, 2, 1)
print(f"X: {X.shape} | y: {y.shape}")

split   = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model     = STGAT(in_channels=2, hidden_channels=64, out_channels=PRED_LEN).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
criterion = nn.MSELoss()
edge_index = edge_index.to(DEVICE)

best_val_loss = float('inf')
history = {"train_loss": [], "val_loss": []}

print(f"\nStarting training — {EPOCHS} epochs, batch {BATCH_SIZE}, device: {DEVICE}\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    num_b = 0

    for i in range(0, len(X_train), BATCH_SIZE):
        xb = X_train[i:i + BATCH_SIZE].to(DEVICE)   # [B, N, SEQ, 2]
        yb = y_train[i:i + BATCH_SIZE].to(DEVICE)   # [B, N, PRED]

        # Process all samples in batch via vectorized stack
        preds  = torch.stack([model(xb[b], edge_index) for b in range(xb.shape[0])])
        loss   = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        num_b += 1

    model.eval()
    val_loss = 0.0
    val_b    = 0
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH_SIZE):
            xb = X_val[i:i + BATCH_SIZE].to(DEVICE)
            yb = y_val[i:i + BATCH_SIZE].to(DEVICE)
            preds     = torch.stack([model(xb[b], edge_index) for b in range(xb.shape[0])])
            val_loss += criterion(preds, yb).item()
            val_b    += 1

    train_loss /= max(num_b, 1)
    val_loss   /= max(val_b, 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    scheduler.step()

    saved = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(DATASET_DIR, "best_stgat.pth"))
        saved = " ✅ saved"

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | Train: {train_loss:.5f} | Val: {val_loss:.5f}{saved}")

print(f"\nTraining complete! Best Val Loss: {best_val_loss:.5f}")
print(f"Model saved: {os.path.join(DATASET_DIR, 'best_stgat.pth')}")

# ── Save metadata ─────────────────────────────────────────────────────────────
with open(os.path.join(DATASET_DIR, "model_meta.json"), "w") as f:
    json.dump({
        "zones":       zones,
        "zone_to_idx": zone_to_idx,
        "seq_len":     SEQ_LEN,
        "pred_len":    PRED_LEN,
        "num_nodes":   NUM_NODES,
        "history":     history
    }, f)
print("Metadata saved.")

# ── Push logs to Django DB ────────────────────────────────────────────────────
try:
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    django.setup()
    from admin_panel.models import ModelTrainingLog
    ModelTrainingLog.objects.all().delete()
    for i, (tl, vl) in enumerate(zip(history['train_loss'], history['val_loss'])):
        ModelTrainingLog.objects.create(
            epoch=i, train_loss=tl, val_loss=vl,
            mse=vl, mae=float(vl**0.5), rmse=float(vl**0.5)
        )
    print("Training logs saved to Django DB.")
except Exception as e:
    print(f"Note: Django DB log skipped ({e}). Logs saved in model_meta.json.")
