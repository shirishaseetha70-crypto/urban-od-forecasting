import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Path fix ──────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'ml_pipeline'))
# ──────────────────────────────────────────────────────────────────────────────

from stgat_model import STGAT

model_path = os.path.join(DATASET_DIR, "best_stgat.pth")
meta_path  = os.path.join(DATASET_DIR, "model_meta.json")
scaler_path = os.path.join(DATASET_DIR, "scaler.pkl")
graph_path  = os.path.join(DATASET_DIR, "road_graph.pkl")

for p in [model_path, meta_path, scaler_path, graph_path]:
    if not os.path.exists(p):
        print(f"ERROR: {p} not found. Train model first.")
        sys.exit(1)

with open(meta_path) as f:
    meta = json.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
with open(graph_path, "rb") as f:
    G = pickle.load(f)

zones       = meta['zones']
zone_to_idx = meta['zone_to_idx']
NUM_NODES   = meta['num_nodes']
SEQ_LEN     = meta['seq_len']
PRED_LEN    = meta['pred_len']

df = pd.read_csv(os.path.join(DATASET_DIR, "od_cleaned.csv"))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
time_steps = sorted(df['timestamp'].unique())

od_matrices = []
for ts in time_steps:
    sub    = df[df['timestamp'] == ts]
    matrix = np.zeros((NUM_NODES, NUM_NODES))
    for _, row in sub.iterrows():
        oi = zone_to_idx.get(row['origin_zone'])
        di = zone_to_idx.get(row['dest_zone'])
        if oi is not None and di is not None:
            matrix[oi, di] = row['od_flow']
    od_matrices.append(matrix)

od_matrices = np.array(od_matrices)
features = np.stack([od_matrices.sum(axis=2), od_matrices.sum(axis=1)], axis=2)
T = len(od_matrices)
F_flat   = scaler.transform(features.reshape(-1, 2))
features = F_flat.reshape(T, NUM_NODES, 2)

X_list, y_list = [], []
for i in range(T - SEQ_LEN - PRED_LEN):
    X_list.append(features[i:i + SEQ_LEN])
    y_list.append(features[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN, :, 0])

X = torch.tensor(np.array(X_list), dtype=torch.float32).permute(0, 2, 1, 3)
y = torch.tensor(np.array(y_list), dtype=torch.float32).permute(0, 2, 1)

split  = int(0.8 * len(X))
X_test = X[split:]
y_test = y[split:]

edges      = list(G.edges())
edge_index = torch.tensor(
    [[zone_to_idx[u], zone_to_idx[v]] for u, v in edges], dtype=torch.long
).t().contiguous()

model = STGAT(in_channels=2, hidden_channels=64, out_channels=PRED_LEN)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for i in range(len(X_test)):
        pred = model(X_test[i], edge_index).numpy()
        true = y_test[i].numpy()
        all_preds.append(pred)
        all_true.append(true)

all_preds = np.array(all_preds).flatten()
all_true  = np.array(all_true).flatten()

mse  = mean_squared_error(all_true, all_preds)
mae  = mean_absolute_error(all_true, all_preds)
rmse = np.sqrt(mse)

print(f"\n{'='*40}")
print(f"  ST-GAT Model Evaluation Results")
print(f"{'='*40}")
print(f"  MSE  : {mse:.6f}")
print(f"  MAE  : {mae:.6f}")
print(f"  RMSE : {rmse:.6f}")
print(f"{'='*40}")

# Save metrics to JSON
metrics = {"mse": mse, "mae": mae, "rmse": rmse}
metrics_path = os.path.join(DATASET_DIR, "evaluation_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to: {metrics_path}")
