from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import ODPrediction, PredictionRequest
import torch, pickle, json, numpy as np, pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ml_pipeline'))

@login_required
def predict_view(request):
    zones_df = pd.read_csv("dataset/zone_metadata.csv")
    zones = zones_df['zone_id'].tolist()
    return render(request, 'prediction/predict.html', {'zones': zones})

@login_required
def run_prediction(request):
    if request.method == 'POST':
        from ml_pipeline.stgat_model import STGAT
        origin = request.POST.get('origin_zone')
        target_time = request.POST.get('target_time')

        with open("dataset/model_meta.json") as f:
            meta = json.load(f)
        with open("dataset/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("dataset/road_graph.pkl", "rb") as f:
            G = pickle.load(f)

        zones = meta['zones']
        zone_to_idx = meta['zone_to_idx']
        NUM_NODES = meta['num_nodes']
        SEQ_LEN = meta['seq_len']
        PRED_LEN = meta['pred_len']

        df = pd.read_csv("dataset/od_cleaned.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_steps = sorted(df['timestamp'].unique())
        recent = time_steps[-SEQ_LEN:]

        features = []
        od_matrix = np.zeros((NUM_NODES, NUM_NODES))
        for ts in recent:
            sub = df[df['timestamp'] == ts]
            mat = np.zeros((NUM_NODES, NUM_NODES))
            for _, row in sub.iterrows():
                oi = zone_to_idx.get(row['origin_zone'])
                di = zone_to_idx.get(row['dest_zone'])
                if oi is not None and di is not None:
                    mat[oi, di] = row['od_flow']
            features.append([mat.sum(axis=1), mat.sum(axis=0)])

        feat = np.array(features)  # [SEQ_LEN, 2, N]
        feat = feat.transpose(2, 0, 1)  # [N, SEQ_LEN, 2]
        feat_flat = feat.reshape(-1, 2)
        feat_flat = scaler.transform(feat_flat)
        feat = feat_flat.reshape(NUM_NODES, SEQ_LEN, 2)

        edges = list(G.edges())
        edge_index = torch.tensor(
            [[zone_to_idx[u], zone_to_idx[v]] for u, v in edges], dtype=torch.long
        ).t().contiguous()

        model = STGAT(in_channels=2, hidden_channels=64, out_channels=PRED_LEN)
        model.load_state_dict(torch.load("dataset/best_stgat.pth", map_location='cpu'))
        model.eval()

        x = torch.tensor(feat, dtype=torch.float32)
        with torch.no_grad():
            pred = model(x, edge_index).numpy()  # [N, PRED_LEN]

        # Inverse scale
        pred_inv = scaler.inverse_transform(
            np.column_stack([pred[:, 0], np.zeros(NUM_NODES)])
        )[:, 0]

        results = [
            {"zone": zones[i], "predicted_outflow": round(float(pred_inv[i]), 2)}
            for i in range(NUM_NODES)
        ]
        results = sorted(results, key=lambda x: x['predicted_outflow'], reverse=True)

        # Save to DB
        for r in results:
            ODPrediction.objects.create(
                user=request.user,
                origin_zone=r['zone'],
                dest_zone="All",
                predicted_flow=r['predicted_outflow'],
                target_interval=target_time,
                confidence_score=0.85
            )

        return render(request, 'prediction/results.html', {'results': results, 'origin': origin})
    return redirect('predict')
