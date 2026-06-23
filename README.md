# Mobile Data-Driven Spatio-Temporal Graph Network for Urban Forecasting

A deep learning system for **Origin-Destination (OD) flow prediction** in urban areas using mobile data and Spatio-Temporal Graph Neural Networks (STGNN).

---

## Overview

Traditional OD estimation methods like surveys and fixed sensors are costly and fail to capture dynamic travel behavior. This project leverages mobile data and graph neural networks to predict urban travel demand accurately and in near real-time.

---

## Features

- Mobile data ingestion (GPS, CDR, IoT sensors)
- Dynamic graph construction — zones as nodes, travel demand as edges
- STGNN model combining graph convolutions with LSTM/GRU layers
- Interactive dashboard with heatmaps and flow visualizations
- Admin panel for dataset management, training logs, and metrics
- Anonymized and encrypted data handling

---



## Methodology

1. Data Preprocessing
Missing value handling
Noise removal
Timestamp formatting
OD matrix generation
Feature normalization
2. Graph Construction
Each zone is represented as a node.
Edges represent:
Geographic adjacency
Mobility connectivity
Travel interactions
3. Spatial Learning
A Graph Attention Network (GAT) learns relationships between neighboring zones using attention mechanisms.
4. Temporal Learning
Historical OD flow sequences are processed to capture temporal mobility patterns.
5. ST-GAT Model
The spatial and temporal components are integrated to generate future OD flow predictions.
---


## Tech Stack

| Layer            | Tools 
| Web Framework    | Django 
| Deep Learning    | PyTorch, PyTorch Geometric 
| Data Processing  | Pandas, NumPy, Scikit-learn 
| Visualization    | Plotly, Matplotlib 
| Database         | MySQL / PostgreSQL 
| Language         | Python 3.9+ 
---



## Setup and Installation
```bash
git clone https://github.com/<shirisha seetha70>/urban-od-forecasting.git
cd urban-od-forecasting


#  Always stay in the project root folder
# step 1 - create virtual environment 
python -m venv mobile_venv (****if not created*)

# step 2 - activate environment
mobile_venv\Scripts\activate

# step 3 - install requiremts
pip install -r requirements.txt

# step 4 - create database in mysql command line client 
create database urban_od_db;

# step 5 - make migrations
python manage.py makemigrations
python manage.py migrate

-----------------ALREADY EXITS--------------------------

# Step 6 — Generate dataset
python generate_dataset.py

# Step 7 — DTW cleaning
python ml_pipeline/dtw_cleaner.py

# Step 8 — Build graph
python ml_pipeline/graph_builder.py

# Step 9 — Train model
python ml_pipeline/train_stgat.py

# Step 10 — Evaluate (optional, shows MSE/MAE/RMSE)
python ml_pipeline/evaluate_model.py

---------------------------------------------------



-----CONTINUE FROM HERE-------

# step 11 - createsuperuser
python manage.py createsuperuser

# Step 12 — Run Django server
python manage.py runserver
```
---

## Usage

| URL | Description |
|---|---|
| `/admin_panel/data/` | Upload mobility datasets |
| `/admin_panel/train/` | View training logs |
| `/admin_panel/metrics/` | MAE, RMSE, MSE scores |
| `/admin_panel/graph/` | Road graph visualization |
| `/trends/` | Zone-wise temporal mobility trends |
---

## Conclusion
The ST-GAT Resident OD Prediction Model demonstrates how spatial-temporal deep learning can be used to predict Origin-Destination (OD) flows from mobile signaling data. By combining spatial relationships between zones with temporal travel patterns, the model is able to generate accurate mobility forecasts.
The project provides a complete pipeline including data preprocessing, graph construction, model training, evaluation, and prediction. The results highlight the potential of Graph Neural Networks for transportation analysis, urban planning, and smart city applications. Future enhancements can focus on real-time prediction, additional data sources, and deployment for practical transportation management systems.

---
