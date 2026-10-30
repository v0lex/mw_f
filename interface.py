

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/cycling_gnn/data_2025_26/"

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import xgboost as xgb
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear

HIDDEN_DIM = 128
HEADS = 32
device = torch.device("cpu")

class RacePredictionGNN2Layer(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, heads=HEADS, dropout_p=0.3):
        super().__init__()
        self.input_proj = Linear(in_dim, hidden_dim)
        self.gat_h2h_1  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team_1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.combine_1  = Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.dropout_1  = nn.Dropout(dropout_p)
        self.gat_h2h_2  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team_2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.combine_2  = Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.dropout_2  = nn.Dropout(dropout_p)
        self.head = Linear(hidden_dim, 1)

    def forward(self, data):
        h2h_ei = data["rider", "head_to_head", "rider"].edge_index
        h2h_ea = data["rider", "head_to_head", "rider"].edge_attr
        team_ei = data["rider", "teammate_of", "rider"].edge_index
        team_ea = data["rider", "teammate_of", "rider"].edge_attr
        x = F.relu(self.input_proj(data["rider"].x))
        x_h2h  = F.elu(self.gat_h2h_1(x, h2h_ei, edge_attr=h2h_ea)) if h2h_ei.shape[1] > 0 else torch.zeros_like(x)
        x_team = F.elu(self.gat_team_1(x, team_ei, edge_attr=team_ea)) if team_ei.shape[1] > 0 else torch.zeros_like(x)
        x1 = self.dropout_1(F.relu(self.combine_1(torch.cat([x, x_h2h, x_team], dim=1))))
        x_h2h_2  = F.elu(self.gat_h2h_2(x1, h2h_ei, edge_attr=h2h_ea)) if h2h_ei.shape[1] > 0 else torch.zeros_like(x1)
        x_team_2 = F.elu(self.gat_team_2(x1, team_ei, edge_attr=team_ea)) if team_ei.shape[1] > 0 else torch.zeros_like(x1)
        x2 = self.dropout_2(F.relu(self.combine_2(torch.cat([x1, x_h2h_2, x_team_2], dim=1))))
        return self.head(x2).squeeze(-1)

class RacePredictionGNN2LayerTop10(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, heads=HEADS, dropout_p=0.3):
        super().__init__()
        self.input_proj = Linear(in_dim, hidden_dim)
        self.gat_h2h_1  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team_1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.combine_1  = Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.dropout_1  = nn.Dropout(dropout_p)
        self.gat_h2h_2  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team_2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.combine_2  = Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.dropout_2  = nn.Dropout(dropout_p)
        self.head = Linear(hidden_dim, 1)
        self.win_head = Linear(hidden_dim, 1)

    def forward(self, data):
        h2h_ei = data["rider", "head_to_head", "rider"].edge_index
        h2h_ea = data["rider", "head_to_head", "rider"].edge_attr
        team_ei = data["rider", "teammate_of", "rider"].edge_index
        team_ea = data["rider", "teammate_of", "rider"].edge_attr
        x = F.relu(self.input_proj(data["rider"].x))
        x_h2h  = F.elu(self.gat_h2h_1(x, h2h_ei, edge_attr=h2h_ea)) if h2h_ei.shape[1] > 0 else torch.zeros_like(x)
        x_team = F.elu(self.gat_team_1(x, team_ei, edge_attr=team_ea)) if team_ei.shape[1] > 0 else torch.zeros_like(x)
        x1 = self.dropout_1(F.relu(self.combine_1(torch.cat([x, x_h2h, x_team], dim=1))))
        x_h2h_2  = F.elu(self.gat_h2h_2(x1, h2h_ei, edge_attr=h2h_ea)) if h2h_ei.shape[1] > 0 else torch.zeros_like(x1)
        x_team_2 = F.elu(self.gat_team_2(x1, team_ei, edge_attr=team_ea)) if team_ei.shape[1] > 0 else torch.zeros_like(x1)
        x2 = self.dropout_2(F.relu(self.combine_2(torch.cat([x1, x_h2h_2, x_team_2], dim=1))))
        return self.head(x2).squeeze(-1), self.win_head(x2).squeeze(-1)

def slice_graph_features(g, feature_indices):
    g2 = g.clone()
    g2["rider"].x = g["rider"].x[:, feature_indices]
    return g2

def discipline_for_parcours(parcours_type):
    return {"flat": "sprint", "punchy": "punchy", "hilly": "versatile", "mountain": "climbing"}.get(parcours_type, "sprint")

FULL_COLS = list(range(14))
DISCIPLINES = ["sprint", "punchy", "versatile", "climbing"]

races = json.load(open(DATA_DIR + "races_2025_26_1.json"))
riders = json.load(open(DATA_DIR + "riders_2025_26_1.json"))
results = json.load(open(DATA_DIR + "results_2025_26_1.json"))
race_map = {r["race_id"]: r for r in races}
rider_name_map = {r["rider_id"]: r.get("rider_name", r["rider_id"]) for r in riders}

checkpoint = torch.load(DATA_DIR + "race_prediction_graphs_cutoff_wf_15.pt", weights_only=False)
test_graphs = checkpoint["test_graphs"]
rider_to_idx = checkpoint["rider_to_idx"]
idx_to_rider = checkpoint["idx_to_rider"]

race_id_to_discipline_scores = torch.load(DATA_DIR + "race_id_to_discipline_scores_15.pt", weights_only=False)

model_xgb = xgb.XGBRanker()
model_xgb.load_model(DATA_DIR + "interface_xgb.json")

ckpt_listnet = torch.load(DATA_DIR + "interface_listnet_2L.pt", map_location="cpu", weights_only=False)
model_listnet = RacePredictionGNN2Layer(in_dim=ckpt_listnet["in_dim"], dropout_p=ckpt_listnet["dropout_p"])
model_listnet.load_state_dict(ckpt_listnet["state_dict"])
model_listnet.eval()
LISTNET_FEATURES = ckpt_listnet["feature_indices"]

ckpt_top10 = torch.load(DATA_DIR + "interface_top10_2L_win.pt", map_location="cpu", weights_only=False)
model_top10 = RacePredictionGNN2LayerTop10(in_dim=ckpt_top10["in_dim"], dropout_p=ckpt_top10["dropout_p"])
model_top10.load_state_dict(ckpt_top10["state_dict"])
model_top10.eval()
TOP10_FEATURES = ckpt_top10["feature_indices"]

RACE_CHOICES = sorted(test_graphs.keys())
MODEL_CHOICES = ["Pure sort (discipline score)", "XGBoost", "ListNet (2-layer)", "Top-10 classifier (2-layer, win head)"]

final_model_scores = torch.load(DATA_DIR + "final_model_discipline_scores.pt", weights_only=False)

def get_top20(discipline):
    scores_full = final_model_scores[discipline]
    order = np.argsort(-scores_full)[:20]
    rows = []
    for rank, idx in enumerate(order, 1):
        rider_id = idx_to_rider.get(idx, {}).get("rider_id") if isinstance(idx_to_rider.get(idx), dict) else idx_to_rider.get(idx)
        name = rider_name_map.get(rider_id, str(rider_id))
        rows.append([rank, name, round(float(scores_full[idx]), 4)])
    return rows

rider_race_contributions = torch.load(DATA_DIR + "rider_race_contributions.pt", weights_only=False)

def get_rider_contributions(discipline, evt: gr.SelectData):
    row_data = evt.row_value
    rider_name = row_data[1]
    rider_id = next((rid for rid, name in rider_name_map.items() if name == rider_name), None)
    if rider_id is None:
        return [["(rider not found)", "", ""]]

    records = rider_race_contributions.get(rider_id, {}).get(discipline)
    if not records:
        return [["(no results in this discipline)", "", ""]]

    rows = [[r["race_id"], round(r["rank_normalized"], 3), round(r["contribution"], 4)] for r in records]
    return rows

def predict_race(race_id, model_choice):
    if race_id not in test_graphs:
        return [["(race not found)", "", ""]]

    g, rider_ids, targets = test_graphs[race_id]
    target_normalized = g["rider"].y.cpu().numpy()
    n = len(rider_ids)
    actual_rank = {i: int(round(target_normalized[i] * n)) for i in range(n)}

    if model_choice == "Pure sort (discipline score)":
        parcours = race_map.get(race_id, {}).get("parcours_type", "flat")
        disc = discipline_for_parcours(parcours)
        scores_full = race_id_to_discipline_scores[race_id][disc]
        global_idx = [rider_to_idx[rid] for rid in rider_ids if rid in rider_to_idx]
        pred = -np.array([scores_full[gidx] for gidx in global_idx])

    elif model_choice == "XGBoost":
        x = g["rider"].x.cpu().numpy()[:, FULL_COLS]
        pred = -model_xgb.predict(x)

    elif model_choice == "ListNet (2-layer)":
        g_dev = slice_graph_features(g, LISTNET_FEATURES)
        with torch.no_grad():
            pred = model_listnet(g_dev).cpu().numpy()

    else:
        g_dev = slice_graph_features(g, TOP10_FEATURES)
        with torch.no_grad():
            top10_logit, _ = model_top10(g_dev)
            pred = -torch.sigmoid(top10_logit).cpu().numpy()

    pred_order = np.argsort(pred)
    rows = []
    for pred_rank, idx in enumerate(pred_order, 1):
        name = rider_name_map.get(rider_ids[idx], rider_ids[idx])
        rows.append([pred_rank, name, actual_rank[idx]])
    return rows

with gr.Blocks(title="Women's Cycling Discipline Rankings & Race Prediction") as demo:
    gr.Markdown("# Discipline Rankings & Race Prediction")

    with gr.Tab("Discipline Rankings"):
        disc_dropdown = gr.Dropdown(choices=DISCIPLINES, value="sprint", label="Discipline")
        top20_table = gr.Dataframe(headers=["Rank", "Rider", "Score"], interactive=False)
        gr.Markdown("Click a row above to see that rider's race-by-race results in this discipline.")
        contrib_table = gr.Dataframe(
            headers=["Race", "Rank (normalized)", "Contribution to score"],
            interactive=False,
        )
        disc_dropdown.change(get_top20, inputs=disc_dropdown, outputs=top20_table)
        top20_table.select(get_rider_contributions, inputs=disc_dropdown, outputs=contrib_table)
        demo.load(lambda: get_top20("sprint"), outputs=top20_table)

    with gr.Tab("Race Prediction"):
        race_dropdown = gr.Dropdown(choices=RACE_CHOICES, label="Race")
        model_dropdown = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")
        predict_btn = gr.Button("Predict")
        gr.Markdown(
            "Actual results are shown for comparison because these are past races from the "
            "test set. Prediction for races without a known result is not yet supported."
        )
        pred_table = gr.Dataframe(headers=["Predicted rank", "Rider", "Actual rank"], interactive=False)
        predict_btn.click(predict_race, inputs=[race_dropdown, model_dropdown], outputs=pred_table)

demo.launch(share = True)
