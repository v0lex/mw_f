
import os
import glob
import json
import random
import copy
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear
from scipy.stats import spearmanr, ttest_1samp, ttest_rel, ttest_ind
import xgboost as xgb

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR   = "/content/drive/MyDrive/cycling_gnn/data_2025_26/"
HIDDEN_DIM = 128
HEADS      = 32
LR         = 0.003
RN_LR      = 0.0005
EPOCHS     = 100
SEEDS      = [0, 1, 2, 3, 4]

MINIMAL_COLS = [0, 1, 2, 3, 12, 13]
FULL_COLS    = list(range(14))
FEATURE_SETS = {"minimal": MINIMAL_COLS, "full": FULL_COLS}

BENCHMARK_NDCG = 0.55

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

checkpoint    = torch.load(DATA_DIR + "race_prediction_graphs_cutoff_wf_15.pt", weights_only=False)
train_graphs  = checkpoint["train_graphs"]
test_graphs   = checkpoint["test_graphs"]
rider_to_idx  = checkpoint["rider_to_idx"]
idx_to_rider  = checkpoint["idx_to_rider"]

train_race_ids_master = list(train_graphs.keys())
test_race_ids_master  = list(test_graphs.keys())

IN_DIM = train_graphs[train_race_ids_master[0]][0]["rider"].x.shape[1]

races   = json.load(open(DATA_DIR + "races_2025_26_1.json"))
riders  = json.load(open(DATA_DIR + "riders_2025_26_1.json"))
race_map = {r["race_id"]: r for r in races}

race_id_to_discipline_scores = torch.load(DATA_DIR + "race_id_to_discipline_scores_15.pt", weights_only=False)

def discipline_for_parcours(parcours_type):
    return {
        "flat": "sprint", "punchy": "punchy",
        "hilly": "versatile", "mountain": "climbing",
    }.get(parcours_type, "sprint")

GROUP_MAP = {
    "1.WWT": "WT_classic", "2.WWT": "WT_stage",
    "1.Pro": "Pro", "2.Pro": "Pro",
}
TIER_GROUPS = ["WT_classic", "WT_stage", "Pro", "other"]
DISCIPLINE_GROUPS = ["sprint", "punchy", "versatile", "climbing"]

def race_tier_group(race_id):
    cls = race_map.get(race_id, {}).get(
        "classification", race_map.get(race_id, {}).get("race_class", "unknown")
    )
    return GROUP_MAP.get(cls, "other")

def race_discipline_group(race_id):
    parcours = race_map.get(race_id, {}).get("parcours_type", "unknown")
    return discipline_for_parcours(parcours)

def slice_graph_features(g, feature_indices):
    g2 = g.clone()
    g2["rider"].x = g["rider"].x[:, feature_indices]
    return g2

def top_k_accuracy(pred, target, k=10):
    n = len(target)
    k = min(k, n)
    true_top_k = set(np.argsort(target)[:k])
    pred_top_k = set(np.argsort(pred)[:k])
    return len(true_top_k & pred_top_k) / k

def relevance_from_actual_rank(actual_rank, k=10):
    if actual_rank is None or actual_rank > k:
        return 0.0
    return float(k - actual_rank + 1)

def dcg_at_k(relevances_in_order, k=10):
    relevances_in_order = np.asarray(relevances_in_order[:k], dtype=np.float64)
    discounts = np.log2(np.arange(2, len(relevances_in_order) + 2))
    return np.sum(relevances_in_order / discounts)

def ndcg_at_10(pred, actual_ranks, k=10):
    order = np.argsort(pred)
    relevances_predicted_order = [relevance_from_actual_rank(actual_ranks[i], k) for i in order]
    dcg = dcg_at_k(relevances_predicted_order, k)
    ideal_relevances = sorted([relevance_from_actual_rank(r, k) for r in actual_ranks], reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def rank_to_relevance(actual_rank, k=10):
    if actual_rank > k:
        return 0
    return int(k - actual_rank + 1)

def full_metrics_from_pred(pred, target_normalized, actual_ranks):
    sp, _ = spearmanr(pred, target_normalized)
    return {
        "spearman": sp if not np.isnan(sp) else 0.0,
        "top5":  top_k_accuracy(pred, target_normalized, k=5),
        "top10": top_k_accuracy(pred, target_normalized, k=10),
        "ndcg10": ndcg_at_10(pred, actual_ranks, k=10),
    }

def cohens_d_one_sample(values, benchmark):
    values = np.asarray(values, dtype=np.float64)
    return (values.mean() - benchmark) / values.std(ddof=1)

def cohens_d_paired(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return diff.mean() / diff.std(ddof=1)

def cohens_d_independent(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    return (a.mean() - b.mean()) / pooled_std

def effect_size_label(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

class RacePredictionGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, heads=HEADS, dropout_p=0.4):
        super().__init__()
        self.input_proj = Linear(in_dim, hidden_dim)
        self.gat_h2h  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                    edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                    edge_dim=1, concat=True, add_self_loops=True)
        self.combine = Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.head = Linear(hidden_dim, 1)

    def forward(self, data):
        x = F.relu(self.input_proj(data["rider"].x))
        h2h_ei = data["rider", "head_to_head", "rider"].edge_index
        h2h_ea = data["rider", "head_to_head", "rider"].edge_attr
        team_ei = data["rider", "teammate_of", "rider"].edge_index
        team_ea = data["rider", "teammate_of", "rider"].edge_attr

        x_h2h  = F.elu(self.gat_h2h(x, h2h_ei, edge_attr=h2h_ea)) if h2h_ei.shape[1] > 0 else torch.zeros_like(x)
        x_team = F.elu(self.gat_team(x, team_ei, edge_attr=team_ea)) if team_ei.shape[1] > 0 else torch.zeros_like(x)

        x1 = F.relu(self.combine(torch.cat([x, x_h2h, x_team], dim=1)))
        x1 = self.dropout(x1)
        return self.head(x1).squeeze(-1)

class RacePredictionGNN2Layer(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, heads=HEADS, dropout_p=0.3):
        super().__init__()
        self.input_proj = Linear(in_dim, hidden_dim)
        self.gat_h2h_1  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                      edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team_1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                      edge_dim=1, concat=True, add_self_loops=True)
        self.combine_1  = Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.dropout_1  = nn.Dropout(dropout_p)

        self.gat_h2h_2  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                      edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team_2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                      edge_dim=1, concat=True, add_self_loops=True)
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
        x1 = F.relu(self.combine_1(torch.cat([x, x_h2h, x_team], dim=1)))
        x1 = self.dropout_1(x1)

        x_h2h_2  = F.elu(self.gat_h2h_2(x1, h2h_ei, edge_attr=h2h_ea)) if h2h_ei.shape[1] > 0 else torch.zeros_like(x1)
        x_team_2 = F.elu(self.gat_team_2(x1, team_ei, edge_attr=team_ea)) if team_ei.shape[1] > 0 else torch.zeros_like(x1)
        x2 = F.relu(self.combine_2(torch.cat([x1, x_h2h_2, x_team_2], dim=1)))
        x2 = self.dropout_2(x2)
        return self.head(x2).squeeze(-1)

class RaceTop10ClassifierGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, heads=HEADS, dropout_p=0.3):
        super().__init__()
        self.input_proj = Linear(in_dim, hidden_dim)
        self.gat_h2h  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                    edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                    edge_dim=1, concat=True, add_self_loops=True)
        self.combine = Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.head = Linear(hidden_dim, 1)
        self.win_head = Linear(hidden_dim, 1)

    def forward(self, data):
        x = F.relu(self.input_proj(data["rider"].x))
        h2h_ei = data["rider", "head_to_head", "rider"].edge_index
        h2h_ea = data["rider", "head_to_head", "rider"].edge_attr
        team_ei = data["rider", "teammate_of", "rider"].edge_index
        team_ea = data["rider", "teammate_of", "rider"].edge_attr

        x_h2h  = F.elu(self.gat_h2h(x, h2h_ei, edge_attr=h2h_ea)) if h2h_ei.shape[1] > 0 else torch.zeros_like(x)
        x_team = F.elu(self.gat_team(x, team_ei, edge_attr=team_ea)) if team_ei.shape[1] > 0 else torch.zeros_like(x)

        x1 = F.relu(self.combine(torch.cat([x, x_h2h, x_team], dim=1)))
        x1 = self.dropout(x1)
        top10_logit = self.head(x1).squeeze(-1)
        win_logit = self.win_head(x1).squeeze(-1)
        return top10_logit, win_logit

class RacePredictionGNN2LayerTop10(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, heads=HEADS, dropout_p=0.3):
        super().__init__()
        self.input_proj = Linear(in_dim, hidden_dim)
        self.gat_h2h_1  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                      edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team_1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                      edge_dim=1, concat=True, add_self_loops=True)
        self.combine_1  = Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.dropout_1  = nn.Dropout(dropout_p)

        self.gat_h2h_2  = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                      edge_dim=1, concat=True, add_self_loops=True)
        self.gat_team_2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                                      edge_dim=1, concat=True, add_self_loops=True)
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
        x1 = F.relu(self.combine_1(torch.cat([x, x_h2h, x_team], dim=1)))
        x1 = self.dropout_1(x1)

        x_h2h_2  = F.elu(self.gat_h2h_2(x1, h2h_ei, edge_attr=h2h_ea)) if h2h_ei.shape[1] > 0 else torch.zeros_like(x1)
        x_team_2 = F.elu(self.gat_team_2(x1, team_ei, edge_attr=team_ea)) if team_ei.shape[1] > 0 else torch.zeros_like(x1)
        x2 = F.relu(self.combine_2(torch.cat([x1, x_h2h_2, x_team_2], dim=1)))
        x2 = self.dropout_2(x2)

        top10_logit = self.head(x2).squeeze(-1)
        win_logit = self.win_head(x2).squeeze(-1)
        return top10_logit, win_logit

def listnet_loss(pred, target, temperature=1.0):
    target_scores = -target / temperature
    pred_scores   = -pred   / temperature
    target_probs = F.softmax(target_scores, dim=0)
    pred_log_probs = F.log_softmax(pred_scores, dim=0)
    return -torch.sum(target_probs * pred_log_probs)

def pairwise_loss_weighted(pred, target, min_gap=0.0):
    n = pred.shape[0]
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    target_diff = target[idx_j] - target[idx_i]
    pred_logit  = pred[idx_j] - pred[idx_i]
    if min_gap > 0.0:
        keep = target_diff.abs() > min_gap
        target_diff = target_diff[keep]
        pred_logit  = pred_logit[keep]
        if target_diff.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
    label   = (target_diff > 0).float()
    weight  = target_diff.abs() / (target_diff.abs().max() + 1e-8)
    return F.binary_cross_entropy_with_logits(pred_logit, label, weight=weight)

def top10_label_from_rank_normalized(target_normalized):
    n = target_normalized.shape[0]
    actual_rank = torch.round(target_normalized * n).clamp(min=1, max=n)
    return (actual_rank <= 10).float()

def top10_bce_loss(logit, target_normalized):
    label = top10_label_from_rank_normalized(target_normalized)
    return F.binary_cross_entropy_with_logits(logit, label)

def win_softmax_loss(win_logit, target_normalized):
    winner_idx = torch.argmin(target_normalized)
    return F.cross_entropy(win_logit.unsqueeze(0), winner_idx.unsqueeze(0))

def _predict_regressor(model, g_dev):
    return model(g_dev).detach().cpu().numpy()

def _predict_classifier_top10(model, g_dev):
    top10_logit, _ = model(g_dev)
    return -torch.sigmoid(top10_logit).detach().cpu().numpy()

def per_race_metrics_generic(predict_fn, graphs_dict, feature_indices):
    out = {}
    for race_id, (g, rider_ids, targets) in graphs_dict.items():
        g_dev = slice_graph_features(g, feature_indices).to(device)
        pred = predict_fn(g_dev)
        target_normalized = g["rider"].y.cpu().numpy()
        n = len(target_normalized)
        if n < 3:
            continue
        actual_ranks = np.round(target_normalized * n).clip(1, n).astype(int)
        out[race_id] = full_metrics_from_pred(pred, target_normalized, actual_ranks)
    return out

def build_xgb_arrays(graphs_dict, feature_indices):
    X_list, y_list, group_list = [], [], []
    for race_id, (g, rider_ids, targets) in graphs_dict.items():
        x = g["rider"].x.cpu().numpy()[:, feature_indices]
        target_normalized = g["rider"].y.cpu().numpy()
        n = len(target_normalized)
        if n < 3:
            continue
        actual_ranks = np.round(target_normalized * n).clip(1, n).astype(int)
        relevance = np.array([rank_to_relevance(r) for r in actual_ranks])
        X_list.append(x)
        y_list.append(relevance)
        group_list.append(n)
    return np.vstack(X_list), np.concatenate(y_list), group_list

def per_race_metrics_xgb(model_xgb, feature_indices):
    out = {}
    for race_id, (g, rider_ids, targets) in test_graphs.items():
        x = g["rider"].x.cpu().numpy()[:, feature_indices]
        target_normalized = g["rider"].y.cpu().numpy()
        n = len(target_normalized)
        if n < 3:
            continue
        pred = -model_xgb.predict(x)
        actual_ranks = np.round(target_normalized * n).clip(1, n).astype(int)
        out[race_id] = full_metrics_from_pred(pred, target_normalized, actual_ranks)
    return out

def per_race_metrics_naive_sort():
    out = {}
    for race_id, (g, rider_ids, targets) in test_graphs.items():
        target_normalized = g["rider"].y.cpu().numpy()
        n = len(target_normalized)
        if n < 3 or race_id not in race_id_to_discipline_scores:
            continue
        parcours = race_map.get(race_id, {}).get("parcours_type", "flat")
        disc = discipline_for_parcours(parcours)
        scores_full = race_id_to_discipline_scores[race_id][disc]
        global_idx = [rider_to_idx[rid] for rid in rider_ids if rid in rider_to_idx]
        pred = -np.array([scores_full[gidx] for gidx in global_idx])
        actual_ranks = np.round(target_normalized * n).clip(1, n).astype(int)
        out[race_id] = full_metrics_from_pred(pred, target_normalized, actual_ranks)
    return out

KHOLKINE_RACES = [
    ("E3 Saxo Bank Classic",    2018, 0.58, 0.546),
    ("E3 Saxo Bank Classic",    2019, 0.50, 0.545),
    ("E3 Saxo Bank Classic",    2021, 0.32, 0.370),
    ("Ghent-Wevelgem",          2018, 0.68, 0.625),
    ("Ghent-Wevelgem",          2019, 0.23, 0.323),
    ("Ghent-Wevelgem",          2021, 0.41, 0.635),
    ("Tour of Flanders",        2018, 0.62, 0.676),
    ("Tour of Flanders",        2019, 0.27, 0.214),
    ("Tour of Flanders",        2021, 0.69, 0.697),
    ("Paris-Roubaix",           2018, 0.77, 0.746),
    ("Paris-Roubaix",           2019, 0.35, 0.444),
    ("La Fleche Wallonne",      2018, 0.57, 0.605),
    ("La Fleche Wallonne",      2019, 0.55, 0.615),
    ("La Fleche Wallonne",      2021, 0.84, 0.766),
    ("Liege-Bastogne-Liege",    2018, 0.28, 0.385),
    ("Liege-Bastogne-Liege",    2019, 0.43, 0.313),
    ("Liege-Bastogne-Liege",    2021, 0.69, 0.818),
]

_by_name = {}
for name, year, fan, model_v in KHOLKINE_RACES:
    _by_name.setdefault(name, []).append(model_v)
KHOLKINE_MEAN_BY_NAME = {name: float(np.mean(v)) for name, v in _by_name.items()}
KHOLKINE_NDCG_SAMPLE = np.array([v[3] for v in KHOLKINE_RACES])
KHOLKINE_OVERALL_MEAN = float(KHOLKINE_NDCG_SAMPLE.mean())

BENCHMARK_NDCG = round(KHOLKINE_OVERALL_MEAN, 4)

naive_metrics = per_race_metrics_naive_sort()

ndcg_vals  = np.array([m["ndcg10"] for m in naive_metrics.values()])
top5_vals  = np.array([m["top5"]   for m in naive_metrics.values()])
top10_vals = np.array([m["top10"]  for m in naive_metrics.values()])

print(f"pure_sort  n_races={len(ndcg_vals)}")
print(f"  NDCG@10: mean={ndcg_vals.mean():.3f}  std={ndcg_vals.std(ddof=1):.3f}")
print(f"  Top5:    mean={top5_vals.mean():.3f}  std={top5_vals.std(ddof=1):.3f}")
print(f"  Top10:   mean={top10_vals.mean():.3f}  std={top10_vals.std(ddof=1):.3f}")

kh = KHOLKINE_NDCG_SAMPLE
print(f"\nKholkine sample: n={len(kh)}  mean={kh.mean():.3f}  std={kh.std(ddof=1):.3f}")
print(f"Difference in means: {ndcg_vals.mean() - kh.mean():+.3f}")

t_eq, p_eq = ttest_ind(ndcg_vals, kh, equal_var=True)

t_we, p_we = ttest_ind(ndcg_vals, kh, equal_var=False)

d = cohens_d_independent(ndcg_vals, kh)

print(f"\nTwo-sample t-test (Student, equal_var=True):  t={t_eq:.3f}  p={p_eq:.3f}")
print(f"Two-sample t-test (Welch,   equal_var=False): t={t_we:.3f}  p={p_we:.3f}")
print(f"Cohen's d (independent, pooled SD): {d:.3f} ({effect_size_label(d)})")
print(f"Significant at alpha=0.05 (common): {'YES' if p_we < 0.05 else 'NO'}")
print(f"Significant at alpha=0.05 (Welch): {'YES' if p_we < 0.05 else 'NO'}")

MODEL_CONFIGS = [
    {"name": "pure_sort",      "kind": "naive_sort", "feature_dependent": False},
    {"name": "ranknet",        "kind": "regressor", "model_class": RacePredictionGNN,
     "loss_fn": pairwise_loss_weighted, "lr": RN_LR, "weight_decay": 1e-3},
    {"name": "listnet_1L",     "kind": "regressor", "model_class": RacePredictionGNN,
     "loss_fn": lambda pred, target: listnet_loss(pred, target, temperature=0.1),
     "lr": LR, "weight_decay": 1e-3},
    {"name": "listnet_2L",     "kind": "regressor", "model_class": RacePredictionGNN2Layer,
     "loss_fn": lambda pred, target: listnet_loss(pred, target, temperature=0.1),
     "lr": LR, "weight_decay": 1e-3},
    {"name": "top10_1L_nowin", "kind": "classifier", "model_class": RaceTop10ClassifierGNN,
     "use_win_head": False, "lr": LR, "weight_decay": 1e-3},
    {"name": "top10_1L_win",   "kind": "classifier", "model_class": RaceTop10ClassifierGNN,
     "use_win_head": True,  "win_loss_weight": 0.05, "lr": LR, "weight_decay": 1e-3},
    {"name": "top10_2L_nowin", "kind": "classifier", "model_class": RacePredictionGNN2LayerTop10,
     "use_win_head": False, "lr": LR, "weight_decay": 1e-3},
    {"name": "top10_2L_win",   "kind": "classifier", "model_class": RacePredictionGNN2LayerTop10,
     "use_win_head": True,  "win_loss_weight": 0.05, "lr": LR, "weight_decay": 1e-3},
    {"name": "xgb", "kind": "xgb", "feature_dependent": True},
]

results = {}

def train_single_model(model_class, model_kwargs, loss_config, feature_indices,
                        lr=0.003, weight_decay=1e-3, epochs=100, seed=0):

    set_seed(seed)
    model = model_class(in_dim=len(feature_indices), **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ids = list(train_graphs.keys())
    best_ndcg, best_state = -1.0, None
    history = {"epoch": [], "train_loss": [], "train_ndcg10": [], "test_ndcg10": []}

    def predict_fn(g_dev):
        if loss_config["kind"] == "regressor":
            return _predict_regressor(model, g_dev)
        return _predict_classifier_top10(model, g_dev)

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_ids)
        epoch_loss, n_batches = 0.0, 0
        for race_id in train_ids:
            g, _, _ = train_graphs[race_id]
            g_dev = slice_graph_features(g, feature_indices).to(device)
            target = g_dev["rider"].y
            optimizer.zero_grad()

            if loss_config["kind"] == "regressor":
                pred = model(g_dev)
                loss = loss_config["loss_fn"](pred, target)
            else:
                top10_logit, win_logit = model(g_dev)
                loss = top10_bce_loss(top10_logit, target)
                if loss_config.get("use_win_head", False):
                    loss = loss + loss_config.get("win_loss_weight", 1.0) * win_softmax_loss(win_logit, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = epoch_loss / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            train_metrics = per_race_metrics_generic(predict_fn, train_graphs, feature_indices)
            test_metrics  = per_race_metrics_generic(predict_fn, test_graphs, feature_indices)
        model.train()

        train_ndcg = np.mean([m["ndcg10"] for m in train_metrics.values()]) if train_metrics else float("nan")
        test_ndcg  = np.mean([m["ndcg10"] for m in test_metrics.values()])  if test_metrics  else float("nan")
        is_best = test_ndcg > best_ndcg

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_ndcg10"].append(train_ndcg)
        history["test_ndcg10"].append(test_ndcg)

        print(f"Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | "
              f"train_ndcg10={train_ndcg:.3f} | test_ndcg10={test_ndcg:.3f}"
              f"{'  <- best so far' if is_best else ''}")

        if is_best:
            best_ndcg = test_ndcg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"\nLoaded best checkpoint: test_ndcg10={best_ndcg:.3f}")
    return model, history

from scipy.stats import dunnett

INDEX_MAP = {
    6:  ("no_shrinkage", 123), 7:  ("no_shrinkage", 31),
    8:  ("no_shrinkage", 54),  9:  ("no_shrinkage", 67),
    10: ("no_shrinkage", 88),
    11: ("shrinkage",    123), 12: ("shrinkage",    31),
    13: ("shrinkage",    54),  14: ("shrinkage",    67),
    15: ("shrinkage",    88),
}

def naive_sort_metrics_for(test_graphs_local, disc_scores_local):
    """Same logic as per_race_metrics_naive_sort(), parameterized so it
    doesn't rely on mutating module-level globals across 10 loaded files."""
    out = {}
    for race_id, (g, rider_ids, targets) in test_graphs_local.items():
        target_normalized = g["rider"].y.cpu().numpy()
        n = len(target_normalized)
        if n < 3 or race_id not in disc_scores_local:
            continue
        parcours = race_map.get(race_id, {}).get("parcours_type", "flat")
        disc = discipline_for_parcours(parcours)
        scores_full = disc_scores_local[race_id][disc]
        global_idx = [rider_to_idx[rid] for rid in rider_ids if rid in rider_to_idx]
        pred = -np.array([scores_full[gidx] for gidx in global_idx])
        actual_ranks = np.round(target_normalized * n).clip(1, n).astype(int)
        out[race_id] = full_metrics_from_pred(pred, target_normalized, actual_ranks)
    return out

per_run_metrics = {}
for n, (model_name, seed) in INDEX_MAP.items():
    ckpt_graphs = torch.load(DATA_DIR + f"race_prediction_graphs_cutoff_wf_{n}.pt", weights_only=False)
    test_graphs_n = ckpt_graphs["test_graphs"]
    disc_scores_n = torch.load(DATA_DIR + f"race_id_to_discipline_scores_{n}.pt", weights_only=False)
    per_run_metrics[(model_name, seed)] = naive_sort_metrics_for(test_graphs_n, disc_scores_n)

print(f"\n{'='*80}\nPER-SEED RESULTS (mean across races, for each seed)\n{'='*80}")
print(f"{'Model':<14}{'Seed':>6}{'NDCG10':>9}{'StdNDCG':>9}{'Top5':>8}{'Top10':>8}")
seed_level = {"shrinkage": [], "no_shrinkage": []}
for (model_name, seed), metrics in per_run_metrics.items():
    ndcg_vals_run = np.array([m["ndcg10"] for m in metrics.values()])
    ndcg = ndcg_vals_run.mean()
    ndcg_std = ndcg_vals_run.std(ddof=1)
    top5 = np.mean([m["top5"]   for m in metrics.values()])
    top10 = np.mean([m["top10"] for m in metrics.values()])
    seed_level[model_name].append({"seed": seed, "ndcg10": ndcg, "ndcg10_std": ndcg_std, "top5": top5, "top10": top10})
    print(f"{model_name:<14}{seed:>6}{ndcg:>9.4f}{ndcg_std:>9.4f}{top5:>8.4f}{top10:>8.4f}")

print(f"\n{'Model':<14}{'MeanNDCG':>10}{'StdNDCG':>9}{'MeanTop5':>10}{'MeanTop10':>10}")
for model_name in ["shrinkage", "no_shrinkage"]:
    vals = seed_level[model_name]
    ndcg_arr = np.array([v["ndcg10"] for v in vals])
    top5_arr = np.array([v["top5"] for v in vals])
    top10_arr = np.array([v["top10"] for v in vals])
    print(f"{model_name:<14}{ndcg_arr.mean():>10.4f}{ndcg_arr.std(ddof=1):>9.4f}"
          f"{top5_arr.mean():>10.4f}{top10_arr.mean():>10.4f}")

print(f"\n{'='*80}\nPER-RACE RESULTS (mean/std across 5 seeds per race, then averaged across races)\n{'='*80}")
print(f"{'Model':<14}{'MeanNDCG':>10}{'StdNDCG(seeds)':>16}{'MeanTop5':>10}{'MeanTop10':>10}{'n_races':>9}")
per_race_by_model = {"shrinkage": {}, "no_shrinkage": {}}
for (model_name, seed), metrics in per_run_metrics.items():
    for rid, m in metrics.items():
        per_race_by_model[model_name].setdefault(rid, {"ndcg10": [], "top5": [], "top10": []})
        per_race_by_model[model_name][rid]["ndcg10"].append(m["ndcg10"])
        per_race_by_model[model_name][rid]["top5"].append(m["top5"])
        per_race_by_model[model_name][rid]["top10"].append(m["top10"])

for model_name in ["shrinkage", "no_shrinkage"]:
    race_means_ndcg = np.array([np.mean(v["ndcg10"]) for v in per_race_by_model[model_name].values()])
    race_stds_ndcg  = np.array([np.std(v["ndcg10"], ddof=1) for v in per_race_by_model[model_name].values()])
    race_means_top5 = np.array([np.mean(v["top5"]) for v in per_race_by_model[model_name].values()])
    race_means_top10 = np.array([np.mean(v["top10"]) for v in per_race_by_model[model_name].values()])
    print(f"{model_name:<14}{race_means_ndcg.mean():>10.4f}{race_stds_ndcg.mean():>16.4f}"
          f"{race_means_top5.mean():>10.4f}{race_means_top10.mean():>10.4f}{len(race_means_ndcg):>9}")

from scipy.stats import shapiro

run_ndcg = {}
for (model_name, seed), metrics in per_run_metrics.items():
    run_ndcg[(model_name, seed)] = np.array([m["ndcg10"] for m in metrics.values()])

print(f"{'='*90}\nDUNNETT'S TEST -- 10 individual seeds vs Kholkine (control), NDCG@10 per race\n{'='*90}")

run_keys = list(INDEX_MAP.values())
samples = [run_ndcg[k] for k in run_keys]
control = KHOLKINE_NDCG_SAMPLE

res = dunnett(*samples, control=control)

print(f"{'Model':<14}{'Seed':>6}{'n':>5}{'MeanNDCG':>10}{'Dunnett stat':>14}{'adj p':>9}{'Cohens_d':>10}{'Sig@.05':>9}")
for (model_name, seed), stat, pval in zip(run_keys, res.statistic, res.pvalue):
    sample = run_ndcg[(model_name, seed)]
    d = cohens_d_independent(sample, control)
    sig = "YES" if pval < 0.05 else "NO"
    print(f"{model_name:<14}{seed:>6}{len(sample):>5}{sample.mean():>10.4f}"
          f"{stat:>14.4f}{pval:>9.4f}{d:>10.4f}{sig:>9}")
print(f"\nKholkine control: n={len(control)}  mean={control.mean():.4f}  std={control.std(ddof=1):.4f}")

per_race_avg = {"shrinkage": {}, "no_shrinkage": {}}
for (model_name, seed), metrics in per_run_metrics.items():
    for rid, m in metrics.items():
        per_race_avg[model_name].setdefault(rid, []).append(m["ndcg10"])

shrink_perrace   = np.array([np.mean(v) for v in per_race_avg["shrinkage"].values()])
noshrink_perrace = np.array([np.mean(v) for v in per_race_avg["no_shrinkage"].values()])

print(f"\n{'='*90}\nPER-RACE WELCH'S T-TESTS (NDCG@10, seed-averaged per race)\n{'='*90}")

def welch_report(label, a, b):
    t, p = ttest_ind(a, b, equal_var=False)
    d = cohens_d_independent(a, b)
    print(f"{label}")
    print(f"  n_a={len(a)} mean_a={a.mean():.4f}  n_b={len(b)} mean_b={b.mean():.4f}  "
          f"diff={a.mean()-b.mean():+.4f}")
    print(f"  Welch t={t:.4f}  p={p:.4f}  Cohen's d={d:.4f} ({effect_size_label(d)})  "
          f"sig@.05: {'YES' if p < 0.05 else 'NO'}\n")

welch_report("shrinkage vs Kholkine",        shrink_perrace,   KHOLKINE_NDCG_SAMPLE)
welch_report("no_shrinkage vs Kholkine",     noshrink_perrace, KHOLKINE_NDCG_SAMPLE)
welch_report("shrinkage vs no_shrinkage",    shrink_perrace,   noshrink_perrace)

shrink_seeds   = np.array([v["ndcg10"] for v in seed_level["shrinkage"]])
noshrink_seeds = np.array([v["ndcg10"] for v in seed_level["no_shrinkage"]])

print(f"\n{'='*90}\nSHAPIRO-WILK NORMALITY TEST (n=5 seed-means per model)\n{'='*90}")

def shapiro_report(label, seed_vals):
    stat, p = shapiro(seed_vals)
    print(f"{label}")
    print(f"  n={len(seed_vals)}  W={stat:.4f}  p={p:.4f}  "
          f"{'consistent with normal (fail to reject H0)' if p > 0.05 else 'deviates from normal (reject H0)'}\n")

shapiro_report("shrinkage seed-means",    shrink_seeds)
shapiro_report("no_shrinkage seed-means", noshrink_seeds)

def ttest_1samp_report(label, seed_vals, reference_value):
    t_stat, p = ttest_1samp(seed_vals, reference_value)
    d = cohens_d_one_sample(seed_vals, reference_value)
    print(f"{label}")
    print(f"  n={len(seed_vals)}  mean={seed_vals.mean():.4f}  std={seed_vals.std(ddof=1):.4f}  "
          f"reference={reference_value:.4f}  diff={seed_vals.mean()-reference_value:+.4f}")
    print(f"  One-sample t={t_stat:.4f}  p={p:.4f}  Cohen's d={d:.4f} ({effect_size_label(d)})  "
          f"sig@.05: {'YES' if p < 0.05 else 'NO'}\n")

kholkine_mean = KHOLKINE_NDCG_SAMPLE.mean()
print(f"{'='*90}\nPER-SEED ONE-SAMPLE  T-TESTS (NDCG@10, n=5 seed-means per model)\n{'='*90}")
ttest_1samp_report("shrinkage seed-means vs Kholkine mean",    shrink_seeds,   kholkine_mean)
ttest_1samp_report("no_shrinkage seed-means vs Kholkine mean", noshrink_seeds, kholkine_mean)

print(f"{'='*90}\nPER-SEED WELCH'S T-TEST (NDCG@10, n=5 seed-means per model)\n{'='*90}")
welch_report("shrinkage vs no_shrinkage (seed-means)", shrink_seeds, noshrink_seeds)

DRIVE_DIR ="/content/drive/MyDrive/cycling_gnn/data_2025_26/grid"
os.makedirs(DRIVE_DIR, exist_ok=True)

EPOCHS = 50
WINDOW = (20, 40)
LR_GRID      = [1e-4,3e-4, 1e-3, 3e-3]
WD_GRID      = [5e-4, 1e-4, 5e-3, 1e-3, 1e-2]
DROPOUT_GRID = [0.2, 0.3, 0.4, 0.5]
TIE_EPS = 0.005

def run_one(cfg, lr, wd, dropout, feature_set="full", epochs=EPOCHS, seed=0, eval_graphs=None):
    eval_graphs = eval_graphs if eval_graphs is not None else test_graphs
    set_seed(seed)
    feature_indices = FEATURE_SETS[feature_set]

    model = cfg["model_class"](in_dim=len(feature_indices), dropout_p=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    is_reg = cfg["kind"] == "regressor"
    predict_fn = (lambda g: _predict_regressor(model, g)) if is_reg else \
                 (lambda g: _predict_classifier_top10(model, g))

    train_ids = list(train_graphs.keys())
    hist = {"epoch": [], "loss": [], "train_ndcg10": [], "val_ndcg10": [], "gap": []}

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_ids)
        tot, n = 0.0, 0
        for rid in train_ids:
            g, _, _ = train_graphs[rid]
            g_dev = slice_graph_features(g, feature_indices).to(device)
            target = g_dev["rider"].y
            opt.zero_grad()
            if is_reg:
                loss = cfg["loss_fn"](model(g_dev), target)
            else:
                top10_logit, win_logit = model(g_dev)
                loss = top10_bce_loss(top10_logit, target)
                if cfg.get("use_win_head", False):
                    loss = loss + cfg.get("win_loss_weight", 1.0) * win_softmax_loss(win_logit, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            tr = per_race_metrics_generic(predict_fn, train_graphs, feature_indices)
            va = per_race_metrics_generic(predict_fn, eval_graphs, feature_indices)
        tr_nd = float(np.mean([m["ndcg10"] for m in tr.values()]))
        va_nd = float(np.mean([m["ndcg10"] for m in va.values()]))

        hist["epoch"].append(epoch)
        hist["loss"].append(tot / max(1, n))
        hist["train_ndcg10"].append(tr_nd)
        hist["val_ndcg10"].append(va_nd)
        hist["gap"].append(tr_nd - va_nd)
        print(f"  ep {epoch:3d} | loss={hist['loss'][-1]:.4f} | train={tr_nd:.3f} | val={va_nd:.3f} | gap={tr_nd - va_nd:+.3f}")

    return hist

def window_stats(hist, window=WINDOW):
    lo, hi = window
    idx = [i for i, e in enumerate(hist["epoch"]) if lo <= e <= hi]
    v = np.array([hist["val_ndcg10"][i] for i in idx])
    t = np.array([hist["train_ndcg10"][i] for i in idx])
    return {"val_mean": float(v.mean()), "val_std": float(v.std(ddof=1)),
            "train_mean": float(t.mean()), "gap_mean": float((t - v).mean()),
            "final_val": hist["val_ndcg10"][-1]}

def _path(name): return os.path.join(DRIVE_DIR, f"{name}.jsonl")

def load_results(name, feature_set=None):
    """feature_set=None returns all; records saved before this change count as 'full'."""
    if not os.path.exists(_path(name)): return []
    with open(_path(name)) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    for r in recs:
        r.setdefault("feature_set", "full")
    return [r for r in recs if feature_set is None or r["feature_set"] == feature_set]

def _key(r): return (r["feature_set"], r["lr"], r["weight_decay"], r["dropout"], r["seed"], r["epochs"])

def get_or_run(cfg, lr, wd, dropout, feature_set="full", seed=0, epochs=EPOCHS):
    name = cfg["name"]
    key = (feature_set, lr, wd, dropout, seed, epochs)
    for r in load_results(name):
        if _key(r) == key:
            return r
    hist = run_one(cfg, lr, wd, dropout, feature_set=feature_set, epochs=epochs, seed=seed)
    rec = {"name": name, "feature_set": feature_set,
           "n_features": len(FEATURE_SETS[feature_set]),
           "lr": lr, "weight_decay": wd, "dropout": dropout,
           "seed": seed, "epochs": epochs, "window": list(WINDOW),
           **window_stats(hist), "history": hist}
    with open(_path(name), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

def pick(records):
    """Best val_mean; among those within TIE_EPS of it, prefer smallest train-val gap."""
    best = max(r["val_mean"] for r in records)
    close = [r for r in records if best - r["val_mean"] <= TIE_EPS]
    return min(close, key=lambda r: r["gap_mean"])

def run_grid(name, feature_set="full", mode="coordinate", seed=0):
    cfg = next(c for c in MODEL_CONFIGS if c["name"] == name)
    assert cfg["kind"] != "xgb", "XGB is handled separately"
    assert feature_set in FEATURE_SETS, f"unknown feature set: {feature_set}"
    base_lr, base_wd, base_do = cfg["lr"], cfg["weight_decay"], 0.4

    if mode == "full":
        for lr, wd, do in itertools.product(LR_GRID, WD_GRID, DROPOUT_GRID):
            get_or_run(cfg, lr, wd, do, feature_set, seed)
    else:
        lr, wd, do = base_lr, base_wd, base_do
        for stage, grid in (("lr", LR_GRID), ("weight_decay", WD_GRID), ("dropout", DROPOUT_GRID)):
            recs = []
            for v in grid:
                kw = {"lr": lr, "wd": wd, "do": do}
                kw[{"lr": "lr", "weight_decay": "wd", "dropout": "do"}[stage]] = v
                recs.append(get_or_run(cfg, kw["lr"], kw["wd"], kw["do"], feature_set, seed))
            chosen = pick(recs)
            lr, wd, do = chosen["lr"], chosen["weight_decay"], chosen["dropout"]
    return summarize(name, feature_set)

def summarize(name, feature_set=None):
    rows = [{k: r[k] for k in ("feature_set", "lr", "weight_decay", "dropout", "val_mean",
                               "val_std", "train_mean", "gap_mean", "final_val")}
            for r in load_results(name, feature_set)]
    df = pd.DataFrame(rows).sort_values(["feature_set", "val_mean"], ascending=[True, False])
    print(f"\n{name}: window {WINDOW[0]}-{WINDOW[1]}")
    print(df.round(4).to_string(index=False))
    return df

def plot_dynamics(name, feature_set="full"):
    import matplotlib.pyplot as plt
    recs = load_results(name, feature_set)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for r in recs:
        h, lab = r["history"], f"lr={r['lr']:g} wd={r['weight_decay']:g} do={r['dropout']}"
        ax[0].plot(h["epoch"], h["val_ndcg10"], label=lab)
        ax[1].plot(h["epoch"], h["gap"], label=lab)
    for a, t in zip(ax, ("Validation NDCG@10", "Train - val gap")):
        a.axvspan(*WINDOW, alpha=0.1); a.set_title(f"{name} [{feature_set}]: {t}"); a.set_xlabel("epoch")
    ax[0].legend(fontsize=6); plt.tight_layout(); plt.show()

run_grid("listnet_1L", feature_set="full")

run_grid("listnet_1L", feature_set="minimal")

run_grid("top10_1L_nowin", feature_set="full")

run_grid("ranknet", feature_set="full")

summarize("listnet_1L")

plot_dynamics("listnet_1L", "minimal")

recs = load_results("listnet_1L", "full")

best = pick(recs)

def status():
    rows = []
    for path in sorted(glob.glob(os.path.join(DRIVE_DIR, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        for r in load_results(name):
            rows.append({"model": name, "feature_set": r["feature_set"], "epochs": r["epochs"], "seed": r["seed"]})
    if not rows:
        print("no results found in", DRIVE_DIR); return
    df = pd.DataFrame(rows)
    print(df.groupby(["model", "feature_set", "epochs", "seed"]).size().rename("runs_done").reset_index().to_string(index=False))

status()

src = os.path.join(DRIVE_DIR, "top10_1L_win.jsonl")
if os.path.exists(src):
    os.rename(src, src.replace(".jsonl", "_wrong_weight_backup.jsonl"))

run_grid("top10_1L_win", feature_set="full")

run_grid("top10_2L_nowin", feature_set="full")

run_grid("top10_2L_win", feature_set="full")

run_grid("top10_2L_win", feature_set="minimal")

run_grid("top10_2L_nowin", feature_set="minimal")

run_grid("ranknet", feature_set="minimal")

run_grid("top10_1L_win", feature_set="minimal")

run_grid("top10_1L_nowin", feature_set="minimal")

status()

common_race_ids = sorted(set(per_race_avg["shrinkage"].keys()) & set(per_race_avg["no_shrinkage"].keys()))
shrink_perrace_aligned   = np.array([np.mean(per_race_avg["shrinkage"][rid])    for rid in common_race_ids])
noshrink_perrace_aligned = np.array([np.mean(per_race_avg["no_shrinkage"][rid]) for rid in common_race_ids])

def paired_report(label, a, b, n_warn=5):
    diff = a - b
    t, p = ttest_rel(a, b)
    d = cohens_d_paired(a, b)
    warn = "  (n < 5, treat with caution)" if len(a) < n_warn else ""
    print(f"{label}  n={len(a)}  mean_a={a.mean():.4f}  mean_b={b.mean():.4f}  "
          f"diff={diff.mean():+.4f}  paired t={t:.4f}  p={p:.4f}  "
          f"Cohen's d={d:.4f} ({effect_size_label(d)})  sig@.05: {'YES' if p < 0.05 else 'NO'}{warn}")

print(f"{'='*95}\nPER-RACE PAIRED T-TEST -- shrinkage vs no_shrinkage (n=51, overall)\n{'='*95}")
paired_report("shrinkage vs no_shrinkage (overall)", shrink_perrace_aligned, noshrink_perrace_aligned)

print(f"\n{'='*95}\nPER-RACE PAIRED T-TEST -- shrinkage vs no_shrinkage, BY TIER GROUP\n{'='*95}")
for grp in TIER_GROUPS:
    idx = [i for i, rid in enumerate(common_race_ids) if race_tier_group(rid) == grp]
    if len(idx) < 2:
        print(f"{grp}: n={len(idx)} (too few races to test)")
        continue
    a = shrink_perrace_aligned[idx]
    b = noshrink_perrace_aligned[idx]
    paired_report(grp, a, b)

print(f"\n{'='*95}\nPER-RACE PAIRED T-TEST -- shrinkage vs no_shrinkage, BY DISCIPLINE GROUP\n{'='*95}")
for grp in DISCIPLINE_GROUPS:
    idx = [i for i, rid in enumerate(common_race_ids) if race_discipline_group(rid) == grp]
    if len(idx) < 2:
        print(f"{grp}: n={len(idx)} (too few races to test)")
        continue
    a = shrink_perrace_aligned[idx]
    b = noshrink_perrace_aligned[idx]
    paired_report(grp, a, b)

def seed_group_means(model_name, group_fn, group_val):
    out = []
    for seed in sorted({s for (m, s) in per_run_metrics if m == model_name}):
        metrics = per_run_metrics[(model_name, seed)]
        vals = [m["ndcg10"] for rid, m in metrics.items() if group_fn(rid) == group_val]
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)

print(f"\n{'='*95}\nPER-SEED PAIRED T-TEST (n=5) -- shrinkage vs no_shrinkage, BY TIER GROUP\n{'='*95}")
for grp in TIER_GROUPS:
    a = seed_group_means("shrinkage", race_tier_group, grp)
    b = seed_group_means("no_shrinkage", race_tier_group, grp)
    if np.isnan(a).any() or np.isnan(b).any():
        print(f"{grp}: missing data for at least one seed (no races in this group for that seed)")
        continue
    paired_report(grp, a, b)

print(f"\n{'='*95}\nPER-SEED PAIRED T-TEST (n=5) -- shrinkage vs no_shrinkage, BY DISCIPLINE GROUP\n{'='*95}")
for grp in DISCIPLINE_GROUPS:
    a = seed_group_means("shrinkage", race_discipline_group, grp)
    b = seed_group_means("no_shrinkage", race_discipline_group, grp)
    if np.isnan(a).any() or np.isnan(b).any():
        print(f"{grp}: missing data for at least one seed (no races in this group for that seed)")
        continue
    paired_report(grp, a, b)

def group_descriptive(model_name, group_fn, group_vals, label):
    print(f"\n{label} (vs Kholkine overall mean = {kholkine_mean:.4f}, descriptive only, no test)")
    for grp in group_vals:
        idx = [rid for rid in common_race_ids if group_fn(rid) == grp]
        vals = [np.mean(per_race_avg[model_name][rid]) for rid in idx]
        if not vals:
            print(f"  {grp:<12} n=0")
            continue
        arr = np.array(vals)
        print(f"  {grp:<12} n={len(arr):>3}  mean={arr.mean():.4f}  std={arr.std(ddof=1):.4f}  "
              f"diff vs Kholkine={arr.mean()-kholkine_mean:+.4f}")

print(f"\n{'='*95}\nDESCRIPTIVE GROUP BREAKDOWNS vs KHOLKINE (no significance test)\n{'='*95}")
group_descriptive("shrinkage",    race_tier_group,       TIER_GROUPS,       "shrinkage -- by tier")
group_descriptive("no_shrinkage", race_tier_group,       TIER_GROUPS,       "no_shrinkage -- by tier")
group_descriptive("shrinkage",    race_discipline_group, DISCIPLINE_GROUPS, "shrinkage -- by discipline")
group_descriptive("no_shrinkage", race_discipline_group, DISCIPLINE_GROUPS, "no_shrinkage -- by discipline")

from statsmodels.stats.multitest import multipletests

def run_group_family(group_vals, idx_fn, a_full, b_full, label):
    """idx_fn(grp) -> list of positions into a_full/b_full for that group.
    Collects raw p-values across the family, then applies Holm-Bonferroni."""
    rows = []
    for grp in group_vals:
        idx = idx_fn(grp)
        if len(idx) < 2:
            rows.append((grp, len(idx), np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        a, b = a_full[idx], b_full[idx]
        diff = a - b
        t, p = ttest_rel(a, b)
        d = cohens_d_paired(a, b)
        rows.append((grp, len(idx), a.mean(), b.mean(), t, p, d))

    valid = [(i, r) for i, r in enumerate(rows) if not np.isnan(r[5])]
    if valid:
        raw_p = [r[5] for _, r in valid]
        reject, adj_p, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
        adj_lookup = {i: (adj_p[k], reject[k]) for k, (i, _) in enumerate(valid)}
    else:
        adj_lookup = {}

    print(f"\n{label}")
    print(f"{'Group':<14}{'n':>5}{'MeanA':>9}{'MeanB':>9}{'t':>8}{'raw p':>9}{'adj p':>9}{'Cohens_d':>10}{'Sig@.05':>9}")
    for i, (grp, n, ma, mb, t, p, d) in enumerate(rows):
        if np.isnan(p):
            print(f"{grp:<14}{n:>5}  (too few races to test)")
            continue
        adj_p, rej = adj_lookup[i]
        print(f"{grp:<14}{n:>5}{ma:>9.4f}{mb:>9.4f}{t:>8.3f}{p:>9.4f}{adj_p:>9.4f}{d:>10.4f}{'YES' if rej else 'NO':>9}")

run_group_family(
    TIER_GROUPS,
    lambda grp: [i for i, rid in enumerate(common_race_ids) if race_tier_group(rid) == grp],
    shrink_perrace_aligned, noshrink_perrace_aligned,
    "PER-RACE PAIRED T-TEST, Holm-Bonferroni corrected -- BY TIER GROUP",
)

run_group_family(
    DISCIPLINE_GROUPS,
    lambda grp: [i for i, rid in enumerate(common_race_ids) if race_discipline_group(rid) == grp],
    shrink_perrace_aligned, noshrink_perrace_aligned,
    "PER-RACE PAIRED T-TEST, Holm-Bonferroni corrected -- BY DISCIPLINE GROUP",
)

def run_seed_group_family(group_vals, group_fn, label):
    rows = []
    for grp in group_vals:
        a = seed_group_means("shrinkage", group_fn, grp)
        b = seed_group_means("no_shrinkage", group_fn, grp)
        if np.isnan(a).any() or np.isnan(b).any():
            rows.append((grp, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        t, p = ttest_rel(a, b)
        d = cohens_d_paired(a, b)
        rows.append((grp, a.mean(), b.mean(), t, p, d))

    valid = [(i, r) for i, r in enumerate(rows) if not np.isnan(r[4])]
    if valid:
        raw_p = [r[4] for _, r in valid]
        reject, adj_p, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
        adj_lookup = {i: (adj_p[k], reject[k]) for k, (i, _) in enumerate(valid)}
    else:
        adj_lookup = {}

    print(f"\n{label}")
    print(f"{'Group':<14}{'n':>5}{'MeanA':>9}{'MeanB':>9}{'t':>8}{'raw p':>9}{'adj p':>9}{'Cohens_d':>10}{'Sig@.05':>9}")
    for i, (grp, ma, mb, t, p, d) in enumerate(rows):
        if np.isnan(p):
            print(f"{grp:<14}  (missing data for at least one seed)")
            continue
        adj_p, rej = adj_lookup[i]
        print(f"{grp:<14}{5:>5}{ma:>9.4f}{mb:>9.4f}{t:>8.3f}{p:>9.4f}{adj_p:>9.4f}{d:>10.4f}{'YES' if rej else 'NO':>9}")

run_seed_group_family(TIER_GROUPS, race_tier_group,
    "PER-SEED PAIRED T-TEST (n=5), Holm-Bonferroni corrected -- BY TIER GROUP")
run_seed_group_family(DISCIPLINE_GROUPS, race_discipline_group,
    "PER-SEED PAIRED T-TEST (n=5), Holm-Bonferroni corrected -- BY DISCIPLINE GROUP")

print(f"\n{'='*95}\nWT CLASSICS: shrinkage & no_shrinkage vs Kholkine (Dunnett's method, shared control)\n{'='*95}")
print("Control = Kholkine's full 17-race sample (all six races are WT-classics-tier events).")

wt_idx = [i for i, rid in enumerate(common_race_ids) if race_tier_group(rid) == "WT_classic"]
shrink_wt = shrink_perrace_aligned[wt_idx]
noshrink_wt = noshrink_perrace_aligned[wt_idx]

if len(wt_idx) < 2:
    print(f"Only {len(wt_idx)} WT_classic race(s) in test set -- not enough to test.")
else:
    res_wt = dunnett(shrink_wt, noshrink_wt, control=KHOLKINE_NDCG_SAMPLE)
    print(f"{'Group':<16}{'n':>5}{'Mean':>9}{'Dunnett stat':>14}{'adj p':>9}{'Cohens_d':>10}{'Sig@.05':>9}")
    for label, sample, stat, p in zip(["shrinkage", "no_shrinkage"], [shrink_wt, noshrink_wt], res_wt.statistic, res_wt.pvalue):
        d = cohens_d_independent(sample, KHOLKINE_NDCG_SAMPLE)
        print(f"{label:<16}{len(sample):>5}{sample.mean():>9.4f}{stat:>14.4f}{p:>9.4f}{d:>10.4f}{'YES' if p < 0.05 else 'NO':>9}")
    print(f"Kholkine control: n={len(KHOLKINE_NDCG_SAMPLE)}  mean={KHOLKINE_NDCG_SAMPLE.mean():.4f}")

status()

run_grid("listnet_2L", feature_set="minimal")

run_grid("listnet_2L", feature_set="full")

XGB_NAME        = "xgb"
XGB_ROUNDS      = 300
XGB_EVAL_EVERY  = 10
XGB_WINDOW      = (150, 300)
XGB_SEEDS       = [0, 1, 2]

XGB_DEPTH_GRID  = [1, 2, 3, 4]
XGB_LR_GRID     = [0.02, 0.05, 0.1]
XGB_MCW_GRID    = [1, 5, 10]
XGB_LAMBDA_GRID = [1, 10, 30]
XGB_BASE = {"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 5, "reg_lambda": 10}
PARAM_KEYS = ("max_depth", "learning_rate", "min_child_weight", "reg_lambda")

def _prep_eval(graphs, feature_indices):
    Xs, meta, off = [], [], 0
    for rid, (g, _, _) in graphs.items():
        t = g["rider"].y.cpu().numpy()
        n = len(t)
        if n < 3:
            continue
        Xs.append(g["rider"].x.cpu().numpy()[:, feature_indices])
        ranks = np.round(t * n).clip(1, n).astype(int)
        meta.append((off, off + n, t, ranks))
        off += n
    return np.vstack(Xs), meta

def _ndcg_at(model, X, meta, rounds):
    p = model.predict(X, iteration_range=(0, rounds))
    return float(np.mean([full_metrics_from_pred(-p[s:e], t, rk)["ndcg10"]
                          for s, e, t, rk in meta]))

def run_one_xgb(params, feature_set, seed, eval_graphs=None):
    eval_graphs = eval_graphs if eval_graphs is not None else test_graphs
    fi = FEATURE_SETS[feature_set]
    X, y, grp = build_xgb_arrays(train_graphs, fi)
    model = xgb.XGBRanker(
        objective="rank:ndcg", eval_metric="ndcg@10", n_estimators=XGB_ROUNDS,
        subsample=0.7, colsample_bytree=0.7, random_state=seed, **params)
    model.fit(X, y, group=grp, verbose=False)

    Xtr, mtr = _prep_eval(train_graphs, fi)
    Xva, mva = _prep_eval(eval_graphs, fi)
    hist = {"round": [], "train_ndcg10": [], "val_ndcg10": [], "gap": []}
    for r in range(XGB_EVAL_EVERY, XGB_ROUNDS + 1, XGB_EVAL_EVERY):
        tr, va = _ndcg_at(model, Xtr, mtr, r), _ndcg_at(model, Xva, mva, r)
        hist["round"].append(r); hist["train_ndcg10"].append(tr)
        hist["val_ndcg10"].append(va); hist["gap"].append(tr - va)
    return hist

def _window_stats_xgb(hist):
    lo, hi = XGB_WINDOW
    idx = [i for i, r in enumerate(hist["round"]) if lo <= r <= hi]
    v = np.array([hist["val_ndcg10"][i] for i in idx])
    t = np.array([hist["train_ndcg10"][i] for i in idx])
    return {"val_mean": float(v.mean()), "val_std": float(v.std(ddof=1)),
            "train_mean": float(t.mean()), "gap_mean": float((t - v).mean()),
            "final_val": hist["val_ndcg10"][-1]}

def load_xgb(feature_set=None, tag=None):
    recs = load_results(XGB_NAME, feature_set)
    for r in recs:
        r.setdefault("tag", "default")
    return [r for r in recs if tag is None or r["tag"] == tag]

def _xgb_key(feature_set, tag, p):
    return (feature_set, tag, *(p[k] for k in PARAM_KEYS), XGB_ROUNDS)

def get_or_run_xgb(params, feature_set="full", tag="default"):
    key = _xgb_key(feature_set, tag, params)
    for r in load_xgb(feature_set, tag):
        if _xgb_key(feature_set, tag, r) == key:
            return r
    hists = [run_one_xgb(params, feature_set, s) for s in XGB_SEEDS]
    per_seed = [_window_stats_xgb(h) for h in hists]
    mean_hist = {k: list(np.mean([h[k] for h in hists], axis=0)) for k in hists[0]}
    rec = {"name": XGB_NAME, "feature_set": feature_set, "tag": tag,
           "n_features": len(FEATURE_SETS[feature_set]), **params,
           "epochs": XGB_ROUNDS, "seed": ",".join(map(str, XGB_SEEDS)),
           "window": list(XGB_WINDOW),
           "val_mean":   float(np.mean([s["val_mean"] for s in per_seed])),
           "val_std":    float(np.mean([s["val_std"] for s in per_seed])),
           "val_seed_std": float(np.std([s["val_mean"] for s in per_seed], ddof=1)),
           "train_mean": float(np.mean([s["train_mean"] for s in per_seed])),
           "gap_mean":   float(np.mean([s["gap_mean"] for s in per_seed])),
           "final_val":  float(np.mean([s["final_val"] for s in per_seed])),
           "per_seed_val_mean": [s["val_mean"] for s in per_seed],
           "history": mean_hist}
    with open(_path(XGB_NAME), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

def run_grid_xgb(feature_set="full", tag="default"):
    cur = dict(XGB_BASE)
    stages = (("max_depth", XGB_DEPTH_GRID), ("learning_rate", XGB_LR_GRID),
              ("min_child_weight", XGB_MCW_GRID), ("reg_lambda", XGB_LAMBDA_GRID))
    for key, grid in stages:
        recs = [get_or_run_xgb({**cur, key: v}, feature_set, tag) for v in grid]
        chosen = pick(recs)
        cur = {k: chosen[k] for k in PARAM_KEYS}
    return summarize_xgb(feature_set, tag)

def select_xgb(feature_set="full"):
    """Average each config over all tags it was run on (only configs present in every tag), then apply pick()."""
    recs = load_xgb(feature_set)
    tags = {r["tag"] for r in recs}
    groups = {}
    for r in recs:
        groups.setdefault(tuple(r[k] for k in PARAM_KEYS), []).append(r)
    pooled = []
    for rs in groups.values():
        if {r["tag"] for r in rs} == tags:
            pooled.append({**rs[0],
                           "val_mean": float(np.mean([r["val_mean"] for r in rs])),
                           "gap_mean": float(np.mean([r["gap_mean"] for r in rs]))})
    return pick(pooled)

def summarize_xgb(feature_set=None, tag=None):
    cols = ("feature_set", "tag", *PARAM_KEYS, "val_mean", "val_std", "val_seed_std",
            "train_mean", "gap_mean", "final_val")
    df = pd.DataFrame([{k: r[k] for k in cols} for r in load_xgb(feature_set, tag)])
    df = df.sort_values(["feature_set", "tag", "val_mean"], ascending=[True, True, False])
    print(f"\nxgb: rounds {XGB_WINDOW[0]}-{XGB_WINDOW[1]}, seeds {XGB_SEEDS}")
    print(df.round(4).to_string(index=False))
    return df

print(len(train_graphs), "train races,", len(test_graphs), "val races")
run_grid_xgb("full",    tag="cut141")
run_grid_xgb("minimal", tag="cut141")

print(select_xgb("full"), select_xgb("minimal"))

status()

SELECTED_PATH = os.path.join(DRIVE_DIR, "selected_configs.json")

def _cfg_by_name(name):
    return next((c for c in MODEL_CONFIGS if c["name"] == name), None)

def _gnn_records(name, fs):
    cfg = _cfg_by_name(name)
    recs = [r for r in load_results(name, fs) if r["epochs"] == EPOCHS and r["seed"] == 0]
    if cfg is not None and cfg.get("use_win_head"):
        w = cfg.get("win_loss_weight")
        recs = [r for r in recs if r.get("win_loss_weight") in (None, w)]
    return recs

def select_all(save=True):
    rows, selected = [], {}
    for path in sorted(glob.glob(os.path.join(DRIVE_DIR, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        if "backup" in name:
            continue
        for fs in ("full", "minimal"):
            if name == XGB_NAME:
                if not load_xgb(fs):
                    continue
                try:
                    c = select_xgb(fs)
                except ValueError:
                    print(f"[warn] xgb [{fs}]: no config present in every tag, skipped")
                    continue
                params = {k: c[k] for k in PARAM_KEYS}
                edge = []
            else:
                recs = _gnn_records(name, fs)
                if not recs:
                    continue
                c = pick(recs)
                params = {"lr": c["lr"], "weight_decay": c["weight_decay"], "dropout": c["dropout"]}
                edge = [k for k, grid in (("lr", LR_GRID), ("weight_decay", WD_GRID), ("dropout", DROPOUT_GRID))
                        if params[k] in (min(grid), max(grid))]
            selected[f"{name}|{fs}"] = params
            rows.append({"model": name, "fs": fs, **params,
                         "val_mean": round(c["val_mean"], 4), "gap": round(c["gap_mean"], 4),
                         "n_runs": len(_gnn_records(name, fs)) if name != XGB_NAME else len(load_xgb(fs)),
                         "at_grid_edge": ",".join(edge)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if save:
        with open(SELECTED_PATH, "w") as f:
            json.dump(selected, f, indent=2)
        df.to_csv(SELECTED_PATH.replace(".json", ".csv"), index=False)
    return selected

selected = select_all()

XGB_FINAL_ROUNDS = 225

def build_final_configs(feature_set, selected=None):
    if selected is None:
        with open(SELECTED_PATH) as f:
            selected = json.load(f)
    final = []
    for cfg in MODEL_CONFIGS:
        c = copy.copy(cfg)
        key = f"{cfg['name']}|{feature_set}"

        if cfg["kind"] == "naive_sort":
            final.append(c)
            continue
        if key not in selected:
            print(f"[warn] no selected config for {key}, skipped")
            continue

        p = selected[key]
        if cfg["kind"] == "xgb":
            c["xgb_params"] = {**p, "subsample": 0.7, "colsample_bytree": 0.7}
            c["n_estimators"] = XGB_FINAL_ROUNDS
        else:
            c["lr"] = p["lr"]
            c["weight_decay"] = p["weight_decay"]
            c["dropout"] = p["dropout"]
            c["epochs"] = EPOCHS
        c["feature_set"] = feature_set
        final.append(c)
    return final

FINAL_CONFIGS = {fs: build_final_configs(fs) for fs in ("full", "minimal")}

for fs, cfgs in FINAL_CONFIGS.items():
    print(f"\n=== {fs} ===")
    for c in cfgs:
        show = {k: v for k, v in c.items() if k in
                ("name", "kind", "lr", "weight_decay", "dropout", "epochs",
                 "win_loss_weight", "xgb_params", "n_estimators")}
        print(show)

status()

FINAL_DIR = "/content/drive/MyDrive/cycling_gnn/data_2025_26/final_2"
os.makedirs(FINAL_DIR, exist_ok=True)
FINAL_WINDOW = (20, 40)
XGB_FINAL_EVAL_EVERY = 5
KEEP = ("ndcg10", "top5", "top10")

def _fpath(name, fs, seed): return os.path.join(FINAL_DIR, f"{name}__{fs}__s{seed}.json")

def _slim(per_race):
    return {str(rid): {k: float(m[k]) for k in KEEP} for rid, m in per_race.items()}

def run_final_gnn(cfg, fs, seed):
    fi = FEATURE_SETS[fs]
    epochs = cfg["epochs"]
    set_seed(seed)
    model = cfg["model_class"](in_dim=len(fi), dropout_p=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    is_reg = cfg["kind"] == "regressor"
    predict_fn = (lambda g: _predict_regressor(model, g)) if is_reg else \
                 (lambda g: _predict_classifier_top10(model, g))

    train_ids = list(train_graphs.keys())
    hist = {"epoch": [], "loss": [], "train_ndcg10": [], "test_ndcg10": [], "gap": []}
    per_race_by_epoch = {}

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_ids)
        tot, n = 0.0, 0
        for rid in train_ids:
            g, _, _ = train_graphs[rid]
            g_dev = slice_graph_features(g, fi).to(device)
            target = g_dev["rider"].y
            opt.zero_grad()
            if is_reg:
                loss = cfg["loss_fn"](model(g_dev), target)
            else:
                top10_logit, win_logit = model(g_dev)
                loss = top10_bce_loss(top10_logit, target)
                if cfg.get("use_win_head", False):
                    loss = loss + cfg.get("win_loss_weight", 1.0) * win_softmax_loss(win_logit, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            tr = per_race_metrics_generic(predict_fn, train_graphs, fi)
            te = per_race_metrics_generic(predict_fn, test_graphs, fi)
        tr_nd = float(np.mean([m["ndcg10"] for m in tr.values()]))
        te_nd = float(np.mean([m["ndcg10"] for m in te.values()]))
        hist["epoch"].append(epoch); hist["loss"].append(tot / max(1, n))
        hist["train_ndcg10"].append(tr_nd); hist["test_ndcg10"].append(te_nd)
        hist["gap"].append(tr_nd - te_nd)
        for rid, m in te.items():
            per_race_by_epoch.setdefault(str(rid), []).append(float(m["ndcg10"]))
        print(f"  ep {epoch:3d} | loss={hist['loss'][-1]:.4f} | train={tr_nd:.3f} | test={te_nd:.3f}")

    lo, hi = FINAL_WINDOW
    win = [i for i, e in enumerate(hist["epoch"]) if lo <= e <= hi]
    return {
        "name": cfg["name"], "feature_set": fs, "seed": seed, "kind": cfg["kind"],
        "hparams": {k: cfg[k] for k in ("lr", "weight_decay", "dropout", "epochs")},
        "per_race_final": _slim(te),
        "final_test_ndcg10": te_nd, "final_train_ndcg10": tr_nd,
        "final_train_loss": hist["loss"][-1],
        "window": list(FINAL_WINDOW),
        "window_test_ndcg10": float(np.mean([hist["test_ndcg10"][i] for i in win])),
        "per_race_window_ndcg": {rid: float(np.mean([v[i] for i in win]))
                                 for rid, v in per_race_by_epoch.items()},
        "history": hist,
    }

def run_final_xgb(cfg, fs, seed):
    fi = FEATURE_SETS[fs]
    X, y, grp = build_xgb_arrays(train_graphs, fi)
    model = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg@10",
                          n_estimators=cfg["n_estimators"], random_state=seed, **cfg["xgb_params"])
    model.fit(X, y, group=grp, verbose=False)

    Xtr, mtr = _prep_eval(train_graphs, fi)
    Xte, mte = _prep_eval(test_graphs, fi)
    hist = {"round": [], "train_ndcg10": [], "test_ndcg10": [], "gap": []}
    for r in range(XGB_FINAL_EVAL_EVERY, cfg["n_estimators"] + 1, XGB_FINAL_EVAL_EVERY):
        tr, te = _ndcg_at(model, Xtr, mtr, r), _ndcg_at(model, Xte, mte, r)
        hist["round"].append(r); hist["train_ndcg10"].append(tr)
        hist["test_ndcg10"].append(te); hist["gap"].append(tr - te)

    per_race = per_race_metrics_xgb(model, fi)
    te_nd = float(np.mean([m["ndcg10"] for m in per_race.values()]))
    print(f"  mean_test_ndcg10={te_nd:.3f}")
    return {
        "name": cfg["name"], "feature_set": fs, "seed": seed, "kind": "xgb",
        "hparams": {**cfg["xgb_params"], "n_estimators": cfg["n_estimators"]},
        "per_race_final": _slim(per_race),
        "final_test_ndcg10": te_nd, "final_train_ndcg10": hist["train_ndcg10"][-1],
        "final_train_loss": None, "history": hist,
    }

def run_all_final(seeds, feature_sets=("full", "minimal"), only=None):
    print(len(train_graphs), "train races,", len(test_graphs), "test races")
    for fs in feature_sets:
        for cfg in FINAL_CONFIGS[fs]:
            if cfg["kind"] == "naive_sort" or (only and cfg["name"] not in only):
                continue
            for seed in seeds:
                p = _fpath(cfg["name"], fs, seed)
                if os.path.exists(p):
                    continue
                print(f"\n--- {cfg['name']} | {fs} | seed={seed} ---")
                rec = run_final_xgb(cfg, fs, seed) if cfg["kind"] == "xgb" else run_final_gnn(cfg, fs, seed)
                with open(p, "w") as f:
                    json.dump(rec, f)

def build_results(seeds, feature_sets=("full", "minimal")):
    results = {}
    naive = per_race_metrics_naive_sort()
    results["pure_sort"] = {}
    for fs in feature_sets:
        results["pure_sort"][fs] = {
            "per_race_seed_ndcg":  {rid: [m["ndcg10"]] * len(seeds) for rid, m in naive.items()},
            "per_race_seed_top5":  {rid: [m["top5"]]   * len(seeds) for rid, m in naive.items()},
            "per_race_seed_top10": {rid: [m["top10"]]  * len(seeds) for rid, m in naive.items()},
            "seed_means_ndcg": [float(np.mean([m["ndcg10"] for m in naive.values()]))] * len(seeds),
            "overfit": [],
        }
    names = sorted({os.path.basename(p).split("__")[0] for p in os.listdir(FINAL_DIR)})
    for name in names:
        results[name] = {}
        for fs in feature_sets:
            recs = []
            for s in seeds:
                p = _fpath(name, fs, s)
                if os.path.exists(p):
                    recs.append(json.load(open(p)))
            if not recs:
                continue
            d = {"per_race_seed_ndcg": {}, "per_race_seed_top5": {}, "per_race_seed_top10": {},
                 "seed_means_ndcg": [r["final_test_ndcg10"] for r in recs], "overfit": []}
            for r in recs:
                for rid, m in r["per_race_final"].items():
                    d["per_race_seed_ndcg"].setdefault(rid, []).append(m["ndcg10"])
                    d["per_race_seed_top5"].setdefault(rid, []).append(m["top5"])
                    d["per_race_seed_top10"].setdefault(rid, []).append(m["top10"])
                d["overfit"].append({"final_train_loss": r["final_train_loss"],
                                     "train_ndcg10": r["final_train_ndcg10"],
                                     "test_ndcg10": r["final_test_ndcg10"]})
            if "window_test_ndcg10" in recs[0]:
                d["seed_means_window_ndcg"] = [r["window_test_ndcg10"] for r in recs]
            d["histories"] = [r["history"] for r in recs]
            results[name][fs] = d
    return results

def plot_final_curves(name, fs, seeds):
    import matplotlib.pyplot as plt
    recs = [json.load(open(_fpath(name, fs, s))) for s in seeds if os.path.exists(_fpath(name, fs, s))]
    xk = "epoch" if "epoch" in recs[0]["history"] else "round"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for r in recs:
        h = r["history"]
        ax[0].plot(h[xk], h["train_ndcg10"], alpha=.5, color="C0")
        ax[0].plot(h[xk], h["test_ndcg10"], alpha=.5, color="C1")
        ax[1].plot(h[xk], h["gap"], alpha=.6)
    ax[0].set_title(f"{name} [{fs}] train (blue) / test (orange) NDCG@10")
    ax[1].set_title("train - test gap")
    for a in ax: a.set_xlabel(xk)
    plt.tight_layout(); plt.show()

run_all_final(SEEDS)

results = build_results(SEEDS)
plot_final_curves("listnet_1L", "full", SEEDS)

import matplotlib.pyplot as plt
from scipy import stats

CURVES_DIR = os.path.join(FINAL_DIR, "curves")
STATS_DIR  = os.path.join(FINAL_DIR, "stats")
os.makedirs(CURVES_DIR, exist_ok=True); os.makedirs(STATS_DIR, exist_ok=True)
METRICS = ("last", "window", "best")

def load_final(seeds):
    recs = {}
    for p in sorted(glob.glob(os.path.join(FINAL_DIR, "*__s*.json"))):
        r = json.load(open(p))
        if r["seed"] in seeds:
            recs.setdefault((r["name"], r["feature_set"]), []).append(r)
    for k, rs in recs.items():
        if len(rs) != len(seeds):
            print(f"[warn] {k}: {len(rs)}/{len(seeds)} seeds found")
    return recs

def plot_config_curves(key, rs, save=True):
    name, fs = key
    H = [r["history"] for r in rs]
    xk = "epoch" if "epoch" in H[0] else "round"
    has_loss = "loss" in H[0]
    x = H[0][xk]
    fig, ax = plt.subplots(1, 3 if has_loss else 2, figsize=(17 if has_loss else 12, 4.2))

    for h in H:
        ax[0].plot(x, h["train_ndcg10"], color="C0", alpha=.25, lw=1)
        ax[0].plot(x, h["test_ndcg10"],  color="C1", alpha=.25, lw=1)
        ax[1].plot(x, h["gap"], color="C2", alpha=.25, lw=1)
    ax[0].plot(x, np.mean([h["train_ndcg10"] for h in H], 0), color="C0", lw=2, label="train")
    ax[0].plot(x, np.mean([h["test_ndcg10"] for h in H], 0),  color="C1", lw=2, label="test")
    ax[0].axhline(KHOLKINE_OVERALL_MEAN, color="k", ls="--", lw=1, label="Kholkine mean")
    ax[1].plot(x, np.mean([h["gap"] for h in H], 0), color="C2", lw=2)
    ax[1].axhline(0, color="k", lw=.5)
    if has_loss:
        for h in H: ax[2].plot(x, h["loss"], color="C3", alpha=.25, lw=1)
        ax[2].plot(x, np.mean([h["loss"] for h in H], 0), color="C3", lw=2)
        ax[2].set_title("train loss")
    if xk == "epoch":
        for a in ax: a.axvspan(*FINAL_WINDOW, alpha=.08, color="gray")
    ax[0].set_title(f"{name} [{fs}]  NDCG@10 (thin = seeds, thick = mean)")
    ax[0].legend(fontsize=8); ax[1].set_title("train - test gap")
    for a in ax: a.set_xlabel(xk)
    plt.tight_layout()
    if save: fig.savefig(os.path.join(CURVES_DIR, f"{name}__{fs}.png"), dpi=150)
    plt.close(fig)

def plot_overview(recs, save=True):
    """Mean test curve of every GNN config, one panel per feature set."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for a, fs in zip(ax, ("full", "minimal")):
        for (name, f), rs in sorted(recs.items()):
            if f != fs or "epoch" not in rs[0]["history"]:
                continue
            h = rs[0]["history"]["epoch"]
            a.plot(h, np.mean([r["history"]["test_ndcg10"] for r in rs], 0), label=name)
        a.axhline(KHOLKINE_OVERALL_MEAN, color="k", ls="--", lw=1)
        a.axvspan(*FINAL_WINDOW, alpha=.08, color="gray")
        a.set_title(f"mean test NDCG@10 [{fs}]"); a.set_xlabel("epoch")
    ax[0].legend(fontsize=7); plt.tight_layout()
    if save: fig.savefig(os.path.join(CURVES_DIR, "overview_gnn_test.png"), dpi=150)
    plt.close(fig)

def save_all_curves(seeds):
    recs = load_final(seeds)
    for key, rs in recs.items():
        plot_config_curves(key, rs)
    plot_overview(recs)

def seed_value(r, metric):
    h = r["history"]["test_ndcg10"]
    if metric == "last":   return h[-1]
    if metric == "best":   return max(h)
    return r.get("window_test_ndcg10")

def race_vector(rs, metric, rids):
    """Per-race NDCG averaged over seeds, or None if not available in the stored files."""
    mats = []
    for r in rs:
        if metric == "last":
            v = [r["per_race_final"][i]["ndcg10"] for i in rids]
        elif metric == "window":
            if "per_race_window_ndcg" not in r: return None
            v = [r["per_race_window_ndcg"][i] for i in rids]
        else:
            if "per_race_by_epoch" not in r: return None
            be = int(np.argmax(r["history"]["test_ndcg10"]))
            v = [r["per_race_by_epoch"][i][be] for i in rids]
        mats.append(v)
    return np.mean(mats, axis=0)

def holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(np.argsort(p)):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj

def dunnett_per_race(recs, metric, alternative="two-sided"):
    rids = sorted(set.intersection(*[set(r["per_race_final"]) for rs in recs.values() for r in rs]))
    samples = {}
    for key, rs in recs.items():
        v = race_vector(rs, metric, rids)
        if v is None:
            continue
        samples[key] = v
    if not samples:
        return pd.DataFrame()
    res = stats.dunnett(*samples.values(), control=KHOLKINE_NDCG_SAMPLE, alternative=alternative)
    ci = res.confidence_interval()
    rows = []
    for i, (key, v) in enumerate(samples.items()):
        rows.append({"level": "per_race", "metric": metric, "model": key[0], "fs": key[1],
                     "n": len(v), "mean": v.mean(), "kholkine_mean": KHOLKINE_OVERALL_MEAN,
                     "diff": v.mean() - KHOLKINE_NDCG_SAMPLE.mean(),
                     "ci95_low": ci.low[i], "ci95_high": ci.high[i],
                     "stat": res.statistic[i], "p_adj": res.pvalue[i], "test": "Dunnett"})
    return pd.DataFrame(rows)

def ttest_per_seed(recs, metric, alternative="two-sided"):
    rows = []
    for key, rs in recs.items():
        vals = [seed_value(r, metric) for r in rs]
        if any(v is None for v in vals):
            continue
        t, p = stats.ttest_1samp(vals, KHOLKINE_OVERALL_MEAN, alternative=alternative)
        rows.append({"level": "per_seed", "metric": metric, "model": key[0], "fs": key[1],
                     "n": len(vals), "mean": np.mean(vals), "seed_std": np.std(vals, ddof=1),
                     "kholkine_mean": KHOLKINE_OVERALL_MEAN,
                     "diff": np.mean(vals) - KHOLKINE_OVERALL_MEAN,
                     "stat": t, "p_raw": p, "test": "one-sample t + Holm"})
    df = pd.DataFrame(rows)
    if len(df): df["p_adj"] = holm(df["p_raw"])
    return df

def run_all_stats(seeds, alternative="two-sided"):
    recs = load_final(seeds)
    out = []
    for m in METRICS:
        out += [dunnett_per_race(recs, m, alternative), ttest_per_seed(recs, m, alternative)]
    df = pd.concat([d for d in out if len(d)], ignore_index=True)
    df["significant_0.05"] = df["p_adj"] < 0.05
    df.to_csv(os.path.join(STATS_DIR, "kholkine_tests.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for (lvl, m), g in df.groupby(["level", "metric"]):
            print(f"\n=== {lvl} | {m} ===")
            print(g.drop(columns=["level", "metric"]).round(4).to_string(index=False))
    return df

save_all_curves(SEEDS)
stats_df = run_all_stats(SEEDS)

N_BOOT   = 10000
BOOT_SEED = 12345

def _holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(np.argsort(p)):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj

def get_vectors(seeds, metric, naive):
    recs = load_final(seeds)
    naive_s = {str(k): v for k, v in naive.items()}
    rids = sorted(set.intersection(*[set(r["per_race_final"]) for rs in recs.values() for r in rs])
                  & set(naive_s))
    vecs = {}
    for key, rs in recs.items():
        v = race_vector(rs, metric, rids)
        if v is not None:
            vecs[key] = v
    vecs[("pure_sort", "-")] = np.array([naive_s[i]["ndcg10"] for i in rids])
    return vecs, rids

def maxt_bootstrap(diffs, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05):
    """diffs: (K, n_races) paired differences. Races are resampled jointly, so the
    correlation between comparisons is preserved."""
    K, n = diffs.shape
    rng = np.random.default_rng(seed)
    d = diffs.mean(1)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[:, idx].mean(2).T
    se = np.maximum(boot.std(0, ddof=1), 1e-12)

    t_obs  = np.abs(d / se)
    max_t  = np.abs((boot - d) / se).max(1)
    p_maxT = (1 + (max_t[:, None] >= t_obs).sum(0)) / (n_boot + 1)
    crit   = np.quantile(max_t, 1 - alpha)

    p_raw = np.clip(2 * np.minimum((boot <= 0).mean(0), (boot >= 0).mean(0)), 1 / n_boot, 1.0)
    return {"diff": d, "se": se, "p_raw": p_raw, "p_maxT": p_maxT,
            "ci_low": np.percentile(boot, 2.5, axis=0), "ci_high": np.percentile(boot, 97.5, axis=0),
            "ci_sim_low": d - crit * se, "ci_sim_high": d + crit * se, "crit": crit}

def make_families(vecs):
    names = sorted({k[0] for k in vecs if k[0] != "pure_sort"})
    gnn = [n for n in names if n != "xgb"]
    ref = ("pure_sort", "-")
    fam = {
        "vs_pure_sort":       [(k, ref) for k in sorted(vecs) if k != ref],
        "full_minus_minimal": [((n, "full"), (n, "minimal")) for n in names],
        "gnn_minus_xgb":      [((n, fs), ("xgb", fs)) for n in gnn for fs in ("full", "minimal")],
    }
    arch = []
    for fs in ("full", "minimal"):
        arch += [(("listnet_2L", fs),     ("listnet_1L", fs)),
                 (("top10_2L_nowin", fs), ("top10_1L_nowin", fs)),
                 (("top10_2L_win", fs),   ("top10_1L_win", fs)),
                 (("top10_1L_win", fs),   ("top10_1L_nowin", fs)),
                 (("top10_2L_win", fs),   ("top10_2L_nowin", fs))]
    fam["architecture"] = arch
    return fam

def _lab(k): return f"{k[0]}[{k[1]}]" if k[1] != "-" else k[0]

def run_paired_bootstrap(seeds, metrics=("last", "window")):
    naive = per_race_metrics_naive_sort()
    rows = []
    for m in metrics:
        vecs, rids = get_vectors(seeds, m, naive)
        print(f"[{m}] {len(rids)} races | pure_sort mean NDCG@10 = {vecs[('pure_sort','-')].mean():.4f}")
        for fam_name, pairs in make_families(vecs).items():
            pairs = [(a, b) for a, b in pairs if a in vecs and b in vecs]
            if not pairs:
                continue
            diffs = np.stack([vecs[a] - vecs[b] for a, b in pairs])
            res = maxt_bootstrap(diffs)
            p_holm = _holm(res["p_raw"])
            for i, (a, b) in enumerate(pairs):
                rows.append({"family": fam_name, "metric": m, "comparison": f"{_lab(a)} - {_lab(b)}",
                             "n_races": len(rids), "K": len(pairs),
                             "mean_A": vecs[a].mean(), "mean_B": vecs[b].mean(),
                             "diff": res["diff"][i], "se": res["se"][i],
                             "ci_low": res["ci_low"][i], "ci_high": res["ci_high"][i],
                             "ci_sim_low": res["ci_sim_low"][i], "ci_sim_high": res["ci_sim_high"][i],
                             "win_frac": float((diffs[i] > 0).mean()),
                             "p_raw": res["p_raw"][i], "p_holm": p_holm[i], "p_maxT": res["p_maxT"][i],
                             "sig_maxT": res["p_maxT"][i] < 0.05})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATS_DIR, "paired_bootstrap.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for (m, fam), g in df.groupby(["metric", "family"], sort=False):
            print(f"\n=== {fam} | {m} | K={g['K'].iloc[0]} ===")
            cols = ["comparison", "mean_A", "mean_B", "diff", "se", "ci_sim_low", "ci_sim_high",
                    "win_frac", "p_raw", "p_holm", "p_maxT", "sig_maxT"]
            print(g.sort_values("diff", ascending=False)[cols].round(4).to_string(index=False))
    return df

def plot_forest(df, metric="last", family="vs_pure_sort"):
    d = df[(df.metric == metric) & (df.family == family)].sort_values("diff")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(d) + 1.8))
    ax.errorbar(d["diff"], y, xerr=[d["diff"] - d["ci_sim_low"], d["ci_sim_high"] - d["diff"]],
                fmt="o", capsize=3, color="C0", label="simultaneous 95% (max-t)")
    ax.errorbar(d["diff"], y, xerr=[d["diff"] - d["ci_low"], d["ci_high"] - d["diff"]],
                fmt="none", ecolor="C1", lw=4, alpha=.5, label="unadjusted 95%")
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(d["comparison"], fontsize=8)
    ax.set_xlabel("difference in mean NDCG@10 (paired, per race)")
    ax.set_title(f"{family} | {metric}"); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(STATS_DIR, f"forest_{family}_{metric}.png"), dpi=150)
    plt.show()

pb = run_paired_bootstrap(SEEDS)
plot_forest(pb, "last", "vs_pure_sort")
plot_forest(pb, "window", "vs_pure_sort")

def paired_ttest(diffs, alpha=0.05):
    """diffs: (K, n_races) paired differences A - B."""
    K, n = diffs.shape
    d  = diffs.mean(1)
    sd = diffs.std(1, ddof=1)
    se = sd / np.sqrt(n)
    t  = d / se
    p_raw = 2 * stats.t.sf(np.abs(t), df=n - 1)
    p_holm = _holm(p_raw)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    tcrit_bonf = stats.t.ppf(1 - alpha / (2 * K), n - 1)

    p_wil = np.array([stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0 for x in diffs])
    return {"diff": d, "se": se, "t": t, "dz": d / sd, "p_raw": p_raw, "p_holm": p_holm,
            "p_wilcoxon": p_wil, "p_wilcoxon_holm": _holm(p_wil),
            "ci_low": d - tcrit * se, "ci_high": d + tcrit * se,
            "ci_adj_low": d - tcrit_bonf * se, "ci_adj_high": d + tcrit_bonf * se}

def run_paired_ttests(seeds, metrics=("last", "window")):
    naive = per_race_metrics_naive_sort()
    rows = []
    for m in metrics:
        vecs, rids = get_vectors(seeds, m, naive)
        print(f"[{m}] {len(rids)} races | pure_sort mean NDCG@10 = {vecs[('pure_sort','-')].mean():.4f}")
        for fam_name, pairs in make_families(vecs).items():
            pairs = [(a, b) for a, b in pairs if a in vecs and b in vecs]
            if not pairs:
                continue
            diffs = np.stack([vecs[a] - vecs[b] for a, b in pairs])
            res = paired_ttest(diffs)
            for i, (a, b) in enumerate(pairs):
                rows.append({"family": fam_name, "metric": m, "comparison": f"{_lab(a)} - {_lab(b)}",
                             "n_races": len(rids), "K": len(pairs),
                             "mean_A": vecs[a].mean(), "mean_B": vecs[b].mean(),
                             "win_frac": float((diffs[i] > 0).mean()),
                             **{k: res[k][i] for k in res},
                             "sig_holm": res["p_holm"][i] < 0.05})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATS_DIR, "paired_ttests.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for (m, fam), g in df.groupby(["metric", "family"], sort=False):
            print(f"\n=== {fam} | {m} | K={g['K'].iloc[0]} ===")
            cols = ["comparison", "mean_A", "mean_B", "diff", "se", "ci_adj_low", "ci_adj_high",
                    "dz", "win_frac", "p_raw", "p_holm", "p_wilcoxon_holm", "sig_holm"]
            print(g.sort_values("diff", ascending=False)[cols].round(4).to_string(index=False))
    return df

def plot_forest_t(df, metric="last", family="vs_pure_sort"):
    d = df[(df.metric == metric) & (df.family == family)].sort_values("diff")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(d) + 1.8))
    ax.errorbar(d["diff"], y, xerr=[d["diff"] - d["ci_adj_low"], d["ci_adj_high"] - d["diff"]],
                fmt="o", capsize=3, color="C0", label="Bonferroni-adjusted 95% CI")
    ax.errorbar(d["diff"], y, xerr=[d["diff"] - d["ci_low"], d["ci_high"] - d["diff"]],
                fmt="none", ecolor="C1", lw=4, alpha=.5, label="unadjusted 95% CI")
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(d["comparison"], fontsize=8)
    ax.set_xlabel("difference in mean NDCG@10 (paired, per race)")
    ax.set_title(f"{family} | {metric}"); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(STATS_DIR, f"forest_t_{family}_{metric}.png"), dpi=150)
    plt.show()

pt = run_paired_ttests(SEEDS)
plot_forest_t(pt, "last", "vs_pure_sort")

INTERFACE_FS   = "full"
INTERFACE_SEED = SEEDS[0]
INTERFACE_DIR  = DATA_DIR

def _final_cfg(name, fs=INTERFACE_FS):
    cfg = next((c for c in FINAL_CONFIGS[fs] if c["name"] == name), None)
    assert cfg is not None, f"no selected config for {name}|{fs} (check selected_configs.json)"
    return cfg

def train_interface_gnn(cfg, fs, seed):
    """Identical training procedure to run_final_gnn, without per-epoch evaluation."""
    fi = FEATURE_SETS[fs]
    set_seed(seed)
    model = cfg["model_class"](in_dim=len(fi), dropout_p=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    is_reg = cfg["kind"] == "regressor"

    train_ids = list(train_graphs.keys())
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        random.shuffle(train_ids)
        for rid in train_ids:
            g, _, _ = train_graphs[rid]
            g_dev = slice_graph_features(g, fi).to(device)
            target = g_dev["rider"].y
            opt.zero_grad()
            if is_reg:
                loss = cfg["loss_fn"](model(g_dev), target)
            else:
                top10_logit, win_logit = model(g_dev)
                loss = top10_bce_loss(top10_logit, target)
                if cfg.get("use_win_head", False):
                    loss = loss + cfg.get("win_loss_weight", 1.0) * win_softmax_loss(win_logit, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model

def save_interface_gnn(model, cfg, fs, path):
    torch.save({
        "state_dict": model.state_dict(),
        "model_class": cfg["model_class"].__name__,
        "in_dim": len(FEATURE_SETS[fs]),
        "hidden_dim": HIDDEN_DIM,
        "heads": HEADS,
        "dropout_p": cfg["dropout"],
        "feature_indices": FEATURE_SETS[fs],
    }, path)

def interface_test_ndcg(model, cfg, fs):
    predict_fn = (lambda g: _predict_regressor(model, g)) if cfg["kind"] == "regressor" else \
                 (lambda g: _predict_classifier_top10(model, g))
    with torch.no_grad():
        m = per_race_metrics_generic(predict_fn, test_graphs, FEATURE_SETS[fs])
    return float(np.mean([v["ndcg10"] for v in m.values()]))

cfg_xgb = _final_cfg("xgb")
X_train, y_train, group_train = build_xgb_arrays(train_graphs, FEATURE_SETS[INTERFACE_FS])
model_xgb = xgb.XGBRanker(
    objective="rank:ndcg", eval_metric="ndcg@10", n_estimators=cfg_xgb["n_estimators"],
    random_state=INTERFACE_SEED, **cfg_xgb["xgb_params"],
)
model_xgb.fit(X_train, y_train, group=group_train, verbose=False)
model_xgb.save_model(INTERFACE_DIR + "interface_xgb.json")
test_metrics_xgb = per_race_metrics_xgb(model_xgb, FEATURE_SETS[INTERFACE_FS])
print(f"xgb [{INTERFACE_FS}] seed={INTERFACE_SEED} test NDCG@10 = "
      f"{np.mean([m['ndcg10'] for m in test_metrics_xgb.values()]):.4f}")

cfg_listnet = _final_cfg("listnet_2L")
model_listnet_2L = train_interface_gnn(cfg_listnet, INTERFACE_FS, INTERFACE_SEED)
save_interface_gnn(model_listnet_2L, cfg_listnet, INTERFACE_FS, INTERFACE_DIR + "interface_listnet_2L.pt")
print(f"listnet_2L [{INTERFACE_FS}] seed={INTERFACE_SEED} test NDCG@10 = "
      f"{interface_test_ndcg(model_listnet_2L, cfg_listnet, INTERFACE_FS):.4f}")

cfg_top10 = _final_cfg("top10_2L_win")
model_top10_2L_win = train_interface_gnn(cfg_top10, INTERFACE_FS, INTERFACE_SEED)
save_interface_gnn(model_top10_2L_win, cfg_top10, INTERFACE_FS, INTERFACE_DIR + "interface_top10_2L_win.pt")
print(f"top10_2L_win [{INTERFACE_FS}] seed={INTERFACE_SEED} test NDCG@10 = "
      f"{interface_test_ndcg(model_top10_2L_win, cfg_top10, INTERFACE_FS):.4f}")
