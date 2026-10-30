import json
import numpy as np
import torch
from collections import defaultdict
from datetime import date, datetime
from torch_geometric.data import HeteroData

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/cycling_gnn/data_2025_26/"

TEST_CUTOFF_DATE     = date(2026, 3, 1)

MIN_SHARED_RACES = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

races   = json.load(open(DATA_DIR + "races_2025_26_1.json"))
results = json.load(open(DATA_DIR + "results_2025_26_1.json"))
riders  = json.load(open(DATA_DIR + "riders_2025_26_1.json"))

rider_to_idx = {r["rider_id"]: i for i, r in enumerate(riders)}
race_to_idx  = {r["race_id"]:  i for i, r in enumerate(races)}
race_map     = {r["race_id"]: r for r in races}
idx_to_rider = {i: riders[i] for i in range(len(riders))}


walk_forward_snapshots = torch.load(DATA_DIR + "race_id_to_discipline_scores_2.pt", weights_only=False)
race_id_to_form = torch.load(DATA_DIR + "race_id_to_form_features_2.pt", weights_only=False)
def normalize_scores(arr):
    if arr is None:
        return None
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

print(f"Loaded discipline scores for {len(riders)} riders")

def parse_iso_date(s):
    if s is None:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()

train_race_ids = set()
test_race_ids  = set()

for r in races:
    d = parse_iso_date(r["date"])
    if d is None:
        continue
    if d < TEST_CUTOFF_DATE:
        train_race_ids.add(r["race_id"])
    else:
        test_race_ids.add(r["race_id"])

print(f"Train races: {len(train_race_ids)}")
print(f"Test races:  {len(test_race_ids)}")

train_results = [r for r in results if r["race_id"] in train_race_ids]
test_results  = [r for r in results if r["race_id"] in test_race_ids]

def build_head_to_head(train_results, rider_to_idx, min_shared=MIN_SHARED_RACES):
    race_groups = defaultdict(list)
    for r in train_results:
        if r["status"] not in ("finished", "otl") or r["rank"] is None:
            continue
        if r["rider_id"] not in rider_to_idx:
            continue
        race_groups[r["race_id"]].append(r)

    wins  = defaultdict(int)
    total = defaultdict(int)

    for race_id, group in race_groups.items():
        group_sorted = sorted(group, key=lambda x: x["rank"])
        n = len(group_sorted)
        for i in range(n):
            for j in range(i + 1, n):
                a_id = rider_to_idx[group_sorted[i]["rider_id"]]
                b_id = rider_to_idx[group_sorted[j]["rider_id"]]
                wins[(a_id, b_id)]  += 1
                total[(a_id, b_id)] += 1
                total[(b_id, a_id)] += 1

    h2h = {}
    seen_pairs = set()
    for (a, b), t in total.items():
        pair_key = tuple(sorted((a, b)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        shared = total.get((a, b), 0)
        if shared < min_shared:
            continue

        a_wins = wins.get((a, b), 0)
        b_wins = wins.get((b, a), 0)
        shared_races = a_wins + b_wins
        if shared_races < min_shared:
            continue

        win_rate_a = a_wins / shared_races
        h2h[(a, b)] = win_rate_a
        h2h[(b, a)] = 1.0 - win_rate_a

    print(f"Head-to-head pairs built: {len(h2h):,} (from {len(seen_pairs)} candidate pairs)")
    return h2h

head_to_head = build_head_to_head(train_results, rider_to_idx)

def discipline_for_parcours(parcours_type):
    return {
        "flat":     "sprint",
        "punchy":   "punchy",
        "hilly":    "versatile",
        "mountain": "climbing",
    }.get(parcours_type, "sprint")

_all_race_feats = np.array([[
    r.get("classification_encoded", 0),
    r.get("profile_score", 0),
    r.get("vertical_meters", 0),
    r.get("gradient_final_km", 0) or 0,
    r.get("distance_km", 0) or 0,
    r.get("startlist_quality", 0),
    r.get("parcours_encoded", 0),
] for r in races], dtype=np.float32)
RACE_FEAT_MIN   = _all_race_feats.min(axis=0)
RACE_FEAT_RANGE = _all_race_feats.max(axis=0) - RACE_FEAT_MIN + 1e-8

def build_race_graph(race_id, race_results, head_to_head, min_h2h_for_edge=0.0):
    race_info = race_map[race_id]
    parcours  = race_info["parcours_type"]
    discipline = discipline_for_parcours(parcours)

    finished = [
        r for r in race_results
        if r["status"] in ("finished", "otl") and r["rank"] is not None
        and r["rider_id"] in rider_to_idx
    ]
    if len(finished) < 2:
        return None

    rider_ids = [r["rider_id"] for r in finished]
    local_idx = {rid: i for i, rid in enumerate(rider_ids)}
    global_idx = [rider_to_idx[rid] for rid in rider_ids]

    n = len(rider_ids)

    if race_id not in walk_forward_snapshots:
        raise ValueError()
    scores_dict = walk_forward_snapshots[race_id]
    s_sc, p_sc, v_sc, c_sc = scores_dict["sprint"], scores_dict["punchy"], scores_dict["versatile"], scores_dict["climbing"]

    rider_feats = np.zeros((n, 4), dtype=np.float32)
    for i, gidx in enumerate(global_idx):
        rider_feats[i, 0] = s_sc[gidx] if s_sc is not None else 0.0
        rider_feats[i, 1] = p_sc[gidx] if p_sc is not None else 0.0
        rider_feats[i, 2] = v_sc[gidx] if v_sc is not None else 0.0
        rider_feats[i, 3] = c_sc[gidx] if c_sc is not None else 0.0

    form_feats = race_id_to_form[race_id][global_idx]
    rider_feats = np.concatenate([rider_feats, form_feats], axis=1)

    relevant_col = {"sprint": 0, "punchy": 1, "versatile": 2, "climbing": 3}[discipline]

    team_members_local = defaultdict(list)
    for i, r in enumerate(finished):
        team_members_local[r["team_id"]].append(i)

    team_rank = np.zeros((n, 1), dtype=np.float32)
    for team_id, members in team_members_local.items():
        if len(members) == 1:
            team_rank[members[0], 0] = 0.0
            continue
        sorted_members = sorted(members, key=lambda i: rider_feats[i, relevant_col], reverse=True)
        for rank_pos, i in enumerate(sorted_members):
            team_rank[i, 0] = rank_pos / (len(members) - 1)

    race_feat_vec = np.array([
        race_info.get("classification_encoded", 0),
        race_info.get("profile_score", 0),
        race_info.get("vertical_meters", 0),
        race_info.get("gradient_final_km", 0) or 0,
        race_info.get("distance_km", 0) or 0,
        race_info.get("startlist_quality", 0),
        race_info.get("parcours_encoded", 0),
    ], dtype=np.float32)
    race_feat_vec_norm = (race_feat_vec - RACE_FEAT_MIN) / RACE_FEAT_RANGE
    race_feat_broadcast = np.tile(race_feat_vec_norm, (n, 1))

    x = np.concatenate([rider_feats, team_rank, race_feat_broadcast], axis=1)

    team_groups = defaultdict(list)
    for i, r in enumerate(finished):
        team_groups[r["team_id"]].append(i)

    teammate_src, teammate_dst, teammate_weight = [], [], []
    for team_id, members in team_groups.items():
        for i in members:
            for j in members:
                if i != j:
                    teammate_src.append(i)
                    teammate_dst.append(j)
                    teammate_weight.append(float(rider_feats[i, relevant_col]))

    h2h_src, h2h_dst, h2h_weight = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a_gidx = global_idx[i]
            b_gidx = global_idx[j]
            wr = head_to_head.get((a_gidx, b_gidx))
            if wr is not None and wr >= min_h2h_for_edge:
                h2h_src.append(i)
                h2h_dst.append(j)
                h2h_weight.append(wr)

    targets = np.array([
        r["rank_normalized"] if r["rank_normalized"] is not None else 1.0
        for r in finished
    ], dtype=np.float32)

    data = HeteroData()
    data["rider"].x = torch.tensor(x, dtype=torch.float)
    data["rider"].y = torch.tensor(targets, dtype=torch.float)

    if teammate_src:
        data["rider", "teammate_of", "rider"].edge_index = torch.tensor(
            [teammate_src, teammate_dst], dtype=torch.long
        )
        data["rider", "teammate_of", "rider"].edge_attr = torch.tensor(
            teammate_weight, dtype=torch.float
        ).unsqueeze(1)
    else:
        data["rider", "teammate_of", "rider"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["rider", "teammate_of", "rider"].edge_attr  = torch.zeros((0, 1), dtype=torch.float)

    if h2h_src:
        data["rider", "head_to_head", "rider"].edge_index = torch.tensor(
            [h2h_src, h2h_dst], dtype=torch.long
        )
        data["rider", "head_to_head", "rider"].edge_attr = torch.tensor(
            h2h_weight, dtype=torch.float
        ).unsqueeze(1)
    else:
        data["rider", "head_to_head", "rider"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["rider", "head_to_head", "rider"].edge_attr  = torch.zeros((0, 1), dtype=torch.float)

    return data, rider_ids, targets

def build_all_graphs(race_ids, results, head_to_head):
    race_results_grouped = defaultdict(list)
    for r in results:
        race_results_grouped[r["race_id"]].append(r)

    graphs = {}
    skipped = 0
    for race_id in race_ids:
        if race_id not in race_results_grouped:
            continue
        out = build_race_graph(race_id, race_results_grouped[race_id], head_to_head)
        if out is None:
            skipped += 1
            continue
        graphs[race_id] = out

    print(f"Built {len(graphs)} race graphs ({skipped} skipped, too few finishers)")
    return graphs

print("\nBuilding training race graphs...")
train_graphs = build_all_graphs(train_race_ids, train_results, head_to_head)

print("\nBuilding test race graphs...")
test_graphs = build_all_graphs(test_race_ids, test_results, head_to_head)

example_race_id = next(iter(train_graphs))
g, rider_ids, targets = train_graphs[example_race_id]
print(f"\nExample graph: {example_race_id}")
print(f"  riders: {g['rider'].x.shape}")
print(f"  teammate edges: {g['rider','teammate_of','rider'].edge_index.shape[1]}")
print(f"  head-to-head edges: {g['rider','head_to_head','rider'].edge_index.shape[1]}")
print(f"  targets shape: {g['rider'].y.shape}")
print(f"  any nan in x: {torch.isnan(g['rider'].x).any().item()}")

torch.save({
    "train_graphs": train_graphs,
    "test_graphs":  test_graphs,
    "head_to_head": head_to_head,
    "rider_to_idx": rider_to_idx,
    "idx_to_rider": idx_to_rider,
}, DATA_DIR + "race_prediction_graphs_cutoff_wf_3.pt")
print(f"\n Saved to {DATA_DIR}race_prediction_graphs_cutoff_wf_3.pt")

