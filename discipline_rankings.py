
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from datetime import date, datetime
from torch.nn import Linear
from collections import Counter, defaultdict

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR      = "/content/drive/MyDrive/cycling_gnn/data_2025_26/"
EMBEDDING_DIM = 128
HIDDEN_DIM    = 128
LR            = 0.005
EPOCHS        = 500
MARGIN        = 2.0
SEED          = 88
TEST_CUTOFF_DATE     = date(2025, 9, 15)
DATA_CUTOFF_DATE = date(2026, 2, 28)

torch.manual_seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

races   = json.load(open(DATA_DIR + "races_2025_26_1.json"))

def parse_iso_date(s):
    if s is None:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()

races = [
    r for r in races
    if (d := parse_iso_date(r.get("date"))) is not None and d <= DATA_CUTOFF_DATE
]

train_race_ids = set()
for r in races:
    d = parse_iso_date(r["date"])
    if d is None:
        continue
    if d < TEST_CUTOFF_DATE:
        train_race_ids.add(r["race_id"])

races = [race for race in races if race["race_id"] in train_race_ids]

results = json.load(open(DATA_DIR + "results_2025_26_1.json"))
results = [r for r in results if r["race_id"] in train_race_ids]
riders  = json.load(open(DATA_DIR + "riders_2025_26_1.json"))

race_map = {r["race_id"]: r for r in races}

rider_to_idx = {r["rider_id"]: i for i, r in enumerate(riders)}
race_to_idx  = {r["race_id"]:  i for i, r in enumerate(races)}
idx_to_rider = {i: riders[i] for i in range(len(riders))}

def normalize(vals):
    vals = np.array(vals, dtype=np.float32)
    mn, mx = vals.min(), vals.max()
    return (vals - mn) / (mx - mn + 1e-8)

race_feats = np.stack([
    normalize([r["classification_encoded"] for r in races]),
    normalize([r["profile_score"]          for r in races]),
    normalize([r["vertical_meters"]        for r in races]),
    normalize([r["gradient_final_km"]      for r in races]),
    normalize([r["distance_km"]            for r in races]),
    normalize([r["startlist_quality"]      for r in races]),
    normalize([r["parcours_encoded"]       for r in races]),
], axis=1)

src_rider, dst_race, edge_attrs = [], [], []

for r in results:
    if r["status"] not in ("finished", "otl"):
        continue
    if r["rider_id"] not in rider_to_idx or r["race_id"] not in race_to_idx:
        continue
    src_rider.append(rider_to_idx[r["rider_id"]])
    dst_race.append(race_to_idx[r["race_id"]])
    edge_attrs.append([
        r["rank_normalized"]        or 1.0,
        r["time_behind_normalized"] or 0.0,
        float(r["finished"]),
    ])

edge_index = torch.tensor([src_rider, dst_race], dtype=torch.long)
edge_attr  = torch.tensor(edge_attrs,             dtype=torch.float)

data = HeteroData()
data["rider"].num_nodes = len(riders)
data["race"].x          = torch.tensor(race_feats, dtype=torch.float)
data["rider", "competed_in", "race"].edge_index = edge_index
data["rider", "competed_in", "race"].edge_attr  = edge_attr

sprint_mask   = torch.tensor([
    races[dst_race[i]]["parcours_type"] == "flat"
    for i in range(len(src_rider))
], dtype=torch.bool)

punchy_mask   = torch.tensor([
    races[dst_race[i]]["parcours_type"] == "punchy"
    for i in range(len(src_rider))
], dtype=torch.bool)

versatile_mask = torch.tensor([
    races[dst_race[i]]["parcours_type"] == "hilly"
    for i in range(len(src_rider))
], dtype=torch.bool)

climbing_mask = torch.tensor([
    races[dst_race[i]]["parcours_type"] == "mountain"
    for i in range(len(src_rider))
], dtype=torch.bool)

data["rider", "competed_in", "race"].sprint_mask    = sprint_mask
data["rider", "competed_in", "race"].punchy_mask    = punchy_mask
data["rider", "competed_in", "race"].versatile_mask = versatile_mask
data["rider", "competed_in", "race"].climbing_mask  = climbing_mask

data = data.to(device)

RACE_FEAT_DIM = data["race"].x.shape[1]
EDGE_FEAT_DIM = edge_attr.shape[1]

def build_pairs(results, rider_to_idx, race_to_idx, races, discipline="sprint"):
    parcours_filter = {
        "sprint":   ["flat"],
        "punchy":   ["punchy"],
        "versatile":["hilly"],
        "climbing": ["mountain"],
    }
    allowed = parcours_filter[discipline]

    race_results = defaultdict(list)
    for r in results:
        if r["status"] not in ("finished", "otl") or r["rank"] is None:
            continue
        if r["rider_id"] not in rider_to_idx or r["race_id"] not in race_to_idx:
            continue
        race_results[r["race_id"]].append(r)

    pairs = []
    race_counts = defaultdict(int)
    for race_id, res in race_results.items():
        race_idx = race_to_idx[race_id]

        if races[race_idx]["parcours_type"] not in allowed:
            continue

        res_sorted = sorted(res, key=lambda x: x["rank"])
        sqs        = races[race_idx]["startlist_quality"]
        top10      = res_sorted[:10]
        rest       = res_sorted[10:]

        for r in res_sorted:
          race_counts[rider_to_idx[r["rider_id"]]] += 1

        def time_weight(ri, rj):
          time_i = ri.get("time_behind_sec") or 0
          time_j = rj.get("time_behind_sec") or 0
          gap = (time_j or 0) - (time_i or 0)
          if gap > 0:
              return max(1.0, 1.0 + gap / 300.0)
          return 1.0

        for i, ri in enumerate(top10):
            for rj in res_sorted[i+1:]:
                pairs.append((
                    rider_to_idx[ri["rider_id"]],
                    rider_to_idx[rj["rider_id"]],
                    race_idx,
                    sqs,
                    time_weight(ri, rj),
                ))

        non_top = []
        for i in range(len(rest)):
            for j in range(i+1, len(rest)):
                non_top.append((
                    rider_to_idx[rest[i]["rider_id"]],
                    rider_to_idx[rest[j]["rider_id"]],
                    race_idx,
                    sqs,
                    time_weight(rest[i], rest[j]),
                ))
        k = max(1, len(non_top) // 5)
        pairs.extend(random.sample(non_top, min(k, len(non_top))))

    random.shuffle(pairs)
    print(f"{discipline} pairs: {len(pairs):,}")
    return pairs, dict(race_counts)

class EdgeAwareSAGE(nn.Module):
    def __init__(self, num_riders, race_feat_dim, edge_feat_dim,
                 embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.num_riders    = num_riders
        self.embedding_dim = embedding_dim

        self.rider_embedding_sprint   = nn.Embedding(num_riders, embedding_dim)
        self.rider_embedding_punchy   = nn.Embedding(num_riders, embedding_dim)
        self.rider_embedding_versatile = nn.Embedding(num_riders, embedding_dim)
        self.rider_embedding_climbing = nn.Embedding(num_riders, embedding_dim)
        nn.init.xavier_uniform_(self.rider_embedding_sprint.weight)
        nn.init.xavier_uniform_(self.rider_embedding_punchy.weight)
        nn.init.xavier_uniform_(self.rider_embedding_versatile.weight)
        nn.init.xavier_uniform_(self.rider_embedding_climbing.weight)

        self.race_proj = Linear(race_feat_dim, embedding_dim)

        self.message_net = nn.Sequential(
            Linear(embedding_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.sage_combine = nn.Sequential(
            Linear(embedding_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.message_net2 = nn.Sequential(
            Linear(embedding_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.sage_combine2 = nn.Sequential(
            Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.sprint_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, 1),
        )

        self.punchy_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, 1),
        )

        self.versatile_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, 1),
        )

        self.climbing_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, 1),
        )

        self.dropout = nn.Dropout(0.3)

    def aggregate(self, rider_x, race_x, edge_index, edge_attr, message_net):
        race_emb  = race_x[edge_index[1]]
        edge_input = torch.cat([race_emb, edge_attr], dim=1)

        messages = message_net(edge_input)

        rider_agg = torch.zeros(
            rider_x.shape[0], messages.shape[1], device=device
        )
        counts = torch.zeros(rider_x.shape[0], device=device)
        rider_agg.scatter_add_(
            0, edge_index[0].unsqueeze(1).expand_as(messages), messages
        )
        counts.scatter_add_(
            0, edge_index[0], torch.ones(edge_index[0].shape[0], device=device)
        )
        rider_agg = rider_agg / (counts.unsqueeze(1) + 1e-8)

        return rider_agg

    def forward(self, data):
        rider_x_sprint = self.rider_embedding_sprint(
            torch.arange(self.num_riders, device=device)
        )
        rider_x_climbing = self.rider_embedding_climbing(
            torch.arange(self.num_riders, device=device)
        )

        rider_x_punchy    = self.rider_embedding_punchy(
            torch.arange(self.num_riders, device=device)
        )
        rider_x_versatile = self.rider_embedding_versatile(
            torch.arange(self.num_riders, device=device)
        )

        race_x     = F.relu(self.race_proj(data["race"].x))
        num_races  = data["race"].x.shape[0]
        edge_index = data["rider", "competed_in", "race"].edge_index
        edge_attr  = data["rider", "competed_in", "race"].edge_attr

        sprint_mask   = data["rider", "competed_in", "race"].sprint_mask
        punchy_mask   = data["rider", "competed_in", "race"].punchy_mask
        versatile_mask   = data["rider", "competed_in", "race"].versatile_mask
        climbing_mask = data["rider", "competed_in", "race"].climbing_mask

        def aggregate_discipline(mask, message_net, combine, rider_x):
            ei   = edge_index[:, mask]
            ea   = edge_attr[mask]
            agg  = self.aggregate(rider_x, race_x, ei, ea, message_net)
            return F.relu(combine(torch.cat([rider_x, agg], dim=1)))

        sprint_x = aggregate_discipline(
            sprint_mask, self.message_net, self.sage_combine, rider_x_sprint
        )
        sprint_x = self.dropout(sprint_x)

        ei_s = edge_index[:, sprint_mask]
        ea_s = edge_attr[sprint_mask]

        race_x_sprint = race_x.clone()
        rider_to_race = torch.zeros(num_races, sprint_x.shape[1], device=device)
        race_counts   = torch.zeros(num_races, device=device)
        rider_to_race.scatter_add_(
            0,
            ei_s[1].unsqueeze(1).expand(-1, sprint_x.shape[1]),
            sprint_x[ei_s[0]]
        )
        race_counts.scatter_add_(
            0, ei_s[1], torch.ones(ei_s[1].shape[0], device=device)
        )
        race_x_sprint = race_x + F.relu(
            rider_to_race / (race_counts.unsqueeze(1) + 1e-8)
        )

        sprint_agg2 = self.aggregate(
            sprint_x, race_x_sprint, ei_s, ea_s, self.message_net2
        )
        sprint_x2 = F.relu(self.sage_combine2(
            torch.cat([sprint_x, sprint_agg2], dim=1)
        ))
        sprint_x2 = self.dropout(sprint_x2)

        versatile_x = aggregate_discipline(
            versatile_mask, self.message_net, self.sage_combine, rider_x_versatile
        )
        versatile_x = self.dropout(versatile_x)

        ei_c = edge_index[:, versatile_mask]
        ea_c = edge_attr[versatile_mask]

        rider_to_race_c = torch.zeros(num_races, versatile_x.shape[1], device=device)
        race_counts_c   = torch.zeros(num_races, device=device)
        rider_to_race_c.scatter_add_(
            0,
            ei_c[1].unsqueeze(1).expand(-1, versatile_x.shape[1]),
            versatile_x[ei_c[0]]
        )
        race_counts_c.scatter_add_(
            0, ei_c[1], torch.ones(ei_c[1].shape[0], device=device)
        )
        race_x_versatile = race_x + F.relu(
            rider_to_race_c / (race_counts_c.unsqueeze(1) + 1e-8)
        )

        versatile_agg2 = self.aggregate(
            versatile_x, race_x_versatile, ei_c, ea_c, self.message_net2
        )
        versatile_x2 = F.relu(self.sage_combine2(
            torch.cat([versatile_x, versatile_agg2], dim=1)
        ))
        versatile_x2 = self.dropout(versatile_x2)

        punchy_x = aggregate_discipline(
            punchy_mask, self.message_net, self.sage_combine, rider_x_punchy
        )
        punchy_x = self.dropout(punchy_x)

        ei_c = edge_index[:, punchy_mask]
        ea_c = edge_attr[punchy_mask]

        rider_to_race_c = torch.zeros(num_races, punchy_x.shape[1], device=device)
        race_counts_c   = torch.zeros(num_races, device=device)
        rider_to_race_c.scatter_add_(
            0,
            ei_c[1].unsqueeze(1).expand(-1, punchy_x.shape[1]),
            punchy_x[ei_c[0]]
        )
        race_counts_c.scatter_add_(
            0, ei_c[1], torch.ones(ei_c[1].shape[0], device=device)
        )
        race_x_punchy = race_x + F.relu(
            rider_to_race_c / (race_counts_c.unsqueeze(1) + 1e-8)
        )

        punchy_agg2 = self.aggregate(
            punchy_x, race_x_punchy, ei_c, ea_c, self.message_net2
        )
        punchy_x2 = F.relu(self.sage_combine2(
            torch.cat([punchy_x, punchy_agg2], dim=1)
        ))
        punchy_x2 = self.dropout(punchy_x2)

        climbing_x = aggregate_discipline(
            climbing_mask, self.message_net, self.sage_combine, rider_x_climbing
        )
        climbing_x = self.dropout(climbing_x)

        ei_c = edge_index[:, climbing_mask]
        ea_c = edge_attr[climbing_mask]

        rider_to_race_c = torch.zeros(num_races, climbing_x.shape[1], device=device)
        race_counts_c   = torch.zeros(num_races, device=device)
        rider_to_race_c.scatter_add_(
            0,
            ei_c[1].unsqueeze(1).expand(-1, climbing_x.shape[1]),
            climbing_x[ei_c[0]]
        )
        race_counts_c.scatter_add_(
            0, ei_c[1], torch.ones(ei_c[1].shape[0], device=device)
        )
        race_x_climbing = race_x + F.relu(
            rider_to_race_c / (race_counts_c.unsqueeze(1) + 1e-8)
        )

        climbing_agg2 = self.aggregate(
            climbing_x, race_x_climbing, ei_c, ea_c, self.message_net2
        )
        climbing_x2 = F.relu(self.sage_combine2(
            torch.cat([climbing_x, climbing_agg2], dim=1)
        ))
        climbing_x2 = self.dropout(climbing_x2)

        sprint_scores   = self.sprint_head(sprint_x2).squeeze(-1)
        versatile_scores = self.versatile_head(versatile_x2).squeeze(-1)
        punchy_scores = self.punchy_head(punchy_x2).squeeze(-1)
        climbing_scores = self.climbing_head(climbing_x2).squeeze(-1)

        return sprint_scores, punchy_scores, versatile_scores, climbing_scores, sprint_x2

def ranking_loss(scores, pairs_batch, race_counts, margin=MARGIN, confidence_k=10):
    if not pairs_batch:
        return torch.tensor(0.0, device=device)
    rider_i      = torch.tensor([p[0] for p in pairs_batch], dtype=torch.long,  device=device)
    rider_j      = torch.tensor([p[1] for p in pairs_batch], dtype=torch.long,  device=device)
    sqs          = torch.tensor([p[3] for p in pairs_batch], dtype=torch.float, device=device)
    time_weights = torch.tensor([p[4] for p in pairs_batch], dtype=torch.float, device=device)

    sqs_norm = (sqs - sqs.min()) / (sqs.max() - sqs.min() + 1e-8)

    conf_i = torch.tensor(
        [min(1.0, race_counts.get(p[0], 0) / confidence_k) for p in pairs_batch],
        dtype=torch.float, device=device)
    conf_j = torch.tensor(
        [min(1.0, race_counts.get(p[1], 0) / confidence_k) for p in pairs_batch],
        dtype=torch.float, device=device)
    confidence = conf_i * conf_j

    weight = (0.3 + 0.7 * sqs_norm) * time_weights  * confidence

    loss = weight * F.relu(margin - (scores[rider_i] - scores[rider_j]))
    return loss.mean()

def shrinkage_loss(scores, race_counts, num_riders, confidence_k=10, prior=0.5):
    idx = torch.arange(num_riders, device=device)
    counts = torch.tensor(
        [race_counts.get(i.item(), 0) for i in idx], dtype=torch.float, device=device
    )
    confidence = torch.clamp(counts / confidence_k, max=1.0)
    pull_strength = 1.0 - confidence
    return (pull_strength * (scores - prior) ** 2).mean()

model = EdgeAwareSAGE(
    num_riders    = len(riders),
    race_feat_dim = RACE_FEAT_DIM,
    edge_feat_dim = EDGE_FEAT_DIM,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

sprint_pairs, sprint_race_counts       = build_pairs(results, rider_to_idx, race_to_idx, races, discipline="sprint")
punchy_pairs, punchy_race_counts       = build_pairs(results, rider_to_idx, race_to_idx, races, discipline="punchy")
versatile_pairs, versatile_race_counts = build_pairs(results, rider_to_idx, race_to_idx, races, discipline="versatile")
climbing_pairs, climbing_race_counts   = build_pairs(results, rider_to_idx, race_to_idx, races, discipline="climbing")

best_loss  = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    sprint_scores, punchy_scores, versatile_scores, climbing_scores, _ = model(data)

    sprint_batch   = random.sample(sprint_pairs,   min(1000, len(sprint_pairs)))
    punchy_batch   = random.sample(punchy_pairs,   min(1000, len(punchy_pairs)))
    versatile_batch   = random.sample(versatile_pairs,   min(1000, len(versatile_pairs)))
    climbing_batch = random.sample(climbing_pairs, min(1000,  len(climbing_pairs)))

    loss_sprint    = ranking_loss(sprint_scores, sprint_batch, sprint_race_counts)
    loss_punchy    = ranking_loss(punchy_scores, punchy_batch, punchy_race_counts)
    loss_versatile = ranking_loss(versatile_scores, versatile_batch, versatile_race_counts)
    loss_climbing  = ranking_loss(climbing_scores, climbing_batch, climbing_race_counts)
    loss = (loss_sprint + loss_punchy + loss_versatile + loss_climbing
        + 0.3 * shrinkage_loss(sprint_scores, sprint_race_counts, len(riders))
        + 0.15 * shrinkage_loss(punchy_scores, punchy_race_counts, len(riders))
        + 0.2 * shrinkage_loss(versatile_scores, versatile_race_counts, len(riders))
        + 0.075 * shrinkage_loss(climbing_scores, climbing_race_counts, len(riders)))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if loss.item() < best_loss:
        best_loss  = loss.item()
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Best: {best_loss:.4f} "
              f"| S: {loss_sprint.item():.4f} | C: {loss_climbing.item():.4f} | V: {loss_versatile.item():.4f} | P: {loss_punchy.item():.4f}")

model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    sprint_scores, punchy_scores, versatile_scores, climbing_scores, embeddings = model(data)
    sprint_scores    = sprint_scores.cpu().numpy()
    punchy_scores    = punchy_scores.cpu().numpy()
    versatile_scores = versatile_scores.cpu().numpy()
    climbing_scores  = climbing_scores.cpu().numpy()

sprint_appearances = Counter(
    rider_to_idx[r["rider_id"]]
    for r in results
    if r["status"] in ("finished", "otl")
    and r["rider_id"] in rider_to_idx
    and races[race_to_idx[r["race_id"]]]["parcours_type"] == "flat"
)
punchy_appearances = Counter(
    rider_to_idx[r["rider_id"]]
    for r in results
    if r["status"] in ("finished", "otl")
    and r["rider_id"] in rider_to_idx
    and races[race_to_idx[r["race_id"]]]["parcours_type"] == "punchy"
)
versatile_appearances = Counter(
    rider_to_idx[r["rider_id"]]
    for r in results
    if r["status"] in ("finished", "otl")
    and r["rider_id"] in rider_to_idx
    and races[race_to_idx[r["race_id"]]]["parcours_type"] == "hilly"
)
climbing_appearances = Counter(
    rider_to_idx[r["rider_id"]]
    for r in results
    if r["status"] in ("finished", "otl")
    and r["rider_id"] in rider_to_idx
    and races[race_to_idx[r["race_id"]]]["parcours_type"] == "mountain"
)

def print_rankings(scores, appearances, title, min_races=5):
    fs       = torch.tensor(scores)
    gnn_norm = (fs - fs.min()) / (fs.max() - fs.min() + 1e-8)
    gnn_norm = gnn_norm.numpy()

    ranked = sorted(
        [(idx, float(gnn_norm[idx])) for idx in range(len(riders))
         if appearances.get(idx, 0) >= min_races],
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\n🏆 {title}\n")
    print(f"{'Rank':<5} {'Rider':<30} {'Score':>7} {'Races':>6}")
    print("-" * 85)
    for rank, (idx, score) in enumerate(ranked[:30], 1):
        rider = idx_to_rider[idx]
        print(f"{rank:<5} {rider['rider_name']:<30} "
              f"{score:>7.3f} {appearances.get(idx,0):>6}")

print_rankings(sprint_scores,   sprint_appearances,   "Sprint Rankings", min_races = 0)
print_rankings(climbing_scores, climbing_appearances, "Climbing Rankings", min_races=0)
print_rankings(punchy_scores,    punchy_appearances,    "Punchy Rankings",    min_races=0)
print_rankings(versatile_scores, versatile_appearances, "Versatile Rankings", min_races=0)

all_races_raw   = json.load(open(DATA_DIR + "races_2025_26_1.json"))
all_results_raw = json.load(open(DATA_DIR + "results_2025_26_1.json"))

def parse_iso_date_local(s):
    if s is None:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()

all_races_raw = [
    r for r in all_races_raw
    if (d := parse_iso_date(r.get("date"))) is not None and d <= DATA_CUTOFF_DATE
]

training_races_sorted = sorted(
    [r for r in all_races_raw if parse_iso_date_local(r["date"]) is not None
     and parse_iso_date_local(r["date"]) < TEST_CUTOFF_DATE],
    key=lambda r: parse_iso_date_local(r["date"])
)
test_races_sorted = sorted(
    [r for r in all_races_raw if parse_iso_date_local(r["date"]) is not None
     and parse_iso_date_local(r["date"]) >= TEST_CUTOFF_DATE],
    key=lambda r: parse_iso_date_local(r["date"])
)

all_results_by_race = defaultdict(list)
for r in all_results_raw:
    all_results_by_race[r["race_id"]].append(r)

def rebuild_graph_tensors(races_wf, results_wf, raw_race_feats_wf, rider_to_idx, race_to_idx_wf):
    mn, mx = raw_race_feats_wf.min(axis=0), raw_race_feats_wf.max(axis=0)
    race_feats_norm = (raw_race_feats_wf - mn) / (mx - mn + 1e-8)

    src_rider, dst_race, edge_attrs = [], [], []
    for r in results_wf:
        if r["status"] not in ("finished", "otl"):
            continue
        if r["rider_id"] not in rider_to_idx or r["race_id"] not in race_to_idx_wf:
            continue
        src_rider.append(rider_to_idx[r["rider_id"]])
        dst_race.append(race_to_idx_wf[r["race_id"]])
        edge_attrs.append([
            r["rank_normalized"] or 1.0,
            r["time_behind_normalized"] or 0.0,
            float(r["finished"]),
        ])

    edge_index_new = torch.tensor([src_rider, dst_race], dtype=torch.long)
    edge_attr_new  = torch.tensor(edge_attrs, dtype=torch.float)

    def mask_for(ptype):
        return torch.tensor(
            [races_wf[dst_race[i]]["parcours_type"] == ptype for i in range(len(src_rider))],
            dtype=torch.bool)

    new_data = data.__class__()
    new_data["rider"].num_nodes = len(riders)
    new_data["race"].x = torch.tensor(race_feats_norm, dtype=torch.float)
    new_data["rider", "competed_in", "race"].edge_index = edge_index_new
    new_data["rider", "competed_in", "race"].edge_attr  = edge_attr_new
    new_data["rider", "competed_in", "race"].sprint_mask    = mask_for("flat")
    new_data["rider", "competed_in", "race"].punchy_mask    = mask_for("punchy")
    new_data["rider", "competed_in", "race"].versatile_mask = mask_for("hilly")
    new_data["rider", "competed_in", "race"].climbing_mask  = mask_for("mountain")
    return new_data.to(device)

def normalize_scores(arr):
    if arr is None:
        return None
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

def full_retrain(data_wf, results_wf, races_wf, race_to_idx_wf, epochs=EPOCHS, lr=LR):
    fresh_model = EdgeAwareSAGE(
        num_riders=len(riders), race_feat_dim=RACE_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM,
    ).to(device)
    optimizer = torch.optim.Adam(fresh_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    sprint_pairs, sprint_rc       = build_pairs(results_wf, rider_to_idx, race_to_idx_wf, races_wf, discipline="sprint")
    punchy_pairs, punchy_rc       = build_pairs(results_wf, rider_to_idx, race_to_idx_wf, races_wf, discipline="punchy")
    versatile_pairs, versatile_rc = build_pairs(results_wf, rider_to_idx, race_to_idx_wf, races_wf, discipline="versatile")
    climbing_pairs, climbing_rc   = build_pairs(results_wf, rider_to_idx, race_to_idx_wf, races_wf, discipline="climbing")

    best_loss, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        fresh_model.train()
        optimizer.zero_grad()

        s_sc, p_sc, v_sc, c_sc, _ = fresh_model(data_wf)

        sprint_batch    = random.sample(sprint_pairs, min(1000, len(sprint_pairs)))
        punchy_batch    = random.sample(punchy_pairs, min(1000, len(punchy_pairs)))
        versatile_batch = random.sample(versatile_pairs, min(1000, len(versatile_pairs)))
        climbing_batch  = random.sample(climbing_pairs, min(1000, len(climbing_pairs)))

        loss = (ranking_loss(s_sc, sprint_batch, sprint_rc)
                + ranking_loss(p_sc, punchy_batch, punchy_rc)
                + ranking_loss(v_sc, versatile_batch, versatile_rc)
                + ranking_loss(c_sc, climbing_batch, climbing_rc)
                + 0.3 * shrinkage_loss(s_sc, sprint_rc, len(riders))
                + 0.15 * shrinkage_loss(p_sc, punchy_rc, len(riders))
                + 0.2 * shrinkage_loss(v_sc, versatile_rc, len(riders))
                + 0.075 * shrinkage_loss(c_sc, climbing_rc, len(riders)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fresh_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in fresh_model.state_dict().items()}

    fresh_model.load_state_dict(best_state)
    return fresh_model

def walk_forward_full_retrain(race_list, races_wf, results_wf, raw_race_feats_wf,
                                race_to_idx_wf, batch_size, phase_label, snapshots_dict,
                                save_path):
    batch = []
    batch_num = 0

    def process_and_snapshot(batch):
        nonlocal races_wf, results_wf, raw_race_feats_wf, race_to_idx_wf, batch_num
        for race_dict in batch:
            race_id = race_dict["race_id"]
            if race_id not in all_results_by_race:
                continue
            new_idx = len(races_wf)
            races_wf.append(race_dict)
            race_to_idx_wf[race_id] = new_idx
            results_wf.extend(all_results_by_race[race_id])

            new_raw_row = np.array([[
                race_dict.get("classification_encoded", 0), race_dict.get("profile_score", 0),
                race_dict.get("vertical_meters", 0), race_dict.get("gradient_final_km", 0) or 0,
                race_dict.get("distance_km", 0) or 0, race_dict.get("startlist_quality", 0),
                race_dict.get("parcours_encoded", 0),
            ]], dtype=np.float32)
            raw_race_feats_wf = np.vstack([raw_race_feats_wf, new_raw_row])

        data_wf_local = rebuild_graph_tensors(races_wf, results_wf, raw_race_feats_wf, rider_to_idx, race_to_idx_wf)
        retrained_model = full_retrain(data_wf_local, results_wf, races_wf, race_to_idx_wf)

        batch_num += 1
        label = f"{phase_label}_batch{batch_num}_n{len(races_wf)}"
        snapshots_dict[label] = snapshot_scores(retrained_model, data_wf_local, label)

        torch.save(snapshots_dict, save_path)
        print(f"  [{label}] retrained on {len(races_wf)} total races")

    for race_dict in race_list:
        batch.append(race_dict)
        if len(batch) >= batch_size:
            process_and_snapshot(batch)
            batch = []
    if batch:
        process_and_snapshot(batch)

    return races_wf, results_wf, raw_race_feats_wf, race_to_idx_wf

def snapshot_scores(model, data_wf, label):
    model.eval()
    with torch.no_grad():
        s_sc, p_sc, v_sc, c_sc, _ = model(data_wf)
    scores = {
        "sprint": normalize_scores(s_sc.cpu().numpy()),
        "punchy": normalize_scores(p_sc.cpu().numpy()),
        "versatile": normalize_scores(v_sc.cpu().numpy()),
        "climbing": normalize_scores(c_sc.cpu().numpy()),
    }
    return scores

races_wf = list(races)
results_wf = list(results)
race_to_idx_wf = dict(race_to_idx)

raw_race_feats_wf = np.array([[
    r.get("classification_encoded", 0), r.get("profile_score", 0),
    r.get("vertical_meters", 0), r.get("gradient_final_km", 0) or 0,
    r.get("distance_km", 0) or 0, r.get("startlist_quality", 0),
    r.get("parcours_encoded", 0),
] for r in races_wf], dtype=np.float32)

TRAIN_BATCH_SIZE = len(training_races_sorted)
TEST_BATCH_SIZE = 3
SNAPSHOT_SAVE_PATH = DATA_DIR + "discipline_snapshots_full_retrain_16.pt"

all_snapshots = {}

all_snapshots["static_train"] = snapshot_scores(model, data, "static_train")
torch.save(all_snapshots, SNAPSHOT_SAVE_PATH)

races_wf, results_wf, raw_race_feats_wf, race_to_idx_wf = walk_forward_full_retrain(
    test_races_sorted, races_wf, results_wf, raw_race_feats_wf, race_to_idx_wf,
    batch_size=TEST_BATCH_SIZE, phase_label="test", snapshots_dict=all_snapshots,
    save_path=SNAPSHOT_SAVE_PATH
)

import re

all_snapshots = torch.load(DATA_DIR + "discipline_snapshots_full_retrain_16.pt", weights_only=False)

all_results_by_race_local = {}
for r in all_results_raw:
    all_results_by_race_local.setdefault(r["race_id"], []).append(r)

def build_batches(race_list, batch_size):
    batches = []
    current = []
    for race_dict in race_list:
        if race_dict["race_id"] not in all_results_by_race_local:
            continue
        current.append(race_dict["race_id"])
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches

train_batches = build_batches(training_races_sorted, TRAIN_BATCH_SIZE)
test_batches = build_batches(test_races_sorted, TEST_BATCH_SIZE)

def batch_num_from_label(label, prefix):
    m = re.match(rf"{prefix}_batch(\d+)_", label)
    return int(m.group(1)) if m else -1

train_labels = sorted(
    [k for k in all_snapshots if k.startswith("static_train")],
    key=lambda k: batch_num_from_label(k, "train")
)
test_labels = sorted(
    [k for k in all_snapshots if k.startswith("test_batch")],
    key=lambda k: batch_num_from_label(k, "test")
)
ordered_labels = train_labels + test_labels

race_id_to_label = {}

preceding_label = "static_train"
for batch_race_ids, label in zip(train_batches, train_labels):
    for race_id in batch_race_ids:
        race_id_to_label[race_id] = preceding_label
    preceding_label = label

for batch_race_ids, label in zip(test_batches, test_labels):
    for race_id in batch_race_ids:
        race_id_to_label[race_id] = preceding_label
    preceding_label = label

race_id_to_scores = {
    race_id: all_snapshots[label] for race_id, label in race_id_to_label.items()
}

torch.save(race_id_to_scores, DATA_DIR + "race_id_to_discipline_scores_16.pt")

FORM_WINDOW_DAYS = 30
FORM_LOOKBACK_N = 5

from datetime import timedelta
def compute_form_features(results_so_far, races_so_far_map, num_riders):
    if not results_so_far:
        out = np.zeros((num_riders, 2), dtype=np.float32)
        out[:, 1] = 0.5
        return out
    all_dates = [parse_iso_date_local(races_so_far_map[r["race_id"]]["date"])
                 for r in results_so_far if r["race_id"] in races_so_far_map]
    all_dates = [d for d in all_dates if d is not None]
    if not all_dates:
        out = np.zeros((num_riders, 2), dtype=np.float32)
        out[:, 1] = 0.5
        return out
    now = max(all_dates)
    window_start = now - timedelta(days=FORM_WINDOW_DAYS)

    rider_results = defaultdict(list)
    for r in results_so_far:
        if r["status"] not in ("finished", "otl") or r["race_id"] not in races_so_far_map:
            continue
        if r["rider_id"] not in rider_to_idx:
            continue
        d = parse_iso_date_local(races_so_far_map[r["race_id"]]["date"])
        if d is None:
            continue
        gidx = rider_to_idx[r["rider_id"]]
        rn = r["rank_normalized"] if r["rank_normalized"] is not None else 1.0
        rider_results[gidx].append((d, rn))

    out = np.zeros((num_riders, 2), dtype=np.float32)
    out[:, 1] = 0.5
    for gidx, entries in rider_results.items():
        entries.sort(key=lambda x: x[0])
        races_in_window = sum(1 for d, _ in entries if window_start <= d <= now)
        recent_n = entries[-FORM_LOOKBACK_N:]
        mean_recent_rank = np.mean([rn for _, rn in recent_n])
        out[gidx, 0] = races_in_window
        out[gidx, 1] = mean_recent_rank

    out[:, 0] = np.clip(out[:, 0] / 15.0, 0, 1)
    return out

race_id_to_form = {}

preceding_results = []
races_so_far_map = {}

for batch_race_ids in train_batches:
    form = compute_form_features(preceding_results, races_so_far_map, len(riders))
    for race_id in batch_race_ids:
        race_id_to_form[race_id] = form
    for race_id in batch_race_ids:
        preceding_results.extend(all_results_by_race[race_id])
        races_so_far_map[race_id] = next(r for r in all_races_raw if r["race_id"] == race_id)

for batch_race_ids in test_batches:
    form = compute_form_features(preceding_results, races_so_far_map, len(riders))
    for race_id in batch_race_ids:
        race_id_to_form[race_id] = form
    for race_id in batch_race_ids:
        preceding_results.extend(all_results_by_race[race_id])
        races_so_far_map[race_id] = next(r for r in all_races_raw if r["race_id"] == race_id)

torch.save(race_id_to_form, DATA_DIR + "race_id_to_form_features_16.pt")

import pandas as pd

"""
Attributes how much each of a rider's individual race results contributed to
her final discipline score, via edge ablation: for each race she competed in,
remove just that (rider, race) edge from the graph, rerun the model, and
measure how much her score shifts. This is the honest way to do this for a
transductive model -- her score isn't a sum of per-race terms you can just
read off, it's the output of one forward pass over the WHOLE graph, so the
only way to isolate one race's effect is to actually remove it and re-run.

contribution = baseline_score - score_without_that_race
  positive contribution = that race pulled her score UP (removing it hurts her)
  negative contribution = that race pulled her score DOWN (removing it helps her)
"""

DISCIPLINE_INDEX = {"sprint": 0, "punchy": 1, "versatile": 2, "climbing": 3}
DISCIPLINE_MASK_ATTR = {
    "sprint": "sprint_mask", "punchy": "punchy_mask",
    "versatile": "versatile_mask", "climbing": "climbing_mask",
}

def save_model_for_analysis(model, data, tag="default"):
    torch.save(model.state_dict(), DATA_DIR + f"discipline_model_for_analysis_{tag}.pt")
    torch.save(data.cpu(), DATA_DIR + f"discipline_data_for_analysis_{tag}.pt")

def load_model_for_analysis(tag="default"):
    model = EdgeAwareSAGE(
        num_riders=len(riders), race_feat_dim=RACE_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM,
    ).to(device)
    model.load_state_dict(
        torch.load(DATA_DIR + f"discipline_model_for_analysis_{tag}.pt", weights_only=False)
    )
    model.eval()

    data = torch.load(DATA_DIR + f"discipline_data_for_analysis_{tag}.pt", weights_only=False)
    data = data.to(device)
    return model, data

@torch.no_grad()
def race_contributions_for_rider(model, data, rider_id, discipline, rider_to_idx,
                                  races_list, race_to_idx_map, results_by_race):
    """
    Returns a DataFrame: one row per race the rider competed in, with her
    result in that race and how much it contributed to her final discipline
    score, sorted by contribution magnitude (most influential first).
    """
    if rider_id not in rider_to_idx:
        raise ValueError(f"{rider_id} not found in rider_to_idx")
    rider_gidx = rider_to_idx[rider_id]

    model.eval()
    edge_type = ("rider", "competed_in", "race")
    edge_index = data[edge_type].edge_index
    edge_attr = data[edge_type].edge_attr
    mask_attr = DISCIPLINE_MASK_ATTR[discipline]
    disc_mask = data[edge_type][mask_attr]

    scores = model(data)
    disc_idx = DISCIPLINE_INDEX[discipline]
    baseline_score = scores[disc_idx][rider_gidx].item()

    rider_edges = ((edge_index[0] == rider_gidx) & disc_mask).nonzero(as_tuple=True)[0]

    rows = []
    for edge_pos in rider_edges.tolist():
        race_gidx = edge_index[1, edge_pos].item()
        race_id = races_list[race_gidx]["race_id"] if race_gidx < len(races_list) else f"race_idx_{race_gidx}"

        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        keep_mask[edge_pos] = False

        data_ablated = data.clone()
        data_ablated[edge_type].edge_index = edge_index[:, keep_mask]
        data_ablated[edge_type].edge_attr = edge_attr[keep_mask]
        for m_attr in DISCIPLINE_MASK_ATTR.values():
            data_ablated[edge_type][m_attr] = data[edge_type][m_attr][keep_mask]

        scores_ablated = model(data_ablated)
        ablated_score = scores_ablated[disc_idx][rider_gidx].item()

        contribution = baseline_score - ablated_score

        her_rank_normalized = edge_attr[edge_pos, 0].item()

        rows.append({
            "race_id": race_id,
            "rank_normalized": her_rank_normalized,
            "contribution": contribution,
        })

    df = pd.DataFrame(rows)
    df["abs_contribution"] = df["contribution"].abs()
    return df.sort_values("abs_contribution", ascending=False).drop(columns="abs_contribution")

all_races_full  = json.load(open(DATA_DIR + "races_2025_26_1.json"))
all_results_full = json.load(open(DATA_DIR + "results_2025_26_1.json"))
races_wf = list(all_races_full)
results_wf = list(all_results_full)
race_to_idx_wf = dict(race_to_idx)

raw_race_feats_wf = np.array([[
    r.get("classification_encoded", 0), r.get("profile_score", 0),
    r.get("vertical_meters", 0), r.get("gradient_final_km", 0) or 0,
    r.get("distance_km", 0) or 0, r.get("startlist_quality", 0),
    r.get("parcours_encoded", 0),
] for r in races_wf], dtype=np.float32)

race_to_idx_full = {r["race_id"]: i for i, r in enumerate(all_races_full)}

data_final = rebuild_graph_tensors(
    all_races_full, all_results_full, raw_race_feats_wf,
    rider_to_idx, race_to_idx_full,
)

final_model = full_retrain(data_final, all_results_full, all_races_full, race_to_idx_full)

save_model_for_analysis(final_model, data_final, tag="final_walk_forward")

def precompute_all_contributions(model, data, rider_to_idx, races_list,
                                   race_to_idx_map, results_by_race, out_path):
    """Runs race_contributions_for_rider for every rider, every discipline
    she has edges in, and saves the result as a plain dict -- no model or
    graph object involved, safe to load into the Space with just torch."""
    all_contributions = {}
    for rider_id in rider_to_idx:
        all_contributions[rider_id] = {}
        for discipline in DISCIPLINE_INDEX:
            try:
                df = race_contributions_for_rider(
                    model, data, rider_id, discipline, rider_to_idx,
                    races_list, race_to_idx_map, results_by_race,
                )
            except Exception:
                continue
            if len(df) == 0:
                continue
            all_contributions[rider_id][discipline] = df.to_dict("records")
        print(f"done: {rider_id}")

    torch.save(all_contributions, out_path)

model, data = load_model_for_analysis(tag="final_walk_forward")

final_model_scores = snapshot_scores(model, data, "final_walk_forward")
torch.save(final_model_scores, DATA_DIR + "final_model_discipline_scores.pt")

precompute_all_contributions(
    model, data, rider_to_idx, all_races_full, race_to_idx_full, all_results_by_race,
    out_path=DATA_DIR + "rider_race_contributions.pt",
)

FILE_A, LABEL_A = DATA_DIR + "discipline_snapshots_full_retrain_15.pt", "run15"
FILE_B, LABEL_B = DATA_DIR + "discipline_snapshots_full_retrain_10.pt", "run10"
SNAPSHOT_LABEL  = "test_batch17_n222"
MAX_RACES       = 5

DISCIPLINES = ["sprint", "punchy", "versatile", "climbing"]
PARCOURS_TO_DISC = {"flat": "sprint", "punchy": "punchy",
                    "hilly": "versatile", "mountain": "climbing"}

races   = json.load(open(DATA_DIR + "races_2025_26_1.json"))
results = json.load(open(DATA_DIR + "results_2025_26_1.json"))

race_to_disc = {r["race_id"]: PARCOURS_TO_DISC.get(r.get("parcours_type"), "sprint") for r in races}
idx_to_rider = {i: r for i, r in enumerate(riders)}

counts = {d: np.zeros(len(riders), dtype=int) for d in DISCIPLINES}
for r in results:
    if r["status"] not in ("finished", "otl"):
        continue
    if r["rider_id"] not in rider_to_idx or r["race_id"] not in race_to_disc:
        continue
    counts[race_to_disc[r["race_id"]]][rider_to_idx[r["rider_id"]]] += 1

def describe(obj, name="snap", depth=0):
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict[{len(obj)}] keys={list(obj.keys())[:6]}")
        if depth < 2 and obj:
            k = list(obj.keys())[-1]
            describe(obj[k], f"[{k!r}]", depth + 1)
    elif torch.is_tensor(obj):
        print(f"{pad}{name}: tensor {tuple(obj.shape)}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}[{len(obj)}]")
    else:
        print(f"{pad}{name}: {type(obj).__name__}")

def load_scores(path, label=None):
    snaps = torch.load(path, weights_only=False)
    key = label if label is not None else list(snaps.keys())[-1]
    snap = snaps[key]
    out = {}
    for i, d in enumerate(DISCIPLINES):
        s = snap[d] if isinstance(snap, dict) else snap[i]
        out[d] = (s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)).astype(float)
    return out

scores_A = load_scores(FILE_A, SNAPSHOT_LABEL)
scores_B = load_scores(FILE_B, SNAPSHOT_LABEL)

assert len(scores_A["sprint"]) == len(riders) == len(scores_B["sprint"]), \
    f"rider count mismatch: scores={len(scores_A['sprint'])}, riders.json={len(riders)}"

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 500)

def rider_name(i):
    r = idx_to_rider[i]
    return r.get("rider_name", r["rider_id"])

for d in DISCIPLINES:
    n = counts[d]
    active = n >= 1
    df = pd.DataFrame({
        "rider":   [rider_name(i) for i in range(len(n))],
        "n_races": n,
        f"score_{LABEL_A}": scores_A[d],
        f"score_{LABEL_B}": scores_B[d],
    })[active].copy()

    df[f"rank_{LABEL_A}"] = df[f"score_{LABEL_A}"].rank(ascending=False, method="min").astype(int)
    df[f"rank_{LABEL_B}"] = df[f"score_{LABEL_B}"].rank(ascending=False, method="min").astype(int)
    df["score_diff"] = df[f"score_{LABEL_A}"] - df[f"score_{LABEL_B}"]
    df["rank_diff"]  = df[f"rank_{LABEL_B}"] - df[f"rank_{LABEL_A}"]

    few  = df[df["n_races"] <= MAX_RACES].sort_values(f"rank_{LABEL_A}")
    many = df[df["n_races"] >  MAX_RACES]

    print(f"\n{'=' * 95}\n{d.upper()}  |  active riders: {len(df)}  |  with <= {MAX_RACES} races: {len(few)}\n{'=' * 95}")
    print(few.round(4).to_string(index=False))

    print(f"\nMean |score_diff|  few-race: {few['score_diff'].abs().mean():.4f}   many-race: {many['score_diff'].abs().mean():.4f}")
    print(f"Mean |rank_diff|   few-race: {few['rank_diff'].abs().mean():.1f}   many-race: {many['rank_diff'].abs().mean():.1f}")
    print(f"Few-race riders in top-50:  {LABEL_A}={int((few[f'rank_{LABEL_A}'] <= 50).sum())}   "
          f"{LABEL_B}={int((few[f'rank_{LABEL_B}'] <= 50).sum())}")
