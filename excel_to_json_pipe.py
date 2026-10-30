
import pandas as pd
from openpyxl import load_workbook
from datetime import datetime, date
import numpy as np
import json
import re
import sys
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

EXCEL_PATH   = "/content/drive/MyDrive/cycling_gnn/data_2025_26/cycling_results_2025_2026_1.xlsx"
OUTPUT_DIR = Path("/content/drive/MyDrive/cycling_gnn/data_2025_26")
REFERENCE_DATE = date.today()

GRACE_DAYS     = 9999
HALF_LIFE_DAYS = 30
FLOOR          = 0.05

CLASSIFICATION_MAP = {
    "1.WWT": 6,
    "2.WWT": 5,
    "1.Pro": 4,
    "2.Pro": 3,
    "1.1":   2,
    "2.1": 1,
    "1.2":   0,
}

PARCOURS_MAP = {
    "flat":     0,
    "punchy":   1,
    "hilly":    2,
    "mountain": 3,
}

def derive_finish_type(gradient_final_km):
    if gradient_final_km > 6:
        return "uphill"
    elif gradient_final_km > 2:
        return "false_flat"
    else:
        return "flat"

def time_decay(race_date, reference_date):
    days_ago = (reference_date - race_date).days
    if days_ago <= 0:
        return 1.0
    if days_ago <= GRACE_DAYS:
        return 1.0
    days_since_grace = days_ago - GRACE_DAYS
    decay = np.exp(-np.log(2) * days_since_grace / HALF_LIFE_DAYS)
    return float(max(decay, FLOOR))

def extract_slug(cell):
    if cell.hyperlink and cell.hyperlink.target:
        return cell.hyperlink.target.rstrip("/").split("/")[-1]
    return None

def clean_distance(val):
    if val is None:
        return None
    return float(re.sub(r"[^\d.]", "", str(val)))

def clean_gradient(val):
    if val is None:
        return None
    return float(re.sub(r"[^\d.]", "", str(val)))

def clean_classification(val):
    s = str(val).strip()
    if re.match(r"^01\.\w+$", s):
        return "1.1"
    if re.match(r"^02\.\w+$", s):
        return "2.1"
    return s

def parse_date(val):
    if isinstance(val, (datetime,)):
        return val.date()
    if isinstance(val, date):
        return val
    for fmt in ("%d %B %Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(val).strip(), fmt).date()
        except ValueError:
            continue
    print(f"Could not parse date: {val}")
    return None

def time_to_seconds(val, is_winner=False):
    if is_winner:
        return 0

    if isinstance(val, datetime):
        total_sec = int(val.total_seconds())
        hh = (total_sec % 86400) // 3600
        mm = (total_sec % 3600) // 60
        return (24 + hh) * 60 + mm

    s = str(val).strip().lstrip("*")
    if s in (",,", "", "None", "nan"):
        return None

    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])

        elif len(parts) == 3:
            p0, p1, p2 = int(parts[0]), int(parts[1]), int(parts[2])

            if p2 == 0 and p0 <= 23:
                return p0 * 60 + p1

            elif p2 == 0 and p0 > 23:
                return p0 * 60 + p1

            else:
                return p0 * 3600 + p1 * 60 + p2

    except ValueError:
        pass

    print(f"Could not parse time: {val!r}")
    return None

def parse_races(wb):
    ws = wb["races"]
    rows = list(ws.iter_rows(min_row=2, values_only=False))

    races = []
    for row in rows:
        if not row[0].value:
            continue

        race_id      = str(row[0].value).strip()
        raw_date     = row[1].value
        raw_class    = row[2].value
        raw_dist     = row[3].value
        raw_grad     = row[4].value
        raw_ps       = row[5].value
        raw_vm       = row[6].value
        raw_sqs      = row[7].value
        raw_parcours = row[8].value if len(row) > 8 else None

        race_date      = parse_date(raw_date)
        classification = clean_classification(raw_class)
        distance       = clean_distance(raw_dist)
        gradient       = clean_gradient(raw_grad)
        profile_score  = int(raw_ps) if raw_ps is not None else 0
        vert_meters    = int(raw_vm) if raw_vm is not None else 0

        sqs_str = str(raw_sqs).split("(")[0].strip() if raw_sqs else "0"
        sqs     = int(re.sub(r"[^\d]", "", sqs_str)) if sqs_str else 0

        if raw_parcours and str(raw_parcours).strip().lower() in PARCOURS_MAP:
            parcours_type = str(raw_parcours).strip().lower()
        else:
            print(f"Missing/unknown parcours_type for {race_id}, defaulting to flat")
            parcours_type = "flat"

        parcours_encoded = PARCOURS_MAP[parcours_type]
        finish_type      = derive_finish_type(gradient or 0.0)
        class_encoded    = CLASSIFICATION_MAP.get(classification, 1)
        decay            = time_decay(race_date, REFERENCE_DATE) if race_date else FLOOR

        races.append({
            "race_id":               race_id,
            "date":                  race_date.isoformat() if race_date else None,
            "classification":        classification,
            "classification_encoded": class_encoded,
            "distance_km":           distance,
            "gradient_final_km":     gradient,
            "profile_score":         profile_score,
            "vertical_meters":       vert_meters,
            "startlist_quality":     sqs,
            "parcours_type":         parcours_type,
            "parcours_encoded":      parcours_encoded,
            "finish_type":           finish_type,
            "decay_weight":          decay,
        })

    print(f"Parsed {len(races)} races")
    return races

def parse_results(wb):
    ws = wb["results"]
    rows = list(ws.iter_rows(min_row=2, values_only=False))

    results = []
    prev_time_sec = None
    prev_race_id  = None

    for row in rows:
        if not row[0].value:
            continue

        race_id   = str(row[0].value).strip()
        rank_val  = row[1].value
        rider_cell = row[2]
        team_cell  = row[3]
        time_val  = row[6].value if len(row) > 6 else None

        rider_id = extract_slug(rider_cell) or str(rider_cell.value).strip().lower().replace(" ", "-")
        team_id  = extract_slug(team_cell)  or str(team_cell.value).strip().lower().replace(" ", "-")
        rider_name = str(rider_cell.value).strip() if rider_cell.value else ""
        team_name  = str(team_cell.value).strip()  if team_cell.value  else ""

        rank = None
        status = "finished"
        if rank_val is not None:
            rank_str = str(rank_val).strip().upper()
            if rank_str in ("DNF", "DNS", "DSQ", "NQ"):
                status = rank_str.lower()
            elif rank_str == "OTL":
                status = "otl"
                try:
                    rank = int(rank_val)
                except (ValueError, TypeError):
                    pass
            else:
                try:
                    rank = int(rank_val)
                except (ValueError, TypeError):
                    status = "unknown"

        if status == "dns":
            continue

        if race_id != prev_race_id:
            prev_time_sec = None
            prev_race_id  = race_id

        if status in ("finished", "otl"):
            is_winner = (rank == 1)
            raw_time_sec = time_to_seconds(time_val, is_winner=is_winner)
            if raw_time_sec is None:
                time_behind_sec = prev_time_sec if prev_time_sec is not None else 0
            else:
                time_behind_sec = raw_time_sec
                prev_time_sec = time_behind_sec
        else:
            time_behind_sec = None

        results.append({
            "race_id":          race_id,
            "rider_id":         rider_id,
            "rider_name":       rider_name,
            "team_id":          team_id,
            "team_name":        team_name,
            "rank":             rank,
            "status":           status,
            "time_behind_sec":  time_behind_sec
        })

    print(f"Parsed {len(results)} result rows")
    return results

def normalize_results(results, races):
    df       = pd.DataFrame(results)
    race_map = {r["race_id"]: r for r in races}

    for race_id, group in df.groupby("race_id"):
        finished      = group[group["status"] == "finished"]
        num_finishers = len(finished)
        max_time      = finished["time_behind_sec"].max()
        parcours      = race_map.get(race_id, {}).get("parcours_type", "flat")

        df.loc[finished.index, "rank_normalized"] = (
            finished["rank"] / num_finishers
        )

        if max_time and max_time > 0:
            df.loc[finished.index, "time_behind_normalized"] = (
                finished["time_behind_sec"] / max_time
            )
        else:
            df.loc[finished.index, "time_behind_normalized"] = 0.0

        if parcours == "flat":
            df.loc[finished.index, "time_behind_normalized"] = 0.0

        dnf_mask = (df["race_id"] == race_id) & (df["status"] != "finished")
        df.loc[dnf_mask, "rank_normalized"]        = 1.0
        df.loc[dnf_mask, "time_behind_normalized"] = 1.0

    df["finished"] = (df["status"] == "finished").astype(int)

    print(f"Normalized {len(df)} results across {df['race_id'].nunique()} races")
    return df.to_dict(orient="records")

def build_riders(results):
    seen = {}
    for r in results:
        rid = r["rider_id"]
        if rid not in seen:
            seen[rid] = {
                "rider_id":   rid,
                "rider_name": r["rider_name"],
                "team_id":    r["team_id"],
                "team_name":  r["team_name"],
            }
        else:
            seen[rid]["team_id"]   = r["team_id"]
            seen[rid]["team_name"] = r["team_name"]

    riders = list(seen.values())
    print(f"Built index of {len(riders)} unique riders")
    return riders


def print_stats(races, results, riders):
    df = pd.DataFrame(results)
    finished = df[df["status"] == "finished"]
    dnf_dsq  = df[df["status"].isin(["dnf", "dsq"])]

    print("\nDataset Stats")
    print(f"   Races:            {len(races)}")
    print(f"   Unique riders:    {len(riders)}")
    print(f"   Total rows:       {len(results)}")
    print(f"   Finished edges:   {len(finished)}")
    print(f"   DNF/DSQ:          {len(dnf_dsq)}")
    print(f"   Avg starters/race:{len(finished) / len(races):.1f}")

    appearances = finished.groupby("rider_id").size()
    print(f"\n   Rider appearances (finished results):")
    print(f"     min:    {appearances.min()}")
    print(f"     median: {appearances.median():.1f}")
    print(f"     mean:   {appearances.mean():.1f}")
    print(f"     max:    {appearances.max()}")
    print(f"     riders with ≥5 races: {(appearances >= 5).sum()}")
    print(f"     riders with ≥10 races: {(appearances >= 10).sum()}")

    print(f"\n   Parcours distribution:")
    for ptype, count in pd.DataFrame(races).groupby("parcours_type").size().items():
        print(f"     {ptype}: {count}")

def run(excel_path = EXCEL_PATH):
    print(f"\nCycling Data Pipeline")
    print(f"   Input:  {excel_path}")
    print(f"   Output: {OUTPUT_DIR}/\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading workbook...")
    wb = load_workbook(excel_path)

    print("\nParsing races...")
    races = parse_races(wb)

    print("\nParsing results...")
    results = parse_results(wb)

    print("\nNormalizing...")
    results = normalize_results(results, races)

    print("\nBuilding riders index...")
    riders = build_riders(results)

    print_stats(races, results, riders)

    print("\n💾 Saving JSON files...")

    with open(OUTPUT_DIR / "races_2025_26_1.json", "w", encoding="utf-8") as f:
        json.dump(races, f, indent=2, ensure_ascii=False)
    print(f"races.json ({len(races)} races)")

    with open(OUTPUT_DIR / "results_2025_26_1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"results.json ({len(results)} results)")

    with open(OUTPUT_DIR / "riders_2025_26_1.json", "w", encoding="utf-8") as f:
        json.dump(riders, f, indent=2, ensure_ascii=False)
    print(f"riders.json ({len(riders)} riders)")

    print("\n Pipeline complete.")
    return races, results, riders

run(EXCEL_PATH)

