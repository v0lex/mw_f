# GNN-based prediction and ranking for women's professional road cycling
 
Code for my master's thesis. It contains everything used to produce the results,
figures and the demo application in the thesis:
 
1. **Data pipeline**: Excel files → JSON (`races`, `riders`, `results`)
2. **Discipline rankings**: a graph neural network that gives every rider a score in four
   disciplines (sprint, punchy, versatile, climbing), trained with a walk-forward scheme
3. **Graph builder**: builds one graph per race (head-to-head and teammate edges, 14 rider features)
4. **Race prediction and comparison**: pure sort by discipline score, RankNet, ListNet,
   top-10 classifiers (1 and 2 layers, with/without win head) and XGBoost, with
   hyperparameter search and statistical comparison against a published benchmark
5. **Interface**: a Gradio app (discipline Top-20, per-race contributions, race prediction)
## Please read first: how this code was run
 
All code was developed and run in **Google Colab** as notebooks and exported to `.py`
files. The files in this repository are those exports, **not** single scripts that can be run
top to bottom. Because of this, some parts are "missing" or need context:
 
- **Cell order matters.** Files are sequences of notebook cells. Many cells were run
  individually, some several times, and some only when needed (e.g. the individual
  `run_grid(...)` calls in the comparison file, one per model and feature set).
  Results of the grid search are cached in `.jsonl` files and finished runs are skipped
  when a cell is run again.
- **Discipline ranking part was run multiple times.** The discipline-rankings file was run
  once per configuration and seed, with constants edited by hand between runs (seed, loss
  variant with/without shrinkage, cutoff dates). Each run wrote its outputs with a run
  index in the file name (`..._6.pt`, `..._11.pt`, `..._15.pt`, `..._16.pt`, ...). The mapping
  of run index to configuration and seed used in the thesis is the `INDEX_MAP` dictionary in
  the comparison file. The repository contains the code for one run, not a loop over all runs.
- **Validation threshold vs. test cutoff.** Chronological splits are defined by date constants
  (`TEST_CUTOFF_DATE`, `DATA_CUTOFF_DATE`). For hyperparameter selection the cutoff was moved
  so that the held-out part was a *validation* period. In those runs, variables and graph
  sets named `test_*` hold the **validation** split, even though the code calls them "test".
  Final results were produced with the final cutoff dates and the corresponding
  graph checkpoint. The checkpoint (`race_prediction_graphs_cutoff_wf_N.pt`) used by a
  given run is the one selected by its index/tag.
- **Hard-coded Colab paths.** Files mount Google Drive and read/write
  `/content/drive/MyDrive/cycling_gnn/data_2025_26/`. Change `DATA_DIR` to run elsewhere.
- **Intermediate files are not included.** Graph checkpoints, snapshots, trained models
  and result files were stored on Drive and are not in the repository.
- **Some cells are exploratory** (printing, comparing runs, plots) and may depend on files
  produced by earlier sessions.
Because of the above, this repository is meant to document the method and the exact code
that produced the results, not to be a one-command reproduction.
## Repository contents
 
| File | Purpose |
|---|---|
| `excel_to_json_pipe.py` | Excel → `races_*.json`, `riders_*.json`, `results_*.json` |
| `race_pred_graphs.py` | Builds `race_prediction_graphs_cutoff_wf_N.pt` (per-race graphs, rider features, head-to-head and teammate edges) |
| `discipline_rankings.py` | Discipline-ranking GNN, walk-forward retraining, snapshots, form features, per-race contribution analysis |
| `race_prediction.py` | Models, grid search, selection, final multi-seed evaluation, statistics, training of the models used by the interface |
| `interface.py` | Gradio demo |
 
## Data flow
 
```
Excel files
   └─ [pipeline] ─► races_2025_26_1.json, riders_2025_26_1.json, results_2025_26_1.json
                       │
                       ├─ discipline rankings file ─► discipline_snapshots_full_retrain_N.pt
                       │                              race_id_to_discipline_scores_N.pt
                       │                              race_id_to_form_features_N.pt
                       │                              discipline_model_for_analysis_*.pt / _data_*.pt
                       │                              rider_race_contributions.pt
                       │                              final_model_discipline_scores.pt
                       │
                       └─ [graph builder] (uses form features, discipline scores)
                              └─► race_prediction_graphs_cutoff_wf_N.pt
                                    └─ comparison file ─► grid/*.jsonl, grid/selected_configs.json,
                                                          final_2/*.json (curves, stats),
                                                          interface_xgb.json,
                                                          interface_listnet_2L.pt,
                                                          interface_top10_2L_win.pt
 
interface.py loads: graph checkpoint, race_id_to_discipline_scores_N.pt,
                    final_model_discipline_scores.pt, rider_race_contributions.pt,
                    interface_* models
```
## Author and license

Oleksandr Voznyi

Copyright (c) 2026 Oleksandr Voznyi. All rights reserved.
This code is published for transparency and review of the thesis work.
No permission is granted to copy, modify or redistribute it without the author's consent.
