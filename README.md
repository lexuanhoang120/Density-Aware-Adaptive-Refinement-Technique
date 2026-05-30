# DART: Density-Aware Adaptive Refinement Technique for GUI Grounding

For the private reason, I can not public the code.
---

## Supported Models

| `--model_type` | HuggingFace path |
|---|---|
| `kv_ground` | `vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315` |
| `ui_venus` | `inclusionAI/UI-Venus-1.5-8B` |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` |

## Supported Datasets

| `--dataset` | Description |
|---|---|
| `screenspot_pro` | ScreenSpot-Pro benchmark (1 581 samples) |
| `ui_vision` | UI-Vision element grounding — basic / functional / spatial splits |
| `osworld` | OSWorld-G grounding benchmark (with optional refusal prediction) |
| `screenspot` | ScreenSpot (loaded from HuggingFace Hub) |
| `screenspot_v2` | ScreenSpot-v2 (loaded from HuggingFace Hub) |

---

## Repository Layout

```
.
├── evaluate_multi_threading_all.py            # Multi-GPU evaluation (no refusal)
├── evaluate_multi_threading_all_osworldg.py   # Multi-GPU evaluation with refusal support
├── evaluate_multi_threading_all_osworldg-rvlm.py  # RVLM variant
├── mvp_kv_ground.py      # Inference logic — KV-Ground attention scoring
├── mvp_ui_venus.py       # Inference logic — UI-Venus
├── mvp_qwen3vl.py        # Inference logic — Qwen3-VL
├── qwen3_vl/             # Modified Qwen3-VL implementation (exposes attention maps)
├── qwen2_5_vl/           # Modified Qwen2.5-VL implementation
├── run_all.sh            # Primary run script (KV-Ground, ScreenSpot-Pro)
├── run_all2.sh           # Run script for UI-Venus and Qwen3-VL
├── exps1/                # Result files (JSONL), organised by model and dataset
│   ├── KV-Ground-8B-BaseGuiOwl1.5-0315/
│   │   ├── screenspot_pro/
│   │   ├── ui_vision/
│   │   └── osworld/
│   ├── Qwen3-VL-8B-Instruct/
│   └── UI-Venus-1.5-8B/
├── test.ipynb            # Performance analysis notebook
├── sspro_rst/            # ScreenSpot-Pro group-selection result JSONs
├── uivision_rst/         # UI-Vision group-selection result JSONs
├── MVP/                  # MVP ScreenSpot-Pro scripts and saved group-selection results
├── V2P/                  # V2P and GUI-Actor evaluation scripts, DART result files, and summary tables
├── regionfocus/          # RegionFocus baseline scripts
└── requirements.txt
```

For the V2P-7B and GUI-Actor-7B comparison results, see [`V2P/README.md`](V2P/README.md), especially section **3.4 Classification Evaluation Results**. That section reports overall accuracy and grouped breakdowns for ScreenSpot-Pro, UI-Vision, and OSWorld-G, including baseline (`max_iter=0`) versus DART (`max_iter=3`) comparisons. To reproduce the combined CSV directly, run `bash V2P/scripts/run_v2p_guiactor_all_benchmarks.sh`.

For MVP ScreenSpot-Pro results, see [`MVP/sspro_rst/`](MVP/sspro_rst/). To reproduce the three-model MVP run for Qwen3-VL-8B, UI-Venus-1.5-8B, and KV-Ground-8B with `MAX_INFERENCES=2` and `3`, run `bash MVP/eval_sspro_three_models.sh`.

---

## Installation

```bash
pip install -r requirements.txt
```

```bash
pip install flash-attn --no-build-isolation
```

---

## Running Evaluation

### Quick start (reproducing `run_all.sh`)

```bash
python evaluate_multi_threading_all_osworldg.py \
    --dataset screenspot_pro \
    --data_root /path/to/data \
    --model_type kv_ground \
    --model_path vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315 \
    --max_crop_ratio 0.7 \
    --min_crop_ratio 0.3 \
    --theta 0.3 \
    --score_type nearest \
    --radius 1 \
    --max_iterations 6 \
    --attn_layer 24 \
    --output_dir exps
```

For OSWorld-G with refusal prediction add `--refusal`.

### Multi-GPU

The script automatically detects all visible GPUs and spawns one worker process per GPU. Control which GPUs are used with `CUDA_VISIBLE_DEVICES`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash run_all.sh
```

### MVP ScreenSpot-Pro

Run the MVP ScreenSpot-Pro evaluation for Qwen3-VL-8B, UI-Venus-1.5-8B, and KV-Ground-8B:

```bash
bash MVP/eval_sspro_three_models.sh
```

The script runs each model with `MAX_INFERENCES=2` and `MAX_INFERENCES=3`. Result JSON files are saved in [`MVP/sspro_rst/`](MVP/sspro_rst/).

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | — | HuggingFace model name or local path |
| `--model_type` | — | `kv_ground`, `ui_venus`, or `qwen3vl` |
| `--dataset` | `ui_vision` | Dataset to evaluate |
| `--data_root` | — | Root directory containing dataset folders |
| `--output_dir` | — | Where to write JSONL results |
| `--max_iterations` | `6` | Maximum zoom-in steps per sample |
| `--max_crop_ratio` | `0.7` | Initial crop window (fraction of image) |
| `--min_crop_ratio` | `0.3` | Minimum crop window (fully zoomed in) |
| `--theta` | `0.3` | Attention activation threshold |
| `--score_type` | `None` | Region scoring method: `nearest`, `cir_peak`, `rec_peak`, `bbox` |
| `--radius` | `0` | Neighbourhood radius for attention aggregation |
| `--attn_layer` | `20` | Transformer layer index used for attention maps |
| `--max_samples` | `None` | Limit number of evaluated samples (for debugging) |
| `--refusal` | `False` | Output `[-1,-1]` for infeasible OSWorld-G tasks |

---

## Result Files

Results are written to JSONL files under:

```
exps1/{model_name}/{dataset}/results-max_{max}-min_{min}-threshold_{t}-iter_{n}-agg_{agg}-theta_{theta}-r_{r}_{score_type}-refusal_{bool}.jsonl
```

Each line corresponds to one iteration of one sample and contains:

```json
{
  "id": 42,
  "instruction": "Click the Save button",
  "bbox": [0.1, 0.2, 0.3, 0.4],
  "pred": [0.18, 0.31],
  "pred_text": [0.19, 0.30],
  "correct": true,
  "correct_text": true,
  "score": 0.85,
  "text_point_score": 0.82,
  "repetive_number": 2,
  "output_text": "..."
}
```

`repetive_number` is the 1-based iteration index. Each sample produces exactly `max_iterations` rows.

Additional V2P-7B and GUI-Actor-7B DART result files are stored under [`V2P/exps/cgar/`](V2P/exps/cgar/). Their summarized benchmark tables are documented in [`V2P/README.md`](V2P/README.md).

---

## Computing Performance

### Script: `compute_group_accuracy.py`

The script reads any result JSONL from `exps1/` and reports overall accuracy plus a per-group breakdown. It uses the same early-stopping logic as `test.ipynb`: stop at the first iteration where `text_point_score > conf_threshold`, or at `max_iter`, whichever comes first.

#### ScreenSpot-Pro

Groups embedded in result file: **CAD, Creative, Dev, OS, Office, Scientific**.

```bash
python compute_group_accuracy.py \
    --result_file exps1/KV-Ground-8B-BaseGuiOwl1.5-0315/screenspot_pro/results-max_0.7-min_0.3-threshold_1.0-iter_6-agg_sum-theta_0.3-r_1_nearest-refusal_False.jsonl \
    --dataset screenspot_pro \
    --conf_threshold 0.8 \
    --max_iter 5
```

To sweep every `.jsonl` in a model folder at once:

```bash
python compute_group_accuracy.py \
    --result_dir exps1/KV-Ground-8B-BaseGuiOwl1.5-0315/screenspot_pro \
    --dataset screenspot_pro \
    --conf_threshold 0.8 \
    --max_iter 5 \
    --output_csv sspro_kv_ground.csv
```

#### UI-Vision

Groups derived from annotation file path: **basic, functional, spatial**.

```bash
python compute_group_accuracy.py \
    --result_file exps1/UI-Venus-1.5-8B/ui_vision/results-max_0.7-min_0.3-threshold_1.0-iter_6-agg_sum-theta_0.3-r_1_nearest-refusal_False.jsonl \
    --dataset ui_vision \
    --conf_threshold 0.8 \
    --max_iter 5
```

#### OSWorld-G

Groups loaded from `OSWorld-G/benchmark/classification_result.json`: **element_recognition, fine_grained_manipulation, layout_understanding, text_matching**. Refusal samples are excluded by default.

```bash
python compute_group_accuracy.py \
    --result_file exps1/Qwen3-VL-8B-Instruct/osworld/results-max_0.7-min_0.3-threshold_1.0-iter_6-agg_sum-theta_0.3-r_1_nearest-refusal_True.jsonl \
    --dataset osworld \
    --data_root /media/vli3/data \
    --conf_threshold 0.8 \
    --max_iter 5
```

Add `--include_refusal` to include refusal samples in the count. If the classification JSON is not accessible the script falls back to using `box_type` as the group label.

#### All key arguments

| Argument | Default | Description |
|---|---|---|
| `--result_file` | — | Path to a single `.jsonl` result file |
| `--result_dir` | — | Directory; processes every `.jsonl` found recursively |
| `--dataset` | required | `screenspot_pro`, `ui_vision`, or `osworld` |
| `--data_root` | `/media/vli3/data` | Root for external annotation files (OSWorld) |
| `--conf_threshold` | `0.8` | Early-stop threshold on `text_point_score` |
| `--max_iter` | `3` | Max iteration index (0-based); `5` = use up to 6 steps |
| `--samples_per_case` | `6` | Rows per sample — must match `--max_iterations` used during eval |
| `--corr_col` | `correct_text` | Correctness column: `correct` or `correct_text` |
| `--score_col` | `text_point_score` | Score column that drives early stopping |
| `--exclude_refusal` / `--include_refusal` | exclude | Whether to skip refusal samples (OSWorld only) |
| `--output_csv` | `None` | Save per-group summary table to a CSV file |

---

### Notebook: `test.ipynb`

Open `test.ipynb` for interactive exploration. Key parameters:

```python
CONF_THRESHOLD  = 0.8   # stop early if confidence exceeds this
MAX_ITER        = 5     # maximum iteration index (0-based), i.e. use at most 6 steps
SAMPLES_PER_CASE = 6    # must match --max_iterations used during evaluation
corr            = 'correct_text'  # which correctness column to use
```

**Cell 2** — single-file accuracy.  
**Cell 3** — sweep over all `.jsonl` files across multiple thresholds and iteration budgets → `accuracy_results.csv`.  
**Cell 4** — per-group breakdown (ScreenSpot-Pro).

---

### Getting All Results at Once: `eval_all.py`

Scans every `.jsonl` under `exps/`, infers model and dataset from the path, and prints one table per dataset with models as rows and groups as columns. Supports multiple `--max_iter` values in one run.

```bash
# Compare single-shot (iter=0) vs iterative (iter=3)
python eval_all.py --max_iter 0 3 --conf 0.8

# Custom folder or output CSV
python eval_all.py --exps_dir exps --max_iter 0 3 --output_csv results.csv
```

| Argument | Default | Description |
|---|---|---|
| `--exps_dir` | `exps1` | Root folder to scan for `.jsonl` files |
| `--max_iter` | `3` | One or more iteration budgets, e.g. `--max_iter 0 3 5` |
| `--conf` | `0.8` | Early-stop confidence threshold |
| `--data_root` | `/media/vli3/data` | Path to OSWorld-G benchmark files |
| `--output_csv` | `all_results.csv` | Save full per-group table to CSV |

---

### Results in `exps/` (conf=0.8, score\_type=nearest)

Stopping conditions per iteration: confidence > 0.8 **or** model predicts `[-1,-1]`/`[0,0]` **or** budget exhausted.
OSWorld: files named `refusal_True` include all 564 samples (510 non-refusal + 54 refusal); files named `refusal_False` exclude the 54 refusal tasks (510 samples). All OSWorld results below used `refusal_True`.

#### ScreenSpot-Pro — 1 581 samples

| Model | Overall | CAD | Creative | Dev | OS | Office | Scientific |
|---|---|---|---|---|---|---|---|
| KV-Ground-8B | 72.99% | 69.35% | 67.74% | 77.93% | 65.82% | 86.96% | 70.87% |
| + DART | 80.90% ↑7.91 | 80.46% ↑11.11 | 75.66% ↑7.92 | 82.61% ↑4.68 | 74.49% ↑8.67 | 91.30% ↑4.34 | 81.89% ↑11.02 |
| Qwen3-VL-8B | 54.59% | 48.28% | 48.97% | 52.84% | 49.49% | 73.91% | 57.09% |
| + DART | 68.44% ↑13.85 | 66.67% ↑18.39 | 62.76% ↑13.79 | 64.55% ↑11.71 | 66.33% ↑16.84 | 85.65% ↑11.74 | 68.50% ↑11.41 |
| UI-Venus-1.5-8B | 66.79% | 61.69% | 56.60% | 67.89% | 67.86% | 84.78% | 67.32% |
| + DART | 74.57% ↑7.78 | 73.56% ↑11.87 | 67.16% ↑10.56 | 75.59% ↑7.70 | 69.90% ↑2.04 | 90.43% ↑5.65 | 73.62% ↑6.30 |

#### OSWorld-G — 564 samples (510 non-refusal + 54 refusal)

| Model | Overall | elem\_recog | fine\_manip | layout\_und | refusal | text\_match |
|---|---|---|---|---|---|---|
| KV-Ground-8B | 70.21% | 75.45% | 59.06% | 76.68% | 9.26% | 80.08% |
| + DART | 74.29% ↑4.08 | 79.70% ↑4.25 | 59.73% ↑0.67 | 81.82% ↑5.14 | 20.37% ↑11.11 | 78.93% ↓1.15 |
| Qwen3-VL-8B | 68.79% | 72.42% | 61.07% | 74.31% | 1.85% | 80.46% |
| + DART | 72.69% ↑3.90 | 80.61% ↑8.19 | 56.38% ↓4.69 | 83.79% ↑9.48 | 1.85% — | 78.54% ↓1.92 |
| UI-Venus-1.5-8B | 74.29% | 81.52% | 59.06% | 80.24% | 20.37% | 82.76% |
| + DART | 75.89% ↑1.60 | 82.73% ↑1.21 | 60.40% ↑1.34 | 82.61% ↑2.37 | 29.63% ↑9.26 | 80.84% ↓1.92 |

#### UI-Vision — 5 479 samples (basic, functional, spatial splits)

| Model | Basic | Functional | Spatial | Average |
|---|---|---|---|---|
| Qwen3-VL-8B | 32.7% | 34.1% | 14.7% | 26.8% |
| + DART | 41.8% ↑9.1 | 42.2% ↑8.1 | 25.3% ↑10.6 | 36.1% ↑9.3 |
| UI-Venus-1.5-8B | 56.0% | 50.5% | 30.9% | 45.4% |
| + DART | 59.7% ↑3.7 | 54.8% ↑4.3 | 46.0% ↑15.1 | 53.3% ↑7.9 |
| KV-Ground-8B | 49.2% | 45.4% | 26.4% | 39.9% |
| + DART | 56.2% ↑7.0 | 51.9% ↑6.5 | 38.2% ↑11.8 | 48.5% ↑8.6 |

---

### V2P And GUI-Actor Results

These results use `conf=0.8`, `correct_col=correct`, and `score_col=score`. Baseline is `max_iter=0`; DART is `max_iter=3`. The full grouped breakdowns and reproduction command are in [`V2P/README.md`](V2P/README.md).

| Model | Dataset | Baseline Acc | DART Acc | Gain | DART Correct / Total | DART Avg Iter |
|---|---|---|---|---|---|---|
| GUI-Actor-7B | ScreenSpot-Pro | 43.83% | 60.47% | +16.64% | 956 / 1581 | 2.7761 |
| GUI-Actor-7B | UI-Vision | 21.59% | 30.99% | +9.40% | 1698 / 5479 | 2.8266 |
| GUI-Actor-7B | OSWorld-G | 48.40% | 61.35% | +12.94% | 346 / 564 | 2.2855 |
| V2P-7B | ScreenSpot-Pro | 49.53% | 64.20% | +14.67% | 1015 / 1581 | 2.4491 |
| V2P-7B | UI-Vision | 23.93% | 33.86% | +9.93% | 1855 / 5479 | 2.6832 |
| V2P-7B | OSWorld-G | 51.95% | 62.77% | +10.82% | 354 / 564 | 1.9716 |

To reproduce this table:

```bash
bash V2P/scripts/run_v2p_guiactor_all_benchmarks.sh
```

The combined CSV is saved to:

```text
V2P/exps/classification_results/v2p_guiactor_all_benchmarks_threshold_0.8_maxiters_0_3.csv
```

---

## Comparison with Baseline Refinement Methods

All numbers on **ScreenSpot-Pro** (1 581 samples, `action_acc` = combined text + icon accuracy).

| Method | Model | Overall | CAD | Creative | Dev | OS | Office | Scientific |
|---|---|---|---|---|---|---|---|---|
| **DART (iter=0)** | KV-Ground-8B | 72.99% | 69.35% | 67.74% | 77.93% | 65.82% | 86.96% | 70.87% |
| RegionFocus | KV-Ground-8B | 73.5% | 71.3% | 65.4% | 73.9% | 70.4% | 84.8% | 78.3% |
| **DART (iter=3)** | KV-Ground-8B | **80.90%** | **80.46%** | **75.66%** | **82.61%** | **74.49%** | **91.30%** | **81.89%** |
| MVP (k=3) | KV-Ground-8B | 78.43% | 79.31% | 70.97% | 82.27% | 74.49% | 85.22% | 79.92% |
| | | | | | | | | |
| **DART (iter=0)** | Qwen3-VL-8B | 54.59% | 48.28% | 48.97% | 52.84% | 49.49% | 73.91% | 57.09% |
| RegionFocus | Qwen3-VL-8B | 66.2% | 66.3% | 61.0% | 64.5% | 58.7% | 80.4% | 67.7% |
| **DART (iter=3)** | Qwen3-VL-8B | **68.44%** | **66.67%** | **62.76%** | **64.55%** | **66.33%** | **85.65%** | **68.50%** |
| MVP (k=3) | Qwen3-VL-8B | 65.28% | 62.45% | 59.24% | 63.55% | 64.80% | 81.74% | 63.78% |
| | | | | | | | | |
| **DART (iter=0)** | UI-Venus-1.5-8B | 66.79% | 61.69% | 56.60% | 67.89% | 67.86% | 84.78% | 67.32% |
| RegionFocus | UI-Venus-1.5-8B | 67.7% | 64.8% | 57.2% | 69.9% | 67.3% | 83.0% | 68.9% |
| **DART (iter=3)** | UI-Venus-1.5-8B | **74.57%** | **73.56%** | **67.16%** | **75.59%** | **69.90%** | **90.43%** | **73.62%** |
| MVP (k=3) | UI-Venus-1.5-8B | 72.17% | 68.20% | 65.69% | 74.25% | 68.37% | 88.26% | 70.87% |

RegionFocus numbers are from `regionfocus/results/` using `action_acc` (the combined text + icon metric from `summarize_results.py`). DART numbers use `correct_text` (same definition: prediction falls within the target element's bounding box). MVP numbers use `final_accuracy` from the saved `k=3` JSON files in [`MVP/sspro_rst/`](MVP/sspro_rst/).

---

## Acknowledgments

We thank the authors of **V2P**, **GUI-Actor**, **RegionFocus**, and **MVP** for releasing their models, code, and resources, which serve as strong baselines and support this research.

[[V2P](https://github.com/inclusionAI/AWorld-RL/tree/main/V2P)] | [[GUI-Actor](https://github.com/microsoft/GUI-Actor)] | [[RegionFocus](https://github.com/tiangeluo/regionfocus)] | [[MVP](https://github.com/ZJUSCL/MVP)]
