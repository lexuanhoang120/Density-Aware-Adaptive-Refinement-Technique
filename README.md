# DART: Density-Aware Adaptive Refinement Technique for GUI Grounding

DART is an evaluation framework for GUI element grounding with iterative
attention-guided refinement. A model first predicts a GUI click point on the
full image. If the confidence is low, the evaluator crops around the predicted
point and queries the model again for a fixed number of refinement steps.

The current top-level workflow evaluates three Qwen-style GUI grounding models
on ScreenSpot-Pro, UI-Vision, and OSWorld-G, then computes overall and grouped
accuracy from the saved JSONL outputs.

## Supported Models

| `--model_type` | HuggingFace path |
|---|---|
| `kv_ground` | `vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315` |
| `ui_venus` | `inclusionAI/UI-Venus-1.5-8B` |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` |

## Supported Datasets

| `--dataset` | Description |
|---|---|
| `screenspot_pro` | ScreenSpot-Pro benchmark, grouped into CAD, Creative, Dev, OS, Office, and Scientific |
| `ui_vision` | UI-Vision element grounding, with basic, functional, and spatial annotation splits |
| `osworld` | OSWorld-G grounding benchmark, with optional refusal prediction |
| `screenspot` | ScreenSpot loaded from HuggingFace Hub |
| `screenspot_v2` | ScreenSpot-v2 loaded from HuggingFace Hub |

## Repository Layout

```text
.
├── evaluate_multi_threading_all.py        # Main multi-GPU evaluator
├── mvp_kv_ground.py                       # KV-Ground inference and attention scoring
├── mvp_ui_venus.py                        # UI-Venus inference and attention scoring
├── mvp_qwen3vl.py                         # Qwen3-VL inference and attention scoring
├── compute_group_accuracy.py              # Single-file or folder metric computation
├── eval_all.py                            # Batch metric computation over result folders
├── run_eval.sh                            # Example 3-model x 3-dataset run script
├── exps/                                  # Current JSONL result files
│   ├── KV-Ground-8B-BaseGuiOwl1.5-0315/
│   ├── Qwen3-VL-8B-Instruct/
│   └── UI-Venus-1.5-8B/
├── saved/                                 # Archived scripts and notebooks
│   ├── evaluate_multi_threading_all_osworldg-rvlm.py
│   └── test.ipynb
├── qwen3_vl/                              # Modified Qwen3-VL implementation
├── qwen2_5_vl/                            # Modified Qwen2.5-VL implementation
├── MVP/                                   # Standalone MVP benchmark scripts/results
├── V2P/                                   # V2P and GUI-Actor comparison experiments
├── regionfocus/                           # RegionFocus baseline scripts
└── requirements.txt
```

For V2P-7B and GUI-Actor-7B comparison results, see
[`V2P/README.md`](V2P/README.md). For standalone MVP ScreenSpot-Pro scripts and
outputs, see [`MVP/README.md`](MVP/README.md) and [`MVP/sspro_rst/`](MVP/sspro_rst/).

## Installation

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Flash Attention 2 is required because the modified Qwen model code exposes
attention maps used by the refinement logic.

## Data Layout

Pass the root directory with `--data_root`. The evaluator expects these paths
under that root:

```text
{data_root}/ScreenSpot-Pro/annotations/all.json
{data_root}/ScreenSpot-Pro/images/
{data_root}/ui-vision/images/
{data_root}/ui-vision/annotations/element_grounding/
{data_root}/OSWorld-G/benchmark/OSWorld-G_refined.json
{data_root}/OSWorld-G/benchmark/classification_result.json
```

`classification_result.json` is only needed for OSWorld-G grouped reporting.

## Running Evaluation

### Single Run

```bash
python evaluate_multi_threading_all.py \
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

For OSWorld-G refusal evaluation, add `--refusal`.

### Full 3 x 3 Run

`run_eval.sh` contains commands for three models across ScreenSpot-Pro,
UI-Vision, and OSWorld-G. Update `CUDA_VISIBLE_DEVICES`, `--data_root`, and
`--output_dir` as needed before running:

```bash
bash run_eval.sh
```

The evaluator spawns one worker per visible GPU, capped at 10 workers.

### Key Evaluation Arguments

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | `microsoft/GUI-Actor-7B-Qwen2.5-VL` | HuggingFace model name or local path |
| `--model_type` | same as default path | `kv_ground`, `ui_venus`, `qwen3vl`, or `kv_ground_bbox` |
| `--dataset` | `ui_vision` | Dataset to evaluate |
| `--data_root` | `/media/vli-ws2/Research/Hoang/HOANG/` | Root containing dataset folders |
| `--output_dir` | `None` | Output directory for JSONL results |
| `--max_iterations` | `6` | Rows written per sample; one row per refinement step |
| `--threshold` | `1.0` | Generation-time early exit threshold on `score` |
| `--max_crop_ratio` | `0.7` | Initial crop window as a fraction of image size |
| `--min_crop_ratio` | `0.3` | Minimum crop window as a fraction of image size |
| `--theta` | `0.3` | Attention activation threshold |
| `--score_type` | `None` | Region scoring method, for example `nearest`, `cir_peak`, `rec_peak`, or `bbox` |
| `--radius` | `0` | Neighbourhood radius for attention aggregation |
| `--attn_layer` | `20` | Transformer layer index used for attention maps |
| `--max_samples` | `None` | Optional sample limit for debugging |
| `--refusal` | `False` | Enables OSWorld-G refusal output `[-1,-1]` |

## Result Files

Results are written under:

```text
{output_dir}/{model_name}/{dataset}/results-max_{max}-min_{min}-threshold_{threshold}-iter_{max_iterations}-agg_{agg}-theta_{theta}-r_{radius}_{score_type}-refusal_{bool}.jsonl
```

Each JSONL row is one iteration for one sample. Important fields include:

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
  "iter": true,
  "output_text": "(190, 300)",
  "text_point_source": "nearest_region"
}
```

`repetive_number` is the 1-based refinement step. The metric scripts assume
`--samples_per_case` matches the `--max_iterations` used when the JSONL file was
created.

## Computing Performance

### Single File or Folder

`compute_group_accuracy.py` reports overall accuracy and grouped accuracy. It
selects the first iteration where `text_point_score >= conf_threshold`; if no
iteration reaches the threshold, it uses `max_iter`.

ScreenSpot-Pro example:

```bash
python compute_group_accuracy.py \
    --result_file exps/KV-Ground-8B-BaseGuiOwl1.5-0315/screenspot_pro/results-max_0.7-min_0.3-threshold_1.0-iter_6-agg_sum-theta_0.3-r_1_nearest-refusal_False.jsonl \
    --dataset screenspot_pro \
    --conf_threshold 0.8 \
    --max_iter 5
```

Folder sweep example:

```bash
python compute_group_accuracy.py \
    --result_dir exps/KV-Ground-8B-BaseGuiOwl1.5-0315/screenspot_pro \
    --dataset screenspot_pro \
    --conf_threshold 0.8 \
    --max_iter 5 \
    --output_csv sspro_kv_ground.csv
```

OSWorld-G example:

```bash
python compute_group_accuracy.py \
    --result_file exps/Qwen3-VL-8B-Instruct/osworld/results-max_0.7-min_0.3-threshold_1.0-iter_6-agg_sum-theta_0.3-r_1_nearest-refusal_True.jsonl \
    --dataset osworld \
    --data_root /media/vli3/data \
    --conf_threshold 0.8 \
    --max_iter 5
```

In `compute_group_accuracy.py`, OSWorld-G refusal samples are excluded by
default. Use `--include_refusal` when you want refusal samples included in the
reported overall accuracy. `eval_all.py` infers this from the filename: files
with `refusal_True` are evaluated with refusal samples included, while
`refusal_False` files exclude them and do not stop early on refusal coordinates.

### Batch Metrics

Use `eval_all.py` to compute every result file under an experiment directory.
The current checked-in results are under `exps/`, so pass `--exps_dir exps`:

```bash
python eval_all.py \
    --exps_dir exps \
    --data_root /media/vli3/data \
    --conf 0.8 \
    --max_iter 0 3 5 \
    --output_csv all_results.csv
```

`eval_all.py` prints one table per dataset and saves the full per-group table to
CSV. Its path convention is:

```text
{exps_dir}/{model}/{dataset}/results-*.jsonl
```

### Metric Arguments

| Argument | Default | Description |
|---|---:|---|
| `--result_file` | required unless `--result_dir` is used | Single JSONL result file |
| `--result_dir` | required unless `--result_file` is used | Directory searched recursively for JSONL files |
| `--dataset` | required | `screenspot_pro`, `ui_vision`, or `osworld` |
| `--data_root` | `/media/vli3/data` | Root for OSWorld-G classification files |
| `--conf_threshold` | `0.8` | Analysis-time early stop: `text_point_score >= conf_threshold` |
| `--max_iter` | `5` | Maximum 0-based iteration index; `5` uses up to six rows |
| `--samples_per_case` | `6` | Rows per sample in the JSONL file |
| `--corr_col` | `correct_text` | Correctness column to evaluate |
| `--score_col` | `text_point_score` | Score column used for analysis-time early stopping |
| `--exclude_refusal` / `--include_refusal` | exclude | Whether OSWorld-G refusal samples are counted |
| `--output_csv` | `None` | Optional CSV output path |

## Current Summary Results

`all_results.csv` contains the latest batch summary for `conf=0.8`. Overall
accuracy at `max_iter=3`:

| Dataset | Model | Correct / Total | Accuracy |
|---|---|---:|---:|
| OSWorld-G | KV-Ground-8B | 419 / 564 | 74.29% |
| OSWorld-G | Qwen3-VL-8B | 410 / 564 | 72.70% |
| OSWorld-G | UI-Venus-1.5-8B | 428 / 564 | 75.89% |
| ScreenSpot-Pro | KV-Ground-8B | 1279 / 1581 | 80.90% |
| ScreenSpot-Pro | Qwen3-VL-8B | 1082 / 1581 | 68.44% |
| ScreenSpot-Pro | UI-Venus-1.5-8B | 1179 / 1581 | 74.57% |
| UI-Vision | KV-Ground-8B | 2655 / 5479 | 48.46% |
| UI-Vision | Qwen3-VL-8B | 1979 / 5479 | 36.12% |
| UI-Vision | UI-Venus-1.5-8B | 2920 / 5479 | 53.29% |

## Notes

- The current result files live in `exps/`. Some older scripts or comments may
  still mention `exps1`; pass `--exps_dir exps` when computing metrics from the
  current files.
- `saved/evaluate_multi_threading_all_osworldg-rvlm.py` and `saved/test.ipynb`
  are archived references, not the primary top-level workflow.
- `requirements.txt` is an environment snapshot and may include packages that
  are not needed for a minimal inference-only setup.
