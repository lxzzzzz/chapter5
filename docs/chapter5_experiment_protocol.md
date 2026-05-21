# Chapter 5 Experiment Protocol

This protocol fixes the data path for Chapter 5 tables: every tracker must consume the same detection cache exported from the best Chapter 4 detector. Ablations are run only on `xian`; comparison tables are run on `xian`, `v2x_seq`, and `v2x_real`.

## Unified Detection Input

Use `configs/chapter5_experiments.yaml` as the single experiment matrix. Fill in the `unified_detector.detector_ckpt` entries before exporting caches.

Generate caches:

```bash
python tools/run_chapter5_experiments.py --run_cache --dry_run
python tools/run_chapter5_experiments.py --run_cache
```

The cache threshold is intentionally low (`0.05`) so table-level score thresholds and AMOTA sweeps are evaluated from the same detections.

## Implemented Table Rows

The following rows are locally runnable:

- `ab3dmot`: classic 3D IoU + Hungarian baseline.
- `stage1`: high-confidence 3D association only.
- `stage1_stage2`: high-confidence 3D association followed by low-confidence center-distance recovery.
- `full_5_4`: executable 5.4 tracker row using fused 3D IoU/center association plus low-score recovery from the same unified detection cache.
- `centerpoint_track`, `simpletrack`, and `eagermot`: unified-cache adaptations for the comparison table.
- `baseline_motion`, `baseline_nekfm`, `baseline_nekfm_tlom`, `fmca_single_stage`, `fmca_two_stage`, `fmca_full`, and `full_5_5`: executable 5.5 ablation rows using the online behavior described in the chapter.
- `tlom_tau_030` through `tlom_tau_070`: lifecycle-threshold sweep rows.

Run them:

```bash
python tools/run_chapter5_experiments.py --run_tracking --tables table_5_4_association_ablation --dry_run
python tools/run_chapter5_experiments.py --run_tracking --tables table_5_4_association_ablation
```

Collect table files:

```bash
python tools/run_chapter5_experiments.py --collect --tables table_5_4_association_ablation
```

Outputs are written under `outputs/chapter5/runs/` and summarized under `outputs/chapter5/tables/`.
Each runner writes `runtime_summary.json`; `table_5_runtime` collects tracking FPS, milliseconds per frame, frame count, and tracking time from that file.

## External Rows

The current matrix no longer leaves any paper-table row as `external`. If a later replacement implementation writes a `tracking_results.json`, evaluate it with:

```bash
python tools/evaluate_ab3dmot_json.py \
  --cfg_file <dataset_cfg> \
  --result_json <tracking_results.json> \
  --split val \
  --output <method_output_dir>/ab3dmot_val_metrics_ab3dmot.json \
  --iou_threshold 0.5 \
  --recall_points 40
```

Then rerun collection to update the CSV and Markdown tables.

## Paper Table Mapping

- Table 5.4 association ablation: `table_5_4_association_ablation`, dataset `xian`.
- Table 5.5 core ablation: `table_5_5_core_ablation`, dataset `xian`.
- FMCA ablation: `table_5_5_fmca_ablation`, dataset `xian`.
- TLOM threshold sensitivity: `table_5_5_tlom_threshold`, dataset `xian`.
- Three-dataset comparison: `table_5_comparison`, datasets `xian`, `v2x_seq`, `v2x_real`.
- Runtime table: `table_5_runtime`, dataset `xian`.

Use the same metric columns for all accuracy tables: `sAMOTA`, `AMOTA`, `AMOTP`, `MOTA`, `MOTP`, `IDS`, `FRAG`, `FP`, and `FN`.
