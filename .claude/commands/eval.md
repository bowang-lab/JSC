# JSC Evaluate Model

**Usage:** `/eval <path-to-config.yaml>`

Read the YAML config file provided by the user. An example config with all available fields is at `configs/eval_example.yaml`.

## Config Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `model_path` | yes | -- | Path to trained model dir (e.g. `.../PretrainedMTL__nnUNetResEncUNetLPlans__3d_fullres`) |
| `input_path` | yes | -- | Path to test images (`{ID}_0000.nii.gz`) |
| `output_path` | yes | -- | Path to save predictions |
| `ground_truth_seg_path` | no | `""` | Path to GT segmentation labels (`.nii.gz`) |
| `ground_truth_cls_csv` | no | `""` | Path to GT classification CSV (`identifier`, `label` columns) |
| `fold` | no | `"0"` | Comma-separated folds (e.g. `"0,1,2,3,4"`) or `"all"` |
| `checkpoint` | no | `checkpoint_best.pth` | `checkpoint_best.pth` or `checkpoint_final.pth` |
| `device` | no | `cuda` | `cuda` or `cpu` |
| `cls_mode` | no | `mean` | `mean` or `weighted` |
| `use_softmax` | no | `false` | Apply softmax to segmentation output |
| `num_seg_classes` | no | `2` | Number of segmentation classes (including background) |
| `num_cls_classes` | no | `2` | 2 for binary, >2 for multi-class |

## Steps

### 1. Read and parse the config file

Read the YAML file at the path the user provided. If the file does not exist or cannot be parsed, report the error and stop.

### 2. Validate (BEFORE inference)

1. Check `model_path` exists and contains: `plans.json`, `dataset.json`, `fold_X/` dirs, `fold_X/{checkpoint}` files
2. Check `input_path` exists and contains `*_0000.nii.gz` files
3. If `ground_truth_seg_path` provided, check it exists and contains `.nii.gz` files
4. If `ground_truth_cls_csv` provided, check it exists and has columns `identifier`, `label`

If any check fails, report the issue and stop.

### 3. Run inference

```bash
python /mnt/pool/datasets/CY/JSC/segcls_ensemble_infer.py \
    --input_path {input_path} \
    --output_path {output_path} \
    --model_path {model_path} \
    --fold {fold} \
    --checkpoint {checkpoint} \
    --device {device} \
    --cls_mode {cls_mode}
```

Add `--use_softmax` if `use_softmax: true`. Use the `.venv/bin/python` in the project root.

### 4. Run evaluation

If ground truth paths are provided, run:

```bash
python /mnt/pool/datasets/CY/JSC/eval_metrics.py \
    --pred_seg_path {output_path} \
    --gt_seg_path {ground_truth_seg_path} \
    --pred_cls_csv {output_path}/fold{fold}_results.csv \
    --gt_cls_csv {ground_truth_cls_csv} \
    --num_seg_classes {num_seg_classes} \
    --num_cls_classes {num_cls_classes}
```

### Metrics computed

**Segmentation:** DSC and NSD (mean +/- std), per-class breakdown

**Classification (binary, `num_cls_classes=2`):** Accuracy, AUC, AUPRC, Sensitivity, Specificity, Precision, Recall, F1, confusion matrix

**Classification (multi-class, `num_cls_classes>2`):** Accuracy, Balanced Accuracy, Weighted AUC, Weighted AUPRC, Weighted F1, Weighted Precision, Weighted Recall, per-class metrics, confusion matrix
