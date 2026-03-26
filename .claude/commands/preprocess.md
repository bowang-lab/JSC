# JSC Preprocess Dataset

**Usage:** `/preprocess <path-to-config.yaml>`

Read the YAML config file provided by the user. An example config with all available fields is at `configs/preprocess_example.yaml`.

## Config Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `dataset_id` | yes | -- | Dataset ID number (e.g. 3) |
| `dataset_name` | yes | -- | Must match folder: `Dataset{ID}_{name}` |
| `nnunet_raw` | yes | -- | Path to `nnUNet_raw` directory |
| `nnunet_preprocessed` | yes | -- | Path to `nnUNet_preprocessed` directory |
| `nnunet_results` | yes | -- | Path to `nnUNet_results` directory |
| `configuration` | no | `3d_fullres` | `3d_fullres` or `2d` |
| `planner` | no | `nnUNetPlanner` | `nnUNetPlanner`, `nnUNetPlannerResEncM`, `nnUNetPlannerResEncL`, `nnUNetPlannerResEncXL` |
| `verify_integrity` | no | `true` | Run `--verify_dataset_integrity` |
| `cls_data.input_csv` | no | `""` | Path to clinical CSV/Excel for generating `cls_data.csv` |
| `cls_data.identifier_column` | no | `patient_id` | Column name for patient IDs |
| `cls_data.label_column` | no | `label` | Column name for classification labels |
| `cls_data.age_column` | no | `null` | Column for age stratification |
| `cls_data.gender_column` | no | `null` | Column for gender stratification |

## Steps

### 1. Read and parse the config file

Read the YAML file at the path the user provided. If the file does not exist or cannot be parsed, report the error and stop.

### 2. Validate raw dataset (BEFORE preprocessing)

1. Check raw dataset folder exists: `{nnunet_raw}/Dataset{dataset_id:03d}_{dataset_name}/`
2. Verify `dataset.json` exists and contains: `channel_names`, `labels`, `numTraining`, `file_ending`
3. Verify `imagesTr/` and `labelsTr/` exist and are non-empty
4. Cross-check file counts: unique patient IDs in `imagesTr/` (strip `_000X` suffix) and file count in `labelsTr/` must both match `numTraining`
5. Verify naming convention: images `{ID}_0000.nii.gz`, labels `{ID}.nii.gz`

If any check fails, report the issue and stop.

### 3. Run preprocessing

```bash
export nnUNet_raw="{nnunet_raw}"
export nnUNet_preprocessed="{nnunet_preprocessed}"
export nnUNet_results="{nnunet_results}"

nnUNetv2_plan_and_preprocess -d {dataset_id} -c {configuration} --verify_dataset_integrity

# If planner != nnUNetPlanner:
nnUNetv2_plan_experiment -d {dataset_id} -pl {planner}
```

Use the `.venv/bin/` executables in the project root.

### 4. Post-preprocessing validation

1. Preprocessed folder exists: `{nnunet_preprocessed}/Dataset{dataset_id:03d}_{dataset_name}/`
2. `nnUNetPlans.json` (or planner-specific plans JSON) was created
3. Configuration subfolder (e.g. `nnUNetPlans_3d_fullres/`) contains `.b2nd`, `.pkl`, `_seg.b2nd` files

### 5. Classification data (optional)

If `cls_data.input_csv` is provided and non-empty, run `generate_cls_data.py` and verify `cls_data.csv` and `splits_final.json` are placed under the preprocessed folder.
