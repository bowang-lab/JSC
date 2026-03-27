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

### 5. Classification data

Check if `cls_data.csv` already exists in the raw dataset folder (`{nnunet_raw}/Dataset{dataset_id:03d}_{dataset_name}/cls_data.csv`).

- **If it exists:** Copy `cls_data.csv` from the raw dataset folder to the preprocessed dataset folder (`{nnunet_preprocessed}/Dataset{dataset_id:03d}_{dataset_name}/cls_data.csv`). Also copy `splits_final.json` and `test_data.csv` if they exist alongside it in the raw folder.

- **If it does NOT exist:** Check whether `cls_data.input_csv` is provided and non-empty in the config. If so, run `generate_cls_data.py` with the configured `input_csv`, `identifier_column`, `label_column`, and optional `age_column`/`gender_column` to generate `cls_data.csv`, `splits_final.json`, and `test_data.csv` under the preprocessed folder. If `cls_data.input_csv` is also missing or empty, report that no classification data source was found and skip this step.

In either case, verify that `cls_data.csv` and `splits_final.json` exist under the preprocessed folder after this step completes.
