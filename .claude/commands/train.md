# JSC Train Model

**Usage:** `/train <path-to-config.yaml>`

Read the YAML config file provided by the user. An example config with all available fields is at `configs/train_example.yaml`.

## Config Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `dataset_id` | yes | -- | Dataset ID (e.g. 4 for binary, 5 for multi-class) |
| `configuration` | no | `3d_fullres` | `3d_fullres` or `2d` |
| `fold` | no | `0` | 0-4 for individual fold, or `all` |
| `trainer` | no | `nnUNetCLSTrainerMTL` | `nnUNetCLSTrainerMTL` or `PretrainedMTL` |
| `plan` | no | `nnUNetPlans` | `nnUNetPlans`, `nnUNetResEncUNetMPlans`, `nnUNetResEncUNetLPlans`, `nnUNetResEncUNetXLPlans` |
| `nnunet_raw` | yes | -- | Path to `nnUNet_raw` directory |
| `nnunet_preprocessed` | yes | -- | Path to `nnUNet_preprocessed` directory |
| `nnunet_results` | yes | -- | Path to `nnUNet_results` directory |
| `device` | no | `cuda` | `cuda` or `cpu` |
| `num_gpus` | no | `1` | Number of GPUs for DDP training |
| `continue_training` | no | `false` | Resume from latest checkpoint |
| `pretrained_weights` | no | `""` | Path to pretrained checkpoint (for `PretrainedMTL`) |

## Steps

### 1. Read and parse the config file

Read the YAML file at the path the user provided. If the file does not exist or cannot be parsed, report the error and stop.

### 2. Validate (BEFORE training)

1. Check preprocessed dataset exists: `{nnunet_preprocessed}/Dataset{dataset_id:03d}_*/`
2. Verify plans file exists: `{nnunet_preprocessed}/Dataset{dataset_id:03d}_*/{plan}.json`
3. Verify `cls_data.csv` exists with columns `identifier`, `label`. Report label distribution.
4. Verify `splits_final.json` exists. If fold is 0-4, check fold exists. Report train/val sizes.
5. Check configuration subfolder has `.b2nd` files: `{nnunet_preprocessed}/Dataset{dataset_id:03d}_*/{plan}_{configuration}/`
6. Verify trainer class: `nnUNetCLSTrainerMTL` (multi-task) or `PretrainedMTL` (requires `pretrained_weights`)

If any check fails, report the issue and stop.

### 3. Run training

```bash
export nnUNet_raw="{nnunet_raw}"
export nnUNet_preprocessed="{nnunet_preprocessed}"
export nnUNet_results="{nnunet_results}"

# Single GPU
nnUNetv2_train {dataset_id} {configuration} {fold} -tr {trainer} -p {plan}

# Multi-GPU (if num_gpus > 1)
torchrun --nproc_per_node={num_gpus} $(which nnUNetv2_train) {dataset_id} {configuration} {fold} -tr {trainer} -p {plan}
```

Add `--c` flag if `continue_training: true`. Use the `.venv/bin/` executables in the project root.

### 4. Post-training check

1. Verify output folder: `{nnunet_results}/Dataset{dataset_id:03d}_*/{trainer}__{plan}__{configuration}/fold_{fold}/`
2. Check for `checkpoint_best.pth` and `checkpoint_final.pth`
