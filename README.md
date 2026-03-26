# JSC

JSC is a Python-based project that extends the nnUNet framework with joint segmentation and classification capabilities. It combines the power of nnUNet's segmentation architecture with binary / multi-class classification, making it ideal for medical imaging tasks that require both pixel-level segmentation and image-level classification predictions.

## Features

- **Joint Architecture**: Simultaneous segmentation and classification in a single model
- **Stratified Data Splitting**: Advanced data splitting with demographic stratification (age, gender)
- **Multi-Modal Input**: Support for multi-channel medical images (CT, PET, MRI)
- **Flexible Classification**: Support for both binary and multi-class classification tasks
- **Comprehensive Inference**: Batch processing with sliding window prediction and test-time augmentation

## Installation

To set up JSC, first install the required dependencies:

```bash
pip install wandb
pip install torchmetrics
pip install -e .
```

## Configure nnUNet Paths
Before using JSC, you need to configure the nnUNet environment paths. Modify the paths in nnunetv2/paths.py to point to your desired directories:
```python
# Edit nnunetv2/paths.py
nnUNet_raw = "/path/to/your/nnUNet_raw"
nnUNet_preprocessed = "/path/to/your/nnUNet_preprocessed" 
nnUNet_results = "/path/to/your/nnUNet_results"
```
## Usage

### 1. Prepare Data

## Preprocessing with nnUNet

JSC relies on **nnUNet’s preprocessing pipeline** to standardize image spacing, intensity normalization, and patch extraction. Preprocessing must be completed before training or inference.  

### 1. Required Data Structure  

Your dataset must follow the **nnUNet folder convention**:  

```text
nnUNet_raw/
└── Dataset<ID>_<NAME>/
    ├── dataset.json                # Dataset configuration file
    ├── imagesTr/                   # Training images
    │   ├── PatientID_0000.nii.gz   # Modality 1
    │   ├── PatientID_0001.nii.gz   # Modality 2
    │   └── ...
    ├── labelsTr/                   # Training labels
    │   ├── PatientID.nii.gz        # Matches PatientID (no _0000)
    │   └── ...
    ├── imagesTs/                   # Test images (Inference)
    │   ├── TestID_0000.nii.gz
    │   └── ...
    └── labelsTs/                   # Test labels (Optional/Evaluation)
        ├── TestID.nii.gz           # Matches TestID (no _0000)
        └── ...
```


**Notes:**  
- Each modality is indexed as `_0000`, `_0001`, etc.  
- Segmentation labels must have the same base name as the training images (without modality suffix).  
- `dataset.json` defines modalities, labels, and dataset splits.  

---

### 2. Default Preprocessing  

Run the standard nnUNet preprocessing:  

```bash
nnUNetv2_plan_and_preprocess -d <DATASET_ID> -c 3d_fullres --verify_dataset_integrity

```

For Res Encoder
```bash
nnUNetv2_plan_experiment -d <DATASET_ID> -pl nnUNetPlannerResEncM #nnUNetPlannerResEncL / nnUNetPlannerResEncXL
```

Use `generate_cls_data.py` to create stratified train/validation/test splits from your clinical dataset:

```bash
python generate_cls_data.py \
    --input_path /path/to/clinical_data.csv \
    --output_path /path/to/output/folder \
    --identifier_column PatientID \
    --label_column diagnosis
```

**Arguments:**
- `--input_path, -i`: Path to CSV/Excel file containing clinical and imaging information
- `--output_path, -o`: Directory to save classification data and splits
- `--identifier_column, -id`: Column name for patient identifiers (default: 'patient_id')
- `--label_column, -label`: Column name for classification labels (default: 'label')

**Required CSV columns:**
- Patient identifiers (e.g., 'PatientID')
- Classification labels 
- `Age_at_StudyDate`: For age-based stratification
- `Gender`: For gender-based stratification

**Outputs:**
- `cls_data.csv`: Classification dataset
- `test_data.csv`: Held-out test set (20% of data)
- `splits_final.json`: 5-fold cross-validation splits with stratification
- Automatic filtering of cases without segmentation data
**Notes:**
make sure `cls_data.csv` and `splits_final.json` are under nnUNet_preprocessed

### 2. Training

```bash
nnUNetv2_train 161 3d_fullres 0 -tr <TrainerName>
# Use `nnUNetCLSTrainerMTL` for multi-task learning
# Use `PretrainedMTL` for two-stage warm-up and pretrained fine-tuning

nnUNetv2_train 714 3d_fullres all -p nnUNetResEncUNetMPlans

```
**Notes:**


### 3. Inference

Run joint segmentation and classification inference on NIfTI images:

```bash
python segcls_ensemble_infer.py \
    --input_path /path/to/input/images/ \
    --output_path /path/to/output/ \
    --model_path /path/to/trained/model \
    --fold 0 \ 
    --checkpoint checkpoint_best.pth \
    --device cuda \
    --cls_mode mean
```
For 5-fold ensemble fold should set as (0,1,2,3,4)

**Arguments:**
- `--input_path, -i`: Directory containing input NIfTI images (expects `*_000X.nii.gz` naming convention)
- `--output_path, -o`: Directory to save segmentation masks and classification results
- `--model_path`: Path to trained nnUNet model directory
- `--fold`: Fold number or 'all' for ensemble prediction (default: 'all')
- `--checkpoint`: Checkpoint filename (default: 'checkpoint_best.pth')
- `--use_softmax`: Apply softmax to segmentation output (default: False)
- `--device`: Computing device ('cuda' or 'cpu', default: 'cuda')
- `--cls_mode`: Classification aggregation mode ('mean' or 'weighted', default: 'mean')

**Input Format:**
Images should follow nnUNet naming convention:
- `PatientID_0000.nii.gz` (first modality)
- `PatientID_0001.nii.gz` (second modality)
- etc.

### **Outputs**

- `{PatientID}.nii.gz`: Segmentation mask for each case  
- `results.csv`: Classification predictions for all samples

The **`results.csv`** file contains the model’s classification outputs:

- **Binary classification:**  
  The `probs` field contains a **single probability value** representing the predicted likelihood of the positive class.

- **Multi-class classification:**  
  The `probs` field contains a **list of probabilities**, one for each class (e.g., `[p0, p1, p2, p3]`).

**Columns:**
- `identifier`: Unique sample ID  
- `probs`: Predicted probability (binary) or probability vector (multi-class)


### 4. Key Features

**Stratified Cross-Validation:**
- Creates balanced splits based on age quartiles, gender, and target labels
- Ensures representative distribution across all folds
- 80/20 train-test split with 5-fold cross-validation on training data

**Advanced Inference:**
- Sliding window prediction with Gaussian weighting
- Test-time augmentation with mirroring
- Multi-fold model ensembling
- Memory-efficient processing for large images
- Automatic batch processing of multiple cases

**Classification Modes:**
- `mean`: Average classification scores across all patches
- `weighted`: Weight classification by segmentation confidence

## Model Architecture

The framework extends nnUNet with:
- Shared encoder for both segmentation and classification
- Dual output heads (segmentation + classification)
- Feature aggregation from the final encoder stage
- Support for both binary and multi-class classification

## Evaluation

Use `eval_metrics.py` to compute segmentation and classification metrics after inference:

```bash
python eval_metrics.py \
    --pred_seg_path /path/to/predicted/segmentations \
    --gt_seg_path /path/to/ground_truth/segmentations \
    --pred_cls_csv /path/to/fold0_results.csv \
    --gt_cls_csv /path/to/ground_truth_cls.csv \
    --num_seg_classes 2 \
    --num_cls_classes 2   # 2 for binary, 3+ for multi-class
```

**Segmentation Metrics:**
- DSC (Dice Similarity Coefficient): mean +/- std, per-class
- NSD (Normalized Surface Distance, tau=1.0mm): mean +/- std, per-class

**Classification Metrics (Binary):**
- Accuracy, AUC, AUPRC, Sensitivity, Specificity, Precision, Recall, F1

**Classification Metrics (Multi-class):**
- Accuracy, Balanced Accuracy, Weighted AUC, Weighted AUPRC, Weighted F1, Weighted Precision, Weighted Recall
- Per-class: Precision, Recall, F1, AUC, AUPRC
- Confusion matrix

## Claude Code Skills

JSC provides three Claude Code slash-command skills (`/preprocess`, `/train`, `/eval`) for interactive, guided workflows. Each skill follows a **config-driven** approach: Claude presents a YAML config template with sensible defaults, you customize the values for your experiment, and Claude validates everything before executing.

### Prerequisites

- [Claude Code](https://claude.com/claude-code) CLI installed
- Working directory set to the JSC project root

### How It Works

Each skill takes the **path to a YAML config file** as its argument:

```
/preprocess configs/my_preprocess.yaml
/train configs/my_train.yaml
/eval configs/my_eval.yaml
```

1. **Copy an example config** from the `configs/` directory
2. **Edit the YAML** to match your experiment (dataset ID, paths, trainer, etc.)
3. **Run the skill** with the path to your config
4. Claude reads the config, validates all prerequisites, executes the pipeline step, and checks the output

### Config Files

Example configs are provided in the `configs/` directory. Copy one and edit it for your experiment:

```bash
# Copy and customize for your experiment
cp configs/preprocess_example.yaml configs/my_preprocess.yaml
cp configs/train_example.yaml configs/my_train.yaml
cp configs/eval_example.yaml configs/my_eval.yaml
```

#### `configs/preprocess_example.yaml`

```yaml
dataset_id: 3                          # Dataset ID number
dataset_name: "T1c_crop"               # Must match folder: Dataset{ID}_{name}
nnunet_raw: "/path/to/nnUNet_raw"
nnunet_preprocessed: "/path/to/nnUNet_preprocessed"
nnunet_results: "/path/to/nnUNet_results"
configuration: "3d_fullres"            # 3d_fullres | 2d
planner: "nnUNetPlanner"               # nnUNetPlanner | nnUNetPlannerResEncM/L/XL
verify_integrity: true
cls_data:                              # optional, skip if cls_data.csv exists
  input_csv: ""
  identifier_column: "patient_id"
  label_column: "label"
  age_column: null
  gender_column: null
```

#### `configs/train_example.yaml`

```yaml
dataset_id: 4                          # e.g. 4 for binary, 5 for multi-class
configuration: "3d_fullres"            # 3d_fullres | 2d
fold: 0                                # 0-4 or "all"
trainer: "nnUNetCLSTrainerMTL"         # nnUNetCLSTrainerMTL | PretrainedMTL
plan: "nnUNetPlans"                    # nnUNetPlans | nnUNetResEncUNetM/L/XLPlans
nnunet_raw: "/path/to/nnUNet_raw"
nnunet_preprocessed: "/path/to/nnUNet_preprocessed"
nnunet_results: "/path/to/nnUNet_results"
device: "cuda"
num_gpus: 1
continue_training: false
pretrained_weights: ""                 # required for PretrainedMTL
```

#### `configs/eval_example.yaml`

```yaml
model_path: ""                         # path to trained model dir
input_path: ""                         # path to test images ({ID}_0000.nii.gz)
output_path: ""                        # path to save predictions
ground_truth_seg_path: ""              # path to GT segmentation labels
ground_truth_cls_csv: ""               # path to GT classification CSV
fold: "0"                              # "0" or "0,1,2,3,4" or "all"
checkpoint: "checkpoint_best.pth"
device: "cuda"
cls_mode: "mean"                       # mean | weighted
use_softmax: false
num_seg_classes: 2                     # including background
num_cls_classes: 2                     # 2=binary, >2=multi-class
```

### Example Workflow

```bash
cd /path/to/JSC
claude

# Step 1: Preprocess a new dataset
> /preprocess configs/my_preprocess.yaml

# Step 2: Train a model
> /train configs/my_train.yaml

# Step 3: Evaluate results
> /eval configs/my_eval.yaml
```

### Available Trainers

| Trainer | Description |
|---------|-------------|
| `nnUNetCLSTrainerMTL` | Joint segmentation-classification multi-task learning |
| `PretrainedMTL` | Two-stage warm-up with pretrained encoder fine-tuning |

### Available Plans

| Plan | Description |
|------|-------------|
| `nnUNetPlans` | Default nnUNet planner (PlainConvUNet) |
| `nnUNetResEncUNetMPlans` | Residual encoder, medium |
| `nnUNetResEncUNetLPlans` | Residual encoder, large |
| `nnUNetResEncUNetXLPlans` | Residual encoder, extra-large |

### Test Datasets

| Dataset | Path | Type | Classes |
|---------|------|------|---------|
| Dataset003_T1c_crop | `nnUNet_raw/Dataset003_T1c_crop` | Raw (for preprocessing test) | -- |
| Dataset004_T1c_crop_2class | `nnUNet_preprocessed/Dataset004_T1c_crop_2class` | Binary classification | 0, 1 |
| Dataset005_T1c_crop_3class | `nnUNet_preprocessed/Dataset005_T1c_crop_3class` | Multi-class classification | 0, 1, 2 |

## License

This project is licensed under the [Apache License 2.0](https://github.com/ChingYuanYu/nnunetcls/blob/main/LICENSE).

## Citation

If you use JSC in your research, please cite the original nnUNet paper and this extension.

---

**Note:** Ensure your input data follows the nnUNet preprocessing requirements and naming conventions for optimal performance.