"""
Evaluation script for JSC: computes segmentation (DSC, NSD) and classification metrics.
Supports both binary and multi-class classification.
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
)


def compute_dsc(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    intersection = np.sum(pred_mask & gt_mask)
    denom = np.sum(pred_mask) + np.sum(gt_mask)
    if denom == 0:
        return np.nan
    return 2.0 * intersection / denom


def compute_surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: tuple) -> tuple:
    from scipy.ndimage import binary_erosion
    struct = np.ones((3, 3, 3), dtype=bool)

    pred_border = pred & ~binary_erosion(pred, structure=struct)
    gt_border = gt & ~binary_erosion(gt, structure=struct)

    if np.sum(pred_border) == 0 and np.sum(gt_border) == 0:
        return np.array([0.0]), np.array([0.0])
    if np.sum(pred_border) == 0 or np.sum(gt_border) == 0:
        return np.array([np.inf]), np.array([np.inf])

    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    dist_pred_to_gt = dt_gt[pred_border]
    dist_gt_to_pred = dt_pred[gt_border]

    return dist_pred_to_gt, dist_gt_to_pred


def compute_nsd(pred: np.ndarray, gt: np.ndarray, label: int, spacing: tuple, tau: float = 1.0) -> float:
    pred_mask = (pred == label)
    gt_mask = (gt == label)

    if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
        return np.nan
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return 0.0

    dist_pred_to_gt, dist_gt_to_pred = compute_surface_distances(pred_mask, gt_mask, spacing)
    nsd = (np.sum(dist_pred_to_gt <= tau) + np.sum(dist_gt_to_pred <= tau)) / (len(dist_pred_to_gt) + len(dist_gt_to_pred))
    return nsd


def evaluate_segmentation(pred_seg_path: str, gt_seg_path: str, num_classes: int) -> pd.DataFrame:
    pred_files = sorted(glob.glob(os.path.join(pred_seg_path, "*.nii.gz")))
    gt_files_map = {}
    for f in glob.glob(os.path.join(gt_seg_path, "*.nii.gz")):
        basename = os.path.basename(f)
        gt_files_map[basename] = f

    rows = []
    fg_labels = list(range(1, num_classes))

    for pf in pred_files:
        case_id = os.path.basename(pf)
        if case_id not in gt_files_map:
            print(f"Warning: no ground truth for {case_id}, skipping")
            continue

        pred_sitk = sitk.ReadImage(pf)
        gt_sitk = sitk.ReadImage(gt_files_map[case_id])

        pred_arr = sitk.GetArrayFromImage(pred_sitk)
        gt_arr = sitk.GetArrayFromImage(gt_sitk)
        spacing = gt_sitk.GetSpacing()[::-1]  # zyx order

        for lbl in fg_labels:
            dsc = compute_dsc(pred_arr, gt_arr, lbl)
            nsd = compute_nsd(pred_arr, gt_arr, lbl, spacing, tau=1.0)
            rows.append({
                "case": case_id.replace(".nii.gz", ""),
                "label": lbl,
                "DSC": dsc,
                "NSD": nsd,
            })

    return pd.DataFrame(rows)


def evaluate_classification_binary(gt_labels, pred_probs):
    pred_labels = (np.array(pred_probs) >= 0.5).astype(int)
    gt = np.array(gt_labels)

    tn, fp, fn, tp = confusion_matrix(gt, pred_labels, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    results = {
        "Accuracy": accuracy_score(gt, pred_labels),
        "AUC": roc_auc_score(gt, pred_probs),
        "AUPRC": average_precision_score(gt, pred_probs),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(gt, pred_labels, zero_division=0),
        "Recall": recall_score(gt, pred_labels, zero_division=0),
        "F1": f1_score(gt, pred_labels, zero_division=0),
    }
    cm = confusion_matrix(gt, pred_labels, labels=[0, 1])
    return results, cm


def evaluate_classification_multiclass(gt_labels, pred_probs, num_classes):
    gt = np.array(gt_labels)
    probs = np.array(pred_probs)
    pred_labels = np.argmax(probs, axis=1)

    # One-hot for AUC / AUPRC
    gt_onehot = np.zeros((len(gt), num_classes))
    for i, g in enumerate(gt):
        gt_onehot[i, int(g)] = 1

    try:
        auc = roc_auc_score(gt_onehot, probs, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")

    try:
        auprc = average_precision_score(gt_onehot, probs, average="weighted")
    except ValueError:
        auprc = float("nan")

    results = {
        "Accuracy": accuracy_score(gt, pred_labels),
        "Balanced Accuracy": balanced_accuracy_score(gt, pred_labels),
        "Weighted AUC": auc,
        "Weighted AUPRC": auprc,
        "Weighted F1": f1_score(gt, pred_labels, average="weighted", zero_division=0),
        "Weighted Precision": precision_score(gt, pred_labels, average="weighted", zero_division=0),
        "Weighted Recall": recall_score(gt, pred_labels, average="weighted", zero_division=0),
    }

    # Per-class metrics
    per_class = {}
    for c in range(num_classes):
        c_mask = (gt == c)
        c_pred = (pred_labels == c)
        per_class[f"Class_{c}_Precision"] = precision_score(gt == c, pred_labels == c, zero_division=0)
        per_class[f"Class_{c}_Recall"] = recall_score(gt == c, pred_labels == c, zero_division=0)
        per_class[f"Class_{c}_F1"] = f1_score(gt == c, pred_labels == c, zero_division=0)
        if np.sum(gt_onehot[:, c]) > 0:
            try:
                per_class[f"Class_{c}_AUC"] = roc_auc_score(gt_onehot[:, c], probs[:, c])
            except ValueError:
                per_class[f"Class_{c}_AUC"] = float("nan")
            try:
                per_class[f"Class_{c}_AUPRC"] = average_precision_score(gt_onehot[:, c], probs[:, c])
            except ValueError:
                per_class[f"Class_{c}_AUPRC"] = float("nan")

    results.update(per_class)
    cm = confusion_matrix(gt, pred_labels)
    return results, cm


def parse_probs(probs_str, num_classes):
    if num_classes == 2:
        if isinstance(probs_str, (float, int)):
            return float(probs_str)
        s = str(probs_str).strip().strip("[]")
        vals = [float(x) for x in s.split(",")]
        if len(vals) == 1:
            return vals[0]
        return vals[1]
    else:
        if isinstance(probs_str, list):
            return probs_str
        s = str(probs_str).strip().strip("[]")
        return [float(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Evaluate JSC segmentation and classification")
    parser.add_argument("--pred_seg_path", type=str, default="", help="Path to predicted segmentation masks")
    parser.add_argument("--gt_seg_path", type=str, default="", help="Path to ground truth segmentation masks")
    parser.add_argument("--pred_cls_csv", type=str, default="", help="Path to prediction results CSV")
    parser.add_argument("--gt_cls_csv", type=str, default="", help="Path to ground truth classification CSV")
    parser.add_argument("--num_seg_classes", type=int, default=2, help="Number of segmentation classes (including background)")
    parser.add_argument("--num_cls_classes", type=int, default=2, help="Number of classification classes (2=binary)")
    parser.add_argument("--output_csv", type=str, default="", help="Optional: save metrics to CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("JSC Evaluation")
    print("=" * 60)

    # --- Segmentation ---
    if args.pred_seg_path and args.gt_seg_path:
        print("\n--- Segmentation Metrics ---")
        seg_df = evaluate_segmentation(args.pred_seg_path, args.gt_seg_path, args.num_seg_classes)

        if len(seg_df) > 0:
            for lbl in seg_df["label"].unique():
                lbl_df = seg_df[seg_df["label"] == lbl].dropna()
                dsc_mean = lbl_df["DSC"].mean()
                dsc_std = lbl_df["DSC"].std()
                nsd_mean = lbl_df["NSD"].mean()
                nsd_std = lbl_df["NSD"].std()
                print(f"  Label {lbl}: DSC = {dsc_mean:.4f} +/- {dsc_std:.4f} | NSD = {nsd_mean:.4f} +/- {nsd_std:.4f}")

            overall_dsc = seg_df.dropna()["DSC"]
            overall_nsd = seg_df.dropna()["NSD"]
            print(f"  Overall:  DSC = {overall_dsc.mean():.4f} +/- {overall_dsc.std():.4f} | NSD = {overall_nsd.mean():.4f} +/- {overall_nsd.std():.4f}")

            if args.output_csv:
                seg_df.to_csv(args.output_csv.replace(".csv", "_seg.csv"), index=False)
        else:
            print("  No matching cases found for segmentation evaluation.")
    else:
        print("\nSkipping segmentation evaluation (paths not provided).")

    # --- Classification ---
    if args.pred_cls_csv and args.gt_cls_csv:
        print("\n--- Classification Metrics ---")
        pred_df = pd.read_csv(args.pred_cls_csv)
        gt_df = pd.read_csv(args.gt_cls_csv)

        merged = pd.merge(pred_df, gt_df, on="identifier", suffixes=("_pred", "_gt"))
        if len(merged) == 0:
            print("  No matching cases found for classification evaluation.")
        else:
            gt_labels = merged["label"].values if "label" in merged.columns else merged["label_gt"].values
            pred_probs = [parse_probs(p, args.num_cls_classes) for p in merged["probs"].values]

            if args.num_cls_classes == 2:
                results, cm = evaluate_classification_binary(gt_labels, pred_probs)
                print(f"  Accuracy:    {results['Accuracy']:.4f}")
                print(f"  AUC:         {results['AUC']:.4f}")
                print(f"  AUPRC:       {results['AUPRC']:.4f}")
                print(f"  Sensitivity: {results['Sensitivity']:.4f}")
                print(f"  Specificity: {results['Specificity']:.4f}")
                print(f"  Precision:   {results['Precision']:.4f}")
                print(f"  Recall:      {results['Recall']:.4f}")
                print(f"  F1:          {results['F1']:.4f}")
            else:
                results, cm = evaluate_classification_multiclass(gt_labels, pred_probs, args.num_cls_classes)
                print(f"  Accuracy:          {results['Accuracy']:.4f}")
                print(f"  Balanced Accuracy: {results['Balanced Accuracy']:.4f}")
                print(f"  Weighted AUC:      {results['Weighted AUC']:.4f}")
                print(f"  Weighted AUPRC:    {results['Weighted AUPRC']:.4f}")
                print(f"  Weighted F1:       {results['Weighted F1']:.4f}")
                print(f"  Weighted Precision:{results['Weighted Precision']:.4f}")
                print(f"  Weighted Recall:   {results['Weighted Recall']:.4f}")
                print(f"\n  Per-class metrics:")
                for c in range(args.num_cls_classes):
                    prefix = f"Class_{c}"
                    parts = []
                    for k, v in results.items():
                        if k.startswith(prefix):
                            metric_name = k.replace(f"{prefix}_", "")
                            parts.append(f"{metric_name}={v:.4f}")
                    print(f"    Class {c}: {', '.join(parts)}")

            print(f"\n  Confusion Matrix:\n{cm}")

            if args.output_csv:
                cls_out = args.output_csv.replace(".csv", "_cls.csv")
                pd.DataFrame([results]).to_csv(cls_out, index=False)
                print(f"\n  Classification metrics saved to {cls_out}")
    else:
        print("\nSkipping classification evaluation (paths not provided).")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
