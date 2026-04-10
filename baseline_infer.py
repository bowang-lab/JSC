"""
Per-model inference entry points for the 6 JSC_challenge baselines.

For each baseline model in
    /mnt/pool/datasets/CY/JSC_challenge/baseline_models/
this file exposes a function that:
    - loads the fold_0 `checkpoint_best.pth`
    - runs segmentation + classification on one case
    - writes the segmentation mask (NIfTI) to disk
    - writes an mp4 video overlay of the first modality + segmentation
    - returns (seg_mask_path, overlay_video_path, cls_results_dict)

The cls_results_dict maps the human-readable class name (from
dataset.json["classification_labels"]) to its predicted probability.
"""

import os
import gc
import json
import shutil
import subprocess
import itertools
from typing import List, Sequence, Tuple, Union, Optional

import numpy as np
import torch
import cv2
import SimpleITK as sitk
from tqdm import tqdm

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager, determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.preprocessing.resampling.default_resampling import fast_resample_logit_to_shape
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice


# ─── Registry of baseline models ────────────────────────────────────────────────
BASELINE_ROOT = "/mnt/pool/datasets/CY/JSC_challenge/baseline_models"
RAW_ROOT = "/mnt/pool/datasets/CY/AutoMSC_raw"

MODEL_REGISTRY = {
    "Dataset002_BMLMPS_FLAIR": {
        "plans_folder": "nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres",
        "dataset_json_fallback": f"{RAW_ROOT}/Dataset002_BMLMPS_FLAIR/dataset.json",
    },
    "Dataset003_BMLMPS_T1CE": {
        "plans_folder": "nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres",
        "dataset_json_fallback": f"{RAW_ROOT}/Dataset003_BMLMPS_T1CE/dataset.json",
    },
    "Dataset004_BrainMets": {
        "plans_folder": "nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres",
        "dataset_json_fallback": f"{RAW_ROOT}/Dataset004_BrainMets/dataset.json",
    },
    "Dataset005_MU_Glioma_Post": {
        "plans_folder": "nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres",
        "dataset_json_fallback": f"{RAW_ROOT}/Dataset005_MU_Glioma_Post/dataset.json",
    },
    "Dataset006_JSC_UCSD_PTGB": {
        "plans_folder": "nnUNetCLSTrainerMTL__nnUNetResEncUNetMPlans__3d_fullres",
        "dataset_json_fallback": f"{RAW_ROOT}/Dataset006_JSC_UCSD_PTGB/dataset.json",
    },
    "Dataset007_PICAI": {
        "plans_folder": "nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres",
        "dataset_json_fallback": f"{RAW_ROOT}/Dataset007_PICAI/dataset.json",
    },
}


# ─── Single-fold predictor (adapted from fivefold_eval.py) ─────────────────────
def _logit_to_segment(predicted_logits):
    max_logit, max_class = torch.max(predicted_logits, dim=0)
    return torch.where(max_logit >= 0.5, max_class,
                       torch.tensor(0, device=predicted_logits.device))


def _convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits, plans_manager, configuration_manager,
        label_manager, properties_dict, use_softmax):
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]

    if properties_dict['shape_after_cropping_and_before_resampling'][0] < 600:
        predicted_logits = fast_resample_logit_to_shape(
            predicted_logits,
            properties_dict['shape_after_cropping_and_before_resampling'],
            current_spacing,
            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
        gc.collect()
        empty_cache(predicted_logits.device)
        if use_softmax:
            predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
            del predicted_logits
            segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
        else:
            segmentation = _logit_to_segment(predicted_logits)
    else:
        segmentation = fast_resample_logit_to_shape(
            predicted_logits,
            properties_dict['shape_after_cropping_and_before_resampling'],
            current_spacing,
            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])

    dtype = torch.uint8 if len(label_manager.foreground_labels) < 255 else torch.uint16
    seg_reverted = torch.zeros(properties_dict['shape_before_cropping'], dtype=dtype)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    seg_reverted[slicer] = segmentation
    del segmentation
    seg_reverted = seg_reverted.permute(plans_manager.transpose_backward)
    return seg_reverted.cpu()


class SingleFoldPredictor(nnUNetPredictor):
    """Loads a single fold of an MTL (seg+cls) nnUNet model."""

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             fold: int, checkpoint_name: str):
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        ckpt_path = join(model_training_output_dir, f'fold_{fold}', checkpoint_name)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', None)

        weights = {k.replace('module.', ''): v for k, v in checkpoint['network_weights'].items()}
        safe_state_dict = {}
        for k, v in weights.items():
            if any(x in k for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                safe_state_dict[k] = v.clone()
            else:
                safe_state_dict[k] = v

        configuration_manager = plans_manager.get_configuration(configuration_name)
        if 'cls_class_num' in checkpoint:
            self.cls_class_num = checkpoint['cls_class_num']
        else:
            cls_weight_keys = [k for k in safe_state_dict if 'classifier' in k and 'weight' in k]
            last_cls_key = sorted(cls_weight_keys)[-1]
            self.cls_class_num = safe_state_dict[last_cls_key].shape[0]

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name}')

        cls_head_output = self.cls_class_num if self.cls_class_num > 2 else 1
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
            emb_dim=320,
            cls_class_num=cls_head_output,
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.network.load_state_dict(safe_state_dict)
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

    def preprocess(self, image, props):
        preprocessor = self.configuration_manager.preprocessor_class(verbose=False)
        data = preprocessor.run_case_npy(image, None, props, self.plans_manager,
                                         self.configuration_manager, self.dataset_json)
        return torch.from_numpy(data[0]).to(dtype=torch.float32, memory_format=torch.contiguous_format)

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor):
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction, cls_prediction = self.network(x)
        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                pred_f, cls_f = self.network(torch.flip(x, axes))
                prediction += torch.flip(pred_f, axes)
                cls_prediction += cls_f
            prediction /= (len(axes_combinations) + 1)
            cls_prediction /= (len(axes_combinations) + 1)
        return prediction, cls_prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self, data, slicers, do_on_device=True):
        results_device = self.device if do_on_device else torch.device('cpu')
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)
        data = data.to(results_device)
        predicted_logits = torch.zeros(
            (self.label_manager.num_segmentation_heads, *data.shape[1:]),
            dtype=torch.half, device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
        cls_out_dim = self.cls_class_num if self.cls_class_num > 2 else 1
        class_logits = torch.zeros((cls_out_dim,), dtype=torch.half, device=results_device)

        if self.use_gaussian:
            gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size),
                                        sigma_scale=1. / 8, value_scaling_factor=10,
                                        device=results_device)
        else:
            gaussian = 1

        for sl in tqdm(slicers, disable=not self.allow_tqdm, desc="sliding window"):
            workon = data[sl][None].to(self.device)
            pred, cls_patch = self._internal_maybe_mirror_and_predict(workon)
            pred = pred[0].to(results_device)
            cls_patch = cls_patch[0].to(results_device)
            if self.use_gaussian:
                pred *= gaussian
            predicted_logits[sl] += pred
            n_predictions[sl[1:]] += gaussian
            class_logits += cls_patch

        predicted_logits /= n_predictions
        class_logits /= len(slicers)
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError('Encountered inf in predicted array.')
        return predicted_logits, class_logits

    @torch.inference_mode()
    def inference(self, image, properties_dict, use_softmax: bool = False):
        image = self.preprocess(image, properties_dict)
        empty_cache(self.device)
        data, slicer_revert_padding = pad_nd_image(
            image, self.configuration_manager.patch_size, 'constant', {'value': 0}, True, None)
        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
        seg_logit, class_logit = self._internal_predict_sliding_window_return_logits(
            data, slicers, self.perform_everything_on_device)

        if self.cls_class_num > 2:
            cls_probs = torch.softmax(class_logit.float(), dim=0).cpu()
        else:
            cls_probs = torch.sigmoid(class_logit.float()).cpu()
        seg_logit = seg_logit.cpu()
        empty_cache(self.device)

        seg_logit = seg_logit[(slice(None), *slicer_revert_padding[1:])]
        segmentation = _convert_predicted_logits_to_segmentation_with_correct_shape(
            seg_logit, self.plans_manager, self.configuration_manager,
            self.label_manager, properties_dict, use_softmax)
        return segmentation, cls_probs


# ─── Classification label lookup ───────────────────────────────────────────────
def _load_dataset_json(model_dir: str, fallback_path: str) -> dict:
    """Return dataset.json, preferring the one packaged with the model."""
    primary = join(model_dir, 'dataset.json')
    data = load_json(primary)
    if 'classification_labels' not in data and os.path.isfile(fallback_path):
        fallback = load_json(fallback_path)
        if 'classification_labels' in fallback:
            data['classification_labels'] = fallback['classification_labels']
    return data


def _format_cls_results(cls_probs: torch.Tensor, dataset_json: dict) -> dict:
    """Build {task_name: {class_name: probability}} from raw cls probs."""
    cls_labels = dataset_json.get('classification_labels', {})
    if not cls_labels:
        return {"unknown": {str(i): float(p) for i, p in enumerate(cls_probs.flatten().tolist())}}

    task_name = list(cls_labels.keys())[0]
    name_map = cls_labels[task_name]
    probs = cls_probs.flatten().tolist()

    if len(probs) == 1:
        p_pos = float(probs[0])
        return {task_name: {name_map["0"]: 1.0 - p_pos, name_map["1"]: p_pos}}

    # Multi-class. Model may output more classes than dataset.json names
    # (e.g. Dataset004/006 trained with an extra "Unknown" bucket for -1 labels).
    out = {}
    for i, p in enumerate(probs):
        name = name_map.get(str(i), "Unknown" if i == len(name_map) else f"class_{i}")
        out[name] = float(p)
    return {task_name: out}


# ─── Video overlay helper ───────────────────────────────────────────────────────
# Distinct BGR colors (OpenCV uses BGR)
_LABEL_COLORS = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
]


def _normalize_slice_uint8(img2d: np.ndarray) -> np.ndarray:
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max() if img.max() > img.min() else img.min() + 1)
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo) * 255.0
    return img.astype(np.uint8)


def _overlay_slice(gray_u8: np.ndarray, seg2d: np.ndarray, label_values: Sequence[int],
                   alpha: float = 0.4) -> np.ndarray:
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    overlay = rgb.copy()
    for i, lbl in enumerate(label_values):
        mask = seg2d == lbl
        if mask.any():
            overlay[mask] = _LABEL_COLORS[i % len(_LABEL_COLORS)]
    return cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)


def _build_frames(image_zyx: np.ndarray, seg_zyx: np.ndarray,
                  label_values: Sequence[int], label_names: Sequence[str]):
    """Yield (H, W, 3) BGR uint8 frames: axial slice + seg overlay + legend."""
    assert image_zyx.shape == seg_zyx.shape, \
        f"image/seg shape mismatch: {image_zyx.shape} vs {seg_zyx.shape}"
    z, h, w = image_zyx.shape

    legend_h = 22 + 18 * ((len(label_values) + 2) // 3)
    frame_h = h + legend_h
    # ffmpeg's libx264 requires even dimensions.
    frame_h += frame_h % 2
    frame_w = w + w % 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    for zi in range(z):
        gray = _normalize_slice_uint8(image_zyx[zi])
        frame_rgb = _overlay_slice(gray, seg_zyx[zi], label_values)
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame_rgb

        cv2.putText(canvas, f"slice {zi + 1}/{z}", (6, 16),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        for i, (lbl, name) in enumerate(zip(label_values, label_names)):
            row, col = divmod(i, 3)
            x0 = 6 + col * (frame_w // 3)
            y0 = h + 16 + row * 18
            color = _LABEL_COLORS[i % len(_LABEL_COLORS)]
            cv2.rectangle(canvas, (x0, y0 - 10), (x0 + 14, y0 + 2), color, -1)
            cv2.putText(canvas, name, (x0 + 18, y0),
                        font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        yield canvas


def _write_overlay_video(image_zyx: np.ndarray, seg_zyx: np.ndarray,
                         label_values: Sequence[int], label_names: Sequence[str],
                         out_path: str, fps: int = 10) -> str:
    """Write a browser-playable H.264 mp4 by piping raw BGR frames to ffmpeg.

    Falls back to OpenCV's mp4v VideoWriter if ffmpeg is unavailable.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    frames = list(_build_frames(image_zyx, seg_zyx, label_values, label_names))
    if not frames:
        raise RuntimeError("No frames to write")
    fh, fw = frames[0].shape[:2]

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{fw}x{fh}", "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "23",
            "-movflags", "+faststart",
            out_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                err = proc.stderr.read().decode('utf-8', errors='replace')
                raise RuntimeError(f"ffmpeg exited with {rc}:\n{err}")
        finally:
            if proc.stderr:
                proc.stderr.close()
        return out_path

    # Fallback: OpenCV mp4v (may not play in all viewers)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return out_path


# ─── Core inference runner ──────────────────────────────────────────────────────
_PREDICTOR_CACHE: dict = {}


def _get_predictor(dataset_name: str, device: Union[str, torch.device] = 'cuda',
                   fold: int = 0, checkpoint: str = 'checkpoint_best.pth') -> SingleFoldPredictor:
    cache_key = (dataset_name, str(device), fold, checkpoint)
    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    reg = MODEL_REGISTRY[dataset_name]
    model_dir = join(BASELINE_ROOT, dataset_name, reg['plans_folder'])

    dev = torch.device(device, 0) if isinstance(device, str) and device != 'cpu' else \
        (torch.device('cpu') if device == 'cpu' else device)
    perform_on_device = (dev.type != 'cpu')

    predictor = SingleFoldPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=perform_on_device,
        device=dev,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(model_dir, fold=fold, checkpoint_name=checkpoint)
    predictor.network.to(dev)

    # Prefer the raw-dataset dataset.json when classification_labels is missing
    predictor.dataset_json = _load_dataset_json(model_dir, reg['dataset_json_fallback'])

    _PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def _as_path_list(image: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(image, str):
        return [image]
    return list(image)


def _run_inference(dataset_name: str,
                   image: Union[str, Sequence[str]],
                   output_dir: str,
                   device: Union[str, torch.device] = 'cuda',
                   fold: int = 0,
                   use_softmax: bool = False,
                   case_id: Optional[str] = None) -> Tuple[str, str, dict]:
    """Shared body used by every per-dataset function below."""
    os.makedirs(output_dir, exist_ok=True)

    image_paths = _as_path_list(image)
    predictor = _get_predictor(dataset_name, device=device, fold=fold)

    expected = len(predictor.dataset_json['channel_names'])
    if len(image_paths) != expected:
        raise ValueError(
            f"{dataset_name} expects {expected} modality file(s) "
            f"(channels={list(predictor.dataset_json['channel_names'].values())}), "
            f"but got {len(image_paths)}: {image_paths}"
        )

    if case_id is None:
        first = os.path.basename(image_paths[0])
        for suffix in ('.nii.gz', '.nii', '.mha', '.nrrd'):
            if first.endswith(suffix):
                first = first[: -len(suffix)]
                break
        # strip trailing channel index like _0000
        if len(first) > 5 and first[-5] == '_' and first[-4:].isdigit():
            first = first[:-5]
        case_id = first

    image_npy, props = SimpleITKIO().read_images(image_paths)
    segmentation, cls_probs = predictor.inference(image_npy, props, use_softmax=use_softmax)
    seg_np = segmentation.numpy()

    # ─── Save segmentation NIfTI ──────────────────────────────────────────
    seg_path = join(output_dir, f"{case_id}_seg.nii.gz")
    sitk_img = sitk.GetImageFromArray(seg_np.astype(np.uint8))
    sitk_img.SetSpacing(props['sitk_stuff']['spacing'])
    sitk_img.SetOrigin(props['sitk_stuff']['origin'])
    sitk_img.SetDirection(props['sitk_stuff']['direction'])
    sitk.WriteImage(sitk_img, seg_path)

    # ─── Build overlay video from the FIRST modality ─────────────────────
    label_map = predictor.dataset_json['labels']  # name -> value
    fg = [(name, val) for name, val in label_map.items() if val != 0]
    fg.sort(key=lambda x: x[1])
    label_values = [v for _, v in fg]
    label_names = [n for n, _ in fg]

    first_modality_zyx = image_npy[0]  # read_images returns (C, Z, Y, X)
    video_path = join(output_dir, f"{case_id}_overlay.mp4")
    _write_overlay_video(first_modality_zyx, seg_np, label_values, label_names, video_path)

    # ─── Classification results ──────────────────────────────────────────
    cls_results = _format_cls_results(cls_probs, predictor.dataset_json)
    cls_path = join(output_dir, f"{case_id}_classification.json")
    with open(cls_path, 'w') as f:
        json.dump(cls_results, f, indent=2)

    return seg_path, video_path, cls_results


# ─── Per-model public functions ─────────────────────────────────────────────────
def infer_dataset002_bmlmps_flair(image: Union[str, Sequence[str]],
                                  output_dir: str,
                                  device: Union[str, torch.device] = 'cuda',
                                  fold: int = 0) -> Tuple[str, str, dict]:
    """BMLMPS FLAIR — whole-tumor seg + EGFR status (Wild-Type / Mutation)."""
    return _run_inference("Dataset002_BMLMPS_FLAIR", image, output_dir, device, fold)


def infer_dataset003_bmlmps_t1ce(image: Union[str, Sequence[str]],
                                 output_dir: str,
                                 device: Union[str, torch.device] = 'cuda',
                                 fold: int = 0) -> Tuple[str, str, dict]:
    """BMLMPS T1CE — core-tumor seg + EGFR status (Wild-Type / Mutation)."""
    return _run_inference("Dataset003_BMLMPS_T1CE", image, output_dir, device, fold)


def infer_dataset004_brainmets(image: Union[str, Sequence[str]],
                               output_dir: str,
                               device: Union[str, torch.device] = 'cuda',
                               fold: int = 0) -> Tuple[str, str, dict]:
    """PROTEAS BrainMets — necrotic/enhancing/edema seg + primary tumor origin (NSCLC / Breast)."""
    return _run_inference("Dataset004_BrainMets", image, output_dir, device, fold)


def infer_dataset005_mu_glioma_post(image: Union[str, Sequence[str]],
                                    output_dir: str,
                                    device: Union[str, torch.device] = 'cuda',
                                    fold: int = 0) -> Tuple[str, str, dict]:
    """MU-Glioma-Post — NCR/ED/ET/NET_RC seg + primary diagnosis (GBM / Astrocytoma / Others)."""
    return _run_inference("Dataset005_MU_Glioma_Post", image, output_dir, device, fold)


def infer_dataset006_jsc_ucsd_ptgb(image: Union[str, Sequence[str]],
                                   output_dir: str,
                                   device: Union[str, torch.device] = 'cuda',
                                   fold: int = 0) -> Tuple[str, str, dict]:
    """UCSD Post-Tx GBM — tumor seg + IDH mutation status (Wild-Type / Mutant)."""
    return _run_inference("Dataset006_JSC_UCSD_PTGB", image, output_dir, device, fold)


def infer_dataset007_picai(image: Union[str, Sequence[str]],
                           output_dir: str,
                           device: Union[str, torch.device] = 'cuda',
                           fold: int = 0) -> Tuple[str, str, dict]:
    """PI-CAI — csPCa seg + ISUP grade (6-class)."""
    return _run_inference("Dataset007_PICAI", image, output_dir, device, fold)


DATASET_DISPATCH = {
    "Dataset002_BMLMPS_FLAIR": infer_dataset002_bmlmps_flair,
    "Dataset003_BMLMPS_T1CE": infer_dataset003_bmlmps_t1ce,
    "Dataset004_BrainMets": infer_dataset004_brainmets,
    "Dataset005_MU_Glioma_Post": infer_dataset005_mu_glioma_post,
    "Dataset006_JSC_UCSD_PTGB": infer_dataset006_jsc_ucsd_ptgb,
    "Dataset007_PICAI": infer_dataset007_picai,
}


# ─── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run fold-0 inference for a JSC_challenge baseline model.")
    parser.add_argument('--dataset', required=True, choices=list(DATASET_DISPATCH.keys()),
                        help='Baseline dataset to use.')
    parser.add_argument('--input', required=True, nargs='+',
                        help='One path per modality, in the channel order from dataset.json.')
    parser.add_argument('--output_dir', required=True, help='Where to write seg/video/cls files.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    fn = DATASET_DISPATCH[args.dataset]
    seg_path, video_path, cls_results = fn(
        image=args.input, output_dir=args.output_dir,
        device=args.device, fold=args.fold,
    )
    print(f"Segmentation: {seg_path}")
    print(f"Overlay video: {video_path}")
    print(f"Classification: {json.dumps(cls_results, indent=2)}")
