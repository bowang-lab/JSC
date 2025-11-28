import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class, nnUNetDatasetBlosc2CLS
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoaderCLS, nnUNetDataBalancedLoaderCLS
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.uncertainty_loss import JointUncertaintyLoss
from sklearn import metrics
from torchmetrics.functional import auroc
import wandb
from torch.optim.lr_scheduler import _LRScheduler
import math, warnings

class DynamicClassificationLoss(nn.Module):
    def __init__(self, class_num=2, imbalanced=False):
        super(DynamicClassificationLoss, self).__init__()
        self.class_num = class_num
        self.imbalanced = imbalanced
        if class_num == 2:
            if imbalanced:
                self.loss = FocalBCEWithLogitsLoss()
            else:
                self.loss = nn.BCEWithLogitsLoss()
        else:
            if imbalanced:
                self.loss = FocalCEWithLogitsLoss()
            else:
                self.loss = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        """
        Computes the classification loss.
        :param logits: model outputs
        :param targets: ground truth labels
        :return: classification loss
        """
        if self.class_num == 2:
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            if targets.type() != logits.type():
                targets = targets.type(logits.type())
        else:
            if targets.dim() > 1:
                targets = targets.squeeze().long()
        return self.loss(logits, targets)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * torch.pow((1 - pt), self.gamma)
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class FocalCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        # Get probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

        focal_term = (1 - probs) ** self.gamma
        focal_loss = -self.alpha * focal_term * log_probs
        loss = torch.sum(targets_one_hot * focal_loss, dim=1)

        if self.reduction == 'mean':
            focal_loss = loss.mean()
        elif self.reduction == 'sum':
            focal_loss = loss.sum()
        else:
            focal_loss = loss  # no reduction

        # Combine focal and CE
        return focal_loss

class CosineAnnealingLR_DoubleWarmstart(_LRScheduler):
    """
    Warmup-1: only head 0 -> base
    Warmup-2: head+encoder 0 -> base
    Cosine:   both decay base -> eta_min
    """
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1, verbose=False, warmstart1=0, warmstart2=0):
        self.w1 = max(int(warmstart1), 0)
        self.w2 = max(int(warmstart2), 0)
        self.total_T = max(int(T_max), 1)
        self.decay_T = max(self.total_T - (self.w1 + self.w2), 1)
        self.eta_min = float(eta_min)

        self.head_idx = None
        self.enc_idx = None
        for i, g in enumerate(optimizer.param_groups):
            name = g.get("name")
            if name in ("cls_head", "reg_head"):
                self.head_idx = i
            elif name == "encoder":
                self.enc_idx = i
        if self.head_idx is None:
            raise ValueError("Need param group named 'cls_head' or 'reg_head'.")
        if self.enc_idx is None:
            raise ValueError("Need param group named 'encoder'.")

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        w_total = self.w1 + self.w2

        # Phase 1: only head warms 0 -> base
        if self.w1 > 0 and e < self.w1:
            f = float(e + 1) / self.w1
            lrs = []
            for gi, base_lr in enumerate(self.base_lrs):
                lrs.append(base_lr * f if gi == self.head_idx else 0.0)
            return lrs

        # Phase 2: both warm 0 -> base
        if self.w2 > 0 and e < w_total:
            f = float(e - self.w1 + 1) / self.w2
            return [base_lr * f for base_lr in self.base_lrs]

        # Cosine phase: both decay base -> eta_min
        t = min(max(e - w_total, 0), self.decay_T)
        cos_term = 0.5 * (1.0 + math.cos(math.pi * t / self.decay_T))
        return [self.eta_min + (base_lr - self.eta_min) * cos_term for base_lr in self.base_lrs]

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, strides, out_channels, fusion_layers=3):
        super().__init__()
        
        # Reduce channels for each level
        if fusion_layers == 3:
            self.conv1x1_1 = nn.Conv3d(in_channels_list[0], out_channels, kernel_size=1)
            self.deconv1 = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size=strides[0], stride=strides[0]
            )
            self.smooth1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1_2 = nn.Conv3d(in_channels_list[1], out_channels, kernel_size=1)
        self.conv1x1_3 = nn.Conv3d(in_channels_list[2], out_channels, kernel_size=1)
        
        # Transposed convolutions for upsampling
        self.deconv2 = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size=strides[1], stride=strides[1]
        )

        

        
        # Additional convolutions after addition
        self.smooth2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Normalization and activation
        self.norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fusion_layers = fusion_layers

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize FPN specific weights with custom strategies
        """
        # Initialize 1x1 convolutions
        conv_list = [self.conv1x1_1, self.conv1x1_2, self.conv1x1_3] if self.fusion_layers == 3 else [self.conv1x1_2, self.conv1x1_3]
        deconv_list = [self.deconv1, self.deconv2] if self.fusion_layers == 3 else [self.deconv2]
        smooth_list = [self.smooth1, self.smooth2] if self.fusion_layers == 3 else [self.smooth2]
        for m in conv_list:
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Initialize deconvolution layers
        for m in deconv_list:
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Initialize smoothing convolutions
        for m in smooth_list:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3):
        # Forward pass implementation remains the same
        p3 = self.conv1x1_3(x3)
        p2 = self.conv1x1_2(x2)
        p3_up = self.deconv2(p3)
        p2 = self.relu(self.norm(p2 + p3_up))
        p2 = self.smooth2(p2)
        if self.fusion_layers == 3:
            p1 = self.conv1x1_1(x1)
            p2_up = self.deconv1(p2)
            p1 = self.relu(self.norm(p1 + p2_up))
            p1 = self.smooth1(p1)
            return p1
        else:
            return p2
        
        

class SegmentationNetworkFusionClassificationHead(nn.Module):
    def __init__(self, seg_network: nn.Module, features_per_stage: List[int], strides: List[tuple],
                 num_hidden_features: int, num_classes: int, fusion_layers: int = 3):
        super().__init__()
        self.seg_network = seg_network
        assert hasattr(self.seg_network, 'encoder')
        assert hasattr(self.seg_network, 'decoder')
        self.encoder = self.seg_network.encoder
        self.decoder = self.seg_network.decoder

        self.feature_fusion_block = FeaturePyramidNetwork(features_per_stage[-3:], strides[-2:], num_hidden_features, fusion_layers)
        # Post FPN processing
        self.conv_block = nn.Sequential(
            nn.Conv3d(num_hidden_features, num_hidden_features*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_hidden_features*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_hidden_features*2, num_hidden_features*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_hidden_features*2),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(num_hidden_features*2, num_hidden_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_hidden_features, num_classes)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        #self.freeze_classification_head()
        #self.freeze_seg_network()

    def freeze_seg_network(self):
        """
        Freeze all parameters in seg_network (encoder and decoder)
        """
        for name, param in self.named_parameters():
            param.requires_grad = False
            if name.split('.')[0]  in ['feature_fusion_block', 'conv_block', 'classifier']:
                param.requires_grad = True  # Freeze all other layers
            else:
                print('[FREEZE]', name)

    def freeze_classification_head(self):
        """
        Freeze all parameters in classification_head
        """
        for name, param in self.named_parameters():
            if name.split('.')[0] in ['feature_fusion_block', 'conv_block', 'classifier']:
                print('[FREEZE]', name)
                param.requires_grad = False  # Freeze all other layers


    def forward(self, x):
        skips = self.seg_network.encoder(x)
        seg_output = self.seg_network.decoder(skips)
        x = self.feature_fusion_block(skips[-3], skips[-2], skips[-1])
        x = self.conv_block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return seg_output, x


class nnUNetCLSTrainerMTL(nnUNetTrainer):

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.cls_class_num = self.get_cls_class_num(join(self.preprocessed_dataset_folder, os.pardir, 'cls_data.csv'))
            cls_head_output = self.cls_class_num if self.cls_class_num > 2 else 1
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
                self.configuration_manager.network_arch_init_kwargs["features_per_stage"][-1],
                cls_head_output,
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.uncertainty_loss = JointUncertaintyLoss()

            wandb.init(
                project="JSC_cvpr26",
                name=f"{self.__class__.__name__}_fold{self.fold}",
            )
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def get_cls_class_num(self, df_path: str) -> int:
        """
        Returns the number of classes for classification
        """
        df = pd.read_csv(df_path)

        return df['label'].nunique()
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   emb_dim: int = 256,
                                   cls_class_num: int = 1) -> nn.Module:

        segmentation_network = nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels,
                                                        num_output_channels, enable_deep_supervision)

        return SegmentationNetworkFusionClassificationHead(segmentation_network,
                                                         arch_init_kwargs["features_per_stage"],
                                                         arch_init_kwargs['strides'],
                                                         emb_dim, cls_class_num)
    def _build_classification_loss(self, class_weights):
        imbalanced = False
        if class_weights.max() // class_weights.min() >= 10:
            imbalanced = True
            self.print_to_log_file("Imbalanced classes detected. Using Focal Loss.")
        loss = DynamicClassificationLoss(
            class_num=self.cls_class_num,
            imbalanced=imbalanced
        )
        return loss
    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()
        self.val_cases = len(val_keys)
        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDatasetBlosc2CLS(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = nnUNetDatasetBlosc2CLS(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        self.classification_loss = self._build_classification_loss(dataset_tr.class_weights)
        if self.batch_size >= self.cls_class_num:
            dl_tr = nnUNetDataBalancedLoaderCLS(dataset_tr, self.batch_size,
                                    initial_patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                    probabilistic_oversampling=self.probabilistic_oversampling)
            dl_val = nnUNetDataBalancedLoaderCLS(dataset_val, self.batch_size,
                                    self.configuration_manager.patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                    probabilistic_oversampling=self.probabilistic_oversampling)
        else:
            dl_tr = nnUNetDataLoaderCLS(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
            dl_val = nnUNetDataLoaderCLS(dataset_val, self.batch_size,
                                    self.configuration_manager.patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                    probabilistic_oversampling=self.probabilistic_oversampling)
        self.num_val_iterations_per_epoch = self.val_cases // self.batch_size
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        cls_label = batch['cls_label']

        data = data.to(self.device, non_blocking=True)
        cls_label = cls_label.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_output = self.network(data)
            # del data
            seg_loss = self.loss(seg_output, target)
            cls_loss = self.classification_loss(cls_output, cls_label.unsqueeze(1).long())
        l = self.uncertainty_loss(seg_loss, cls_loss)


        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy(),
                'seg_loss': seg_loss.detach().cpu().numpy(),
                'cls_loss': cls_loss.detach().cpu().numpy()}
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            seg_losses_tr = [None for _ in range(dist.get_world_size())]
            cls_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            dist.all_gather_object(seg_losses_tr, outputs['seg_loss'])
            dist.all_gather_object(cls_losses_tr, outputs['cls_loss'])
            loss_here = np.vstack(losses_tr).mean()
            seg_loss_here = np.vstack(seg_losses_tr).mean()
            cls_loss_here = np.vstack(cls_losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            seg_loss_here = np.mean(outputs['seg_loss'])
            cls_loss_here = np.mean(outputs['cls_loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        #self.seg_weight, self.cls_weight, slope = self.dynamic_loss_weight_updater.update(seg_loss_here)
        wandb.log({
            'train_segmentation_loss': seg_loss_here,
            'train_classification_loss': cls_loss_here,
            # 'seg_loss_slope': slope,
            # 'seg_weight': self.seg_weight,
            # 'cls_weight': self.cls_weight
        })

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        cls_label = batch['cls_label']
        ids = batch['keys']
        all_probs = []
        all_preds = []
        all_labels = []
        all_ids = []
        all_ids.extend(ids)

        data = data.to(self.device, non_blocking=True)
        cls_label = cls_label.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_output = self.network(data)
            del data
            seg_loss = self.loss(seg_output, target)
            cls_loss = self.classification_loss(cls_output, cls_label.long())

            cls_loss = self.classification_loss(cls_output, cls_label.long())
        l = self.uncertainty_loss(seg_loss, cls_loss)


        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            seg_output = seg_output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, seg_output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = seg_output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(seg_output.shape, device=seg_output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = target != self.label_manager.ignore_label
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = (1 - target[:, -1:]).bool()
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1].bool()
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        
        
        if self.cls_class_num == 2:
            class_probs = torch.sigmoid(cls_output)  # Assuming cls_output is logits
            predicted_class = (class_probs > 0.5).float()  # Thresholding at 0.5 for binary classification
        else:
            class_probs = torch.softmax(cls_output, dim=1)
            class_probs /= class_probs.sum(axis=1, keepdims=True)
            predicted_class = class_probs.argmax(dim=1)  # Get the class with the highest probability
        all_probs.extend(class_probs.cpu())  # Move to CPU and convert to numpy
        all_preds.extend(predicted_class.cpu())  # Move to CPU and convert to numpy
        all_labels.extend(cls_label.cpu())
        


        return {'total_loss': l.detach().cpu().numpy(),
                'seg_loss': seg_loss.detach().cpu().numpy(),
                'cls_loss': cls_loss.detach().cpu().numpy(),
                'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard,
                'all_probs': all_probs,
                'all_labels': all_labels,
                'all_preds': all_preds,
                'all_ids': all_ids
                }
        
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['total_loss'])
            loss_here = np.vstack(losses_val).mean()

            segmentation_losses = [None for _ in range(world_size)]
            dist.all_gather_object(segmentation_losses, outputs_collated['seg_loss'])
            seg_loss_here = np.vstack(segmentation_losses).mean()

            classification_losses = [None for _ in range(world_size)]
            dist.all_gather_object(classification_losses, outputs_collated['cls_loss'])
            cls_loss_here = np.vstack(classification_losses).mean()
        else:
            loss_here = np.mean(outputs_collated['total_loss'])
            seg_loss_here = np.mean(outputs_collated['seg_loss'])
            cls_loss_here = np.mean(outputs_collated['cls_loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        all_preds = torch.tensor(outputs_collated['all_preds'])
        if self.cls_class_num > 2:
            all_probs = torch.stack(outputs_collated['all_probs'], dim=0)
        else:
            all_probs = torch.tensor(outputs_collated['all_probs'])
        all_labels = torch.tensor(outputs_collated['all_labels'])
        all_ids = outputs_collated['all_ids']
        if self.cls_class_num == 2:
            auc_metric = auroc(all_probs, all_labels, task="binary")#metrics.roc_auc_score(all_labels, all_probs)
            classification_accuracy = metrics.accuracy_score(all_labels, all_preds)
            balanced_accuracy = metrics.balanced_accuracy_score(all_labels, all_preds)
        else:
            auc_metric = auroc(all_probs, all_labels, task="multiclass", num_classes=self.cls_class_num)#metrics.multiclass.roc_auc_score(all_labels, all_probs, multi_class='ovr')
            classification_accuracy = metrics.accuracy_score(all_labels, all_preds)
            balanced_accuracy = metrics.balanced_accuracy_score(all_labels, all_preds)

        wandb.log({
            'val/segmentation_loss': seg_loss_here,
            'val/classification_loss': cls_loss_here,
            "val/AUC": auc_metric,
            "val/classification_accuracy": classification_accuracy,
            "val/balanced_accuracy": balanced_accuracy
        })


        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_classification_accuracy', classification_accuracy, self.current_epoch)
        self.logger.log('val_cls_losses', cls_loss_here, self.current_epoch)
        self.logger.log('val_auc', auc_metric, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                            self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file('val classification accuracy', np.round(self.logger.my_fantastic_logging['val_classification_accuracy'][-1], decimals=4))
        self.print_to_log_file('val auc', np.round(self.logger.my_fantastic_logging['val_auc'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        eval_metric = self.logger.my_fantastic_logging['ema_fg_dice'][-1] + \
            self.logger.my_fantastic_logging['val_auc'][-1]
        if self._best_ema is None or eval_metric > self._best_ema:
            self._best_ema = eval_metric
            self.print_to_log_file(f"Yayy! New best evaluation metric: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
            self.print_to_log_file("\nTask weights:")
            self.print_to_log_file(self.uncertainty_loss.get_task_weights())

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'cls_class_num': self.cls_class_num,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')
    
class PretrainedMTL(nnUNetCLSTrainerMTL):
    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.cls_class_num = self.get_cls_class_num(join(self.preprocessed_dataset_folder, os.pardir, 'cls_data.csv'))
            cls_head_output = self.cls_class_num if self.cls_class_num > 2 else 1
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
                self.configuration_manager.network_arch_init_kwargs["features_per_stage"][-1],
                cls_head_output,
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.uncertainty_loss = JointUncertaintyLoss()

            wandb.init(
                project="JSC_pretrained_cvpr26",
                name=f"{self.__class__.__name__}_fold{self.fold}",
            )

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   emb_dim: int = 320,
                                   cls_class_num: int = 1) -> nn.Module:

        segmentation_network = nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels,
                                                        num_output_channels, enable_deep_supervision)

        return SegmentationNetworkFusionClassificationHead(segmentation_network,
                                                         arch_init_kwargs["features_per_stage"],
                                                         arch_init_kwargs['strides'],
                                                         emb_dim, cls_class_num)

    def configure_optimizers(self):
        # ----- collect params -----
        # classification head modules living on top of encoder features
        head_modules = [
            getattr(self.network, "feature_fusion_block", None),
            getattr(self.network, "conv_block", None),
            getattr(self.network, "gap", None),
            getattr(self.network, "classifier", None),
        ]
        head_params = []
        for m in head_modules:
            if m is not None:
                head_params += [p for p in m.parameters() if p.requires_grad]

        # encoder & decoder from the UNet backbone
        enc_params  = [p for p in self.network.encoder.parameters()  if p.requires_grad]
        dec_params  = [p for p in self.network.decoder.parameters()  if p.requires_grad]

        # de-overlap to be safe
        head_ids = set(map(id, head_params))
        enc_ids  = set(map(id, enc_params))
        dec_params = [p for p in dec_params if id(p) not in head_ids and id(p) not in enc_ids]

        # lr settings
        head_lr = getattr(self, "head_initial_lr", self.initial_lr)
        enc_lr  = self.initial_lr

        # ----- optimizer with TWO groups: encoder (frozen in warmup-1), head+decoder (warmed in warmup-1) -----
        optimizer = torch.optim.SGD(
            [
                {"params": enc_params,                  "lr": enc_lr,  "name": "encoder"},
                {"params": head_params + dec_params,    "lr": head_lr, "name": "cls_head"},  # includes decoder
            ],
            lr=self.initial_lr, weight_decay=self.weight_decay,
            momentum=0.99, nesterov=True,
        )

        lr_scheduler = CosineAnnealingLR_DoubleWarmstart(
            optimizer,
            T_max=self.num_epochs,
            eta_min=0.0,
            warmstart1=max(1, int(0.10 * self.num_epochs)),
            warmstart2=max(1, int(0.10 * self.num_epochs)),
        )
        return optimizer, lr_scheduler


    
                                                    
