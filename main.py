import os
import json
import time
import math
import random
import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_NAMES = {0: "Background", 1: "Dirt", 2: "Damage"}
CLASS_COLORS = {1: (128, 0, 128), 2: (255, 165, 0)}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def save_json(path: Path, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class WindTurbineSegDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, img_tf=None, mask_tf=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(
            list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        )
        self.img_tf = img_tf
        self.mask_tf = mask_tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        image = Image.open(img_path).convert("RGB")
        if mask_path.exists():
            mask = Image.open(mask_path)
        else:
            mask = Image.new("L", image.size, 0)

        if self.img_tf:
            image_t = self.img_tf(image)
        else:
            image_t = transforms.ToTensor()(image)

        if self.mask_tf:
            mask_t = self.mask_tf(mask).squeeze(0).long()
        else:
            mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image_t, mask_t, img_path.name, str(img_path)


def build_transforms(input_size: int):
    img_tf = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.PILToTensor(),
        ]
    )
    return img_tf, mask_tf


@torch.no_grad()
def confusion_matrix_torch(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    k = (target >= 0) & (target < num_classes)
    idx = num_classes * target[k] + pred[k]
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    iou = tp / (tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    out = {
        "per_class_iou": iou.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "miou_all": float(np.mean(iou)),
        "miou_anom": float(np.mean(iou[1:])),
        "mf1_all": float(np.mean(f1)),
        "mf1_anom": float(np.mean(f1[1:])),
    }

    # binary anomaly (classes > 0)
    fp_a = cm[0, 1:].sum()
    fn_a = cm[1:, 0].sum()
    tp_a = cm[1:, 1:].sum()
    denom_iou = tp_a + fp_a + fn_a
    denom_f1 = 2 * tp_a + fp_a + fn_a
    out.update(
        {
            "anom_iou_binary": float(tp_a / (denom_iou + eps)),
            "anom_f1_binary": float((2 * tp_a) / (denom_f1 + eps)),
            "anom_recall_binary": float(tp_a / (tp_a + fn_a + eps)),
            "anom_precision_binary": float(tp_a / (tp_a + fp_a + eps)),
        }
    )
    return out


class RobustNoisyLabelLoss(nn.Module):
    def __init__(self, threshold=0.90, class_weights=None, warmup_epochs=0):
        super().__init__()
        self.threshold = float(threshold)
        self.warmup_epochs = int(warmup_epochs)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def forward(self, logits, targets, current_epoch=0):
        if current_epoch < self.warmup_epochs:
            return self.ce(logits, targets).mean()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            discovery_mask = (max_probs > self.threshold) & (targets == 0) & (preds > 0)
            corrected = targets.clone()
            corrected[discovery_mask] = preds[discovery_mask]

        return self.ce(logits, corrected).mean()


class AsymmetricDiceLoss(nn.Module):
    def __init__(self, beta=0.4, epsilon=1e-6):
        super().__init__()
        self.beta = float(beta)
        self.epsilon = float(epsilon)

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        if probs.shape[1] == 2:
            p_anom = probs[:, 1]
        else:
            p_anom = probs[:, 1] + probs[:, 2]
        t_anom = (targets > 0).float()

        inter = (p_anom * t_anom).sum()
        fp = p_anom.sum() - inter
        fn = t_anom.sum() - inter

        return 1.0 - (inter + self.epsilon) / (
            inter + self.beta * fp + fn + self.epsilon
        )


class HierarchicalLoss(nn.Module):
    def __init__(
        self,
        fine_weights,
        dice_beta=0.4,
        tau=0.90,
        warmup_epochs=0,
        w_bin=0.4,
        w_fine=0.6,
    ):
        super().__init__()
        self.w_bin = float(w_bin)
        self.w_fine = float(w_fine)
        self.binary_loss = AsymmetricDiceLoss(beta=dice_beta)
        self.fine_loss = RobustNoisyLabelLoss(
            threshold=tau, class_weights=fine_weights, warmup_epochs=warmup_epochs
        )

    def forward(self, inputs, target_fine, current_epoch=0):
        if isinstance(inputs, tuple):
            pred_bin, pred_fine = inputs
            target_bin = (target_fine > 0).long()
            return self.w_bin * self.binary_loss(
                pred_bin, target_bin
            ) + self.w_fine * self.fine_loss(
                pred_fine, target_fine, current_epoch=current_epoch
            )
        return self.fine_loss(inputs, target_fine, current_epoch=current_epoch)


class HierarchicalLossNoSelfCorrect(nn.Module):
    def __init__(self, fine_weights, dice_beta=0.4, w_bin=0.4, w_fine=0.6):
        super().__init__()
        self.w_bin = float(w_bin)
        self.w_fine = float(w_fine)
        self.binary_loss = AsymmetricDiceLoss(beta=dice_beta)
        self.fine_loss = nn.CrossEntropyLoss(weight=fine_weights)

    def forward(self, inputs, target_fine, current_epoch=0):
        if isinstance(inputs, tuple):
            pred_bin, pred_fine = inputs
            target_bin = (target_fine > 0).long()
            return self.w_bin * self.binary_loss(
                pred_bin, target_bin
            ) + self.w_fine * self.fine_loss(pred_fine, target_fine)
        return self.fine_loss(inputs, target_fine)


class ResidualFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_deep, x_skip):
        fused = x_deep + x_skip
        out = self.conv(fused)
        out = out + fused
        return self.relu(out)


class DINOHighResSegmenter(nn.Module):
    def __init__(self, num_classes=3, mode="bitfit", use_detail=True, use_binary=True):
        super().__init__()
        self.use_detail = bool(use_detail)
        self.use_binary = bool(use_binary)

        print("[Model] Loading DINOv2 ViT-S/14 + detail branch...")
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        for p in self.backbone.parameters():
            p.requires_grad = False
        if mode == "bitfit":
            for name, p in self.backbone.named_parameters():
                if ("bias" in name) or ("norm" in name):
                    p.requires_grad = True

        self.extract_indices = [1, 2, 4, 7]
        self.embed_dim = 384
        self.decoder_dim = 256

        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, self.decoder_dim, 1),
                    nn.BatchNorm2d(self.decoder_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(self.extract_indices))
            ]
        )

        self.fusion_blocks = nn.ModuleList(
            [
                ResidualFusionBlock(self.decoder_dim)
                for _ in range(len(self.extract_indices) - 1)
            ]
        )

        self.up_block1 = self._make_pixel_shuffle(self.decoder_dim, 128)
        self.up_block2 = self._make_pixel_shuffle(128, 64)
        self.up_block3 = self._make_pixel_shuffle(64, 32)

        if self.use_detail:
            self.detail_encoder = nn.Sequential(
                nn.Conv2d(3, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            mixer_in = 64
        else:
            self.detail_encoder = None
            mixer_in = 32

        self.mixer = nn.Sequential(
            nn.Conv2d(mixer_in, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        if self.use_binary:
            self.binary_head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(32, 2, 1),
            )
        else:
            self.binary_head = None

        self.fine_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, num_classes, 1),
        )

    def _make_pixel_shuffle(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        layers = self.backbone.get_intermediate_layers(x, n=12, reshape=True)
        feats = [layers[i] for i in self.extract_indices]
        proj_feats = [p(f) for p, f in zip(self.projectors, feats)]

        x_dino = proj_feats[-1]
        for i in range(len(proj_feats) - 2, -1, -1):
            fb = self.fusion_blocks[(len(proj_feats) - 2) - i]
            x_dino = fb(x_dino, proj_feats[i])

        x_dino = self.up_block1(x_dino)
        x_dino = self.up_block2(x_dino)
        x_dino = self.up_block3(x_dino)  # ~H/4

        if self.use_detail:
            x_detail = self.detail_encoder(x)  # H/4
            if x_dino.shape[-2:] != x_detail.shape[-2:]:
                x_dino = F.interpolate(
                    x_dino,
                    size=x_detail.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x_cat = torch.cat([x_dino, x_detail], dim=1)
        else:
            x_cat = x_dino

        x_mixed = self.mixer(x_cat)

        logits_fine = self.fine_head(x_mixed)
        if logits_fine.shape[-2:] != (H, W):
            logits_fine = F.interpolate(
                logits_fine, size=(H, W), mode="bilinear", align_corners=False
            )

        if not self.use_binary:
            return logits_fine

        logits_bin = self.binary_head(x_mixed)
        if logits_bin.shape[-2:] != (H, W):
            logits_bin = F.interpolate(
                logits_bin, size=(H, W), mode="bilinear", align_corners=False
            )

        return logits_bin, logits_fine


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, num_classes=3, base=32):
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    @staticmethod
    def _match_size(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] != ref.shape[-2:]:
            src = F.interpolate(
                src, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )
        return src

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        y = self.up3(x4)
        y = self._match_size(y, x3)
        y = self.dec3(torch.cat([y, x3], dim=1))

        y = self.up2(y)
        y = self._match_size(y, x2)
        y = self.dec2(torch.cat([y, x2], dim=1))

        y = self.up1(y)
        y = self._match_size(y, x1)
        y = self.dec1(torch.cat([y, x1], dim=1))

        y = self.head(y)
        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y


class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_c, out_c, 1, bias=False),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_c, out_c, 3, padding=r, dilation=r, bias=False),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True),
                    )
                )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_c * (len(rates) + 1), out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        gp = self.pool(x)
        gp = F.interpolate(gp, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(gp)
        y = torch.cat(feats, dim=1)
        return self.proj(y)


class DeepLabV3_EffB2(nn.Module):
    def __init__(self, num_classes=3, aspp_c=256, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = self.backbone.features
        self.aspp = ASPP(in_c=1408, out_c=aspp_c)
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_c, aspp_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_c, num_classes, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.features(x)
        y = self.aspp(y)
        y = self.classifier(y)
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y


class SegFormerWrapper(nn.Module):
    """
    Uses HuggingFace SegFormer.
    We default to 'nvidia/mit-b2'
    """

    def __init__(self, num_classes=3, ckpt="nvidia/mit-b2"):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except Exception as e:
            raise RuntimeError(
                "transformers is required for SegFormer. Install transformers or disable segformer runs."
            ) from e

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            ckpt,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        out = self.model(pixel_values=x)
        logits = out.logits  # (B,C,h,w)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in msd.keys():
                if msd[k].dtype.is_floating_point:
                    esd[k].mul_(self.decay).add_(msd[k], alpha=(1.0 - self.decay))

    def get(self):
        return self.ema


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -1e18
        self.count = 0
        self.stop = False

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.count = 0
            return True
        self.count += 1
        if self.count >= self.patience:
            self.stop = True
        return False


def denorm_img(img_t: torch.Tensor) -> np.ndarray:
    if img_t is None:
        raise ValueError("denorm_img got None")
    if not torch.is_tensor(img_t):
        raise TypeError(f"denorm_img expects torch.Tensor, got {type(img_t)}")

    if img_t.dim() == 4:
        img_t = img_t[0]
    if img_t.dim() != 3:
        raise ValueError(
            f"denorm_img expects CHW or BCHW, got shape={tuple(img_t.shape)}"
        )

    mean = torch.tensor([0.485, 0.456, 0.406], device=img_t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_t.device).view(3, 1, 1)

    x = img_t.detach().float()
    x = x * std + mean
    x = x.clamp(0, 1)
    x = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return x


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        out[mask == cls] = np.array(color, dtype=np.uint8)
    return out


def overlay_mask(image: Any, mask: np.ndarray, alpha=0.45) -> np.ndarray:
    if image is None:
        raise ValueError("overlay_mask got image=None")

    if torch.is_tensor(image):
        image = denorm_img(image)

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"overlay_mask expects np.ndarray or torch.Tensor, got {type(image)}"
        )

    if not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError(f"overlay_mask expects HWC RGB image, got shape={image.shape}")

    col = colorize_mask(mask)
    out = image.copy()
    m = mask > 0
    if m.any():
        out[m] = cv2.addWeighted(out[m], 1 - alpha, col[m], alpha, 0)
    return out


def error_overlay(
    image: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha=0.55
) -> np.ndarray:
    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    tp = (gt > 0) & (pred == gt)
    fp = (gt == 0) & (pred > 0)
    fn = (gt > 0) & (pred == 0)
    mis = (gt > 0) & (pred > 0) & (pred != gt)

    overlay[tp] = (0, 255, 0)
    overlay[fp] = (0, 0, 255)
    overlay[fn] = (255, 0, 0)
    overlay[mis] = (255, 255, 0)

    out = image.copy()
    m = tp | fp | fn | mis
    if m.any():
        out[m] = cv2.addWeighted(out[m], 1 - alpha, overlay[m], alpha, 0)
    return out


def load_yolo_bboxes(txt_path: Path, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    bboxes = []
    if not txt_path.exists():
        return bboxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            if cls in [1, 2]:
                bboxes.append({"class": cls, "bbox": [x1, y1, x2, y2]})
    return bboxes


def draw_yolo_overlays(image: np.ndarray, bboxes: List[Dict[str, Any]]) -> np.ndarray:
    out = image.copy()
    for item in bboxes:
        x1, y1, x2, y2 = item["bbox"]
        cls = item["class"]
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(cls, str(cls))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - 20)), (x1 + tw, y1), color, -1)
        cv2.putText(
            out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    return out


def make_test_visualizations(
    results_root: Path,
    project_root: Path,
    input_size: int,
    yolo_label_dir: Optional[Path],
    max_vis: int,
    model_names_for_vis: List[str],
):
    test_img_dir = project_root / "test" / "images"
    test_mask_dir = project_root / "test" / "masks_refined"
    if not test_mask_dir.exists():
        test_mask_dir = project_root / "test" / "masks"

    img_paths = sorted(
        list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    )
    if len(img_paths) == 0:
        return

    out_dir = results_root / "_paper_vis"
    ensure_dir(out_dir)

    chosen = []
    for p in img_paths:
        mp = test_mask_dir / f"{p.stem}.png"
        if mp.exists():
            m = np.array(
                Image.open(mp).resize((input_size, input_size), resample=Image.NEAREST)
            )
            if (m > 0).any():
                chosen.append(p)
        if len(chosen) >= max_vis:
            break
    if len(chosen) == 0:
        chosen = img_paths[:max_vis]

    pred_dirs = {}
    for mn in model_names_for_vis:
        d = results_root / mn / "pred_test"
        if d.exists():
            pred_dirs[mn] = d

    for p in chosen:
        img = (
            Image.open(p)
            .convert("RGB")
            .resize((input_size, input_size), resample=Image.BILINEAR)
        )
        img_np = np.array(img, dtype=np.uint8)

        gt_path = test_mask_dir / f"{p.stem}.png"
        if gt_path.exists():
            gt = np.array(
                Image.open(gt_path).resize(
                    (input_size, input_size), resample=Image.NEAREST
                )
            )
        else:
            gt = np.zeros((input_size, input_size), dtype=np.uint8)

        if yolo_label_dir is not None:
            txt = yolo_label_dir / f"{p.stem}.txt"
            bboxes = load_yolo_bboxes(txt, input_size, input_size)
            yolo_np = draw_yolo_overlays(img_np, bboxes)
        else:
            yolo_np = img_np.copy()

        cols = 2 + len(pred_dirs)
        fig, axes = plt.subplots(1, cols, figsize=(5.2 * cols, 5.8), dpi=150)

        axes[0].imshow(yolo_np)
        axes[0].set_title("Input + BBox")
        axes[1].imshow(overlay_mask(img_np, gt))
        axes[1].set_title("GT (golden)")

        j = 2
        for mn, d in pred_dirs.items():
            pr_path = d / f"{p.stem}.png"
            if pr_path.exists():
                pred = np.array(
                    Image.open(pr_path).resize(
                        (input_size, input_size), resample=Image.NEAREST
                    )
                )
            else:
                pred = np.zeros_like(gt)
            axes[j].imshow(overlay_mask(img_np, pred))
            axes[j].set_title(mn)
            j += 1

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(
            out_dir / f"compare_{p.stem}.png", bbox_inches="tight", pad_inches=0.1
        )
        plt.close(fig)


@dataclass
class RunSpec:
    name: str
    model_type: str  # dino / unet / deeplab_b2 / segformer
    is_ablation: bool
    eval_on_test: bool
    model_kwargs: Dict[str, Any]
    loss_kwargs: Dict[str, Any]


def build_model(spec: RunSpec, num_classes: int, segformer_ckpt: str) -> nn.Module:
    t = spec.model_type
    if t == "dino":
        return DINOHighResSegmenter(num_classes=num_classes, **spec.model_kwargs)
    if t == "unet":
        return UNetSmall(num_classes=num_classes, **spec.model_kwargs)
    if t == "deeplab_b2":
        return DeepLabV3_EffB2(num_classes=num_classes, **spec.model_kwargs)
    if t == "segformer":
        return SegFormerWrapper(num_classes=num_classes, ckpt=segformer_ckpt)
    raise ValueError(f"Unknown model_type: {t}")


def build_criterion(
    loss_kwargs: Dict[str, Any], class_weights: torch.Tensor, use_selfcorr: bool
) -> nn.Module:
    if use_selfcorr:
        return HierarchicalLoss(fine_weights=class_weights, **loss_kwargs)
    return HierarchicalLossNoSelfCorrect(fine_weights=class_weights, **loss_kwargs)


@torch.no_grad()
def eval_split_with_names(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    save_preds_dir: Optional[Path] = None,
    save_prob_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")

    if save_preds_dir is not None:
        ensure_dir(save_preds_dir)
    if save_prob_dir is not None:
        ensure_dir(save_prob_dir)

    for images, masks, names, _ in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        out = model(images)
        logits = out[1] if isinstance(out, tuple) else out
        preds = torch.argmax(logits, dim=1).to(torch.int64)

        cm += confusion_matrix_torch(preds.cpu(), masks.cpu(), num_classes)

        if save_preds_dir is not None:
            for i in range(preds.shape[0]):
                p = preds[i].detach().cpu().numpy().astype(np.uint8)
                Image.fromarray(p).save(save_preds_dir / f"{Path(names[i]).stem}.png")

        if save_prob_dir is not None:
            probs = torch.softmax(logits, dim=1)
            for i in range(probs.shape[0]):
                np.save(
                    save_prob_dir / f"{Path(names[i]).stem}.npy",
                    probs[i].detach().cpu().numpy().astype(np.float16),
                )

    cm_np = cm.numpy()
    return {"cm": cm_np.tolist(), **metrics_from_cm(cm_np)}


def _segformer_param_groups(
    model: SegFormerWrapper, base_lr: float
) -> List[Dict[str, Any]]:
    encoder_params = list(model.model.segformer.encoder.parameters())
    enc_ids = set(map(id, encoder_params))
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]

    groups = [
        {"params": encoder_params, "lr": base_lr * 0.1},
        {"params": other_params, "lr": base_lr},
    ]
    return groups


def train_one_run(
    spec: RunSpec,
    args: argparse.Namespace,
    device: torch.device,
    loaders: Dict[str, DataLoader],
    paths: Dict[str, Path],
    class_weights: torch.Tensor,
) -> Dict[str, Any]:
    run_dir = paths["results_root"] / spec.name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "ckpt")

    save_json(
        run_dir / "run_spec.json",
        {
            "run_name": spec.name,
            "model_type": spec.model_type,
            "is_ablation": spec.is_ablation,
            "eval_on_test": spec.eval_on_test,
            "model_kwargs": spec.model_kwargs,
            "loss_kwargs": spec.loss_kwargs,
            "args": vars(args),
        },
    )

    model = build_model(
        spec, num_classes=args.num_classes, segformer_ckpt=args.segformer_ckpt
    ).to(device)
    pcount = count_params(model)

    # self-correction on/off is encoded by presence of tau in loss_kwargs
    use_selfcorr = "tau" in spec.loss_kwargs

    loss_kwargs = dict(spec.loss_kwargs)
    if not use_selfcorr:
        loss_kwargs.pop("tau", None)
        loss_kwargs.pop("warmup_epochs", None)

    criterion = build_criterion(
        loss_kwargs, class_weights, use_selfcorr=use_selfcorr
    ).to(device)
    print(
        f"[Config] {spec.name}: selfcorr={'ON' if use_selfcorr else 'OFF'} | loss_kwargs={loss_kwargs}"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            f"No trainable parameters for run {spec.name} ({spec.model_type})."
        )

    is_segformer = spec.model_type == "segformer"
    if is_segformer:
        param_groups = _segformer_param_groups(model, base_lr=args.lr)
        optimizer = optim.AdamW(param_groups, weight_decay=args.wd)
        use_amp = bool(args.amp_segformer)  # default False
    else:
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
        use_amp = bool(args.amp)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    stopper = EarlyStopping(patience=args.patience)

    best_path = run_dir / "ckpt" / "best.pth"
    best_epoch = -1
    best_val = -1e18

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        nsteps = 0
        n_skipped = 0

        for images, masks, _, _ in tqdm(
            loaders["train"],
            desc=f"{spec.name} | Ep {epoch+1}/{args.epochs}",
            leave=False,
        ):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(images)
                loss = criterion(out, masks, current_epoch=epoch)
            if not torch.isfinite(loss):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                continue

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            running += float(loss.detach().item())
            nsteps += 1

        scheduler.step()

        eval_model = ema.get() if ema is not None else model
        val_metrics = eval_split_with_names(
            eval_model, loaders["val"], device, args.num_classes
        )
        val_anom_miou = float(val_metrics["miou_anom"])

        is_best = stopper.step(val_anom_miou)
        if is_best:
            best_val = val_anom_miou
            best_epoch = epoch + 1
            torch.save(eval_model.state_dict(), best_path)

        dt = time.time() - t0
        avg_loss = running / max(1, nsteps)
        msg = (
            f"[{spec.name}] ep {epoch+1:03d} | "
            f"tr_loss={avg_loss:.4f} | "
            f"val_mIoU_anom={val_anom_miou:.4f} | "
            f"best={best_val:.4f}@{best_epoch} | "
            f"skipped={n_skipped} | "
            f"time={dt:.1f}s"
        )
        print(msg)

        if stopper.stop:
            break

    if not best_path.exists():
        raise RuntimeError(
            f"[{spec.name}] No best checkpoint saved. (All steps may have been skipped due to NaNs.)"
        )

    # load best and evaluate
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    val_final = eval_split_with_names(model, loaders["val"], device, args.num_classes)

    out = {
        "run": spec.name,
        "model_type": spec.model_type,
        "is_ablation": spec.is_ablation,
        "params_total": pcount["total"],
        "params_trainable": pcount["trainable"],
        "best_epoch": best_epoch,
        "best_val_miou_anom": float(best_val),
        "val": val_final,
    }

    if spec.eval_on_test:
        test_pred_dir = run_dir / "pred_test"
        test_probs_dir = run_dir / "prob_test" if args.save_probs else None
        test_metrics = eval_split_with_names(
            model,
            loaders["test"],
            device,
            args.num_classes,
            save_preds_dir=test_pred_dir,
            save_prob_dir=test_probs_dir,
        )
        out["test"] = test_metrics

    save_json(run_dir / "metrics.json", out)
    return out


def make_suite(args: argparse.Namespace) -> List[RunSpec]:
    dice_beta = args.dice_beta

    base_loss_no_sc = {
        "dice_beta": dice_beta,
        "w_bin": args.w_bin,
        "w_fine": args.w_fine,
    }

    base_loss_sc = {
        "dice_beta": dice_beta,
        "w_bin": args.w_bin,
        "w_fine": args.w_fine,
        "tau": args.selfcorr_tau,
        "warmup_epochs": args.selfcorr_warmup,
    }

    suite_all: List[RunSpec] = [
        RunSpec(
            name="unet",
            model_type="unet",
            is_ablation=False,
            eval_on_test=True,
            model_kwargs={"base": args.unet_base},
            loss_kwargs=base_loss_sc,
        ),
        RunSpec(
            name="deeplabv3_effb2",
            model_type="deeplab_b2",
            is_ablation=False,
            eval_on_test=True,
            model_kwargs={"aspp_c": args.deeplab_aspp_c, "pretrained": True},
            loss_kwargs=base_loss_sc,
        ),
        RunSpec(
            name="segformer_b2",
            model_type="segformer",
            is_ablation=False,
            eval_on_test=True,
            model_kwargs={},
            loss_kwargs=base_loss_sc,
        ),
        RunSpec(
            name="dino_final",
            model_type="dino",
            is_ablation=False,
            eval_on_test=True,
            model_kwargs={"mode": "bitfit", "use_detail": True, "use_binary": True},
            loss_kwargs=base_loss_sc,
        ),
        RunSpec(
            name="abl_dino_final_burnin",
            model_type="dino",
            is_ablation=False,
            eval_on_test=True,
            model_kwargs={"mode": "bitfit", "use_detail": True, "use_binary": True},
            loss_kwargs={**base_loss_sc, "warmup_epochs": 5},
        ),
        RunSpec(
            name="unet_no_selfcorr",
            model_type="unet",
            is_ablation=False,
            eval_on_test=True,
            model_kwargs={"base": args.unet_base},
            loss_kwargs=base_loss_no_sc,
        ),
        RunSpec(
            name="deeplabv3_effb2_no_selfcorr",
            model_type="deeplab_b2",
            is_ablation=False,
            eval_on_test=True,
            model_kwargs={"aspp_c": args.deeplab_aspp_c, "pretrained": True},
            loss_kwargs=base_loss_no_sc,
        ),
        RunSpec(
            name="segformer_b2_no_selfcorr",
            model_type="segformer",
            is_ablation=False,
            eval_on_test=True,
            model_kwargs={},
            loss_kwargs=base_loss_no_sc,
        ),
        RunSpec(
            name="dino_final_no_selfcorr",
            model_type="dino",
            is_ablation=False,
            eval_on_test=True,
            model_kwargs={"mode": "bitfit", "use_detail": True, "use_binary": True},
            loss_kwargs=base_loss_no_sc,
        ),
        RunSpec(
            name="abl_dino_no_detail",
            model_type="dino",
            is_ablation=True,
            eval_on_test=False,
            model_kwargs={"mode": "bitfit", "use_detail": False, "use_binary": True},
            loss_kwargs=base_loss_sc,
        ),
        RunSpec(
            name="abl_dino_no_binary",
            model_type="dino",
            is_ablation=True,
            eval_on_test=True,  # Changed to True to match original final run
            model_kwargs={"mode": "bitfit", "use_detail": True, "use_binary": False},
            loss_kwargs=base_loss_sc,
        ),
        RunSpec(
            name="abl_dino_no_selfcorr",
            model_type="dino",
            is_ablation=True,
            eval_on_test=False,
            model_kwargs={"mode": "bitfit", "use_detail": True, "use_binary": True},
            loss_kwargs=base_loss_no_sc,
        ),
    ]

    if args.suite == "all":
        return suite_all
    if args.suite == "final":
        return [s for s in suite_all if not s.is_ablation]
    if args.suite == "ablations":
        return [s for s in suite_all if s.is_ablation]
    if args.suite == "custom":
        wanted = {x.strip() for x in args.runs.split(",") if x.strip()}
        return [s for s in suite_all if s.name in wanted]
    raise ValueError(f"Unknown suite: {args.suite}")


def to_row(
    run_name: str, split: str, metrics: Dict[str, Any], extra: Dict[str, Any]
) -> Dict[str, Any]:
    row = {"run": run_name, "split": split}
    row.update(extra)
    row.update(
        {
            "miou_all": metrics.get("miou_all"),
            "miou_anom": metrics.get("miou_anom"),
            "mf1_all": metrics.get("mf1_all"),
            "mf1_anom": metrics.get("mf1_anom"),
            "anom_iou_binary": metrics.get("anom_iou_binary"),
            "anom_f1_binary": metrics.get("anom_f1_binary"),
            "anom_recall_binary": metrics.get("anom_recall_binary"),
            "anom_precision_binary": metrics.get("anom_precision_binary"),
        }
    )
    iou = metrics.get("per_class_iou", [None] * 3)
    f1 = metrics.get("per_class_f1", [None] * 3)
    rec = metrics.get("per_class_recall", [None] * 3)
    prec = metrics.get("per_class_precision", [None] * 3)
    for c in range(3):
        row[f"iou_c{c}"] = iou[c] if c < len(iou) else None
        row[f"f1_c{c}"] = f1[c] if c < len(f1) else None
        row[f"rec_c{c}"] = rec[c] if c < len(rec) else None
        row[f"prec_c{c}"] = prec[c] if c < len(prec) else None
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    import csv

    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_results(
    results_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_dirs = [
        p for p in results_root.iterdir() if p.is_dir() and not p.name.startswith("_")
    ]
    all_results: List[Dict[str, Any]] = []

    for rd in sorted(run_dirs):
        mp = rd / "metrics.json"
        if mp.exists():
            try:
                all_results.append(load_json(mp))
            except Exception as e:
                print(f"[Warn] Failed reading {mp}: {e}")

    rows = []
    for r in all_results:
        extra = {
            "model_type": r.get("model_type"),
            "is_ablation": r.get("is_ablation"),
            "params_total": r.get("params_total"),
            "params_trainable": r.get("params_trainable"),
            "best_epoch": r.get("best_epoch"),
            "best_val_miou_anom": r.get("best_val_miou_anom"),
        }
        rows.append(to_row(r["run"], "val", r["val"], extra))
        if "test" in r:
            rows.append(to_row(r["run"], "test", r["test"], extra))

    write_csv(results_root / "summary.csv", rows)
    save_json(
        results_root / "summary.json",
        {"rows": rows, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    )
    print(f"[Done] Aggregated {len(all_results)} runs -> {results_root/'summary.csv'}")
    return all_results, rows


def main():
    parser = argparse.ArgumentParser(
        "Wind turbine segmentation benchmark (CVPR workshop ready)"
    )
    parser.add_argument("--project_root", type=str, default="dataset_dtu_final_split")
    parser.add_argument("--results_root", type=str, default="benchmark_results")
    parser.add_argument(
        "--yolo_label_dir", type=str, default="../dataset_dtu_final_split/test/yolo_labels"
    )

    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-2)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)

    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--use_ema", action="store_true")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp_segformer",
        action="store_true",
        help="Enable AMP for segformer (default off for stability).",
    )

    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--class_weights", type=str, default="0.4,4.0,4.0")

    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "final", "ablations", "custom"],
    )
    parser.add_argument("--runs", type=str, default="")  

    parser.add_argument("--selfcorr_tau", type=float, default=0.90)
    parser.add_argument(
        "--selfcorr_warmup", type=int, default=0
    )  # set to 15 to mimic your "working" behavior

    parser.add_argument("--dice_beta", type=float, default=0.4)
    parser.add_argument("--w_bin", type=float, default=0.5)
    parser.add_argument("--w_fine", type=float, default=0.5)

    parser.add_argument("--unet_base", type=int, default=32)
    parser.add_argument("--deeplab_aspp_c", type=int, default=256)

    parser.add_argument("--segformer_ckpt", type=str, default="nvidia/mit-b2")
    parser.add_argument("--save_probs", action="store_true")

    parser.add_argument("--max_vis", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only aggregate results_root/*/metrics.json into summary.*",
    )
    parser.add_argument(
        "--make_vis",
        action="store_true",
        help="When aggregating, also generate _paper_vis from pred_test.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    project_root = Path(args.project_root)
    results_root = Path(args.results_root)
    ensure_dir(results_root)

    if args.aggregate_only:
        all_results, _ = aggregate_results(results_root)

        if args.make_vis:
            yolo_dir = (
                Path(args.yolo_label_dir)
                if args.yolo_label_dir and Path(args.yolo_label_dir).exists()
                else None
            )
            test_run_names = [r["run"] for r in all_results if ("test" in r)]
            if test_run_names:
                make_test_visualizations(
                    results_root=results_root,
                    project_root=project_root,
                    input_size=args.input_size,
                    yolo_label_dir=yolo_dir,
                    max_vis=args.max_vis,
                    model_names_for_vis=test_run_names,
                )
                print(
                    f"[Done] Paper comparison visuals saved in: {results_root / '_paper_vis'}"
                )
        return

    # paths
    train_img = project_root / "train" / "images"
    train_mask = project_root / "train" / "masks"
    val_img = project_root / "val" / "images"
    val_mask = project_root / "val" / "masks"
    test_img = project_root / "test" / "images"
    test_mask = project_root / "test" / "masks_refined"
    if not test_mask.exists():
        test_mask = project_root / "test" / "masks"

    for p in [train_img, train_mask, val_img, val_mask, test_img, test_mask]:
        if not p.exists():
            raise FileNotFoundError(f"Missing path: {p}")

    img_tf, mask_tf = build_transforms(args.input_size)
    train_ds = WindTurbineSegDataset(train_img, train_mask, img_tf, mask_tf)
    val_ds = WindTurbineSegDataset(val_img, val_mask, img_tf, mask_tf)
    test_ds = WindTurbineSegDataset(test_img, test_mask, img_tf, mask_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    paths = {"results_root": results_root}

    # class weights
    cw = [float(x) for x in args.class_weights.split(",")]
    if len(cw) != args.num_classes:
        raise ValueError(f"--class_weights must have {args.num_classes} values")
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device)

    suite = make_suite(args)
    if args.suite == "custom" and not suite:
        raise ValueError(
            "Custom suite is empty. Provide --runs with comma-separated run names."
        )

    print("[Suite] Runs:")
    for s in suite:
        print(
            f"  - {s.name:>18} | model={s.model_type:10} | ablation={s.is_ablation} | test={s.eval_on_test}"
        )

    all_results: List[Dict[str, Any]] = []
    for spec in suite:
        print(f"\n=== RUN: {spec.name} ===")
        res = train_one_run(spec, args, device, loaders, paths, class_weights)
        all_results.append(res)

    if args.suite == "custom":
        print(
            "[Info] Each run wrote results_root/<run>/metrics.json. Run aggregation after array finishes:"
        )
        print("       python -u main.py --results_root ... --aggregate_only --make_vis")
        return
    aggregate_results(results_root)

    yolo_dir = (
        Path(args.yolo_label_dir)
        if args.yolo_label_dir and Path(args.yolo_label_dir).exists()
        else None
    )
    test_run_names = [r["run"] for r in all_results if ("test" in r)]
    if test_run_names:
        make_test_visualizations(
            results_root=results_root,
            project_root=project_root,
            input_size=args.input_size,
            yolo_label_dir=yolo_dir,
            max_vis=args.max_vis,
            model_names_for_vis=test_run_names,
        )
        print(
            f"[Done] Paper comparison visuals saved in: {results_root / '_paper_vis'}"
        )


if __name__ == "__main__":
    main()