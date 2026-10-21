# -*- coding: utf-8 -*-
"""
植物幼苗 5 分类 —— 多骨干 + Mixup/CutMix + EMA + 强 TTA (Kaggle 单文件版)
路径: plant_classifier/cursor/ (Cursor Agent 方案)
目标: 在 ResNet18(~0.81) / EffNetV2(~0.82) 基础上继续抬升 mean macro-F1

策略摘要:
  - timm 双骨干: ConvNeXt-Tiny + EfficientNetV2-S (概率平均, 互补特征)
  - 5 折 StratifiedKFold, 两阶段微调 + Cosine LR
  - Mixup / CutMix + Label Smoothing + 类别权重 (利于 macro-F1)
  - 训练期 EMA, 推理用 EMA 权重
  - 8 视角 TTA: 中心裁剪 + 水平翻转 × 两种缩放

Kaggle 用法:
  1) 上传含 train/、test/ 的数据集并挂载到 Notebook
  2) 加速器选 GPU (T4 x2 更佳), 关闭 Internet 也可 (timm/torch 已预装)
  3) 整文件粘贴到一个 code cell 运行
  4) 输出: /kaggle/working/submission_advanced_ensemble.csv

本地: 设置环境变量 PLANT_DATA_ROOT 指向含 train/test 的目录后 python 本文件
"""
from __future__ import annotations

import os
import sys
import random
import copy
from collections import Counter

# Kaggle Notebook 里 stdout 常被缓冲，长时间看不到 print
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
print("[cursor] 脚本已加载，正在 import 依赖...", flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

try:
    import timm
except ImportError:
    raise SystemExit("需要 timm: Kaggle 已自带; 本地请 pip install timm")

# ======================== 可调配置 ========================
# fast: 约 1h(T4) — 单骨干 3 折; full: 约 2–4h — 双骨干 5 折
RUN_MODE = os.environ.get("PLANT_RUN_MODE", "full").lower()

if RUN_MODE == "fast":
    BACKBONES = ["convnext_tiny.fb_in22k_ft_in1k"]
    N_FOLDS = 3
    STAGE1_EPOCHS = 2
    STAGE2_EPOCHS = 8
else:
    BACKBONES = [
        "convnext_tiny.fb_in22k_ft_in1k",
        "tf_efficientnetv2_s.in1k",
    ]
    N_FOLDS = 5
    STAGE1_EPOCHS = 3
    STAGE2_EPOCHS = 12

IMG_SIZE = 288
BATCH_SIZE = 24 if IMG_SIZE >= 288 else 32
LR_HEAD = 8e-4
LR_BACKBONE = 1.5e-5
WEIGHT_DECAY = 1e-2
LABEL_SMOOTH = 0.08
MIXUP_ALPHA = 0.35
CUTMIX_ALPHA = 0.8
MIXUP_PROB = 0.5
EMA_DECAY = 0.995
SEED = 42
# Kaggle 上 num_workers>0 常卡在 DataLoader 子进程，表现为 GPU/CPU 都几乎为 0
NUM_WORKERS = 0 if os.path.isdir("/kaggle/input") else 2


def find_data_root():
    env = os.environ.get("PLANT_DATA_ROOT")
    if env and os.path.isdir(os.path.join(env, "train")):
        return os.path.abspath(env)
    for root, dirs, _files in os.walk("/kaggle/input"):
        if "train" in dirs and "test" in dirs:
            return os.path.abspath(root)
    local = r"c:\Users\lenovo\Desktop\dataset-for-task2\dataset-for-task2"
    if os.path.isdir(os.path.join(local, "train")):
        return local
    return None


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.06),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


class EvalTransform:
    """可复现 TTA: resize → center crop → (可选 hflip) → normalize."""

    def __init__(self, size, hflip=False, scale=1.0):
        self.size = size
        self.hflip = hflip
        self.scale = scale

    def __call__(self, img: Image.Image):
        side = int(round(self.size * self.scale * 1.12))
        img = transforms.Resize(side)(img)
        img = transforms.CenterCrop(self.size)(img)
        if self.hflip:
            img = TF.hflip(img)
        t = transforms.ToTensor()(img)
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t)


def tta_transform_list(size):
    views = []
    for scale in (1.0, 1.08):
        for hflip in (False, True):
            views.append(EvalTransform(size, hflip=hflip, scale=scale))
    return views


class TrainSubset(Dataset):
    def __init__(self, folder, transform, indices):
        self.base = datasets.ImageFolder(folder)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.base.samples[self.indices[i]]
        return self.transform(Image.open(path).convert("RGB")), label



class FlatImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.folder, self.names[i])).convert("RGB")
        if callable(self.transform):
            return self.transform(img)
        return self.transform(img)


class ModelEma:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)


def mixup_batch(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[idx]
    return mixed, y, y[idx], lam


def cutmix_batch(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[idx]
    _, _, h, w = x.size()
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x, y_a, y_b, lam


def soft_ce(logits, targets, n_class, smoothing, weight=None):
    with torch.no_grad():
        true = torch.zeros_like(logits)
        true.fill_(smoothing / (n_class - 1))
        true.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    logp = F.log_softmax(logits, dim=1)
    loss = -(true * logp).sum(dim=1)
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def mixed_loss(logits, y_a, y_b, lam, n_class, smoothing, weight=None):
    return lam * _one_hot_ce(logits, y_a, n_class, smoothing, weight) + (
        1.0 - lam
    ) * _one_hot_ce(logits, y_b, n_class, smoothing, weight)


def _one_hot_ce(logits, targets, n_class, smoothing, weight):
    with torch.no_grad():
        true = torch.zeros_like(logits)
        true.fill_(smoothing / (n_class - 1))
        true.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    logp = F.log_softmax(logits, dim=1)
    loss = -(true * logp).sum(dim=1)
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def make_model(name, n_class):
    print(f"  加载 timm 骨干 {name} (首次会下载权重，需开启 Internet)...", flush=True)
    m = timm.create_model(name, pretrained=True, num_classes=n_class)
    return m


def param_groups(model):
    head_keys = ("head.", "classifier", "fc.")
    head, body = [], []
    for n, p in model.named_parameters():
        if n.startswith("head.") or n.startswith("classifier") or ".fc." in n or n in ("fc.weight", "fc.bias"):
            head.append(p)
        else:
            body.append(p)
    if not head:
        ps = list(model.parameters())
        head, body = ps[-2:], ps[:-2]
    return head, body


def train_fold(
    train_loader,
    device,
    n_class,
    class_weight,
    backbone_name,
):
    model = make_model(backbone_name, n_class).to(device)
    ema = ModelEma(model)
    cw = torch.tensor(class_weight, dtype=torch.float32, device=device)

    head, body = param_groups(model)
    for p in body:
        p.requires_grad = False
    for p in head:
        p.requires_grad = True

    opt1 = torch.optim.AdamW(head, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS)

    for _ in range(STAGE1_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt1.zero_grad()
            loss = soft_ce(model(x), y, n_class, LABEL_SMOOTH, cw)
            loss.backward()
            opt1.step()
            ema.update(model)
        sch1.step()

    for p in model.parameters():
        p.requires_grad = True
    head, body = param_groups(model)
    opt2 = torch.optim.AdamW(
        [
            {"params": head, "lr": LR_HEAD * 0.5},
            {"params": body, "lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS)

    for _ in range(STAGE2_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt2.zero_grad()
            if np.random.rand() < MIXUP_PROB:
                if np.random.rand() < 0.5:
                    x, y_a, y_b, lam = mixup_batch(x, y, MIXUP_ALPHA)
                else:
                    x, y_a, y_b, lam = cutmix_batch(x, y, CUTMIX_ALPHA)
                loss = mixed_loss(model(x), y_a, y_b, lam, n_class, LABEL_SMOOTH, cw)
            else:
                loss = soft_ce(model(x), y, n_class, LABEL_SMOOTH, cw)
            loss.backward()
            opt2.step()
            ema.update(model)
        sch2.step()

    return ema.shadow


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    sm = nn.Softmax(dim=1)
    chunks = []
    for x in loader:
        chunks.append(sm(model(x.to(device))).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def predict_tta_train_subset(model, folder, indices, device):
    base = datasets.ImageFolder(folder)
    views = tta_transform_list(IMG_SIZE)
    prob_sum = None
    for tf in views:

        class _DS(Dataset):
            def __len__(self):
                return len(indices)

            def __getitem__(self, i):
                path = base.samples[indices[i]][0]
                return tf(Image.open(path).convert("RGB"))

        loader = DataLoader(_DS(), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        probs = predict_loader(model, loader, device)
        prob_sum = probs if prob_sum is None else prob_sum + probs
    return prob_sum / len(views)


def run_backbone(backbone_name, train_folder, test_folder, y, n_class, classes, device):
    print(f"\n{'=' * 60}\n骨干: {backbone_name}\n{'=' * 60}")
    train_tf = build_train_transform(IMG_SIZE)
    class_weight = compute_class_weight("balanced", classes=np.arange(n_class), y=y).astype(np.float32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_total = len(y)
    oof_probs = np.zeros((n_total, n_class), dtype=np.float32)

    test_plain = FlatImageDataset(test_folder, EvalTransform(IMG_SIZE))
    test_probs = np.zeros((len(test_plain), n_class), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n_total), y), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS}  train={len(tr_idx)} val={len(va_idx)} ---")
        tr_loader = DataLoader(
            TrainSubset(train_folder, train_tf, tr_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=len(tr_idx) > BATCH_SIZE,
        )
        model = train_fold(tr_loader, device, n_class, class_weight, backbone_name)

        va_p = predict_tta_train_subset(model, train_folder, va_idx, device)
        oof_probs[va_idx] = va_p.numpy()
        pred = va_p.argmax(1).numpy()
        print(
            f"Fold {fold} val acc={accuracy_score(y[va_idx], pred):.4f}  "
            f"val macroF1={f1_score(y[va_idx], pred, average='macro', zero_division=0):.4f}"
        )

        # test TTA
        tta_probs = predict_tta_multi(model, test_folder, device)
        test_probs += tta_probs.numpy()

    oof_pred = oof_probs.argmax(1)
    print(
        f"\n[{backbone_name}] OOF acc={accuracy_score(y, oof_pred):.4f}  "
        f"OOF macroF1={f1_score(y, oof_pred, average='macro', zero_division=0):.4f}"
    )
    print("混淆矩阵:\n", confusion_matrix(y, oof_pred))

    test_probs /= N_FOLDS
    return oof_probs, test_probs


@torch.no_grad()
def predict_tta_multi(model, test_folder, device):
    views = tta_transform_list(IMG_SIZE)
    total = None
    n = len([f for f in os.listdir(test_folder) if f.lower().endswith(".png")])
    for tf in views:
        ds = FlatImageDataset(test_folder, tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        probs = predict_loader(model, loader, device)
        total = probs if total is None else total + probs
    assert len(ds) == n
    return total / len(views)


def main():
    print("[cursor] main() 开始", flush=True)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = find_data_root()
    if data_root is None:
        raise SystemExit("未找到 train/test 目录。Kaggle 请挂载数据集; 本地请设置 PLANT_DATA_ROOT。")

    train_folder = os.path.join(data_root, "train")
    test_folder = os.path.join(data_root, "test")
    base = datasets.ImageFolder(train_folder)
    classes = base.classes
    n_class = len(classes)
    y = np.array([s[1] for s in base.samples])

    print("数据:", data_root, flush=True)
    print("device:", device, "| RUN_MODE:", RUN_MODE, "| 折数:", N_FOLDS, "| IMG:", IMG_SIZE, flush=True)
    print("类别:", classes, "| 样本:", len(y), "| 骨干:", BACKBONES, flush=True)
    if device.type == "cpu":
        print("警告: 未检测到 CUDA，请在 Notebook 右侧 Session options 选 GPU 后 Restart Session。", flush=True)

    ensemble_test = None
    oof_list = []

    for name in BACKBONES:
        oof_p, test_p = run_backbone(name, train_folder, test_folder, y, n_class, classes, device)
        oof_list.append(oof_p)
        ensemble_test = test_p if ensemble_test is None else ensemble_test + test_p

    ensemble_test /= len(BACKBONES)
    oof_avg = np.mean(np.stack(oof_list, axis=0), axis=0)
    oof_pred = oof_avg.argmax(1)
    print(
        f"\n===== 多骨干集成 OOF (概率平均) =====\n"
        f"OOF acc={accuracy_score(y, oof_pred):.4f}  "
        f"OOF macroF1={f1_score(y, oof_pred, average='macro', zero_division=0):.4f}"
    )
    print("集成混淆矩阵:\n", confusion_matrix(y, oof_pred))

    test_ds = FlatImageDataset(test_folder, EvalTransform(IMG_SIZE))
    test_pred = ensemble_test.argmax(1)
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "submission_advanced_ensemble.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        fp.write("ID,Category\n")
        for fname, pidx in zip(test_ds.names, test_pred):
            fp.write(f"{fname},{classes[pidx]}\n")

    print("\n已写入:", out_path)
    print("test 预测分布:", dict(Counter(classes[i] for i in test_pred)))
    print("提示: OOF macroF1 通常接近线上 meanF1; 若 OOF≈0.88+ 再提交更有机会到 0.9。")


# Notebook 粘贴整文件时务必执行 main（不要只复制文件末尾几行）
if os.path.isdir("/kaggle/working"):
    main()
elif __name__ == "__main__":
    main()
