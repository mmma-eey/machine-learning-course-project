# -*- coding: utf-8 -*-
"""
植物 5 分类 —— OOF 概率 stacking (Kaggle 单文件)
路径: plant_classifier/cursor/ (Cursor Agent 方案)
思路: 两个差异较大的 timm 骨干分别 5 折出 OOF softmax,
      用 LogisticRegression 在 OOF 上学习融合权重, 再融合 test 概率.
      往往比简单平均再高 0.01~0.03 macro-F1.

用法同 kaggle_advanced_ensemble.py; 输出 submission_meta_stack.csv
RUN_MODE=fast 时 3 折 + 单骨干, 约 40 分钟 (T4).
"""
from __future__ import annotations

import os
import sys
import random
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
print("[cursor] meta_stacking 已加载...", flush=True)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

import timm

RUN_MODE = os.environ.get("PLANT_RUN_MODE", "full").lower()
BACKBONES = (
    ["convnext_tiny.fb_in22k_ft_in1k", "tf_efficientnetv2_s.in1k"]
    if RUN_MODE == "full"
    else ["convnext_tiny.fb_in22k_ft_in1k"]
)
N_FOLDS = 5 if RUN_MODE == "full" else 3
IMG_SIZE = 256
BATCH_SIZE = 32
STAGE1_EPOCHS = 3 if RUN_MODE == "full" else 2
STAGE2_EPOCHS = 10 if RUN_MODE == "full" else 6
LR_HEAD, LR_BACK = 1e-3, 2e-5
WEIGHT_DECAY = 1e-2
SEED = 42
NUM_WORKERS = 0 if os.path.isdir("/kaggle/input") else 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_data_root():
    env = os.environ.get("PLANT_DATA_ROOT")
    if env and os.path.isdir(os.path.join(env, "train")):
        return os.path.abspath(env)
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "train" in dirs and "test" in dirs:
            return os.path.abspath(root)
    local = r"c:\Users\lenovo\Desktop\dataset-for-task2\dataset-for-task2"
    return local if os.path.isdir(os.path.join(local, "train")) else None


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EvalTransform:
    def __init__(self, size, hflip=False):
        self.size = size
        self.hflip = hflip

    def __call__(self, img):
        img = transforms.Resize(int(self.size * 1.12))(img)
        img = transforms.CenterCrop(self.size)(img)
        if self.hflip:
            img = TF.hflip(img)
        t = transforms.ToTensor()(img)
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t)


def train_tf(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TrainSubset(Dataset):
    def __init__(self, folder, transform, indices):
        self.base = datasets.ImageFolder(folder)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, y = self.base.samples[self.indices[i]]
        return self.transform(Image.open(path).convert("RGB")), y


class FlatTest(Dataset):
    def __init__(self, folder, transform):
        self.names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.folder, self.names[i])).convert("RGB")
        return self.transform(img)


def make_model(name, n_class):
    return timm.create_model(name, pretrained=True, num_classes=n_class)


def classifier_parameters(model):
    if hasattr(model, "get_classifier"):
        return model.get_classifier().parameters()
    head, _body = [], []
    for n, p in model.named_parameters():
        if n.startswith("head.") or n.startswith("classifier") or ".fc." in n:
            head.append(p)
    return head if head else [list(model.parameters())[-1]]


def train_one_fold(loader, device, n_class, backbone):
    model = make_model(backbone, n_class).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    for p in model.parameters():
        p.requires_grad = False
    for p in classifier_parameters(model):
        p.requires_grad = True
    opt = torch.optim.AdamW(classifier_parameters(model), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STAGE1_EPOCHS)
    for _ in range(STAGE1_EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sch.step()

    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=LR_BACK, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STAGE2_EPOCHS)
    for _ in range(STAGE2_EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sch.step()
    return model


@torch.no_grad()
def infer_probs(model, folder, device, indices=None, test=False):
    model.eval()
    sm = nn.Softmax(dim=1)
    views = [EvalTransform(IMG_SIZE, False), EvalTransform(IMG_SIZE, True)]
    if test:
        prob = None
        for tf in views:
            ds = FlatTest(folder, tf)
            ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            chunks = []
            for x in ld:
                chunks.append(sm(model(x.to(device))).cpu())
            p = torch.cat(chunks)
            prob = p if prob is None else prob + p
        return prob / len(views)

    base = datasets.ImageFolder(folder)
    prob = None
    for tf in views:

        class DS(Dataset):
            def __len__(self):
                return len(indices)

            def __getitem__(self, i):
                path = base.samples[indices[i]][0]
                return tf(Image.open(path).convert("RGB"))

        ld = DataLoader(DS(), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        chunks = []
        for x in ld:
            chunks.append(sm(model(x.to(device))).cpu())
        p = torch.cat(chunks)
        prob = p if prob is None else prob + p
    return prob / len(views)


def collect_oof(backbone, train_folder, test_folder, y, n_class, device):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), n_class), dtype=np.float32)
    test_sum = None
    tf = train_tf(IMG_SIZE)
    for fold, (tr, va) in enumerate(skf.split(np.arange(len(y)), y), 1):
        print(f"  [{backbone}] fold {fold}/{N_FOLDS}")
        ld = DataLoader(
            TrainSubset(train_folder, tf, tr),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        model = train_one_fold(ld, device, n_class, backbone)
        oof[va] = infer_probs(model, train_folder, device, indices=va).numpy()
        tp = infer_probs(model, test_folder, device, test=True)
        test_sum = tp.numpy() if test_sum is None else test_sum + tp.numpy()
    test_avg = test_sum / N_FOLDS
    return oof, test_avg


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_data_root()
    if not root:
        raise SystemExit("找不到数据目录")
    train_folder = os.path.join(root, "train")
    test_folder = os.path.join(root, "test")
    base = datasets.ImageFolder(train_folder)
    classes = base.classes
    n_class = len(classes)
    y = np.array([s[1] for s in base.samples])
    print("device:", device, "RUN_MODE:", RUN_MODE, "backbones:", BACKBONES)

    oof_parts, test_parts = [], []
    for bb in BACKBONES:
        print("\n训练骨干:", bb)
        oof, tprob = collect_oof(bb, train_folder, test_folder, y, n_class, device)
        oof_parts.append(oof)
        test_parts.append(tprob)
        print(
            f"  OOF macroF1={f1_score(y, oof.argmax(1), average='macro', zero_division=0):.4f}"
        )

    X_oof = np.hstack(oof_parts)
    X_test = np.hstack(test_parts)
    meta = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", solver="lbfgs"),
    )
    meta.fit(X_oof, y)
    oof_meta = meta.predict(X_oof)
    print(
        f"\nStacking OOF acc={accuracy_score(y, oof_meta):.4f}  "
        f"macroF1={f1_score(y, oof_meta, average='macro', zero_division=0):.4f}"
    )
    print("混淆矩阵:\n", confusion_matrix(y, oof_meta))

    test_pred = meta.predict(X_test)
    names = FlatTest(test_folder, EvalTransform(IMG_SIZE)).names
    out = os.path.join(
        "/kaggle/working" if os.path.isdir("/kaggle/working") else ".",
        "submission_meta_stack.csv",
    )
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("ID,Category\n")
        for nm, pi in zip(names, test_pred):
            f.write(f"{nm},{classes[pi]}\n")
    print("已写入:", out)
    print("test 分布:", dict(Counter(classes[i] for i in test_pred)))


if os.path.isdir("/kaggle/working"):
    main()
elif __name__ == "__main__":
    main()
