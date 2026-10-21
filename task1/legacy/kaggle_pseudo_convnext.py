# -*- coding: utf-8 -*-
"""
植物幼苗 5 分类 —— ConvNeXt-Small + 伪标签自训练 (Pseudo-Labeling), Kaggle 单文件版
在已验证 0.92 的 ConvNeXt-Small(320) 配方上, 加两阶段半监督:
  阶段A: 原始 500 张 5 折训练 -> 对 test 做 TTA 预测 -> 挑高置信(>CONF_THR)样本作伪标签
  阶段B: 伪标签图并入训练集 -> 重新 5 折训练 -> 概率平均出最终提交
目的: 把最难分的 Black-grass / Loose Silky-bent 两类的 test 图"收编"进训练, 扩样本冲分
输出: /kaggle/working/submission_pseudo.csv
"""
from __future__ import annotations

import os
import sys
import random
import copy
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
print("[pseudo] 脚本已加载，正在 import 依赖...", flush=True)

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
BACKBONE = "convnext_small.fb_in22k_ft_in1k"
N_FOLDS_A = 5          # 阶段A折数(生成伪标签)
N_FOLDS_B = 5          # 阶段B折数
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 16
CONF_THR = 0.90        # 伪标签置信度阈值, 越高越稳

IMG_SIZE = 320
BATCH_SIZE = 16
LR_HEAD = 8e-4
LR_BACKBONE = 1.5e-5
WEIGHT_DECAY = 1e-2
LABEL_SMOOTH = 0.08
MIXUP_ALPHA = 0.35
CUTMIX_ALPHA = 0.8
MIXUP_PROB = 0.5
EMA_DECAY = 0.995
SEED = 42
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
        return self.transform(img)


class PseudoWrapper(Dataset):
    """原始训练折 + 伪标签 test 图 合并成一个数据集."""
    def __init__(self, train_folder, test_folder, transform, train_indices, pseudo):
        self.base = datasets.ImageFolder(train_folder)
        self.train_indices = list(train_indices)
        self.pseudo = pseudo  # list of (test_filename, class_idx)
        self.transform = transform
        self.test_folder = test_folder

    def __len__(self):
        return len(self.train_indices) + len(self.pseudo)

    def __getitem__(self, i):
        if i < len(self.train_indices):
            path, label = self.base.samples[self.train_indices[i]]
            return self.transform(Image.open(path).convert("RGB")), label
        j = i - len(self.train_indices)
        name, label = self.pseudo[j]
        return self.transform(Image.open(os.path.join(self.test_folder, name)).convert("RGB")), label


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
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


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


def _one_hot_ce(logits, targets, n_class, smoothing, weight=None):
    with torch.no_grad():
        true = torch.zeros_like(logits)
        true.fill_(smoothing / (n_class - 1))
        true.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    logp = F.log_softmax(logits, dim=1)
    loss = -(true * logp).sum(dim=1)
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def soft_ce(logits, targets, n_class, smoothing, weight=None):
    return _one_hot_ce(logits, targets, n_class, smoothing, weight)


def mixed_loss(logits, y_a, y_b, lam, n_class, smoothing, weight=None):
    return lam * _one_hot_ce(logits, y_a, n_class, smoothing, weight) + (
        1.0 - lam
    ) * _one_hot_ce(logits, y_b, n_class, smoothing, weight)


def make_model(name, n_class):
    m = timm.create_model(name, pretrained=True, num_classes=n_class)
    return m


def param_groups(model):
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


def train_fold(train_loader, device, n_class, class_weight):
    model = make_model(BACKBONE, n_class).to(device)
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
        [{"params": head, "lr": LR_HEAD * 0.5}, {"params": body, "lr": LR_BACKBONE}],
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
                return tf(Image.open(base.samples[indices[i]][0]).convert("RGB"))
        loader = DataLoader(_DS(), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        probs = predict_loader(model, loader, device)
        prob_sum = probs if prob_sum is None else prob_sum + probs
    return prob_sum / len(views)


@torch.no_grad()
def predict_tta_test(model, test_folder, device):
    views = tta_transform_list(IMG_SIZE)
    total = None
    for tf in views:
        ds = FlatImageDataset(test_folder, tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        probs = predict_loader(model, loader, device)
        total = probs if total is None else total + probs
    return total / len(views)


def main():
    print("[pseudo] main() 开始", flush=True)
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
    n_total = len(y)
    print("数据:", data_root, "| device:", device, "| 骨干:", BACKBONE, "| IMG:", IMG_SIZE, flush=True)
    print("类别:", classes, "| 样本:", n_total, flush=True)

    train_tf = build_train_transform(IMG_SIZE)
    test_ds = FlatImageDataset(test_folder, EvalTransform(IMG_SIZE))
    n_test = len(test_ds)
    class_weight = compute_class_weight("balanced", classes=np.arange(n_class), y=y).astype(np.float32)

    # ---------- 阶段A: 生成伪标签 ----------
    print("\n===== 阶段A: 原始数据 5 折, 估计 test 置信度 =====", flush=True)
    skfA = StratifiedKFold(n_splits=N_FOLDS_A, shuffle=True, random_state=SEED)
    test_probs = torch.zeros((n_test, n_class))
    for fold, (tr_idx, va_idx) in enumerate(skfA.split(np.arange(n_total), y), 1):
        tr_loader = DataLoader(TrainSubset(train_folder, train_tf, tr_idx),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=True, drop_last=len(tr_idx) > BATCH_SIZE)
        model = train_fold(tr_loader, device, n_class, class_weight)
        test_probs += predict_tta_test(model, test_folder, device)
        va_p = predict_tta_train_subset(model, train_folder, va_idx, device)
        print(f"[A Fold {fold}] val macroF1={f1_score(y[va_idx], va_p.argmax(1).numpy(), average='macro', zero_division=0):.4f}", flush=True)
    test_probs /= N_FOLDS_A
    conf = test_probs.max(1).values.numpy()
    plabel = test_probs.argmax(1).numpy()
    mask = conf >= CONF_THR
    pseudo = [(test_ds.names[i], int(plabel[i])) for i in np.where(mask)[0]]
    print(f"\n高置信伪标签: {len(pseudo)}/{n_test} 张 (阈值 {CONF_THR})", flush=True)
    if pseudo:
        print("伪标签类别分布:", dict(Counter(classes[p] for _, p in pseudo)), flush=True)
    if len(pseudo) == 0:
        print("无样本达标, 建议调低 CONF_THR(如0.80)。", flush=True)
        return

    # ---------- 阶段B: 并入伪标签重训 ----------
    print("\n===== 阶段B: 合并伪标签重训 =====", flush=True)
    skfB = StratifiedKFold(n_splits=N_FOLDS_B, shuffle=True, random_state=SEED)
    test_probs2 = torch.zeros((n_test, n_class))
    oof_probs = np.zeros((n_total, n_class), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skfB.split(np.arange(n_total), y), 1):
        ds = PseudoWrapper(train_folder, test_folder, train_tf, tr_idx, pseudo)
        tr_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=True, drop_last=True)
        model = train_fold(tr_loader, device, n_class, class_weight)
        test_probs2 += predict_tta_test(model, test_folder, device)
        va_p = predict_tta_train_subset(model, train_folder, va_idx, device)
        oof_probs[va_idx] = va_p.numpy()
        print(f"[B Fold {fold}] val macroF1={f1_score(y[va_idx], va_p.argmax(1).numpy(), average='macro', zero_division=0):.4f}", flush=True)

    test_probs2 /= N_FOLDS_B
    oof_pred = oof_probs.argmax(1)
    print(f"\n===== 阶段B OOF =====\nOOF acc={accuracy_score(y, oof_pred):.4f}  "
          f"OOF macroF1={f1_score(y, oof_pred, average='macro', zero_division=0):.4f}", flush=True)

    test_pred = test_probs2.argmax(1)
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "submission_pseudo.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        fp.write("ID,Category\n")
        for fname, pidx in zip(test_ds.names, test_pred):
            fp.write(f"{fname},{classes[pidx]}\n")
    print("\n已写入:", out_path)
    print("test 预测分布:", dict(Counter(classes[i] for i in test_pred)), flush=True)
    print("提示: 看阶段B OOF macroF1, 若 > 阶段A 则伪标签起效。", flush=True)


if os.path.isdir("/kaggle/working"):
    main()
elif __name__ == "__main__":
    main()
