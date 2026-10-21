# -*- coding: utf-8 -*-
"""
干净训练集(1713张, 原竞赛5类剔除与课设test重合图) 5折CV 训练 ConvNeXt-Small@384，
预测课设 test(378张)，输出 Kaggle 提交 CSV。

数据来源(两个并行文件夹, 非父子):
  - task1-extra/clean_dataset/train/  : 训练集(5 类子目录, 1713 张)
  - 2026task1/dataset-for-task2/test/  : 课设测试集(378 张平铺 .png)

脚本会递归扫描 /kaggle/input 全树自动定位:
  - 训练集取"含 5 类子目录且图片总数最多"者(clean 1713 > 原 500)
  - 测试集取"非类别名的叶子目录, 300-400 张平铺 png"

用法: 在 Kaggle Notebook 中粘入本文件全部内容运行(挂载数据集, 加速器 GPU)。
输出: /kaggle/working/submission_clean_384.csv
"""
import os
import glob
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import timm

# ================= 配置 =================
SEED = 42
IMG_SIZE = 384             # 384 分辨率: 利于 Black-grass / Loose Silky-bent 细部纹理区分
BATCH_SIZE = 16            # T4 x2 显存充足, 384 下 16 无压力
N_FOLDS = 5
STAGE1_EPOCHS = 4          # 阶段1: 只训分类头
STAGE2_EPOCHS = 12         # 阶段2: 全量微调
LR_HEAD = 1e-3
LR_FT = 5e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
EMA_DECAY = 0.999
BACKBONE = "convnext_small.fb_in22k_ft_in1k_384"
VAL_RATIO = 0.10           # 大胆: 验证集仅 10% (每折 90/10)

CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent",
           "Scentless Mayweed", "Sugar beet"]
N_CLASS = len(CLASSES)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def find_data():
    """在 /kaggle/input 全树内定位, 不依赖具体挂载路径。
    - train: 含 5 个类别子目录的文件夹; 有多个时取图片总数最多的(clean 1713 > 原 500)
    - test : 非类别名的叶子目录, 约 300-400 张平铺 .png
    """
    cls_set = set(CLASSES)
    train_cands = []   # (total_pngs, root)
    test_cands = []    # (is_named_test, pngs, root)

    def _png_count(files):
        return sum(1 for f in files if f.lower().endswith(".png"))

    for root, dirs, files in os.walk("/kaggle/input"):
        # 训练集候选: 同时含 5 个类别子目录
        if len(set(dirs) & cls_set) == 5:
            total = 0
            for cname in CLASSES:
                cdir = os.path.join(root, cname)
                try:
                    total += len([f for f in os.listdir(cdir)
                                  if f.lower().endswith(".png")])
                except OSError:
                    pass
            train_cands.append((total, root))
        # 测试集候选: 平铺图片的叶子目录(排除类别名子目录)
        if not dirs:
            name = os.path.basename(root)
            if name in cls_set:
                continue
            n = _png_count(files)
            if 300 <= n <= 400:
                test_cands.append((name.lower() == "test", n, root))

    train_cands.sort(key=lambda x: x[0], reverse=True)
    test_cands.sort(key=lambda x: (not x[0], -x[1]))

    if train_cands:
        print("[find_data] 训练集候选(图片数, 路径):",
              [(c, p) for c, p in train_cands], flush=True)
    if test_cands:
        print("[find_data] 测试集候选(图片数, 路径):",
              [(c, p) for _, c, p in test_cands], flush=True)

    train_root = train_cands[0][1] if train_cands else None
    test_root = test_cands[0][2] if test_cands else None
    return train_root, test_root


# ================= 数据增强 =================
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])
val_tf = T.Compose([
    T.Resize(int(IMG_SIZE * 1.15)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


class TestDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.files = sorted([f for f in os.listdir(root)
                             if f.lower().endswith(".png")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = os.path.join(self.root, self.files[i])
        img = Image.open(p).convert("RGB")
        return self.transform(img), self.files[i]


# ================= Mixup =================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


# ================= EMA =================
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])


def build_model():
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=N_CLASS)
    return model


def set_trainable(model, backbone, head):
    for n, p in model.named_parameters():
        is_head = n.startswith("head.") or n.startswith("classifier.") or n.startswith("fc.")
        p.requires_grad = head if is_head else backbone


def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for c in range(N_CLASS):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def run_epoch(model, loader, opt, mixup, device):
    model.train()
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if mixup:
            x, ya, yb, lam = mixup_data(x, y)
            opt.zero_grad()
            logits = model(x)
            loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
        else:
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    probs = []
    ids = []
    for x, *rest in loader:
        x = x.to(device)
        p = F.softmax(model(x), dim=1)
        if tta:
            px = T.functional.hflip(x)
            p = (p + F.softmax(model(px), dim=1)) / 2.0
        probs.append(p.cpu())
        if rest:
            ids.extend(rest[0])
    return torch.cat(probs, dim=0).numpy(), ids


def train_fold(model, train_loader, val_loader, device):
    # 阶段1: 冻结骨干, 只训分类头
    set_trainable(model, backbone=False, head=True)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS)
    for _ in range(STAGE1_EPOCHS):
        run_epoch(model, train_loader, opt1, mixup=False, device=device)
        sched1.step()

    # 阶段2: 全量微调
    set_trainable(model, backbone=True, head=True)
    opt2 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_FT, weight_decay=WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS)
    ema = EMA(model)
    best_f1 = 0.0
    best_state = None
    for ep in range(STAGE2_EPOCHS):
        run_epoch(model, train_loader, opt2, mixup=True, device=device)
        ema.update(model)
        sched2.step()
        ema.apply_shadow(model)
        p, _ = predict(model, val_loader, device, tta=True)
        ema.restore(model)
        pred = p.argmax(1)
        yt = np.array(val_loader.dataset.targets)
        f1 = macro_f1(yt, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in ema.shadow.items()}
    # 用最优 EMA 状态
    for n, p in model.named_parameters():
        if p.requires_grad and n in best_state:
            p.data.copy_(best_state[n].to(p.device))
    return model, best_f1


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    train_root, test_root = find_data()
    if train_root is None or test_root is None:
        raise SystemExit(
            "未找到训练/测试目录。请确认已挂载含 clean_dataset/train 与课设 test 的数据集。")
    print(f"训练集: {train_root}", flush=True)
    print(f"测试集: {test_root}", flush=True)

    ds = ImageFolder(train_root, transform=train_tf)
    y = np.array(ds.targets)
    assert ds.classes == CLASSES, f"类别顺序不一致: {ds.classes} vs {CLASSES}"
    print(f"训练总样本: {len(ds)} | 类别: {ds.classes}", flush=True)

    test_ds = TestDataset(test_root, val_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=False)
    print(f"测试样本: {len(test_ds)}", flush=True)

    skf = StratifiedKFold(n_splits=int(1 / VAL_RATIO), shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(ds)), y))[:N_FOLDS]  # 每折 90/10, 取前 5 折

    oof_prob = np.zeros((len(ds), N_CLASS), dtype=np.float32)
    oof_count = np.zeros(len(ds), dtype=np.int32)
    test_prob = np.zeros((len(test_ds), N_CLASS), dtype=np.float32)

    for fi, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n===== Fold {fi+1}/{N_FOLDS} train={len(tr_idx)} val={len(va_idx)} =====", flush=True)
        tr_ds = torch.utils.data.Subset(ds, tr_idx)
        # 验证集用 val 变换重新构造 Dataset
        from torch.utils.data import Dataset as _DS
        class _ValDS(_DS):
            def __init__(self, base, idx, tf):
                self.base = base
                self.idx = idx
                self.tf = tf
                self.targets = [base.targets[i] for i in idx]
            def __len__(self):
                return len(self.idx)
            def __getitem__(self, i):
                p, _ = self.base.samples[self.idx[i]]
                img = Image.open(p).convert("RGB")
                return self.tf(img), self.targets[i]
        va_ds2 = _ValDS(ds, va_idx, val_tf)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=False)
        va_loader = DataLoader(va_ds2, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=False)

        model = build_model().to(device)
        model, best_f1 = train_fold(model, tr_loader, va_loader, device)
        print(f"Fold {fi+1} 最佳 val macroF1={best_f1:.4f}", flush=True)

        # OOF 概率
        p, _ = predict(model, va_loader, device, tta=True)
        oof_prob[va_idx] += p
        oof_count[va_idx] += 1
        # test 概率
        tp, _ = predict(model, test_loader, device, tta=True)
        test_prob += tp

    # OOF 评估(覆盖 50% 数据, 每张被预测1次)
    mask = oof_count > 0
    oof_pred = oof_prob[mask].argmax(1)
    oof_true = y[mask]
    oof_acc = float((oof_pred == oof_true).mean())
    oof_f1 = macro_f1(oof_true, oof_pred)
    print(f"\n===== OOF (覆盖 {mask.sum()}/{len(ds)} 张) =====", flush=True)
    print(f"OOF acc={oof_acc:.4f}  OOF macroF1={oof_f1:.4f}", flush=True)

    # 生成提交
    test_pred = test_prob.argmax(1)
    test_ids = test_ds.files
    out_path = "/kaggle/working/submission_clean_384.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ID,Category\n")
        for fid, c in zip(test_ids, test_pred):
            f.write(f"{fid},{CLASSES[c]}\n")
    print(f"\n已写入: {out_path}", flush=True)
    from collections import Counter
    print("test 预测分布:", dict(Counter(CLASSES[c] for c in test_pred)), flush=True)


if __name__ == "__main__":
    main()
