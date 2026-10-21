# -*- coding: utf-8 -*-
"""
Black-grass vs Loose Silky-bent 二分类仲裁器 (Kaggle 单文件版)。

目的:
  主 5 分类模型里, 大部分错误都集中在 Black-grass / Loose Silky-bent 这对形态极像的草。
  本脚本专门只拿这两类(clean 数据)训一个高分辨率二分类器, 输出每张课设 test 的
  P(Black-grass) / P(Loose Silky-bent), 用来"仲裁"主模型在这两类边界上的难样本。

数据:
  - 训练: clean_dataset/train/ 下的 Black-grass(222) + Loose Silky-bent(547) = 769 张
  - 测试: 2026task1/dataset-for-task2/test/ 的 378 张平铺 png
  (自动递归定位, 优先选路径含 "clean" 的训练文件夹, 避免误用未清洗的原竞赛集)

关键设计:
  - 逆频率类别权重: 校正 LSB/BG = 2.5x 的训练不平衡(课设 test 上两类≈1:1),
    避免复现 384 模型"系统性偏向 LSB"的问题
  - 5 折 StratifiedKFold(90/10) + Mixup + Label Smoothing + EMA + 水平翻转 TTA
  - OOF 同时报 accuracy(=微观F1)、BG/LSB 各自 recall、2x2 混淆矩阵

用法: 整文件粘到 Kaggle Notebook(挂载数据集, GPU, 开 Internet), 运行。
输出:
  - /kaggle/working/binary_test_probs.csv   (ID, P_Black-grass, P_Loose Silky-bent)
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from collections import Counter
import timm

# ================= 配置 =================
SEED = 42
IMG_SIZE = 384            # 想更激进可改 512 (约 1.5x 耗时)
BATCH_SIZE = 16
N_FOLDS = 5
STAGE1_EPOCHS = 4          # 阶段1: 只训二分类头
STAGE2_EPOCHS = 8          # 阶段2: 全量微调
LR_HEAD = 1e-3
LR_FT = 5e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
EMA_DECAY = 0.999
BACKBONE = "convnext_small.fb_in22k_ft_in1k_384"
VAL_RATIO = 0.10           # 每折 90/10

CLASSES = ["Black-grass", "Loose Silky-bent"]   # 字母序: BG=0, LSB=1
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def find_binary_data():
    """定位含 Black-grass + Loose Silky-bent 两个子目录的文件夹; 优先路径含 'clean'。
    返回 (train_root, test_root)。"""
    targets = {"Black-grass", "Loose Silky-bent"}
    train_cands = []
    test_cands = []

    def png_count(files):
        return sum(1 for f in files if f.lower().endswith(".png"))

    for root, dirs, files in os.walk("/kaggle/input"):
        # 训练集候选: 同时含两类子目录
        if targets.issubset(set(dirs)):
            total = 0
            for cname in CLASSES:
                cdir = os.path.join(root, cname)
                try:
                    total += len([f for f in os.listdir(cdir)
                                  if f.lower().endswith(".png")])
                except OSError:
                    pass
            train_cands.append(("clean" in root.lower(), total, root))
        # 测试集候选: 平铺 png 的叶子目录(排除类别名)
        if not dirs:
            name = os.path.basename(root)
            if name in targets or name in ("Black-grass", "Loose Silky-bent"):
                continue
            n = png_count(files)
            if 300 <= n <= 400:
                test_cands.append((name.lower() == "test", n, root))

    # clean 优先, 其次图片多
    train_cands.sort(key=lambda x: (not x[0], -x[1]))
    test_cands.sort(key=lambda x: (not x[0], -x[1]))

    print("[find_data] 训练集候选(clean?, 图片数, 路径):",
          [(c, n, p) for c, n, p in train_cands], flush=True)
    print("[find_data] 测试集候选(图片数, 路径):",
          [(n, p) for _, n, p in test_cands], flush=True)

    train_root = train_cands[0][2] if train_cands else None
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


class BinDataset(Dataset):
    """二分类数据集: 给定 (paths, labels) 列表 + transform。"""
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


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
    return timm.create_model(BACKBONE, pretrained=True, num_classes=2)


def set_trainable(model, backbone, head):
    for n, p in model.named_parameters():
        is_head = n.startswith("head.") or n.startswith("classifier.") or n.startswith("fc.")
        p.requires_grad = head if is_head else backbone


def run_epoch(model, loader, opt, mixup, crit, device):
    model.train()
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


def train_fold(model, tr_loader, va_loader, crit, device):
    set_trainable(model, backbone=False, head=True)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS)
    for _ in range(STAGE1_EPOCHS):
        run_epoch(model, tr_loader, opt1, mixup=False, crit=crit, device=device)
        sched1.step()

    set_trainable(model, backbone=True, head=True)
    opt2 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_FT, weight_decay=WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS)
    ema = EMA(model)
    best_acc = 0.0
    best_state = None
    for ep in range(STAGE2_EPOCHS):
        run_epoch(model, tr_loader, opt2, mixup=True, crit=crit, device=device)
        ema.update(model)
        sched2.step()
        ema.apply_shadow(model)
        p, _ = predict(model, va_loader, device, tta=True)
        ema.restore(model)
        pred = p.argmax(1)
        yt = np.array(va_loader.dataset.labels)
        acc = float((pred == yt).mean())
        if ep == STAGE2_EPOCHS - 1 or acc > best_acc:
            print(f"    stage2 ep{ep+1}/{STAGE2_EPOCHS} val acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in ema.shadow.items()}
    for n, p in model.named_parameters():
        if p.requires_grad and n in best_state:
            p.data.copy_(best_state[n].to(p.device))
    return model, best_acc


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    train_root, test_root = find_binary_data()
    if train_root is None or test_root is None:
        raise SystemExit("未找到 BG/LSB 训练目录或课设 test 目录。")
    print(f"训练集: {train_root}", flush=True)
    print(f"测试集: {test_root}", flush=True)

    # 收集两类图片
    paths, labels = [], []
    for ci, cname in enumerate(CLASSES):
        cdir = os.path.join(train_root, cname)
        fs = sorted([f for f in os.listdir(cdir) if f.lower().endswith(".png")])
        for f in fs:
            paths.append(os.path.join(cdir, f))
            labels.append(ci)
    labels = np.array(labels)
    print(f"二分类样本: {len(paths)} | 分布: "
          f"{dict(Counter(CLASSES[l] for l in labels))}", flush=True)

    # 逆频率类别权重(校正 LSB/BG 不平衡)
    cnt = Counter(labels.tolist())
    w = torch.tensor([1.0 / cnt[0], 1.0 / cnt[1]], device=device, dtype=torch.float32)
    w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)
    print(f"类别权重: BG={w[0].item():.3f} LSB={w[1].item():.3f}", flush=True)

    # 测试集
    test_ds = TestDataset(test_root, val_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=False)
    print(f"测试样本: {len(test_ds)}", flush=True)

    # 5 折(每折 90/10)
    skf = StratifiedKFold(n_splits=int(1 / VAL_RATIO), shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(paths)), labels))[:N_FOLDS]

    oof_prob = np.zeros((len(paths), 2), dtype=np.float32)
    oof_count = np.zeros(len(paths), dtype=np.int32)
    test_prob = np.zeros((len(test_ds), 2), dtype=np.float32)

    for fi, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n===== Fold {fi+1}/{N_FOLDS} train={len(tr_idx)} val={len(va_idx)} =====", flush=True)
        tr_ds = BinDataset([paths[i] for i in tr_idx], labels[tr_idx], train_tf)
        va_ds = BinDataset([paths[i] for i in va_idx], labels[va_idx], val_tf)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=False)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=False)

        model = build_model().to(device)
        model, best_acc = train_fold(model, tr_loader, va_loader, crit, device)
        print(f"Fold {fi+1} 最佳 val acc={best_acc:.4f}", flush=True)

        p, _ = predict(model, va_loader, device, tta=True)
        oof_prob[va_idx] += p
        oof_count[va_idx] += 1
        tp, _ = predict(model, test_loader, device, tta=True)
        test_prob += tp

    # OOF 评估
    mask = oof_count > 0
    oof_pred = oof_prob[mask].argmax(1)
    oof_true = labels[mask]
    acc = float((oof_pred == oof_true).mean())
    recall_bg = float(((oof_pred == 0) & (oof_true == 0)).sum() / max((oof_true == 0).sum(), 1))
    recall_lsb = float(((oof_pred == 1) & (oof_true == 1)).sum() / max((oof_true == 1).sum(), 1))
    cm = confusion_matrix(oof_true, oof_pred, labels=[0, 1])
    print(f"\n===== OOF (覆盖 {mask.sum()}/{len(paths)} 张) =====", flush=True)
    print(f"OOF acc={acc:.4f}  BG recall={recall_bg:.4f}  LSB recall={recall_lsb:.4f}", flush=True)
    print(f"混淆矩阵(行=真实 BG/LSB, 列=预测 BG/LSB):\n{cm}", flush=True)

    # 输出 test 概率(折平均)
    test_prob /= N_FOLDS
    out_path = "/kaggle/working/binary_test_probs.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ID,P_Black-grass,P_Loose Silky-bent\n")
        for fid, (p0, p1) in zip(test_ds.files, test_prob):
            f.write(f"{fid},{p0:.6f},{p1:.6f}\n")
    print(f"\n已写入: {out_path}", flush=True)


if __name__ == "__main__":
    main()