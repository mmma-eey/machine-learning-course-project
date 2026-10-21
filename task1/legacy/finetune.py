# -*- coding: utf-8 -*-
"""
植物幼苗图片分类 —— 全量微调版 (ResNet18, 两阶段训练 + 测试时增强 TTA)
策略:
  阶段1 冻结骨干, 只训分类头 (lr=1e-3, 5 epochs) -> 稳定初始化
  阶段2 解冻全部层, 全量微调 (lr=3e-5, 20 epochs) -> 提分关键
  用验证集 macro-F1 选最优权重
用法:
  训练: D:\Anaconda3\envs\agent\python.exe finetune.py
  预测生成提交: D:\Anaconda3\envs\agent\python.exe finetune.py --predict
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter
from PIL import Image

DATA_ROOT = r"c:\Users\lenovo\Desktop\dataset-for-task2\dataset-for-task2"
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_RATIO = 0.10          # 训练集划出 10% 做验证 (90/10)
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 20
LR_HEAD = 1e-3
LR_ALL = 3e-5
SEED = 42
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    plain_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, plain_tf


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, yb in loader:
            out = model(x.to(device))
            y_true.extend(yb.tolist())
            y_pred.extend(out.argmax(1).cpu().tolist())
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0),
            y_true, y_pred)


class FlatImageDataset(Dataset):   # test 目录是平铺 png
    def __init__(self, folder, transform):
        self.names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        self.transform = transform
        self.folder = folder

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.transform(Image.open(os.path.join(self.folder, self.names[i])).convert("RGB"))


def _predict(model, classes, plain_tf, device):
    model.eval()
    test_ds = FlatImageDataset(os.path.join(DATA_ROOT, "test"), plain_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # TTA: (原图, 水平翻转) softmax 平均取 argmax
    hflip = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    flip_ds = FlatImageDataset(os.path.join(DATA_ROOT, "test"), hflip)
    flip_loader = DataLoader(flip_ds, batch_size=BATCH_SIZE, shuffle=False)

    sm = nn.Softmax(dim=1)
    probs_orig, probs_flip = [], []
    with torch.no_grad():
        for x, xf in zip(test_loader, flip_loader):
            probs_orig.append(sm(model(x.to(device))).cpu())
            probs_flip.append(sm(model(xf.to(device))).cpu())
    p = (torch.cat(probs_orig) + torch.cat(probs_flip)) / 2.0
    preds = p.argmax(1).tolist()

    sub_path = os.path.join(SAVE_DIR, "submission.csv")
    with open(sub_path, "w", newline="") as fp:
        fp.write("ID,Category\n")
        for name, pidx in zip(test_ds.names, preds):
            fp.write(f"{name},{classes[pidx]}\n")
    print("\n已生成提交文件:", sub_path)
    print("test 各类预测数:", dict(Counter(classes[pidx] for pidx in preds)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true", help="加载权重并对 test 预测生成提交文件")
    parser.add_argument("--ratio", type=float, default=VAL_RATIO, help="验证集比例")
    args = parser.parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_tf, plain_tf = build_transforms()
    full = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
    classes = full.classes
    n_class = len(classes)

    # ------------------- 模型 -------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_class)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 预测模式: 加载已保存权重, 跳过训练
    if args.predict:
        if not os.path.exists(MODEL_PATH):
            raise SystemExit("未找到权重文件 best_model.pth, 请先运行训练: python finetune.py")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("已加载权重:", MODEL_PATH)
        _predict(model, classes, plain_tf, device)
        return

    # ------------------- 数据划分 (90/10) -------------------
    idxs = np.arange(len(full))
    y = np.array([s[1] for s in full.samples])
    tr_idx, va_idx = train_test_split(idxs, test_size=args.ratio, stratify=y, random_state=SEED)
    train_ds = Subset(full, tr_idx)
    val_ds = Subset(full, va_idx)
    val_ds.dataset.transform = plain_tf   # 验证集用无增强预处理
    print(f"类别={classes}  train={len(tr_idx)} val={len(va_idx)}  "
          f"({(1-args.ratio)*100:.0f}/{(args.ratio)*100:.0f})")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    best_f1, best_state = 0.0, None

    def set_trainable(backbone: bool, head: bool):
        for p in model.parameters():
            p.requires_grad = backbone
        for p in model.fc.parameters():
            p.requires_grad = head

    def run_epochs(param_groups, n_epochs, tag):
        nonlocal best_f1, best_state
        optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        for epoch in range(1, n_epochs + 1):
            model.train()
            total, correct, loss_sum = 0, 0, 0.0
            for x, yb in train_loader:
                x, yb = x.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * x.size(0)
                total += yb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
            acc_v, f1_v, _, _ = evaluate(model, val_loader, device)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[{tag} {epoch:02d}/{n_epochs}] loss={loss_sum/len(train_ds):.4f} "
                  f"train_acc={correct/total:.4f} | val_acc={acc_v:.4f} val_macroF1={f1_v:.4f} lr={lr_now:.1e}")
            if f1_v > best_f1:
                best_f1 = f1_v
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print("   * 新最优 (macroF1={:.4f})".format(f1_v))
            sched.step()

    # 阶段1: 冻结骨干, 只训分类头
    set_trainable(backbone=False, head=True)
    run_epochs([{"params": model.fc.parameters(), "lr": LR_HEAD}], STAGE1_EPOCHS, "head")

    # 阶段2: 解冻全网络微调
    set_trainable(backbone=True, head=True)
    fc_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    run_epochs([
        {"params": fc_params, "lr": LR_ALL},
        {"params": backbone_params, "lr": LR_ALL},
    ], STAGE2_EPOCHS, "ft")

    torch.save(best_state, MODEL_PATH)
    print("\n已完成, 最优模型已保存 ->", MODEL_PATH, "best_val_macroF1={:.4f}".format(best_f1))

    # ---------------- 报告 ----------------
    model.load_state_dict(best_state)
    acc_v, f1_v, y_true, y_pred = evaluate(model, val_loader, device)
    print(f"验证集最终: acc={acc_v:.4f}  macroF1={f1_v:.4f}")
    print("混淆矩阵(行=真实, 列=预测):\n", confusion_matrix(y_true, y_pred))
    print("类别对应:", dict(enumerate(classes)))


if __name__ == "__main__":
    main()