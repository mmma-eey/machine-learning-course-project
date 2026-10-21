# -*- coding: utf-8 -*-
"""
植物幼苗图片分类 —— 最小可跑通版本（ResNet18 迁移学习微调）
数据集: Kaggle Plants Seedlings (5 类子集)
用法: conda activate agent && python train.py
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ----------------------------- 基础配置 -----------------------------
DATA_ROOT = r"c:\Users\lenovo\Desktop\dataset-for-task2\dataset-for-task2"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3          # 冻结骨干时分类头用较大学习率
VAL_RATIO = 0.10   # 从训练集分出 10% 做验证(用户的 90/10)
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true", help="加载权重并对 test 预测生成提交文件")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "best_model.pth"))
    args = parser.parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dir = os.path.join(DATA_ROOT, "train")
    test_dir = os.path.join(DATA_ROOT, "test")

    # 训练预处理(带数据增强), 验证/测试只做归一化
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
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

    full = datasets.ImageFolder(train_dir, transform=train_tf)
    classes = full.classes
    n_class = len(classes)
    print("类别:", classes, "总样本:", len(full))

    idxs = np.arange(len(full))
    y = np.array([s[1] for s in full.samples])
    tr_idx, va_idx = train_test_split(idxs, test_size=VAL_RATIO, stratify=y, random_state=SEED)
    train_ds = Subset(full, tr_idx)
    val_ds = Subset(full, va_idx)
    val_ds.dataset.transform = plain_tf   # 验证集用无增强预处理
    print(f"train={len(tr_idx)}  val={len(va_idx)}  ({1-VAL_RATIO:.0f}/{VAL_RATIO*100:.0f})")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ----------------------------- 模型: ResNet18 预训练 -----------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_class)   # 替换分类头为 5 类
    model.to(device)

    # 冻结骨干, 只训练分类头
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
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
        train_acc = correct / total

        # 验证
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, yb in val_loader:
                out = model(x.to(device))
                y_true.extend(yb.tolist())
                y_pred.extend(out.argmax(1).cpu().tolist())
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"[{epoch:02d}/{EPOCHS}] loss={loss_sum/len(train_ds):.4f} train_acc={train_acc:.4f} | "
              f"val_acc={acc:.4f} val_macroF1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.model)
            print("   -> 保存最优模型, macroF1={:.4f}".format(f1))

    # ----------------------------- 最优模型在验证集上的报告 -----------------------------
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, yb in val_loader:
            out = model(x.to(device))
            y_true.extend(yb.tolist())
            y_pred.extend(out.argmax(1).cpu().tolist())
    print("\n验证集最终结果: acc={:.4f}  macroF1={:.4f}".format(
        accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")))
    print("混淆矩阵(行=真实, 列=预测):")
    print(confusion_matrix(y_true, y_pred))
    print("类别对应:", dict(enumerate(classes)))

    # ----------------------------- 可选: 对 test 预测并生成提交文件 -----------------------------
    if args.predict:
        from torch.utils.data import Dataset
        from PIL import Image
        class FlatImageDataset(Dataset):   # test 目录是平铺 png, 无类别子目录
            def __init__(self, folder, transform):
                self.names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
                self.transform = transform
                self.folder = folder
            def __len__(self):
                return len(self.names)
            def __getitem__(self, i):
                img = Image.open(os.path.join(self.folder, self.names[i])).convert("RGB")
                return self.transform(img)
        test_ds = FlatImageDataset(test_dir, plain_tf)
        test_names = [os.path.splitext(f)[0] for f in test_ds.names]
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        sub_path = os.path.join(os.path.dirname(args.model), "submission.csv")
        with open(sub_path, "w", newline="") as fp:
            fp.write("file,species\n")
            for name, p in zip(test_names, preds):
                fp.write(f"{name},{classes[p]}\n")
        print("\n已生成提交文件:", sub_path)
        # 统计各类预测数量, 简单检查是否合理均衡
        from collections import Counter
        print("test 各类预测数:", dict(Counter(preds)))

if __name__ == "__main__":
    main()