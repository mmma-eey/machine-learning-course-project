# -*- coding: utf-8 -*-
"""
植物幼苗图片分类 —— 5 折交叉验证集成版 (ResNet18)
策略:
  用全部 500 张训练图, 5 折 StratifiedKFold
  每折: 400 train / 100 val, 两阶段训练(先训头再全量微调), 加正则抗过拟合
  test 预测: 5 个模型 softmax 概率平均 + 水平翻转 TTA
  报告: 每折 val macro-F1, 以及 OOF(袋外) macro-F1 (接近真实线上水平)
用法: D:\Anaconda3\envs\agent\python.exe kfold_ensemble.py
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter
from PIL import Image

DATA_ROOT = r"c:\Users\lenovo\Desktop\dataset-for-task2\dataset-for-task2"
IMG_SIZE = 224
BATCH_SIZE = 32
N_FOLDS = 5
STAGE1_EPOCHS = 4
STAGE2_EPOCHS = 12
LR_HEAD = 1e-3
LR_ALL = 2e-5
WEIGHT_DECAY = 1e-3
SEED = 42
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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
        transforms.Resize(int(IMG_SIZE * 1.15)), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    flip_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.RandomHorizontalFlip(p=1.0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, plain_tf, flip_tf


class TrainSubset(Dataset):
    """训练折: 按 indices 取训练集文件夹里的样本, 返回 (image, label)."""
    def __init__(self, folder, transform, indices):
        self.base = datasets.ImageFolder(folder)
        self.indices = indices
        self.transform = transform
        self.folder = folder

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.base.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class ImgSubset(Dataset):
    """只返回图像(用于验证 OOF 和 test 预测). 表现与 indices 顺序一致."""
    def __init__(self, folder, transform, indices):
        self.base = datasets.ImageFolder(folder)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, _ = self.base.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class FlatImageDataset(Dataset):
    """test 目录是平铺 png, 无类别子目录."""
    def __init__(self, folder, transform):
        self.names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        self.transform = transform
        self.folder = folder

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.folder, self.names[i])).convert("RGB")
        return self.transform(img)


def _loader(ds, batch_size):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def predict_tta(model, plain_ds, flip_ds, device):
    """原图+水平翻转 softmax 平均, 返回 n x n_class."""
    model.eval()
    sm = nn.Softmax(dim=1)
    sums = []
    for x, xf in zip(_loader(plain_ds, BATCH_SIZE), _loader(flip_ds, BATCH_SIZE)):
        p = sm(model(x.to(device)))
        pf = sm(model(xf.to(device)))
        sums.append(((p + pf) / 2).cpu())
    return torch.cat(sums)


def train_fold(train_loader, device, n_class):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_class)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    def set_trainable(backbone, head):
        for p in model.parameters():
            p.requires_grad = backbone
        for p in model.fc.parameters():
            p.requires_grad = head

    def run(optimizer, scheduler, epochs):
        for _ in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            scheduler.step()

    # 阶段1: 只训分类头
    set_trainable(False, True)
    opt1 = torch.optim.Adam(model.fc.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    run(opt1, torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS), STAGE1_EPOCHS)

    # 阶段2: 全量微调(低学习率) + 正则
    set_trainable(True, True)
    fc_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt2 = torch.optim.Adam([
        {"params": fc_params, "lr": LR_ALL},
        {"params": backbone_params, "lr": LR_ALL},
    ], weight_decay=WEIGHT_DECAY)
    run(opt2, torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS), STAGE2_EPOCHS)
    return model


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| 折数:", N_FOLDS)
    train_tf, plain_tf, flip_tf = build_transforms()

    train_folder = os.path.join(DATA_ROOT, "train")
    base = datasets.ImageFolder(train_folder)
    classes = base.classes
    n_class = len(classes)
    y = np.array([s[1] for s in base.samples])
    n_total = len(y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((n_total, n_class))
    fold_models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n_total), y), 1):
        print(f"\n===== Fold {fold}/{N_FOLDS}  train={len(tr_idx)} val={len(va_idx)} =====")
        tr_ds = TrainSubset(train_folder, train_tf, tr_idx)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        model = train_fold(tr_loader, device, n_class)
        fold_models.append(model)

        probs = predict_tta(model, ImgSubset(train_folder, plain_tf, va_idx),
                            ImgSubset(train_folder, flip_tf, va_idx), device)
        oof_probs[va_idx] = probs.numpy()
        pred = probs.argmax(1).numpy()
        val_acc = accuracy_score(y[va_idx], pred)
        val_f1 = f1_score(y[va_idx], pred, average="macro", zero_division=0)
        print(f"Fold {fold} val acc={val_acc:.4f}  val macroF1={val_f1:.4f}")

    # ---- OOF 总体指标(更接近真实线下水平) ----
    oof_pred = oof_probs.argmax(1)
    print("\n===== OOF (袋外) 总体指标 =====")
    print(f"OOF acc={accuracy_score(y, oof_pred):.4f}  OOF macroF1="
          f"{f1_score(y, oof_pred, average='macro', zero_division=0):.4f}")
    print("OOF 混淆矩阵(行=真实, 列=预测):\n", confusion_matrix(y, oof_pred))
    print("类别对应:", dict(enumerate(classes)))

    # ---- test 预测(5 模型概率平均 + TTA) 并生成提交 ----
    test_folder = os.path.join(DATA_ROOT, "test")
    test_ds = FlatImageDataset(test_folder, plain_tf)
    test_flip = FlatImageDataset(test_folder, flip_tf)
    test_probs = torch.zeros((len(test_ds), n_class))
    for model in fold_models:
        test_probs += predict_tta(model, test_ds, test_flip, device)
    test_pred = test_probs.argmax(1).tolist()

    sub_path = os.path.join(SAVE_DIR, "submission_ensemble.csv")
    with open(sub_path, "w", newline="") as fp:
        fp.write("ID,Category\n")
        for name, pidx in zip(test_ds.names, test_pred):
            fp.write(f"{name},{classes[pidx]}\n")
    print("\n已生成提交文件:", sub_path)
    print("test 各类预测数:", dict(Counter(classes[p] for p in test_pred)))


if __name__ == "__main__":
    main()