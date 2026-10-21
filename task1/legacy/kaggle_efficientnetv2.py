# -*- coding: utf-8 -*-
"""
植物幼苗图片分类 —— EfficientNetV2-S 5折集成 (Kaggle Notebook 版)
- 骨干替换为 EfficientNetV2-S, 特征更强, 应对 Black-grass/Loose Silky-bent 细粒度混淆
- 5 折 StratifiedKFold 集成 + 原图/水平翻转 TTA
- 两阶段训练: 先训分类头, 再低学习率全量微调 + 正则
- 输出 OOF(袋外) macro-F1 作为真实水平估计, 并生成提交提交 CSV

Kaggle 使用步骤:
  1) 把老师给的 train/ 和 test/ 打包上传为私有 Dataset (名字随意, 如 plant5)
  2) 新建 Notebook, 选中该 Dataset
  3) 加速器选 GPU (T4/P100), 不要用 TPU(需要额外 XLA 配置)
  4) 把本文件全部内容粘进第一个 cell 运行
提交文件会生成到 /kaggle/working/submission_efficientnet_s.csv
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

# ---------------- 可调配置 ----------------
IMG_SIZE = 256          # 用 256 兼顾速度与细粒度细节; 想更强可改 288/384(更慢)
BATCH_SIZE = 32
N_FOLDS = 5             # 折数; 时间紧可改 3
STAGE1_EPOCHS = 3       # 只训分类头轮数
STAGE2_EPOCHS = 10      # 全量微调轮数 (数据少, 别太多防过拟合)
LR_HEAD = 1e-3
LR_ALL = 2e-5
WEIGHT_DECAY = 1e-3
SEED = 42


def find_data_root():
    """递归搜索 /kaggle/input 下任意一层同时含 train/ 和 test/ 的目录.
    你的数据挂载在 /kaggle/input/2026task1/dataset-for-task2/dataset-for-task2/{train,test},
    双层嵌套也能自动找到. 本地跑则用绝对路径回退."""
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train" in dirs and "test" in dirs:
            return root
    # 本地回退路径(未挂载 Kaggle 数据时)
    local = r"c:\Users\lenovo\Desktop\dataset-for-task2\dataset-for-task2"
    return local if os.path.isdir(os.path.join(local, "train")) else None


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
    def __init__(self, folder, transform, indices):
        self.base = datasets.ImageFolder(folder)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.base.samples[self.indices[i]]
        return self.transform(Image.open(path).convert("RGB")), label


class ImgSubset(Dataset):
    """仅返回图像(用于验证 OOF / test), 顺序与 indices 一致."""
    def __init__(self, folder, transform, indices):
        self.base = datasets.ImageFolder(folder)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, _ = self.base.samples[self.indices[i]]
        return self.transform(Image.open(path).convert("RGB"))


class FlatImageDataset(Dataset):
    """test 目录是平铺 png, 无类别子目录."""
    def __init__(self, folder, transform):
        self.names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        self.transform = transform
        self.folder = folder

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.transform(Image.open(os.path.join(self.folder, self.names[i])).convert("RGB"))


@torch.no_grad()
def predict_tta(model, plain_ds, flip_ds, device):
    model.eval()
    sm = nn.Softmax(dim=1)
    sums = []
    bl = DataLoader(plain_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    fl = DataLoader(flip_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    for x, xf in zip(bl, fl):
        p = sm(model(x.to(device)))
        pf = sm(model(xf.to(device)))
        sums.append(((p + pf) / 2).cpu())
    return torch.cat(sums)


def make_model(n_class):
    # EfficientNetV2-S, 头部 classifier[1] 是 Linear
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_class)
    return m


def train_fold(train_loader, device, n_class):
    model = make_model(n_class).to(device)
    criterion = nn.CrossEntropyLoss()

    def set_trainable(backbone, head):
        for n, p in model.named_parameters():
            p.requires_grad = head if n.startswith("classifier.") else backbone

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

    # 阶段1: 冻结骨干, 只训分类头
    set_trainable(backbone=False, head=True)
    opt1 = torch.optim.Adam(model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    run(opt1, torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS), STAGE1_EPOCHS)

    # 阶段2: 全量微调
    set_trainable(backbone=True, head=True)
    cp = list(model.classifier.parameters())
    bp = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    opt2 = torch.optim.Adam([
        {"params": cp, "lr": LR_ALL},
        {"params": bp, "lr": LR_ALL},
    ], weight_decay=WEIGHT_DECAY)
    run(opt2, torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS), STAGE2_EPOCHS)
    return model


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_root = find_data_root()
    if data_root is None:
        raise SystemExit("找不到数据目录! 请上传 train/ 和 test/ 为 Kaggle Dataset 并挂载到 Notebook。")
    data_root = os.path.abspath(data_root)
    train_folder = os.path.join(data_root, "train")
    test_folder = os.path.join(data_root, "test")
    print("数据目录:", data_root)
    print("device:", device, "| 折数:", N_FOLDS, "| 骨干: EfficientNetV2-S")

    train_tf, plain_tf, flip_tf = build_transforms()
    base = datasets.ImageFolder(train_folder)
    classes = base.classes
    n_class = len(classes)
    y = np.array([s[1] for s in base.samples])
    n_total = len(y)
    print("类别:", classes, "总样本:", n_total)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((n_total, n_class))
    fold_models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n_total), y), 1):
        print(f"\n===== Fold {fold}/{N_FOLDS}  train={len(tr_idx)} val={len(va_idx)} =====")
        tr_loader = DataLoader(TrainSubset(train_folder, train_tf, tr_idx),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)
        model = train_fold(tr_loader, device, n_class)
        fold_models.append(model)

        probs = predict_tta(model, ImgSubset(train_folder, plain_tf, va_idx),
                            ImgSubset(train_folder, flip_tf, va_idx), device)
        oof_probs[va_idx] = probs.numpy()
        pred = probs.argmax(1).numpy()
        print(f"Fold {fold} val acc={accuracy_score(y[va_idx], pred):.4f}  "
              f"val macroF1={f1_score(y[va_idx], pred, average='macro', zero_division=0):.4f}")

    # ---- OOF 总体指标 (真实水平估计) ----
    oof_pred = oof_probs.argmax(1)
    print("\n===== OOF (袋外) 总体指标 =====")
    print(f"OOF acc={accuracy_score(y, oof_pred):.4f}  OOF macroF1="
          f"{f1_score(y, oof_pred, average='macro', zero_division=0):.4f}")
    print("OOF 混淆矩阵(行=真实, 列=预测):\n", confusion_matrix(y, oof_pred))
    print("类别对应:", dict(enumerate(classes)))

    # ---- test 预测(5 模型概率平均 + TTA) ----
    test_ds = FlatImageDataset(test_folder, plain_tf)
    test_flip = FlatImageDataset(test_folder, flip_tf)
    test_probs = torch.zeros((len(test_ds), n_class))
    for model in fold_models:
        test_probs += predict_tta(model, test_ds, test_flip, device)
    test_pred = test_probs.argmax(1).tolist()

    out = os.path.join("/kaggle/working" if os.path.isdir("/kaggle/working") else ".",
                       "submission_efficientnet_s.csv")
    with open(out, "w", newline="") as fp:
        fp.write("ID,Category\n")
        for name, pidx in zip(test_ds.names, test_pred):
            fp.write(f"{name},{classes[pidx]}\n")
    print("\n已生成提交文件:", out)
    print("test 各类预测数:", dict(Counter(classes[p] for p in test_pred)))


if __name__ == "__main__":
    main()