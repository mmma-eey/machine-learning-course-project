# -*- coding: utf-8 -*-
"""
植物幼苗图片分类 —— 伪标签自训练 (Pseudo-Labeling), EfficientNetV2-S, Kaggle 版
思路(半监督):
  阶段A: 用原始 500 张训练 5 折, 对 test 批量预测, 得到每类 softmax 概率
         只挑选高置信度(>CONF_THR)的 test 图, 用其 argmax 作为"伪标签"
  阶段B: 把伪标签图(以训练样本身份)并入训练集, 再整体重新训练
         这次模型能看到接近全部 test 分布, 对难分的两类更稳
流程:
  1) train_fold()     训练一个折, 返回 model
  2) 阶段A累加 test 概率(含TTA) -> 选出置信 test 图 -> 构造伪标签集
  3) 阶段B每个折在 (原train该折 + 全部伪标签图) 上训练
  4) 最终 pooling 各折概率(含TTA) 出提交
提交: /kaggle/working/submission_final.csv
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
IMG_SIZE = 256
BATCH_SIZE = 32
CONF_THR = 0.90         # 伪标签置信度阈值, 越高越稳(略少), 越低越多(略噪)
N_FOLDS_A = 5           # 阶段A折数(生成伪标签, 建议保持5)
N_FOLDS_B = 5           # 阶段B折数(时间紧可改3)
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 10
LR_HEAD = 1e-3
LR_ALL = 2e-5
WEIGHT_DECAY = 1e-3
SEED = 42


def find_data_root():
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train" in dirs and "test" in dirs:
            return root
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
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        path, label = self.base.samples[self.indices[i]]
        return self.transform(Image.open(path).convert("RGB")), label


class ImgSubset(Dataset):
    def __init__(self, folder, transform, indices):
        self.base = datasets.ImageFolder(folder)
        self.indices = indices
        self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        path, _ = self.base.samples[self.indices[i]]
        return self.transform(Image.open(path).convert("RGB"))


class FlatImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        self.transform = transform
        self.folder = folder
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        return self.transform(Image.open(os.path.join(self.folder, self.names[i])).convert("RGB"))


class PseudoLabeledWrapper(Dataset):
    """把原始训练折 train_idx 与 伪标签 test 图(文件名->class) 合并成一个数据集."""
    def __init__(self, train_folder, test_folder, transform, train_indices, pseudo):
        self.base = datasets.ImageFolder(train_folder)
        self.train_indices = list(train_indices)
        self.pseudo_items = pseudo   # list of (test_filename, class_idx)
        self.transform = transform
        self.test_folder = test_folder
    def __len__(self): return len(self.train_indices) + len(self.pseudo_items)
    def __getitem__(self, i):
        if i < len(self.train_indices):
            path, label = self.base.samples[self.train_indices[i]]
            return self.transform(Image.open(path).convert("RGB")), label
        j = i - len(self.train_indices)
        name, label = self.pseudo_items[j]
        return self.transform(Image.open(os.path.join(self.test_folder, name)).convert("RGB")), label


@torch.no_grad()
def predict_tta(model, ds_plain, ds_flip, device):
    model.eval(); sm = nn.Softmax(dim=1)
    bl = DataLoader(ds_plain, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    fl = DataLoader(ds_flip, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    sums = []
    for x, xf in zip(bl, fl):
        sums.append(((sm(model(x.to(device))) + sm(model(xf.to(device)))) / 2).cpu())
    return torch.cat(sums)


def make_model(n_class):
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_class)
    return m


def train_fold(train_loader, device, n_class):
    model = make_model(n_class).to(device)
    criterion = nn.CrossEntropyLoss()
    def set_trainable(backbone, head):
        for n, p in model.named_parameters():
            p.requires_grad = head if n.startswith("classifier.") else backbone
    def run(opt, sched, epochs):
        for _ in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
            sched.step()
    set_trainable(False, True)
    opt1 = torch.optim.Adam(model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    run(opt1, torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS), STAGE1_EPOCHS)
    set_trainable(True, True)
    cp = list(model.classifier.parameters())
    bp = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    opt2 = torch.optim.Adam([{"params": cp, "lr": LR_ALL}, {"params": bp, "lr": LR_ALL}], weight_decay=WEIGHT_DECAY)
    run(opt2, torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS), STAGE2_EPOCHS)
    return model


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = find_data_root()
    if data_root is None:
        raise SystemExit("找不到数据目录! 请上传 train/ 和 test/ 为 Kaggle Dataset。")
    train_folder = os.path.join(data_root, "train")
    test_folder = os.path.join(data_root, "test")
    print("数据目录:", data_root, "| device:", device)

    train_tf, plain_tf, flip_tf = build_transforms()
    base = datasets.ImageFolder(train_folder)
    classes = base.classes; n_class = len(classes)
    y = np.array([s[1] for s in base.samples]); n_total = len(y)
    print("类别:", classes, "原始训练:", n_total)

    # ---------- 阶段A: 5折估计 test 置信度 ----------
    print("\n===== 阶段A: 训练并估计 test 置信度 =====")
    skf = StratifiedKFold(n_splits=N_FOLDS_A, shuffle=True, random_state=SEED)
    test_ds = FlatImageDataset(test_folder, plain_tf)
    test_ds_flip = FlatImageDataset(test_folder, flip_tf)
    test_probs = torch.zeros((len(test_ds), n_class))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n_total), y), 1):
        tr_loader = DataLoader(TrainSubset(train_folder, train_tf, tr_idx),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        model = train_fold(tr_loader, device, n_class)
        test_probs += predict_tta(model, test_ds, test_ds_flip, device)
        va_prob = predict_tta(model, ImgSubset(train_folder, plain_tf, va_idx),
                              ImgSubset(train_folder, flip_tf, va_idx), device)
        vf = f1_score(y[va_idx], va_prob.argmax(1).numpy(), average="macro", zero_division=0)
        print(f"[阶段A Fold {fold}] val macroF1={vf:.4f}")
    test_probs /= N_FOLDS_A
    conf = test_probs.max(1).values.numpy()
    plabel = test_probs.argmax(1).numpy()
    mask = conf >= CONF_THR
    pseudo = [(test_ds.names[i], int(plabel[i])) for i in np.where(mask)[0]]
    print(f"高置信伪标签: {len(pseudo)}/{len(test_ds)} 张 (阈值 {CONF_THR})")
    if not pseudo:
        print("没有样本达到阈值! 建议把 CONF_THR 调低(如0.75)重跑, 或用原提交。")
        return
    print("伪标签类别分布:", dict(Counter(classes[p] for _, p in pseudo)))

    # ---------- 阶段B: 在原训练 + 伪标签 上重训 ----------
    print("\n===== 阶段B: 合并伪标签重训 =====")
    skfB = StratifiedKFold(n_splits=N_FOLDS_B, shuffle=True, random_state=SEED)
    test_probs2 = torch.zeros((len(test_ds), n_class))
    for fold, (tr_idx, va_idx) in enumerate(skfB.split(np.arange(n_total), y), 1):
        # 该折训练集 = 原始训练折(400) + 全部伪标签图
        ds = PseudoLabeledWrapper(train_folder, test_folder, train_tf, tr_idx, pseudo)
        tr_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        model = train_fold(tr_loader, device, n_class)
        test_probs2 += predict_tta(model, test_ds, test_ds_flip, device)
        va_prob = predict_tta(model, ImgSubset(train_folder, plain_tf, va_idx),
                              ImgSubset(train_folder, flip_tf, va_idx), device)
        vf = f1_score(y[va_idx], va_prob.argmax(1).numpy(), average="macro", zero_division=0)
        print(f"[阶段B Fold {fold}] val macroF1={vf:.4f}")

    # ---------- 提交 ----------
    test_probs2 /= N_FOLDS_B
    pred = test_probs2.argmax(1).tolist()
    out = "/kaggle/working/submission_final.csv"
    with open(out, "w", newline="") as fp:
        fp.write("ID,Category\n")
        for name, pidx in zip(test_ds.names, pred):
            fp.write(f"{name},{classes[pidx]}\n")
    print("\n已生成提交文件:", out)
    print("test 各类预测数:", dict(Counter(classes[p] for p in pred)))


if __name__ == "__main__":
    main()