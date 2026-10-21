"""Task2: 六类人脸表情，Kaggle T4 x2，准确率优先。

Notebook: 上传本文件作为 Input，然后在单元格执行：
    !python /kaggle/input/<代码数据集名称>/task2_kaggle.py
也可将整个文件粘贴进单个代码单元格运行。
开启 Internet（首次下载 ImageNet 预训练权重）和 GPU T4 x2。
输出 /kaggle/working/task2_output/submission.csv。
默认 2 个 Large 模型 × 5 折 × (40+15) epochs，可能需多次 GPU 会话。
已完成的折自动跳过；中途可从最新 epoch 恢复。详见随附说明。
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

# 已按用户提供的 Competition Rules 确认，不能用字母排序生成标签。
CLASS_TO_ID = {"Angry": 0, "Fear": 1, "Happy": 2,
               "Sad": 3, "Surprise": 4, "Neutral": 5}
N_CLASSES = 6
DEFAULT_ROOT = "/kaggle/input/task2forml"
DEFAULT_OUT = "/kaggle/working/task2_output"
SPECS = [
    dict(key="convnext", name="convnext_large.fb_in22k_ft_in1k",
         size=224, final_size=384, lr=8e-5, head_mult=5.0),
    dict(key="eva02", name="eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
         size=448, final_size=448, lr=4e-5, head_mult=5.0),
]


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def inspect_data(root):
    """只读取本任务输入，不下载或使用外部 FER 标注数据。"""
    from PIL import Image
    root = Path(root)
    if not root.exists():
        candidates = [p.parent for p in Path('/kaggle/input').rglob('submission.csv')
                      if (p.parent / 'fer_data').exists()]
        if len(candidates) != 1:
            raise FileNotFoundError(f"找不到 {root}，请显式设置 --data-root")
        root = candidates[0]
    candidates = [p for p in root.rglob('train')
                  if p.is_dir() and (p.parent / 'test').is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"期望唯一 train/test 目录对，实际：{candidates}")
    train_dir, test_dir = candidates[0], candidates[0].parent / 'test'
    folders = {p.name for p in train_dir.iterdir() if p.is_dir()}
    if folders != set(CLASS_TO_ID):
        raise ValueError(f"类别文件夹不匹配：{folders}")
    templates = list(root.rglob('submission.csv'))
    if len(templates) != 1:
        raise ValueError('期望唯一 submission.csv 模板')
    with templates[0].open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['ID', 'Emotion']:
            raise ValueError(f"提交列必须为 ID,Emotion，实际 {reader.fieldnames}")
        ids = [row['ID'] for row in reader]
    if len(ids) != len(set(ids)):
        raise ValueError('提交模板 ID 重复')
    test_paths = {p.name: p for p in test_dir.rglob('*')
                  if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    if set(ids) != set(test_paths):
        raise ValueError('测试图片与模板 ID 集合不一致')
    rows = []
    for label, target in CLASS_TO_ID.items():
        for p in sorted((train_dir / label).iterdir()):
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                rows.append({'path': str(p), 'target': target})

    def image_info(path):
        with Image.open(path) as im:
            im.load()  # 在训练前发现损坏图像
            gray = im.convert('L')
            digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
            return digest, f'{im.width}x{im.height}:{im.mode}'

    all_paths = [r['path'] for r in rows] + [str(test_paths[i]) for i in ids]
    with ThreadPoolExecutor(max_workers=8) as pool:
        info = list(pool.map(image_info, all_paths))
    for row, (digest, _) in zip(rows, info):
        row['group'] = digest
    groups = defaultdict(set)
    for row in rows:
        groups[row['group']].add(row['target'])
    train_hashes = set(groups)
    report = dict(train_count=len(rows), test_count=len(ids),
                  class_to_id=CLASS_TO_ID,
                  class_counts={k: sum(r['target'] == v for r in rows)
                                for k, v in CLASS_TO_ID.items()},
                  image_formats=dict(Counter(fmt for _, fmt in info)),
                  duplicate_train_images=len(rows) - len(groups),
                  conflicting_duplicate_groups=sum(len(v) > 1 for v in groups.values()),
                  test_images_also_in_train=sum(h in train_hashes for h, _ in info[len(rows):]))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return rows, [str(test_paths[i]) for i in ids], ids, report, [h for h, _ in info[len(rows):]]


def setup_ml():
    # Kaggle 自带 torch/torchvision/numpy/sklearn。只补装 timm，不替换 CUDA 的 torch。
    import importlib.metadata
    try:
        parts = importlib.metadata.version('timm').split('.')
        adequate = tuple(int(p) for p in parts[:3]) >= (1, 0, 22)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        adequate = False
    if not adequate:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                               'timm>=1.0.22,<1.1'])
    global np, torch, timm, T, Dataset, DataLoader, StratifiedGroupKFold
    global ModelEmaV2, resolve_model_data_config, F
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    from torchvision import transforms as T
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import StratifiedGroupKFold
    from timm.utils import ModelEmaV2
    from timm.data import resolve_model_data_config
    if not torch.cuda.is_available():
        raise RuntimeError('训练需要 CUDA GPU；请在 Kaggle Settings 选择 GPU。')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print('torch', torch.__version__, 'timm', timm.__version__, flush=True)
    print('GPUs:', [torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())], flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_dataset(paths, targets, transform, crop=1.0):
    # 定义在函数内支持将文件直接粘贴到 Linux Kaggle 单元格；loader 使用 fork。
    class FaceDataset(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, idx):
            from PIL import Image
            with Image.open(paths[idx]) as im:
                im = im.convert('RGB')  # 灰度复制到 3 通道，保留预训练输入格式
            if crop < 1:
                w, h = im.size
                dx, dy = round(w * (1-crop)/2), round(h * (1-crop)/2)
                im = im.crop((dx, dy, w-dx, h-dy))
            return transform(im), int(targets[idx]) if targets is not None else -1
    return FaceDataset()


def transform(size, mean, std, training=False, weak=False):
    interp = T.InterpolationMode.BICUBIC
    ops = []
    if training:
        # FER 图像已是紧致人脸框，避免大角度旋转、重度裁剪、垂直翻转。
        ops += [T.RandomResizedCrop(size, scale=(0.92 if weak else 0.85, 1.0),
                                   ratio=(0.97, 1.03), interpolation=interp),
                T.RandomHorizontalFlip(0.5),
                T.RandomAffine(degrees=5 if weak else 10, translate=(0.025, 0.025),
                               interpolation=interp),
                T.ColorJitter(brightness=0.08 if weak else 0.15,
                              contrast=0.08 if weak else 0.15)]
    else:
        ops += [T.Resize((size, size), interpolation=interp)]
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if training and not weak:
        ops += [T.RandomErasing(p=0.10, scale=(0.01, 0.04), ratio=(0.5, 2.0), value=0)]
    return T.Compose(ops)


def loader(paths, targets, tfm, args, training=False, crop=1.0):
    batch = args.batch_per_gpu * torch.cuda.device_count()
    return DataLoader(make_dataset(paths, targets, tfm, crop), batch_size=batch,
                      shuffle=training, num_workers=args.workers, pin_memory=True,
                      drop_last=False, worker_init_fn=seed_worker,
                      persistent_workers=args.workers > 0)


def make_model(spec, size, pretrained):
    kwargs = dict(pretrained=pretrained, num_classes=N_CLASSES, drop_path_rate=0.15)
    if spec['key'] == 'eva02':
        kwargs['img_size'] = size
    model = timm.create_model(spec['name'], **kwargs)
    model.set_grad_checkpointing(True)  # Large 模型在 T4 上减少激活显存
    return model.cuda()


def parallel(model):
    return torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model


def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def save_pt(obj, path):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def checkpoint(model, score, size, data_cfg, stage, epoch):
    # 最佳推理权重以 fp16 存盘（恢复时加载回 fp32 参数），节约输出空间。
    return dict(state={k: v.detach().cpu().half() if v.is_floating_point()
                       else v.detach().cpu() for k, v in model.state_dict().items()},
                score=score, size=size, mean=data_cfg['mean'], std=data_cfg['std'],
                stage=stage, epoch=epoch)


def optimizer_for(model, spec, base_lr):
    buckets = defaultdict(list)
    for name, param in model.named_parameters():
        head = name.startswith('head.')
        # 保守的骨干分层学习率，分类头更快，前层更慢。
        scale = spec['head_mult'] if head else 1.0
        if name.startswith(('stem.', 'patch_embed.', 'pos_embed')):
            scale *= 0.25
        elif name.startswith(('stages.0.', 'stages.1.', 'blocks.0.', 'blocks.1.', 'blocks.2.')):
            scale *= 0.5
        decay = 0.0 if param.ndim <= 1 or name.endswith('.bias') or 'pos_embed' in name else 0.05
        buckets[(scale, decay)].append(param)
    groups = [dict(params=params, lr=base_lr * scale, initial_lr=base_lr * scale,
                   weight_decay=decay) for (scale, decay), params in buckets.items()]
    return torch.optim.AdamW(groups)


@__import__('contextlib').contextmanager
def amp():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        yield


def smoothed_targets(y, smoothing):
    targets = F.one_hot(y, N_CLASSES).float()
    return targets * (1-smoothing) + smoothing / N_CLASSES


def train_epoch(net, model, ema, dl, opt, scaler, args, mix, smooth, epoch, epochs):
    net.train()
    opt.zero_grad(set_to_none=True)
    total_loss, seen = 0.0, 0
    start = time.time()
    steps = len(dl)
    successful_steps = 0
    for i, (x, y) in enumerate(dl):
        progress = epoch + i / steps
        # 前两轮 warmup，此后 cosine decay 到 5% 初始 LR。
        warmup = min(2, epochs / 5)
        factor = ((progress+1/steps)/warmup if progress < warmup else
                  0.05 + 0.95 * (1+math.cos(math.pi * (progress-warmup)/(epochs-warmup))) / 2)
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * factor
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        target = smoothed_targets(y, smooth)
        if mix and random.random() < 0.5:
            lam = float(np.random.beta(0.2, 0.2))
            order = torch.randperm(len(x), device=x.device)
            x = lam*x + (1-lam)*x[order]
            target = lam*target + (1-lam)*target[order]
        # 最后不足 accum 个批次时按实际窗口长度归一化。
        window = min(args.accum, steps - (i // args.accum)*args.accum)
        with amp():
            logits = net(x)
            loss = -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'非有限 loss，epoch={epoch}, batch={i}')
        scaler.scale(loss/window).backward()
        if (i+1) % args.accum == 0 or i+1 == steps:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                ema.update(model)
                successful_steps += 1
            opt.zero_grad(set_to_none=True)
        seen += len(x)
        total_loss += loss.item() * len(x)
        if (i+1) % 300 == 0:
            print(f'  batch {i+1}/{steps}, loss={total_loss/seen:.4f}', flush=True)
    if successful_steps == 0:
        raise FloatingPointError('本轮所有优化步均被 AMP 跳过，请检查数值稳定性')
    return total_loss/seen, time.time()-start


def predict(net, dl, flip=False):
    net.eval()
    chunks = []
    with torch.inference_mode():
        for x, _ in dl:
            x = x.cuda(non_blocking=True)
            with amp():
                p = net(x).float().softmax(-1)
                if flip:
                    p = (p + net(x.flip(-1)).float().softmax(-1)) / 2
            chunks.append(p.cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise FloatingPointError('推理概率含 NaN/Inf')
    return result


def accuracy(probs, targets):
    return float((probs.argmax(-1) == targets).mean())


def run_fold(spec, fold, rows, test_paths, args, out, signature):
    stem = out / f"{spec['key']}_fold{fold}"
    best_path = stem.with_suffix('.best.pt')
    latest_path = stem.with_suffix('.latest.pt')
    result_path = stem.with_suffix('.predictions.npz')
    if result_path.exists():
        print(f'{stem.name}: 已完成，跳过', flush=True)
        return
    seed_all(args.seed + fold * 100 + (10000 if spec['key'] == 'eva02' else 0))
    tr = [r for r in rows if r['fold'] != fold]
    va = [r for r in rows if r['fold'] == fold]
    tr_paths, tr_y = [r['path'] for r in tr], [r['target'] for r in tr]
    va_paths, va_y = [r['path'] for r in va], np.array([r['target'] for r in va])
    stages = [(spec['size'], args.epochs, spec['lr'], False),
              (spec['final_size'], args.finetune_epochs, spec['lr'] * 0.15, True)]
    resume = load_pt(latest_path) if latest_path.exists() else None
    if resume is not None and resume['signature'] != signature:
        raise ValueError('恢复文件与当前配置/输入不匹配，请使用新的输出目录')
    begin_stage = resume['stage'] if resume else 0
    global_best = load_pt(best_path)['score'] if best_path.exists() else -1
    for stage in range(begin_stage, len(stages)):
        size, epochs, lr, weak = stages[stage]
        if epochs == 0:
            continue
        restoring = resume is not None and resume['stage'] == stage
        model = make_model(spec, size, pretrained=not restoring and stage == 0)
        cfg = resolve_model_data_config(model)
        if stage > 0 and not restoring:
            if not best_path.exists():
                raise RuntimeError('精调阶段缺少上一阶段最佳权重')
            # ConvNeXt 可改变输入分辨率，EVA 两阶段保持 448 不变。
            model.load_state_dict(load_pt(best_path)['state'])
        if restoring:
            model.load_state_dict(resume['model'])
        ema = ModelEmaV2(model, decay=0.999, device='cuda')
        opt = optimizer_for(model, spec, lr)
        scaler = torch.amp.GradScaler('cuda')
        start_epoch, stage_best, stage_best_state = 0, -1, None
        if restoring:
            ema.module.load_state_dict(resume['ema'])
            opt.load_state_dict(resume['optimizer'])
            scaler.load_state_dict(resume['scaler'])
            start_epoch = resume['epoch'] + 1
            stage_best = resume['stage_best']
            stage_best_state = resume['stage_best_state']
            random.setstate(resume['python_rng'])
            np.random.set_state(resume['numpy_rng'])
            torch.set_rng_state(resume['torch_rng'])
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        resume = None
        gc.collect()
        net = parallel(model)
        dl_train = loader(tr_paths, tr_y, transform(size, cfg['mean'], cfg['std'], True, weak),
                          args, True)
        dl_val = loader(va_paths, va_y, transform(size, cfg['mean'], cfg['std']), args)
        # 保存各阶段最佳，最终比较使用各自分辨率，避免升分辨率后分数不可回退。
        stage_path = Path(str(stem) + f'.stage{stage}.pt')
        for epoch in range(start_epoch, epochs):
            loss, elapsed = train_epoch(net, model, ema, dl_train, opt, scaler, args,
                                        mix=not weak, smooth=0.02 if weak else 0.05,
                                        epoch=epoch, epochs=epochs)
            # EMA 与原始权重均检查，以避免单独依赖 EMA 的早期滞后。
            raw_score = accuracy(predict(net, dl_val), va_y)
            ema_score = accuracy(predict(parallel(ema.module), dl_val), va_y)
            score = max(raw_score, ema_score)
            winner = model if raw_score >= ema_score else ema.module
            print(f'{stem.name} stage={stage} epoch={epoch+1}/{epochs} '
                  f'loss={loss:.4f} raw_acc={raw_score:.5f} ema_acc={ema_score:.5f} '
                  f'seconds={elapsed:.0f}', flush=True)
            if score > stage_best:
                stage_best = score
                stage_best_state = checkpoint(winner, score, size, cfg, stage, epoch)
                save_pt(stage_best_state, stage_path)
            if score > global_best:
                global_best = score
                save_pt(stage_best_state, best_path)
            # 只有当前折保留完整恢复状态，已完成折只留较小的最佳推理权重。
            save_pt(dict(signature=signature, stage=stage, epoch=epoch,
                         model=model.state_dict(), ema=ema.module.state_dict(),
                         optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                         stage_best=stage_best, stage_best_state=stage_best_state,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()),
                    latest_path)
        # 如果在某阶段末尾中断，stage 文件仍存在，可继续下一阶段。
        del net, model, ema, opt, scaler, dl_train, dl_val, stage_best_state
        gc.collect()
        torch.cuda.empty_cache()

    # 评估每阶段 clean/flip/crop TTA，阶段与 TTA 选择只用该折验证标签。
    candidates = []
    for stage in range(len(stages)):
        path = Path(str(stem) + f'.stage{stage}.pt')
        if not path.exists():
            continue
        ckpt = load_pt(path)
        model = make_model(spec, ckpt['size'], False)
        model.load_state_dict(ckpt['state'])
        model.set_grad_checkpointing(False)
        net = parallel(model)
        tfm = transform(ckpt['size'], ckpt['mean'], ckpt['std'])
        clean = predict(net, loader(va_paths, va_y, tfm, args))
        flip = predict(net, loader(va_paths, va_y, tfm, args), flip=True)
        crop = predict(net, loader(va_paths, va_y, tfm, args, crop=0.92), flip=True)
        for mode, probs in [('clean', clean), ('flip', flip), ('crop_flip', (flip+crop)/2)]:
            # 同分时优先较低开销的 clean，然后较早阶段。
            candidates.append((accuracy(probs, va_y), stage, mode, probs))
        del net, model, ckpt
        gc.collect()
        torch.cuda.empty_cache()
    score, stage, mode, va_probs = max(candidates, key=lambda c: c[0])
    ckpt = load_pt(Path(str(stem) + f'.stage{stage}.pt'))
    ckpt['tta_mode'] = mode
    ckpt['tta_score'] = score
    save_pt(ckpt, best_path)
    model = make_model(spec, ckpt['size'], False)
    model.load_state_dict(ckpt['state'])
    model.set_grad_checkpointing(False)
    net = parallel(model)
    tfm = transform(ckpt['size'], ckpt['mean'], ckpt['std'])
    test_probs = predict(net, loader(test_paths, None, tfm, args), flip=mode != 'clean')
    if mode == 'crop_flip':
        test_probs = (test_probs + predict(net, loader(test_paths, None, tfm, args, crop=0.92),
                                          flip=True)) / 2
    tmp = Path(str(result_path) + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, indices=np.array([r['index'] for r in va]),
                            val=va_probs, test=test_probs)
    tmp.replace(result_path)
    print(f'{stem.name}: selected stage={stage}, TTA={mode}, val_acc={score:.5f}', flush=True)
    del net, model, ckpt, candidates
    latest_path.unlink(missing_ok=True)
    for s in range(len(stages)):
        Path(str(stem) + f'.stage{s}.pt').unlink(missing_ok=True)
    gc.collect()
    torch.cuda.empty_cache()


def ensemble(rows, ids, test_hashes, specs, args, out):
    ys = np.array([r['target'] for r in rows])
    all_oof, all_test, individual = [], [], {}
    for spec in specs:
        oof = np.zeros((len(rows), N_CLASSES), dtype=np.float32)
        test = np.zeros((len(ids), N_CLASSES), dtype=np.float32)
        covered = np.zeros(len(rows), dtype=bool)
        for fold in range(args.n_folds):
            path = out / f"{spec['key']}_fold{fold}.predictions.npz"
            if not path.exists():
                print(f'尚未完成所有折：缺少 {path.name}，本次不生成最终 submission.csv', flush=True)
                return
            with np.load(path) as z:
                idx = z['indices']
                if covered[idx].any():
                    raise ValueError('OOF 索引重复')
                oof[idx] = z['val']
                covered[idx] = True
                test += z['test'] / args.n_folds
        if not covered.all():
            raise ValueError('OOF 预测不完整')
        all_oof.append(oof)
        all_test.append(test)
        individual[spec['key']] = accuracy(oof, ys)
    # 只有 2 个模型时用粗网格，避免过度调整大量融合参数。
    weights = [np.array([1.0])] if len(specs) == 1 else [
        np.array([w, 1-w]) for w in [0.5, 0.25, 0.75, 0.0, 1.0]]
    best = None
    for w in weights:
        p = sum(wi*pi for wi, pi in zip(w, all_oof))
        score = accuracy(p, ys)
        if best is None or score > best[0]:
            best = (score, w, p)
    score, w, oof = best
    probs = sum(wi*pi for wi, pi in zip(w, all_test))
    # 输入数据内部已有完全相同的图像：仅当训练标签一致时采用已知标签。
    # 不使用外部 FER 标签，不读取测试真值；冲突重复样本交给模型预测。
    matches = []
    if not args.no_exact_match:
        known = defaultdict(set)
        for row in rows:
            known[row['group']].add(row['target'])
        for i, digest in enumerate(test_hashes):
            labels = known.get(digest, set())
            if len(labels) == 1:
                label = next(iter(labels))
                probs[i] = np.eye(N_CLASSES, dtype=np.float32)[label]
                matches.append({'ID': ids[i], 'Emotion': label})
    pred = probs.argmax(-1)
    if probs.shape != (len(ids), N_CLASSES) or not np.isfinite(probs).all():
        raise ValueError('最终概率异常')
    if not np.allclose(probs.sum(1), 1, atol=1e-4):
        raise ValueError('最终概率未归一化')
    path = out / 'submission.csv'
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Emotion'])
        writer.writerows((id_, int(y)) for id_, y in zip(ids, pred))
    tmp.replace(path)
    np.save(out / 'test_probabilities.npy', probs)
    np.save(out / 'oof_probabilities.npy', oof)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(confusion, (ys, oof.argmax(-1)), 1)
    atomic_json(dict(individual_oof_accuracy=individual, selected_oof_accuracy=score,
                     weights={s['key']: float(wi) for s, wi in zip(specs, w)},
                     class_to_id=CLASS_TO_ID, confusion_matrix=confusion.tolist(),
                     exact_match_count=len(matches),
                     note='OOF 用于选择阶段/TTA/融合，不是完全独立的最终测试成绩。'),
                out / 'validation_metrics.json')
    atomic_json(matches, out / 'exact_train_test_matches.json')
    print(f'OOF accuracy={score:.5f}, weights={w.tolist()}, exact_matches={len(matches)}; '
          f'提交文件：{path}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=DEFAULT_ROOT)
    parser.add_argument('--output', default=DEFAULT_OUT)
    parser.add_argument('--models', choices=['both', 'convnext', 'eva02'], default='both')
    parser.add_argument('--folds', default='all', help='例如 0,1；默认全部折')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--finetune-epochs', type=int, default=15)
    parser.add_argument('--batch-per-gpu', type=int, default=1)
    parser.add_argument('--accum', type=int, default=32, help='T4x2 有效 batch=64')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--ensemble-only', action='store_true')
    parser.add_argument('--no-exact-match', action='store_true',
                        help='关闭输入数据内部完全相同且标签一致图像的标签复用')
    args = parser.parse_args(argv)
    if args.n_folds < 2 or args.epochs < 1 or args.finetune_epochs < 0:
        parser.error('n-folds >= 2, epochs >= 1, finetune-epochs >= 0')
    if args.batch_per_gpu < 1 or args.accum < 1 or args.workers < 0:
        parser.error('batch/accum 必须为正数，workers >= 0')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, test_paths, ids, report, test_hashes = inspect_data(args.data_root)
    atomic_json(report, out / 'data_audit.json')
    if args.audit_only:
        return
    setup_ml()
    seed_all(args.seed)
    ys = np.array([r['target'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    folds = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    for fold, (tr, va) in enumerate(folds.split(np.zeros(len(rows)), ys, groups)):
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError('重复图像跨训练/验证折')
        if set(ys[va]) != set(range(N_CLASSES)):
            raise ValueError('某验证折缺少类别，请调整 n-folds')
        for i in va:
            rows[i]['fold'] = fold
    for i, row in enumerate(rows):
        row['index'] = i
    # 同目录不允许不同配置混用。模型/折选择属于调度，可跨会话更改。
    config = {k: v for k, v in vars(args).items()
              if k not in {'models', 'folds', 'audit_only', 'ensemble_only', 'data_root', 'output'}}
    config['specs'] = SPECS
    config['labels'] = CLASS_TO_ID
    config['manifest'] = hashlib.sha256(json.dumps(
        [(Path(r['path']).name, r['target'], r['group'], r['fold']) for r in rows],
        sort_keys=True).encode()).hexdigest()
    config['test_ids'] = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    config['test_pixels'] = hashlib.sha256('\n'.join(test_hashes).encode()).hexdigest()
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cfg_path = out / 'run_config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding='utf-8'))['signature'] != signature:
        raise ValueError('此输出目录含不同配置的结果；请指定新的 --output')
    atomic_json(dict(signature=signature, config=config), cfg_path)
    with (out / 'fold_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'target', 'group', 'fold', 'index'])
        writer.writeheader()
        writer.writerows(rows)
    specs = [s for s in SPECS if args.models == 'both' or s['key'] == args.models]
    selected = list(range(args.n_folds)) if args.folds == 'all' else [int(v) for v in args.folds.split(',')]
    if not selected or any(f < 0 or f >= args.n_folds for f in selected):
        parser.error('--folds 必须为合法折编号')
    if not args.ensemble_only:
        for spec in specs:
            for fold in selected:
                run_fold(spec, fold, rows, test_paths, args, out, signature)
    ensemble(rows, ids, test_hashes, specs, args, out)


if __name__ == '__main__':
    # 直接粘贴入 Notebook 时，忽略 IPython 的 -f kernel.json 参数。
    main([] if 'ipykernel' in sys.modules else None)
