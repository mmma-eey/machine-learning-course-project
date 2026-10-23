"""Task3 中心凹定位：将本文件全部内容复制到 Kaggle 的一个 Code 单元格运行。
开启 GPU 和 Internet，添加 2026task3 数据集；支持已解压目录或原始 ZIP。
默认 ResNet34(ImageNet) + U-Net 热力图，5 折，每折最多 60 轮并早停。
结果 /kaggle/working/task3_fovea_output/submission.csv；指标 RMSE 越低越好。
训练集只有 80 张，不能将本任务的轮数与 Task2 的大数据集直接比较。
"""
print('[启动] Task3 中心凹定位代码开始执行', flush=True)

import argparse
from contextlib import contextmanager
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
import zipfile

# 直接粘贴运行时，在这里修改。默认仅使用第一块 GPU，降低多卡启动的复杂度。
DATA_ROOT = '/kaggle/input/2026task3'
OUTPUT_DIR = '/kaggle/working/task3_fovea_output'
RESUME_DIR = 'auto'  # 续跑：将此前整个输出目录作为 Input 加入，可自动识别。
TRAIN_FOLDS = '0,1,2,3,4'
IMAGE_SIZE = 512
MAX_EPOCHS = 60
PATIENCE = 12
BATCH_SIZE = 4
REPEATS = 4  # 每轮重复训练图并重新随机增强，默认每折约64个 batch。
SEED = 2026
BACKBONE = 'resnet34.a1_in1k'
CODE_VERSION = 'task3-fovea-v1'


def log(message):
    print(message, flush=True)


@contextmanager
def activity(label):
    stop = threading.Event()
    start = time.monotonic()
    log('[开始] ' + label)
    def heartbeat():
        while not stop.wait(30):
            log(f'[等待] {label}，已耗时 {time.monotonic()-start:.0f}s')
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)
    log(f'[完成] {label}，耗时 {time.monotonic()-start:.1f}s')


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def atomic_torch_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def save_npz(path, **values):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def setup():
    global np, pd, cv2, torch, nn, F, timm, StratifiedGroupKFold, DataLoader, Dataset
    for module, package in [('numpy', 'numpy'), ('pandas', 'pandas'),
                            ('cv2', 'opencv-python-headless'), ('sklearn', 'scikit-learn'),
                            ('timm', 'timm==1.0.30')]:
        if importlib.util.find_spec(module) is None:
            with activity('安装依赖 ' + package):
                subprocess.run([sys.executable, '-m', 'pip', 'install', package,
                                '--disable-pip-version-check'], check=True)
    with activity('导入 PyTorch 和图像处理依赖'):
        import numpy as np
        import pandas as pd
        import cv2
        import torch
        from torch import nn
        import torch.nn.functional as F
        import timm
        from sklearn.model_selection import StratifiedGroupKFold
        from torch.utils.data import DataLoader, Dataset
    cv2.setNumThreads(0)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    log(f'[环境] torch={torch.__version__}, timm={timm.__version__}, '
        f'CUDA可用={torch.cuda.is_available()}')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            relative = Path(member.filename.replace('\\', '/'))
            if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
                raise ValueError('ZIP 含不安全路径：' + member.filename)
            resolved = (destination / relative).resolve()
            if not resolved.is_relative_to(destination):
                raise ValueError('ZIP 路径越界')
        z.extractall(destination)
    return destination


def find_data(requested, out):
    requested = Path(requested)
    if requested.is_file() and requested.suffix.lower() == '.zip':
        with activity('解压输入 ZIP'):
            requested = safe_extract(requested, out / 'input_cache')
    roots = [requested] if requested.is_dir() else []
    if not roots and Path('/kaggle/input').is_dir():
        roots = [Path('/kaggle/input')]
    candidates = []
    for root in roots:
        candidates.extend(root.rglob('fovea_localization_train_GT.csv'))
    if not candidates and roots:
        archives = [p for root in roots for p in root.rglob('2026task3.zip')]
        if len(archives) == 1:
            with activity('解压数据集中的 2026task3.zip'):
                extracted = safe_extract(archives[0], out / 'input_cache')
            candidates = list(extracted.rglob('fovea_localization_train_GT.csv'))
    candidates = sorted(set(p.resolve() for p in candidates))
    if len(candidates) != 1:
        raise ValueError(f'找到 {len(candidates)} 份训练标注。请把 DATA_ROOT/--data-root '
                         '设为唯一的原始数据目录或ZIP路径。')
    root = candidates[0].parent
    for name in ['train', 'test', 'sample_submission.csv']:
        if not (root / name).exists():
            raise FileNotFoundError(root / name)
    log('[数据目录] ' + str(root))
    return root


def image_index(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}:
            continue
        if not path.stem.isdigit():
            raise ValueError('图像文件名不是数字ID：' + str(path))
        image_id = int(path.stem)
        if image_id in result:
            raise ValueError(f'重复的图像ID: {image_id}')
        result[image_id] = path
    return result


def parse_submission_ids(template):
    if list(template.columns) != ['ImageID', 'value']:
        raise ValueError('提交模板必须是 ImageID,value 两列')
    pairs = []
    for value in template.ImageID:
        match = re.fullmatch(r'(\d+)_Fovea_([XY])', str(value))
        if not match:
            raise ValueError('无法解析提交行ID：' + str(value))
        pairs.append((int(match.group(1)), match.group(2)))
    if len(set(pairs)) != len(pairs):
        raise ValueError('提交模板行ID重复')
    ids = sorted(set(i for i, _ in pairs))
    if set(pairs) != {(i, axis) for i in ids for axis in ['X', 'Y']}:
        raise ValueError('每张图必须包含X、Y两行')
    return ids, pairs


def crop_and_render(rgb, size):
    """最大眼底连通区域裁剪 + 保持比例的仿射映射，保留精确反变换。"""
    height, width = rgb.shape[:2]
    thumb_scale = 512 / max(height, width)
    small = cv2.resize(rgb, (max(1, round(width*thumb_scale)),
                             max(1, round(height*thumb_scale))))
    mask = (small.max(axis=2) > 12).astype('uint8')
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        box = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
        x, y, w, h, area = map(int, box)
    else:
        x, y, w, h, area = 0, 0, small.shape[1], small.shape[0], 0
    if area < small.size / 3 * 0.05:
        x, y, w, h = 0, 0, small.shape[1], small.shape[0]
    sx, sy = width / small.shape[1], height / small.shape[0]
    margin = 0.025 * max(w*sx, h*sy)
    x0, y0 = max(0, math.floor(x*sx-margin)), max(0, math.floor(y*sy-margin))
    x1, y1 = min(width-1, math.ceil((x+w)*sx+margin)), min(height-1, math.ceil((y+h)*sy+margin))
    scale = min((size-1)/max(1, x1-x0), (size-1)/max(1, y1-y0))
    px = ((size-1) - (x1-x0)*scale)/2
    py = ((size-1) - (y1-y0)*scale)/2
    matrix = np.array([[scale, 0, px-scale*x0], [0, scale, py-scale*y0]], np.float32)
    image = cv2.warpAffine(rgb, matrix, (size, size), flags=cv2.INTER_AREA)
    # 温和 CLAHE，保留原图颜色及病变结构；训练和推理完全一致。
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    enhanced = lab.copy()
    enhanced[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    image = cv2.addWeighted(image, 0.65, enhanced, 0.35, 0)
    image[np.max(cv2.warpAffine(rgb, matrix, (size, size)), axis=2) <= 6] = 0
    return image, matrix, cv2.invertAffineTransform(matrix), [x0, y0, x1, y1]


def apply_xy(matrix, xy):
    return np.asarray(matrix)[:, :2] @ np.asarray(xy) + np.asarray(matrix)[:, 2]


def load_data(root, size, out, n_splits, seed):
    gt = pd.read_csv(root / 'fovea_localization_train_GT.csv')
    if not {'data', 'Fovea_X', 'Fovea_Y'}.issubset(gt.columns):
        raise ValueError('训练标注缺少 data,Fovea_X,Fovea_Y 列')
    if gt['data'].duplicated().any():
        raise ValueError('训练标注ID重复')
    train_files, test_files = image_index(root/'train'), image_index(root/'test')
    template = pd.read_csv(root/'sample_submission.csv', dtype={'ImageID': str})
    test_ids, template_pairs = parse_submission_ids(template)
    if set(test_ids) != set(test_files):
        raise ValueError('测试图像与提交模板ID不一致')
    labels = {int(r['data']): [float(r['Fovea_X']), float(r['Fovea_Y'])]
              for r in gt.to_dict('records')}
    if set(labels) != set(train_files):
        raise ValueError('训练图像与坐标标注ID不一致')
    records = []
    fingerprint = hashlib.sha256()
    for kind, files in [('train', train_files), ('test', test_files)]:
        for image_id, path in sorted(files.items()):
            encoded = np.frombuffer(path.read_bytes(), np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('图像损坏：' + str(path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            xy = np.array(labels.get(image_id, [0, 0]), np.float32)
            visible = bool(xy[0] != 0 or xy[1] != 0) if kind == 'train' else None
            if kind == 'train' and (not np.isfinite(xy).all() or
                    (visible and not (0 < xy[0] < w and 0 < xy[1] < h))):
                raise ValueError(f'非法坐标：{image_id}: {xy}')
            image, matrix, inverse, bbox = crop_and_render(rgb, size)
            mapped = apply_xy(matrix, xy) if visible else np.zeros(2, np.float32)
            if visible and (np.any(mapped < 0) or np.any(mapped > size-1)):
                raise ValueError(f'裁剪排除了标注点：{image_id}，请检查数据')
            record = dict(id=image_id, kind=kind, width=w, height=h, group=digest,
                          xy=xy, visible=visible, image=image, matrix=matrix,
                          inverse=inverse, mapped=mapped, crop=bbox, fold=-1)
            records.append(record)
            fingerprint.update(f'{kind}:{image_id}:{digest}:{w}:{h}'.encode())
            if kind == 'train':
                fingerprint.update(np.asarray(xy, np.float64).tobytes())
            if len(records) % 20 == 0:
                log(f'[数据] 已读取 {len(records)}/{len(train_files)+len(test_files)} 张')
    fingerprint.update(json.dumps(template_pairs).encode())
    train = [r for r in records if r['kind'] == 'train']
    test = [r for r in records if r['kind'] == 'test']
    # 相同像素图像强制同折；按相机尺寸和可见性分层。
    # 不再细分左右位置：本数据部分组合仅2张，不足以分配到5折。
    strata = [f"{int(r['visible'])}_{int(r['width']<2500)}" for r in train]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(train)), strata,
                                                   groups=[r['group'] for r in train])):
        if not any(train[int(i)]['visible'] for i in tr):
            raise ValueError('某折训练集没有可见中心凹样本')
        for i in va:
            train[int(i)]['fold'] = fold
    group_folds = {}
    for r in train:
        if r['group'] in group_folds and group_folds[r['group']] != r['fold']:
            raise AssertionError('重复图像跨折')
        group_folds[r['group']] = r['fold']
    has_negatives = any(not r['visible'] for r in train)
    manifest = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in r.items() if k not in {'image', 'matrix', 'inverse', 'mapped'}}
                for r in records]
    audit = dict(train_count=len(train), test_count=len(test),
                 invisible_train_count=sum(not r['visible'] for r in train),
                 unique_train_images=len(group_folds), visibility_head_enabled=has_negatives,
                 fold_counts={str(f): sum(r['fold']==f for r in train) for f in range(n_splits)},
                 metric='RMSE over all X/Y values in original pixels', records=manifest,
                 limitation=('本训练集无(0,0)样本，不训练不可见判别；默认每张测试图均输出定位坐标。'
                             if not has_negatives else '不可见判别根据训练集中的零坐标样本学习。'))
    atomic_json(audit, out/'data_audit.json')
    with (out/'fold_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'width', 'height', 'Fovea_X', 'Fovea_Y', 'group', 'fold'])
        for r in train:
            writer.writerow([r['id'], r['width'], r['height'], *r['xy'], r['group'], r['fold']])
    make_preview(train, out/'training_labels_preview.jpg', use_labels=True)
    log(f"[数据] train={len(train)}, test={len(test)}, 唯一训练图={len(group_folds)}, "
        f"不可见标注={audit['invisible_train_count']}, folds={audit['fold_counts']}")
    log('[注意] ' + audit['limitation'])
    return train, test, template, fingerprint.hexdigest(), has_negatives


def make_preview(records, path, predictions=None, use_labels=False):
    from PIL import Image, ImageDraw
    count = min(20, len(records))
    columns, cell = 5, 224
    canvas = Image.new('RGB', (columns*cell, math.ceil(count/columns)*(cell+24)), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    for i, record in enumerate(records[:count]):
        image = Image.fromarray(record['image']).resize((cell, cell))
        d = ImageDraw.Draw(image)
        xy = record['xy'] if use_labels else predictions[i]
        if np.any(xy):
            x, y = apply_xy(record['matrix'], xy) * (cell-1)/(record['image'].shape[0]-1)
            d.line((x-6, y, x+6, y), fill='lime', width=2)
            d.line((x, y-6, x, y+6), fill='lime', width=2)
        x0, y0 = (i%columns)*cell, (i//columns)*(cell+24)
        canvas.paste(image, (x0, y0))
        draw.text((x0+4, y0+cell+4), f"{record['id']:04d}", fill='black')
    canvas.save(path)


def augment(image, point, visible):
    size = image.shape[0]
    angle = random.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D(((size-1)/2, (size-1)/2), angle,
                                    random.uniform(0.9, 1.1)).astype(np.float32)
    matrix[:, 2] += np.random.uniform(-0.04, 0.04, 2) * (size-1)
    transformed = apply_xy(matrix, point) if visible else point
    if not visible or (np.all(transformed > 3) and np.all(transformed < size-4)):
        image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)
        point = transformed
    if random.random() < 0.5:
        image = image[:, ::-1]
        if visible:
            point = np.array([size-1-point[0], point[1]], np.float32)
    if random.random() < 0.5:
        image = image[::-1]
        if visible:
            point = np.array([point[0], size-1-point[1]], np.float32)
    if random.random() < 0.8:
        image = image.astype(np.float32)/255
        image = np.power(image, random.uniform(0.85, 1.15))
        image *= np.random.uniform(0.92, 1.08, (1, 1, 3)) * random.uniform(0.9, 1.1)
        image = np.clip(image*255, 0, 255).astype(np.uint8)
    if random.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
    return np.ascontiguousarray(image), np.asarray(point, np.float32)


def dataset_class():
    class FoveaDataset(Dataset):
        def __init__(self, records, training=False, repeats=1):
            self.records, self.training, self.repeats = records, training, repeats
        def __len__(self):
            return len(self.records)*self.repeats
        def __getitem__(self, index):
            r = self.records[index % len(self.records)]
            image, point = r['image'], r['mapped'].copy()
            if self.training:
                image, point = augment(image, point, r['visible'])
            tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()/255
            tensor = (tensor-torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
            return tensor, torch.tensor(point/(image.shape[0]-1), dtype=torch.float32), \
                torch.tensor(float(bool(r['visible'])), dtype=torch.float32)
    return FoveaDataset


def build_model(pretrained, visibility):
    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                     nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                                     nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                     nn.GroupNorm(8, cout), nn.ReLU(inplace=True))
        def forward(self, x):
            return self.net(x)
    class FoveaUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(BACKBONE, pretrained=pretrained,
                                             features_only=True, out_indices=(0, 1, 2, 3, 4))
            channels = self.encoder.feature_info.channels()
            self.decoders = nn.ModuleList()
            previous = channels[-1]
            for skip, output in zip(reversed(channels[:-1]), [192, 128, 64, 32]):
                self.decoders.append(Block(previous+skip, output))
                previous = output
            self.heatmap = nn.Conv2d(previous, 1, 1)
            self.visibility = nn.Linear(channels[-1], 1) if visibility else None
        def forward(self, x):
            features = self.encoder(x)
            y = features[-1]
            for decoder, skip in zip(self.decoders, reversed(features[:-1])):
                y = F.interpolate(y, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                y = decoder(torch.cat([y, skip], dim=1))
            visibility = (self.visibility(features[-1].mean((2, 3))).squeeze(1)
                          if self.visibility is not None else x.new_full((len(x),), 20))
            return self.heatmap(y).squeeze(1), visibility
    return FoveaUNet()


def spatial_coordinates(probabilities):
    h, w = probabilities.shape[-2:]
    xs = torch.linspace(0, 1, w, device=probabilities.device)
    ys = torch.linspace(0, 1, h, device=probabilities.device)
    return torch.stack([(probabilities.sum(1)*xs).sum(1),
                        (probabilities.sum(2)*ys).sum(1)], dim=1)


def loss_function(logits, visibility_logits, xy, visible, visibility_enabled):
    logits = logits.float()
    b, h, w = logits.shape
    xs = torch.linspace(0, 1, w, device=logits.device).view(1, 1, w)
    ys = torch.linspace(0, 1, h, device=logits.device).view(1, h, 1)
    sigma = 0.012  # 相对输入宽度约1.2%，512输入对应约6像素。
    target = torch.exp(-((xs-xy[:, 0, None, None])**2 +
                          (ys-xy[:, 1, None, None])**2)/(2*sigma**2))
    target = target / target.sum((1, 2), keepdim=True).clamp_min(1e-12)
    logprob = logits.flatten(1).log_softmax(1).view(b, h, w)
    heat_loss = -(target*logprob).sum((1, 2))
    coordinates = spatial_coordinates(logprob.exp())
    coordinate_loss = ((coordinates-xy)**2).sum(1)*30
    localization = ((heat_loss+coordinate_loss)*visible).sum()/visible.sum().clamp_min(1)
    classification = (F.binary_cross_entropy_with_logits(visibility_logits.float(), visible)
                      if visibility_enabled else logits.new_zeros(()))
    return localization + classification*0.3


def decode_heatmaps(probabilities, mode):
    if mode == 'integral':
        return spatial_coordinates(probabilities)
    b, h, w = probabilities.shape
    maximum = probabilities.flatten(1).argmax(1)
    xs = torch.arange(w, device=probabilities.device).view(1, 1, w)
    ys = torch.arange(h, device=probabilities.device).view(1, h, 1)
    radius = max(3, round(w*0.035))
    mask = ((xs-(maximum%w)[:, None, None]).abs() <= radius) & \
           ((ys-(maximum//w)[:, None, None]).abs() <= radius)
    local = probabilities*mask
    local = local/local.sum((1, 2), keepdim=True).clamp_min(1e-12)
    return spatial_coordinates(local)


def inverse_coordinates(normalized, records):
    result = []
    for point, r in zip(normalized, records):
        original = apply_xy(r['inverse'], point*(r['image'].shape[0]-1))
        result.append(np.clip(original, [0, 0], [r['width']-1, r['height']-1]))
    return np.asarray(result, np.float64)


@__import__('contextlib').contextmanager
def amp_context(device):
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
        yield


def predict(model, loader, records, device, tta=1):
    model.eval()
    outputs = {'integral': [], 'local': []}
    all_visible = []
    variants = [()] if tta == 1 else [(), (-1,), (-2,), (-2, -1)]
    with torch.inference_mode():
        for images, _, _ in loader:
            images = images.to(device)
            averaged, visible = None, 0
            for dims in variants:
                view = images.flip(dims) if dims else images
                with amp_context(device):
                    logits, vis_logits = model(view)
                probability = logits.float().flatten(1).softmax(1).view_as(logits)
                if dims:
                    probability = probability.flip(dims)
                averaged = probability if averaged is None else averaged+probability
                visible = visible+vis_logits.float().sigmoid()
            averaged = averaged/len(variants)
            for mode in outputs:
                outputs[mode].append(decode_heatmaps(averaged, mode).cpu().numpy())
            all_visible.append((visible/len(variants)).cpu().numpy())
    return {mode: inverse_coordinates(np.concatenate(parts), records)
            for mode, parts in outputs.items()}, np.concatenate(all_visible)


def rmse(target, predictions):
    return float(np.sqrt(np.mean((np.asarray(target)-np.asarray(predictions))**2)))


def choose_decoding(predictions, probability, truth, visibility):
    choices = []
    for mode, points in predictions.items():
        for threshold in ([0.25, 0.5, 0.75] if visibility else [0.0]):
            result = points.copy()
            result[probability < threshold] = 0
            choices.append((rmse(truth, result), mode, threshold, result))
    return min(choices, key=lambda x: x[0])


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def update_ema(ema, model, steps):
    decay = min(0.99, (1+steps)/(10+steps))
    source_state = model.state_dict()
    with torch.no_grad():
        for name, value in ema.state_dict().items():
            source = source_state[name]
            if value.is_floating_point():
                value.mul_(decay).add_(source, alpha=1-decay)
            else:
                value.copy_(source)


def restore_outputs(args, out, signature):
    if args.resume_dir == 'none':
        return
    if args.resume_dir == 'auto':
        configs = list(Path('/kaggle/input').rglob('run_config.json')) if Path('/kaggle/input').exists() else []
        candidates = []
        for p in configs:
            try:
                saved = json.loads(p.read_text(encoding='utf-8'))
                if saved.get('signature') == signature:
                    candidates.append(p.parent)
            except (OSError, ValueError):
                pass
    else:
        candidates = [Path(args.resume_dir)]
        saved = json.loads((candidates[0]/'run_config.json').read_text(encoding='utf-8'))
        if saved.get('signature') != signature:
            raise ValueError('指定续跑目录的配置/数据与本次不一致')
    import shutil
    for source in sorted(candidates):
        if source.resolve() == out.resolve():
            continue
        for pattern in ['fold*.best.pt', 'fold*.last.pt', 'fold*.predictions.npz']:
            for file in source.glob(pattern):
                target = out/file.name
                if not target.exists():
                    log(f'[恢复文件] {file} -> {target}')
                    shutil.copy2(file, target)
                else:
                    # 当前工作目录优先，避免覆盖较新的断点。
                    log(f'[恢复] 保留当前已有文件 {target.name}')


def freeze_existing_folds(args, out, base_config, train):
    """库版本变化时仍使用已有运行的分折，避免续跑重新划分验证集。"""
    current = out/'run_config.json'
    if current.exists():
        configs = [current]
    elif args.resume_dir not in {'auto', 'none'}:
        configs = [Path(args.resume_dir)/'run_config.json']
    elif args.resume_dir == 'auto' and Path('/kaggle/input').exists():
        configs = list(Path('/kaggle/input').rglob('run_config.json'))
    else:
        configs = []
    assignments = []
    for path in configs:
        try:
            config = json.loads(path.read_text(encoding='utf-8'))['config']
            without_folds = {k: v for k, v in config.items() if k != 'fold_assignments'}
            if without_folds == base_config:
                assignments.append(config['fold_assignments'])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not assignments:
        return
    if any(a != assignments[0] for a in assignments[1:]):
        raise ValueError('找到多个不同分折的兼容运行，请显式设置 RESUME_DIR')
    frozen = assignments[0]
    if set(frozen) != {str(r['id']) for r in train}:
        raise ValueError('旧分折的图像ID不匹配')
    groups = {}
    for r in train:
        fold = frozen[str(r['id'])]
        if not isinstance(fold, int) or not 0 <= fold < args.n_splits:
            raise ValueError('旧分折编号无效')
        if r['group'] in groups and groups[r['group']] != fold:
            raise ValueError('旧分折将重复图像放到了不同折')
        groups[r['group']] = fold
        r['fold'] = fold
    if set(frozen.values()) != set(range(args.n_splits)):
        raise ValueError('旧分折缺少某一折')
    with (out/'fold_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'width', 'height', 'Fovea_X', 'Fovea_Y', 'group', 'fold'])
        for r in train:
            writer.writerow([r['id'], r['width'], r['height'], *r['xy'], r['group'], r['fold']])
    audit_path = out/'data_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    for record in audit['records']:
        if record['kind'] == 'train':
            record['fold'] = frozen[str(record['id'])]
    audit['fold_counts'] = {str(f): sum(r['fold']==f for r in train) for f in range(args.n_splits)}
    atomic_json(audit, audit_path)
    log('[分折] 已恢复原运行的固定分折')


def validate_fold_file(path, fold, train, test, signature):
    expected = [r['id'] for r in train if r['fold']==fold]
    with np.load(path, allow_pickle=False) as pack:
        if str(pack['signature'].item()) != signature:
            raise ValueError('预测文件配置不匹配：' + str(path))
        if pack['val_ids'].tolist() != expected or pack['test_ids'].tolist() != [r['id'] for r in test]:
            raise ValueError('预测文件图像ID/顺序不匹配：' + str(path))
        for name, count in [('val_xy', len(expected)), ('test_xy', len(test))]:
            if pack[name].shape != (count, 2) or not np.isfinite(pack[name]).all():
                raise ValueError('预测文件坐标无效：' + str(path))


def train_fold(fold, train, test, out, args, signature, visibility, device):
    prediction_path = out/f'fold{fold}.predictions.npz'
    if prediction_path.exists():
        validate_fold_file(prediction_path, fold, train, test, signature)
        log(f'[跳过] fold={fold} 已完成训练和预测')
        return
    seed_everything(args.seed+fold)
    tr = [r for r in train if r['fold'] != fold]
    va = [r for r in train if r['fold'] == fold]
    DatasetClass = dataset_class()
    training_loader = DataLoader(DatasetClass(tr, True, args.repeats), batch_size=args.batch_size,
                                 shuffle=True, num_workers=0, pin_memory=device.type=='cuda')
    validation_loader = DataLoader(DatasetClass(va), batch_size=args.batch_size,
                                   shuffle=False, num_workers=0)
    test_loader = DataLoader(DatasetClass(test), batch_size=args.batch_size, shuffle=False, num_workers=0)
    best_path, last_path = out/f'fold{fold}.best.pt', out/f'fold{fold}.last.pt'
    with activity(f'构建 fold={fold} 模型 / 首次下载 ImageNet 权重'):
        model = build_model(not args.no_pretrained and not last_path.exists(), visibility).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.lr},
        {'params': [p for name, p in model.named_parameters() if not name.startswith('encoder.')],
         'lr': args.lr*5}], weight_decay=1e-4, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    best, bad, steps, start_epoch = float('inf'), 0, 0, 0
    if last_path.exists():
        with activity('读取训练断点 ' + last_path.name):
            state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state['signature'] != signature:
            raise ValueError('断点签名不匹配')
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        start_epoch, best, bad, steps = state['next_epoch'], state['best'], state['bad'], state['steps']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        log(f'[断点恢复] fold={fold}, 下轮={start_epoch+1}, best_RMSE={best:.3f}')
    truth = np.asarray([r['xy'] for r in va], np.float64)
    log(f'[训练] fold={fold}/{args.n_splits-1}, train={len(tr)}, val={len(va)}, '
        f'每轮batch={len(training_loader)}, 最多{args.epochs}轮')
    for epoch in range(start_epoch, args.epochs):
        # 恢复时延续同一早停规则，不能因重启额外训练更多轮。
        if epoch >= args.min_epochs and bad >= args.patience:
            break
        began = time.monotonic()
        warmup = min(1.0, (epoch+1)/3)
        factor = warmup*(0.08+0.92*0.5*(1+math.cos(math.pi*epoch/max(1,args.epochs-1))))
        for group, base in zip(optimizer.param_groups, [args.lr, args.lr*5]):
            group['lr'] = base*factor
        model.train()
        # 小样本小批次：固定 ImageNet BN 统计量，其仿射参数仍可训练。
        for layer in model.encoder.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        total, seen = 0.0, 0
        for batch, (images, xy, visible) in enumerate(training_loader, 1):
            images, xy, visible = images.to(device), xy.to(device), visible.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                heatmap, vis_logits = model(images)
                loss = loss_function(heatmap, vis_logits, xy, visible, visibility)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'fold={fold}, epoch={epoch+1}: loss 非有限')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            steps += 1
            update_ema(ema, model, steps)
            total += loss.item()*len(images); seen += len(images)
            if batch == 1 or batch%10 == 0 or batch==len(training_loader):
                elapsed = time.monotonic()-began
                log(f'  fold={fold} epoch={epoch+1}/{args.epochs} batch={batch}/{len(training_loader)} '
                    f'loss={total/seen:.4f} elapsed={elapsed:.0f}s '
                    f'train_ETA={elapsed/batch*(len(training_loader)-batch):.0f}s')
            if args.smoke_test and batch >= 2:
                break
        candidate = []
        for name, network in [('raw', model), ('ema', ema)]:
            predictions, probability = predict(network, validation_loader, va, device, tta=1)
            score, mode, threshold, _ = choose_decoding(predictions, probability, truth, visibility)
            candidate.append((score, name, mode, threshold, network))
        score, name, mode, threshold, network = min(candidate, key=lambda x: x[0])
        improved = score < best
        if improved:
            best, bad = score, 0
            atomic_torch_save(dict(signature=signature, model=cpu_state(network), fold=fold,
                                   epoch=epoch+1, rmse=best, weights=name, mode=mode,
                                   threshold=threshold), best_path)
        else:
            bad += 1
        seconds = time.monotonic()-began
        log(f'[验证] fold={fold} epoch={epoch+1}/{args.epochs} '
            f'raw_RMSE={candidate[0][0]:.3f} ema_RMSE={candidate[1][0]:.3f} '
            f'best={best:.3f}px 保存最佳={improved} 本轮={seconds:.1f}s '
            f'预计剩余上限={(args.epochs-epoch-1)*seconds/60:.1f}min（本折）')
        checkpoint = dict(signature=signature, model=cpu_state(model), ema=cpu_state(ema),
                          optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                          next_epoch=epoch+1, best=best, bad=bad, steps=steps,
                          python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                          torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        with activity(f'保存 fold={fold} 第{epoch+1}轮断点'):
            atomic_torch_save(checkpoint, last_path)
        del checkpoint
        with (out/'training_history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(fold=fold, epoch=epoch+1, loss=total/seen,
                                    raw_rmse=candidate[0][0], ema_rmse=candidate[1][0],
                                    best_rmse=best, seconds=seconds))+'\n')
        if epoch+1>=args.min_epochs and bad>=args.patience:
            log(f'[早停] fold={fold}, 连续{bad}轮未改善')
            break
    del optimizer, ema, scaler
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
    with activity(f'fold={fold} 最佳权重 + {args.tta}视角预测'):
        state = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        val_modes, val_probability = predict(model, validation_loader, va, device, args.tta)
        score, mode, threshold, val_xy = choose_decoding(val_modes, val_probability, truth, visibility)
        test_modes, test_probability = predict(model, test_loader, test, device, args.tta)
        test_xy = test_modes[mode].copy()
        test_xy[test_probability < threshold] = 0
    save_npz(prediction_path, signature=np.array(signature),
             val_ids=np.array([r['id'] for r in va]), test_ids=np.array([r['id'] for r in test]),
             val_xy=val_xy, test_xy=test_xy, test_conditional_xy=test_modes[mode],
             val_probability=val_probability, test_probability=test_probability,
             mode=np.array(mode), threshold=np.array(threshold),
             best_epoch=np.array(state['epoch']), rmse=np.array(score))
    atomic_json(dict(fold=fold, validation_rmse=score, best_epoch=state['epoch'],
                     weights=state['weights'], decoding=mode, visibility_threshold=threshold,
                     tta=args.tta, val_ids=[r['id'] for r in va]), out/f'fold{fold}.metrics.json')
    log(f'[完成一折] fold={fold}, RMSE={score:.3f}px, 最佳epoch={state["epoch"]}, '
        f'weights={state["weights"]}, decode={mode}')
    del model, state
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()


def assemble(train, test, template, folds, out, signature, visibility):
    test_predictions, test_probabilities = [], []
    val_ids, val_xy, fold_metrics = [], [], {}
    for fold in folds:
        path = out/f'fold{fold}.predictions.npz'
        if not path.exists():
            raise FileNotFoundError(f'fold={fold} 未完成，不能生成所选折的提交')
        validate_fold_file(path, fold, train, test, signature)
        with np.load(path, allow_pickle=False) as p:
            test_predictions.append(p['test_conditional_xy'].copy())
            test_probabilities.append(p['test_probability'].copy())
            val_ids.extend(p['val_ids'].tolist()); val_xy.extend(p['val_xy'].tolist())
            fold_metrics[str(fold)] = float(p['rmse'])
    points = np.mean(test_predictions, axis=0)
    probability = np.mean(test_probabilities, axis=0)
    # 无负样本时始终输出坐标；有负样本时融合概率，以固定0.5阈值判别。
    threshold = 0.0
    if visibility:
        threshold = 0.5
        points[probability < threshold] = 0
    _, pairs = parse_submission_ids(template)
    mapping = {r['id']: xy for r, xy in zip(test, points)}
    result = template.copy()
    result['value'] = [float(mapping[i][0 if axis=='X' else 1]) for i, axis in pairs]
    if len(result)!=2*len(test) or not np.isfinite(result['value'].to_numpy()).all():
        raise ValueError('最终提交行数或数值无效')
    tmp = out/'submission.csv.tmp'
    result.to_csv(tmp, index=False)
    tmp.replace(out/'submission.csv')
    lookup = {r['id']: r['xy'] for r in train}
    truth = np.asarray([lookup[i] for i in val_ids])
    validation = np.asarray(val_xy)
    scores = dict(oof_rmse_pixels=rmse(truth, validation),
                  oof_rmse_x=rmse(truth[:, 0], validation[:, 0]),
                  oof_rmse_y=rmse(truth[:, 1], validation[:, 1]),
                  fold_rmse=fold_metrics, selected_folds=folds, validation_count=len(val_ids),
                  fold_weights={str(f): 1/len(folds) for f in folds},
                  visibility_threshold=threshold,
                  note='OOF为各折未见样本预测的拼接分数，并非五模型共同预测同一独立验证集。')
    atomic_json(scores, out/'validation_metrics.json')
    save_npz(out/'final_predictions.npz', test_ids=np.array([r['id'] for r in test]),
             test_xy=points, visible_probability=probability,
             val_ids=np.array(val_ids), val_xy=validation, val_truth=truth)
    make_preview(test, out/'test_predictions_preview.jpg', predictions=points)
    log(f'[提交完成] {out/"submission.csv"}，{len(result)}行，OOF RMSE={scores["oof_rmse_pixels"]:.3f}px')


def main():
    parser = argparse.ArgumentParser(description='Task3 中心凹热力图定位')
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--resume-dir', default=RESUME_DIR)
    parser.add_argument('--folds', default=TRAIN_FOLDS)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--size', type=int, default=IMAGE_SIZE)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--tta', type=int, choices=[1, 4], default=4)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--ensemble-only', action='store_true')
    parser.add_argument('--cpu', action='store_true', help='仅本地检查，不推荐CPU正式训练')
    parser.add_argument('--no-pretrained', action='store_true', help='仅烟雾测试；正式训练请保留预训练')
    parser.add_argument('--smoke-test', action='store_true', help='128分辨率/一轮/两批次，仅验证流程')
    args = parser.parse_args([] if 'ipykernel' in sys.modules else None)
    if args.smoke_test:
        args.size, args.epochs, args.min_epochs, args.repeats, args.tta = 128, 1, 1, 1, 1
        args.folds = '0'
        if args.output == OUTPUT_DIR:
            args.output = str(Path(OUTPUT_DIR).with_name('task3_smoke_output'))
        log('[烟雾测试] 此模式不产生可用竞赛模型；不要提交它的结果')
    folds = sorted(set(int(f.strip()) for f in args.folds.split(',')))
    if not folds or min(folds)<0 or max(folds)>=args.n_splits:
        raise ValueError('--folds 超出范围')
    if args.size<64 or args.size%32 or min(args.epochs,args.batch_size,args.repeats,args.patience)<1:
        raise ValueError('size必须>=64且为32倍数；epochs/batch/repeats/patience必须为正')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    seed_everything(args.seed)
    with activity('发现并检查图像、裁剪眼底、建立分折'):
        root = find_data(args.data_root, out)
        train, test, template, data_sha, visibility = load_data(root, args.size, out, args.n_splits, args.seed)
    excluded = {'data_root', 'output', 'resume_dir', 'folds', 'prepare_only', 'ensemble_only', 'cpu'}
    config = {k: v for k, v in vars(args).items() if k not in excluded}
    config.update(version=CODE_VERSION, backbone=BACKBONE, data_sha=data_sha)
    freeze_existing_folds(args, out, config, train)
    config['fold_assignments'] = {str(r['id']): r['fold'] for r in train}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = out/'run_config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous.get('signature') != signature:
            raise ValueError('输出目录已有不同配置/数据的结果，请改用新 OUTPUT_DIR，不要混用断点')
    atomic_json(dict(signature=signature, config=config), config_path)
    if args.prepare_only:
        log('[检查完成] 未训练；数据预览和分折已保存到 '+str(out))
        return
    restore_outputs(args, out, signature)
    if not args.ensemble_only:
        if not args.cpu and not torch.cuda.is_available():
            raise RuntimeError('未检测到 GPU。请在 Kaggle Session options 中开启 GPU，不能用CPU悄悄训练。')
        device = torch.device('cpu' if args.cpu else 'cuda:0')
        if device.type=='cuda':
            log(f'[设备] {torch.cuda.get_device_name(0)}；仅使用cuda:0')
        for fold in folds:
            train_fold(fold, train, test, out, args, signature, visibility, device)
    assemble(train, test, template, folds, out, signature, visibility)
    log('[结束] 正式运行请提交 submission.csv；请保留整个输出目录以备续跑和分析。')


if __name__ == '__main__':
    main()
