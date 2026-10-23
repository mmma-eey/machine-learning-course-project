"""Task3 只预测：读取原 Notebook 的 fold0..4.best.pt，关闭翻转增强，不训练。
将本文件全部内容复制到 Kaggle 的一个 Code 单元格运行。
Input 添加原始 task3fordata 数据集和原 task3 Notebook 输出；二者均自动查找。
输出 /kaggle/working/task3_no_tta_output/submission.csv。
"""
print('[启动] Task3 只预测版开始；不会训练，不会下载预训练权重', flush=True)
import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile

DATA_ROOT = 'auto'      # 图中 task3fordata，自动查找；也可填写具体目录/ZIP。
BASELINE_DIR = 'auto'   # 图中 Notebook task3 的 task3_fovea_output，自动查找。
OUTPUT_DIR = '/kaggle/working/task3_no_tta_output'
BATCH_SIZE = 4
BACKBONE = 'resnet34.a1_in1k'
PREDICTION_VERSION = 'task3-no-tta-v1'

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



def load_data(root, size, out, n_splits, seed, frozen):
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
    # 严格复用旧模型对应的分折，不调用新的随机分折。
    if set(frozen) != {str(r['id']) for r in train}:
        raise ValueError('旧分折的训练ID与当前数据不一致')
    for r in train:
        fold = frozen[str(r['id'])]
        if not isinstance(fold, int) or not 0 <= fold < n_splits:
            raise ValueError('旧分折编号无效')
        r['fold'] = fold
    if set(frozen.values()) != set(range(n_splits)):
        raise ValueError('旧配置缺少某一折的分配')
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



def rmse(target, predictions):
    return float(np.sqrt(np.mean((np.asarray(target)-np.asarray(predictions))**2)))



# 本片段由 build_prediction_only.py 接在相同预处理/模型定义后面；不单独运行。

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def find_baseline(requested):
    if requested != 'auto':
        candidates = [Path(requested)]
    elif Path('/kaggle/input').is_dir():
        candidates = sorted(p.parent for p in Path('/kaggle/input').rglob('run_config.json'))
    else:
        raise FileNotFoundError('未找到 /kaggle/input，请设置 BASELINE_DIR/--baseline-dir')
    valid = []
    for folder in candidates:
        try:
            saved = read_json(folder/'run_config.json')
            cfg = saved['config']
            if cfg.get('version') == 'task3-fovea-v1' and cfg.get('backbone') == BACKBONE:
                valid.append((folder, saved))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if len(valid) != 1:
        raise ValueError(f'找到 {len(valid)} 份兼容的旧运行。请将 BASELINE_DIR 指向唯一含 '
                         'run_config.json 和 fold0.best.pt 等文件的 task3_fovea_output 目录。')
    folder, saved = valid[0]
    expected = hashlib.sha256(json.dumps(saved['config'], sort_keys=True).encode()).hexdigest()
    if saved.get('signature') != expected:
        raise ValueError('旧 run_config.json 的签名校验失败，不应手动修改原配置')
    cfg = saved['config']
    if not isinstance(cfg.get('fold_assignments'), dict):
        raise ValueError('旧运行缺少固定分折信息')
    if cfg['size'] < 64 or cfg['size'] % 32:
        raise ValueError('旧运行图像尺寸无效')
    if cfg.get('smoke_test') or cfg.get('no_pretrained'):
        log('[注意] 当前输入是测试/无预训练权重；它只可用于程序验证，不能代表正式竞赛模型。')
    log('[旧结果目录] ' + str(folder))
    return folder.resolve(), saved


def find_prediction_data(requested, out, baseline):
    if requested != 'auto':
        root = Path(requested)
        if root.is_file() and root.suffix.lower() == '.zip':
            with activity('解压原始数据 ZIP'):
                root = safe_extract(root, out/'input_cache')
        if not root.is_dir():
            raise FileNotFoundError('原始数据路径不存在：' + str(root))
        roots = [root]
    else:
        if not Path('/kaggle/input').is_dir():
            raise FileNotFoundError('未找到 /kaggle/input，请设置 DATA_ROOT/--data-root')
        roots = [Path('/kaggle/input')]
    candidates = sorted(set(p.resolve() for root in roots
                            for p in root.rglob('fovea_localization_train_GT.csv')))
    # 原输出可能自带 input_cache；优先使用独立原始数据集，避免同份数据重复定位。
    external = [p for p in candidates if not p.is_relative_to(baseline)]
    if external:
        candidates = external
    if not candidates:
        archives = sorted(set(p for root in roots for p in root.rglob('2026task3.zip')))
        if len(archives)==1:
            with activity('解压数据集中的 2026task3.zip'):
                cache = safe_extract(archives[0], out/'input_cache')
            candidates = list(cache.rglob('fovea_localization_train_GT.csv'))
    if len(candidates) != 1:
        raise ValueError(f'找到 {len(candidates)} 份原始坐标标注。请把 DATA_ROOT 指向 '
                         'task3fordata 数据集的准确目录或唯一 ZIP。')
    root = candidates[0].parent
    for name in ['train', 'test', 'sample_submission.csv']:
        if not (root/name).exists():
            raise FileNotFoundError(root/name)
    log('[原始数据目录] ' + str(root))
    return root


def inference_dataset_class():
    class InferenceDataset(Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, index):
            image = self.records[index]['image']
            tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()/255
            mean = torch.tensor([.485, .456, .406])[:, None, None]
            std = torch.tensor([.229, .224, .225])[:, None, None]
            return (tensor-mean)/std
    return InferenceDataset


def predict_without_tta(model, loader, records, device, label):
    model.eval()
    outputs = {'integral': [], 'local': []}
    all_visible = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch, images in enumerate(loader, 1):
            images = images.to(device)
            with amp_context(device):
                logits, vis_logits = model(images)
            probabilities = logits.float().flatten(1).softmax(1).view_as(logits)
            for mode in outputs:
                outputs[mode].append(decode_heatmaps(probabilities, mode).cpu().numpy())
            all_visible.append(vis_logits.float().sigmoid().cpu().numpy())
            if batch==1 or batch%10==0 or batch==len(loader):
                log(f'[预测] {label} batch={batch}/{len(loader)}, '
                    f'耗时={time.monotonic()-started:.1f}s，TTA=关闭')
    return {mode: inverse_coordinates(np.concatenate(parts), records)
            for mode, parts in outputs.items()}, np.concatenate(all_visible)


def load_old_predictions(path, signature, val_ids, test_ids):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as p:
        if str(p['signature'].item()) != signature:
            raise ValueError('旧预测文件签名不匹配：' + str(path))
        if p['val_ids'].tolist() != val_ids or p['test_ids'].tolist() != test_ids:
            raise ValueError('旧预测的验证/测试ID与固定分折不一致：' + str(path))
        points = p['val_xy'].copy()
        if points.shape != (len(val_ids), 2) or not np.isfinite(points).all():
            raise ValueError('旧验证预测坐标无效')
    return points


def write_submission(points, test, template, path):
    points = np.asarray(points, np.float64)
    if points.shape != (len(test), 2) or not np.isfinite(points).all():
        raise ValueError('最终坐标形状或数值无效')
    for r, xy in zip(test, points):
        if not (0 <= xy[0] < r['width'] and 0 <= xy[1] < r['height']):
            raise ValueError('最终坐标超出原图范围')
    _, pairs = parse_submission_ids(template)
    mapping = {r['id']: xy for r, xy in zip(test, points)}
    result = template.copy()
    result['value'] = [float(mapping[i][0 if axis=='X' else 1]) for i, axis in pairs]
    temporary = path.with_suffix('.csv.tmp')
    result.to_csv(temporary, index=False)
    temporary.replace(path)


def run_inference(train, test, template, baseline, saved, folds, out, batch_size, device, visibility):
    DatasetClass = inference_dataset_class()
    test_loader = DataLoader(DatasetClass(test), batch_size=batch_size,
                             shuffle=False, num_workers=0)
    test_ids = [r['id'] for r in test]
    all_val_ids, all_val_xy, all_truth = [], [], []
    all_old_xy, test_points, test_probabilities = [], [], []
    fold_metrics = {}
    for position, fold in enumerate(folds, 1):
        va = [r for r in train if r['fold']==fold]
        if not va:
            raise ValueError(f'fold={fold} 没有验证样本')
        val_ids = [r['id'] for r in va]
        truth = np.asarray([r['xy'] for r in va], np.float64)
        val_loader = DataLoader(DatasetClass(va), batch_size=batch_size,
                                shuffle=False, num_workers=0)
        path = baseline/f'fold{fold}.best.pt'
        log(f'[进度] 模型 {position}/{len(folds)}：读取 {path.name}（不训练）')
        with activity('加载最佳权重 ' + path.name):
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state.get('signature') != saved['signature'] or state.get('fold') != fold:
                raise ValueError('权重签名或折号不匹配：' + str(path))
            mode = state.get('mode')
            if mode not in {'integral', 'local'}:
                raise ValueError('旧权重缺少合法的热力图解码方式')
            threshold = float(state.get('threshold', 0.0)) if visibility else 0.0
            model = build_model(pretrained=False, visibility=visibility).to(device)
            model.load_state_dict(state['model'], strict=True)
        with activity(f'fold={fold} 原图视角验证和预测'):
            val_modes, val_probability = predict_without_tta(model, val_loader, va, device,
                                                             f'fold={fold}验证')
            test_modes, test_probability = predict_without_tta(model, test_loader, test, device,
                                                               f'fold={fold}测试')
        # 固定使用最佳权重保存时的解码方式，不重新扩大验证集上的参数搜索。
        val_xy = val_modes[mode].copy()
        val_xy[val_probability < threshold] = 0
        score = rmse(truth, val_xy)
        old = load_old_predictions(baseline/f'fold{fold}.predictions.npz', saved['signature'],
                                   val_ids, test_ids)
        if old is not None:
            old_score = rmse(truth, old)
            all_old_xy.extend(old.tolist())
        else:
            old_score = None
        checkpoint_score = float(state['rmse'])
        comparison = (f'原四视角RMSE={old_score:.3f}, ' if old_score is not None else '')
        log(f'[验证结果] fold={fold}: {comparison}本次无TTA_RMSE={score:.3f}, '
            f'权重保存时RMSE={checkpoint_score:.3f}, 解码={mode}')
        if abs(score-checkpoint_score) > max(0.5, checkpoint_score*.05):
            log('[提示] 本次验证与权重记录差异较大，请先检查环境、数据和运行结果，暂勿据此保证提分。')
        info = dict(validation_count=len(va), no_tta_rmse_pixels=score,
                    original_tta_rmse_pixels=old_score,
                    checkpoint_no_tta_rmse_pixels=checkpoint_score,
                    checkpoint_epoch=int(state['epoch']), checkpoint_weights=state['weights'],
                    decoding=mode, visibility_threshold=threshold)
        fold_metrics[str(fold)] = info
        save_npz(out/f'fold{fold}.no_tta_predictions.npz',
                 source_signature=np.array(saved['signature']),
                 val_ids=np.asarray(val_ids), test_ids=np.asarray(test_ids),
                 val_xy=val_xy, test_conditional_xy=test_modes[mode],
                 val_probability=val_probability, test_probability=test_probability,
                 mode=np.array(mode), threshold=np.array(threshold))
        all_val_ids.extend(val_ids); all_val_xy.extend(val_xy.tolist()); all_truth.extend(truth.tolist())
        test_points.append(test_modes[mode]); test_probabilities.append(test_probability)
        del model, state
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()
    # 与旧版相同：等权融合，不根据公开榜分数调整权重。
    points = np.mean(test_points, axis=0)
    probability = np.mean(test_probabilities, axis=0)
    threshold = 0.5 if visibility else 0.0
    points[probability < threshold] = 0
    truth = np.asarray(all_truth, np.float64)
    predictions = np.asarray(all_val_xy, np.float64)
    new_score = rmse(truth, predictions)
    old_score = rmse(truth, np.asarray(all_old_xy)) if len(all_old_xy)==len(truth) else None
    full_folds = set(folds)==set(range(saved['config']['n_splits']))
    # 只在覆盖与旧提交相同的全部模型时复制备份，避免拿部分折与全折混比。
    baseline_csv = baseline/'submission.csv'
    if baseline_csv.exists() and full_folds:
        prior = pd.read_csv(baseline_csv, dtype={'ImageID': str})
        if list(prior.columns)!=list(template.columns) or prior.ImageID.tolist()!=template.ImageID.tolist():
            raise ValueError('原提交CSV与本次模板不一致')
        if not np.isfinite(pd.to_numeric(prior['value'], errors='coerce')).all():
            raise ValueError('原提交CSV含非数值坐标')
        shutil.copy2(baseline_csv, out/'submission_baseline.csv')
    write_submission(points, test, template, out/'submission.csv')
    metrics = dict(source_signature=saved['signature'], prediction_version=PREDICTION_VERSION,
                   selected_folds=folds, fold_weights={str(f):1/len(folds) for f in folds},
                   tta=1, retraining_performed=False, validation_count=len(truth),
                   no_tta_oof_rmse_pixels=new_score, original_tta_oof_rmse_pixels=old_score,
                   no_tta_oof_rmse_x=rmse(truth[:,0],predictions[:,0]),
                   no_tta_oof_rmse_y=rmse(truth[:,1],predictions[:,1]),
                   validation_improved=(new_score<old_score) if old_score is not None else None,
                   fold_metrics=fold_metrics, visibility_threshold=threshold,
                   note='OOF为各折未参与训练样本的拼接分数，不是五模型对同一独立验证集的融合分数；公开榜可能不同。')
    atomic_json(metrics, out/'validation_comparison.json')
    save_npz(out/'final_predictions.npz', test_ids=np.asarray(test_ids), test_xy=points,
             visible_probability=probability, val_ids=np.asarray(all_val_ids),
             val_xy=predictions, val_truth=truth)
    make_preview(test, out/'test_predictions_preview.jpg', predictions=points)
    log(f'[完成] 新提交：{out/"submission.csv"}，{len(template)}行')
    log(f'[汇总] 无TTA OOF RMSE={new_score:.3f}px')
    if old_score is not None:
        log(f'[对照] 同一批验证样本，原TTA OOF RMSE={old_score:.3f}px')
        if new_score < old_score:
            log('[建议] 验证误差下降，可尝试提交新 submission.csv；公开榜是否下降仍需提交验证。')
        else:
            log('[建议] 本次验证未改善，先保留原13.64分结果，不必急于提交新CSV。')


def main():
    parser = argparse.ArgumentParser(description='Task3 复用最佳模型、关闭翻转增强、仅预测')
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--baseline-dir', default=BASELINE_DIR)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--folds', default='auto', help='默认使用原配置的全部折')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--cpu', action='store_true', help='CPU只预测，无需重新训练')
    parser.add_argument('--check-only', action='store_true', help='只验证目录、数据和固定分折')
    args = parser.parse_args([] if 'ipykernel' in sys.modules else None)
    if args.batch_size<1:
        raise ValueError('batch-size必须为正数')
    baseline, saved = find_baseline(args.baseline_dir)
    out = Path(args.output).resolve()
    if out == baseline or out.is_relative_to(baseline) or baseline.is_relative_to(out):
        raise ValueError('新输出目录必须独立于原运行目录，不能覆盖或嵌套原结果')
    cfg = saved['config']
    folds = (list(range(cfg['n_splits'])) if args.folds=='auto'
             else sorted(set(int(f.strip()) for f in args.folds.split(','))))
    if not folds or min(folds)<0 or max(folds)>=cfg['n_splits']:
        raise ValueError('所选折号无效')
    for fold in folds:
        if not (baseline/f'fold{fold}.best.pt').is_file():
            raise FileNotFoundError(f'缺少 fold{fold}.best.pt；请添加原 Notebook 的完整输出')
    out.mkdir(parents=True, exist_ok=True)
    prediction_config = dict(version=PREDICTION_VERSION, source_signature=saved['signature'],
                             folds=folds, size=cfg['size'], tta=1,
                             decoding='saved-best-checkpoint', weights='equal')
    config_path = out/'prediction_config.json'
    if config_path.exists() and read_json(config_path)!=prediction_config:
        raise ValueError('新输出目录已有不同配置结果，请使用新的 OUTPUT_DIR')
    setup()
    with activity('读取真实图像、核对原模型数据与分折'):
        root = find_prediction_data(args.data_root, out, baseline)
        train, test, template, data_sha, visibility = load_data(
            root, cfg['size'], out, cfg['n_splits'], cfg['seed'], cfg['fold_assignments'])
    if data_sha != cfg['data_sha']:
        raise ValueError('当前图像、标注或模板与原模型的数据指纹不同，停止预测，避免混用数据')
    atomic_json(prediction_config, config_path)
    log(f'[校验通过] 数据与旧运行相同，固定原分折，size={cfg["size"]}，models={folds}')
    if args.check_only:
        log('[检查结束] 未加载网络、未训练、未生成提交CSV')
        return
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    torch.backends.cudnn.benchmark = False
    if device.type=='cuda':
        log('[设备] ' + torch.cuda.get_device_name(0) + '；只用cuda:0')
    else:
        log('[设备] CPU；只预测也可运行，但可能比GPU慢')
    run_inference(train, test, template, baseline, saved, folds, out,
                  args.batch_size, device, visibility)
    log('[结束] 未训练、未修改原Input权重。所有新结果均保存在 '+str(out))


if __name__ == '__main__':
    main()
