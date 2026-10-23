"""Task3 局部精修：Input保留原始数据和第一版task3的完整输出，复制全部代码运行。
复用原五折权重初始化；只训练局部定位，固定原分折；验证/测试都用相同4视角。
推荐提交 /kaggle/working/task3_refine_output/submission.csv。
"""
print('[启动] Task3 局部精修版；先核对原始数据和第一版输出',flush=True)
import argparse
from contextlib import contextmanager
import copy,csv,gc,hashlib,importlib.util,json,math,os,random,re,shutil,subprocess,sys,threading,time,zipfile
from pathlib import Path
DATA_ROOT='auto'
BASELINE_DIR='auto'
RESUME_DIR='auto'
OUTPUT_DIR='/kaggle/working/task3_refine_output'
TRAIN_FOLDS='0,1,2,3,4'
PATCH_SIZE=320
EPOCHS=40
PATIENCE=10
BATCH_SIZE=4
REPEATS=4
BACKBONE='resnet34.a1_in1k'
PREDICTION_VERSION='task3-refine-v1'

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
    sigma = 0.015  # 相对输入宽度约1.2%，512输入对应约6像素。
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




def make_patch(record, center, side, size):
    """矩阵始终从原图像素坐标出发，缓存图只影响采样，不影响坐标单位。"""
    scale=(size-1)/side
    matrix=np.array([[scale,0,(size-1)/2-scale*center[0]],
                     [0,scale,(size-1)/2-scale*center[1]]],np.float32)
    cached=record['full_image']
    sx=cached.shape[1]/record['width']; sy=cached.shape[0]/record['height']
    sampling=matrix.copy(); sampling[:,0]/=sx; sampling[:,1]/=sy
    sampling[:,2]-=sampling[:,:2]@np.array([(sx-1)/2,(sy-1)/2],np.float32)
    image=cv2.warpAffine(cached,sampling,(size,size),flags=cv2.INTER_LINEAR)
    lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB); enhanced=lab.copy()
    enhanced[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
    enhanced=cv2.cvtColor(enhanced,cv2.COLOR_LAB2RGB)
    enhanced=cv2.addWeighted(image,.65,enhanced,.35,0)
    enhanced[image.max(2)<=6]=0
    return dict(record,image=enhanced,matrix=matrix,inverse=cv2.invertAffineTransform(matrix),
                mapped=apply_xy(matrix,record['xy']) if record['visible'] else np.zeros(2,np.float32))


def patch_side(r):
    x0,y0,x1,y1=r['crop']
    return max(384.,min(x1-x0,y1-y0)*.32)


def patch_training_class():
    class PatchDataset(Dataset):
        def __init__(self,records,size,repeats,training):
            self.records,self.size,self.repeats,self.training=records,size,repeats,training
        def __len__(self): return len(self.records)*self.repeats
        def __getitem__(self,index):
            r=self.records[index%len(self.records)]
            if self.training:
                side=patch_side(r)*random.uniform(.9,1.1)
                jitter=np.clip(np.random.normal(0,.12,2),-.28,.28)*side
                p=make_patch(r,r['xy']+jitter,side,self.size)
                image,point=augment(p['image'],p['mapped'].copy(),True)
            else:
                image,point=r['image'],r['mapped']
            tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255
            tensor=(tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            return tensor,torch.tensor(point/(self.size-1),dtype=torch.float32),torch.tensor(1.,dtype=torch.float32)
    return PatchDataset


def bounded_refinement(points,centers,records):
    delta=points-centers
    # 对所有样本使用同一固定规则；避免局部热力图异常时发生任意大跳动。
    limit=np.array([patch_side(r)*.25 for r in records])
    norm=np.linalg.norm(delta,axis=1)
    delta*=np.minimum(1.,limit/np.maximum(norm,1e-9))[:,None]
    result=centers+delta
    return np.array([np.clip(x,[0,0],[r['width']-1,r['height']-1]) for x,r in zip(result,records)])


def read_coarse(baseline,saved,fold,train,test):
    va=[r for r in train if r['fold']==fold]
    with np.load(baseline/f'fold{fold}.predictions.npz',allow_pickle=False) as p:
        if str(p['signature'].item())!=saved['signature']:
            raise ValueError('原预测签名不匹配')
        if p['val_ids'].tolist()!=[r['id'] for r in va] or p['test_ids'].tolist()!=[r['id'] for r in test]:
            raise ValueError('原预测ID或分折不一致')
        a,b=p['val_xy'].copy(),p['test_xy'].copy()
    for points,records in [(a,va),(b,test)]:
        if points.shape!=(len(records),2) or not np.isfinite(points).all():
            raise ValueError('原始定位预测无效')
    return va,a,b


def refine_fold(fold,train,test,baseline,saved,args,out,signature,device):
    result_path=out/f'refine_fold{fold}.predictions.npz'
    va,coarse_val,coarse_test=read_coarse(baseline,saved,fold,train,test)
    val_ids=np.array([r['id'] for r in va]); test_ids=np.array([r['id'] for r in test])
    if result_path.exists():
        with np.load(result_path,allow_pickle=False) as p:
            if str(p['signature'].item())!=signature or not np.array_equal(p['val_ids'],val_ids) or not np.array_equal(p['test_ids'],test_ids):
                raise ValueError('已有精修结果签名/ID不一致')
        log(f'[跳过] 精修fold={fold}已完成')
        return
    seed_everything(args.seed+fold)
    tr=[r for r in train if r['fold']!=fold]
    val_patches=[make_patch(r,c,patch_side(r),args.size) for r,c in zip(va,coarse_val)]
    test_patches=[make_patch(r,c,patch_side(r),args.size) for r,c in zip(test,coarse_test)]
    DS=patch_training_class()
    train_loader=DataLoader(DS(tr,args.size,args.repeats,True),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(DS(val_patches,args.size,1,False),batch_size=args.batch_size,shuffle=False,num_workers=0)
    test_loader=DataLoader(DS(test_patches,args.size,1,False),batch_size=args.batch_size,shuffle=False,num_workers=0)
    truth=np.array([r['xy'] for r in va],np.float64)
    coarse_score=rmse(truth,coarse_val)
    best_path=out/f'refine_fold{fold}.best.pt'; last_path=out/f'refine_fold{fold}.last.pt'
    with activity(f'初始化精修 fold={fold}，复用该折原模型'):
        initial=torch.load(baseline/f'fold{fold}.best.pt',map_location='cpu',weights_only=False)
        if initial['signature']!=saved['signature'] or initial['fold']!=fold: raise ValueError('原权重不匹配')
        model=build_model(False,False).to(device); model.load_state_dict(initial['model'],strict=True)
        del initial
    ema=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW([
        dict(params=model.encoder.parameters(),lr=2e-5),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')],lr=1e-4)
    ],weight_decay=1e-4,foreach=False)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best,bad,start,steps=coarse_score,0,0,0
    if last_path.exists():
        with activity('恢复精修断点'):
            state=torch.load(last_path,map_location='cpu',weights_only=False)
        if state['signature']!=signature: raise ValueError('精修断点配置不一致')
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        best,bad,start,steps=state['best'],state['bad'],state['next_epoch'],state['steps']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        log(f'[恢复] 精修fold={fold}, 下轮={start+1}')
    log(f'[训练] 精修fold={fold}, train={len(tr)}, val={len(va)}, 原RMSE={coarse_score:.3f}, 每轮{len(train_loader)}批')
    for epoch in range(start,args.epochs):
        if epoch>=args.min_epochs and bad>=args.patience: break
        began=time.monotonic()
        factor=min(1.,(epoch+1)/2)*(.15+.85*.5*(1+math.cos(math.pi*epoch/max(1,args.epochs-1))))
        for group,lr in zip(optimizer.param_groups,[2e-5,1e-4]):group['lr']=lr*factor
        model.train()
        for layer in model.encoder.modules():
            if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()
        total=0.; count=0
        for batch,(images,xy,visible) in enumerate(train_loader,1):
            images,xy,visible=images.to(device),xy.to(device),visible.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                heat,vis=model(images); loss=loss_function(heat,vis,xy,visible,False)
            if not torch.isfinite(loss): raise FloatingPointError('精修loss非有限')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update();steps+=1;update_ema(ema,model,steps)
            total+=loss.item()*len(images);count+=len(images)
            if batch==1 or batch%10==0 or batch==len(train_loader):
                elapsed=time.monotonic()-began
                log(f'  fold={fold} epoch={epoch+1}/{args.epochs} batch={batch}/{len(train_loader)} loss={total/count:.4f} train_ETA={elapsed/batch*(len(train_loader)-batch):.0f}s')
            if args.smoke_test and batch>=2:break
        scores=[]
        for label,network in [('raw',model),('ema',ema)]:
            modes,_=predict(network,val_loader,val_patches,device,tta=args.tta)
            refined=bounded_refinement(modes['local'],coarse_val,va)
            score=rmse(truth,(coarse_val+refined)*.5)
            scores.append((score,label,network,rmse(truth,refined)))
        score,label,network,pure=min(scores,key=lambda x:x[0])
        if score<best:
            best,bad=score,0
            atomic_torch_save(dict(signature=signature,model=cpu_state(network),epoch=epoch+1,
                                   score=best,weights=label),best_path)
        else:bad+=1
        seconds=time.monotonic()-began
        log(f'[验证] fold={fold} epoch={epoch+1}/{args.epochs}: 原={coarse_score:.3f} '
            f'raw融合={scores[0][0]:.3f} ema融合={scores[1][0]:.3f} best={best:.3f}, {seconds:.1f}s/轮；验证与测试均TTA={args.tta}')
        state=dict(signature=signature,model=cpu_state(model),ema=cpu_state(ema),
                   optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                   next_epoch=epoch+1,best=best,bad=bad,steps=steps,
                   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        with activity(f'保存精修fold={fold} epoch={epoch+1}断点'):atomic_torch_save(state,last_path)
        del state
        with (out/'refine_history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(fold=fold,epoch=epoch+1,loss=total/count,baseline=coarse_score,
                                    raw_blend=scores[0][0],ema_blend=scores[1][0],best=best,seconds=seconds))+'\n')
        if epoch+1>=args.min_epochs and bad>=args.patience:
            log(f'[早停] fold={fold}');break
    del optimizer,scaler,ema
    scores=None;network=None
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    if best_path.exists():
        state=torch.load(best_path,map_location='cpu',weights_only=False)
        if state['signature']!=signature:raise ValueError('最佳精修权重签名不一致')
        model.load_state_dict(state['model'])
        with activity(f'fold={fold}最佳精修权重正式预测'):
            vp,_=predict(model,val_loader,val_patches,device,args.tta)
            tp,_=predict(model,test_loader,test_patches,device,args.tta)
        refined_val=bounded_refinement(vp['local'],coarse_val,va)
        refined_test=bounded_refinement(tp['local'],coarse_test,test)
        val_result=(coarse_val+refined_val)*.5;test_result=(coarse_test+refined_test)*.5
        used=True
    else:
        val_result,test_result=coarse_val,coarse_test
        used=False
        log(f'[回退] fold={fold}精修验证未超过原模型，保留该折原预测')
    save_npz(result_path,signature=np.array(signature),val_ids=val_ids,test_ids=test_ids,
             val_xy=val_result,test_xy=test_result,coarse_val=coarse_val,coarse_test=coarse_test,
             refinement_used=np.array(used))
    log(f'[完成一折] fold={fold} 原RMSE={coarse_score:.3f} 新RMSE={rmse(truth,val_result):.3f}, 采用精修={used}')
    del model
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()


def assemble_refinement(train,test,template,folds,out,signature,baseline):
    predictions=[];original=[];truth=[];values=[];coarse=[];ids=[];metrics={}
    lookup={r['id']:r['xy'] for r in train}
    for fold in folds:
        with np.load(out/f'refine_fold{fold}.predictions.npz',allow_pickle=False) as p:
            if str(p['signature'].item())!=signature or p['test_ids'].tolist()!=[r['id'] for r in test] or p['val_ids'].tolist()!=[r['id'] for r in train if r['fold']==fold]:
                raise ValueError('精修结果签名或ID不一致')
            for key,n in [('val_xy',len(p['val_ids'])),('test_xy',len(test)),('coarse_val',len(p['val_ids'])),('coarse_test',len(test))]:
                if p[key].shape!=(n,2) or not np.isfinite(p[key]).all():raise ValueError('精修预测无效')
            target=np.array([lookup[int(i)] for i in p['val_ids']])
            metrics[str(fold)]=dict(baseline_rmse=rmse(target,p['coarse_val']),
                                    refined_blend_rmse=rmse(target,p['val_xy']),used=bool(p['refinement_used']))
            predictions.append(p['test_xy'].copy()); original.append(p['coarse_test'].copy())
            truth.extend(target);values.extend(p['val_xy']);coarse.extend(p['coarse_val']);ids.extend(p['val_ids'].tolist())
    test_xy=np.mean(predictions,axis=0); base_xy=np.mean(original,axis=0)
    baseline_score=rmse(truth,coarse);refine_score=rmse(truth,values)
    # 默认只推荐验证改善的方案；不能把验证下降包装为必然公开提分。
    use_refinement=refine_score<baseline_score-1e-6
    write_submission(test_xy,test,template,out/'submission_refine_candidate.csv')
    write_submission(base_xy,test,template,out/'submission_baseline.csv')
    write_submission(test_xy if use_refinement else base_xy,test,template,out/'submission.csv')
    make_preview(test,out/'test_predictions_preview.jpg',predictions=test_xy if use_refinement else base_xy)
    atomic_json(dict(baseline_oof_rmse=baseline_score,new_oof_rmse=refine_score,
                     recommendation='refined_blend' if use_refinement else 'baseline',selected_folds=folds,
                     fold_metrics=metrics,blend_alpha=.5,tta_consistent=True,
                     note='OOF用于模型选择，有选择偏差；并非五模型对同一独立集的融合分数，不能保证公开榜改善。'),out/'validation_comparison.json')
    save_npz(out/'final_predictions.npz',test_ids=np.array([r['id'] for r in test]),test_xy=test_xy,
             coarse_test_xy=base_xy,val_ids=np.array(ids),val_xy=np.array(values),val_truth=np.array(truth))
    log(f'[完成] 原OOF={baseline_score:.3f}，新OOF={refine_score:.3f}；推荐={"精修融合" if use_refinement else "原预测"}')
    log('[提交文件] '+str(out/'submission.csv'))


def main():
    parser=argparse.ArgumentParser(description='Task3 局部精修训练')
    parser.add_argument('--data-root',default=DATA_ROOT);parser.add_argument('--baseline-dir',default=BASELINE_DIR)
    parser.add_argument('--resume-dir',default=RESUME_DIR);parser.add_argument('--output',default=OUTPUT_DIR)
    parser.add_argument('--folds',default=TRAIN_FOLDS);parser.add_argument('--size',type=int,default=PATCH_SIZE)
    parser.add_argument('--epochs',type=int,default=EPOCHS);parser.add_argument('--min-epochs',type=int,default=12)
    parser.add_argument('--patience',type=int,default=PATIENCE);parser.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    parser.add_argument('--repeats',type=int,default=REPEATS);parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--tta',type=int,choices=[1,4],default=4)
    parser.add_argument('--cpu',action='store_true');parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args([] if 'ipykernel' in sys.modules else None)
    if args.smoke_test:
        args.size,args.epochs,args.min_epochs,args.repeats,args.tta,args.folds=128,1,1,1,1,'0'
        if args.output==OUTPUT_DIR:args.output='/kaggle/working/task3_refine_smoke'
        log('[测试模式] 只验证流程；不能作为正式竞赛成绩')
    if args.size<64 or args.size%32 or min(args.epochs,args.patience,args.batch_size,args.repeats)<1:raise ValueError('训练参数无效')
    baseline,saved=find_baseline(args.baseline_dir)
    folds=sorted(set(int(f) for f in args.folds.split(',')))
    if not folds or min(folds)<0 or max(folds)>=saved['config']['n_splits']:raise ValueError('折号无效')
    for fold in folds:
        for name in [f'fold{fold}.best.pt',f'fold{fold}.predictions.npz']:
            if not (baseline/name).exists():raise FileNotFoundError('原输出缺少 '+name)
    out=Path(args.output).resolve()
    if out==baseline or out.is_relative_to(baseline) or baseline.is_relative_to(out):raise ValueError('新输出必须独立于原输出')
    out.mkdir(parents=True,exist_ok=True)
    setup();seed_everything(args.seed)
    with activity('检查原图、坐标和固定分折'):
        root=find_prediction_data(args.data_root,out,baseline);cfg=saved['config']
        train,test,template,data_sha,visibility=load_data(root,cfg['size'],out,cfg['n_splits'],cfg['seed'],cfg['fold_assignments'])
    if data_sha!=cfg['data_sha']:raise ValueError('数据指纹与第一版模型不同')
    if visibility:raise ValueError('此精修版针对当前全可见数据，不能直接用于含不可见标注的数据')
    keys=['size','epochs','min_epochs','patience','batch_size','repeats','seed','tta','smoke_test']
    config={k:getattr(args,k) for k in keys};config.update(version=PREDICTION_VERSION,source_signature=saved['signature'],blend=.5)
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    path=out/'refine_config.json'
    if path.exists() and read_json(path)['signature']!=signature:raise ValueError('输出中已有不同配置，请换新OUTPUT_DIR')
    atomic_json(dict(signature=signature,config=config),path)
    sources=[]
    if args.resume_dir=='auto' and Path('/kaggle/input').exists():
        sources=[p.parent for p in Path('/kaggle/input').rglob('refine_config.json') if read_json(p).get('signature')==signature]
    elif args.resume_dir not in ['auto','none']:
        src=Path(args.resume_dir)
        if read_json(src/'refine_config.json')['signature']!=signature:raise ValueError('精修续跑配置不一致')
        sources=[src]
    if len(sources)>1:raise ValueError('多个精修输出，请显式设置RESUME_DIR')
    for source in sources:
        for file in source.glob('refine_fold*'):
            if file.is_file() and not (out/file.name).exists():shutil.copy2(file,out/file.name)
    if args.prepare_only:
        log('[检查完成] 数据与第一版相同；未训练');return
    if not args.cpu and not torch.cuda.is_available():raise RuntimeError('请开启GPU；不要无意中用CPU正式训练')
    device=torch.device('cpu' if args.cpu else 'cuda:0')
    with activity('建立高清原图缓存（最大边1536，仅保存在内存）'):
        for kind,records in [('train',train),('test',test)]:
            paths=image_index(root/kind)
            for r in records:
                rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(paths[r['id']].read_bytes(),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                scale=min(1.,1536/max(rgb.shape[:2]))
                r['full_image']=cv2.resize(rgb,(round(r['width']*scale),round(r['height']*scale)),interpolation=cv2.INTER_AREA)
    for fold in folds:refine_fold(fold,train,test,baseline,saved,args,out,signature,device)
    assemble_refinement(train,test,template,folds,out,signature,baseline)
    log('[结束] 保留整个精修输出可续跑；原13.64分结果保持不变')


if __name__=='__main__':main()
