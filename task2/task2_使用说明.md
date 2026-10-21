# Task2 Kaggle 人脸表情分类

## 数据与标签

已检查全部 35,340 张图片，均为 48×48 灰度 JPEG，全部可以解码。训练集 28,275 张，测试集 7,065 张。提交编码采用你提供的竞赛规则：Angry=0、Fear=1、Happy=2、Sad=3、Surprise=4、Neutral=5。Neutral 不能按文件夹字母顺序编码为 3。

训练集存在 1,194 张额外重复图像，39 个完全相同像素的分组具有冲突标签；520 张测试图像与训练图像像素完全相同，其中 511 张对应无冲突训练标签。代码使用灰度像素 SHA-256 分组进行 StratifiedGroupKFold，避免重复图跨训练/验证导致虚高分数。保留冲突标签而不猜测哪一个正确。默认对测试集中与训练图像完全一致且训练标签无冲突的图像复用已有标签，输出记录可审查；可以用 `--no-exact-match` 关闭。不会下载其他 FER 数据或借用外部测试标签。

## 模型与训练选择

采用两种互补的骨干：

- `convnext_large.fb_in22k_ft_in1k`：ImageNet-22K 预训练、ImageNet-1K 微调的 CNN，先 224 分辨率训练，再 384 精调。官方模型卡：https://huggingface.co/timm/convnext_large.fb_in22k_ft_in1k
- `eva02_large_patch14_448.mim_in22k_ft_in22k_in1k`：ImageNet-22K 掩码图像建模预训练，再经 ImageNet-22K/1K 微调的 Transformer，两阶段保持原生 448 分辨率。官方模型卡：https://huggingface.co/timm/eva02_large_patch14_448.mim_in22k_ft_in22k_in1k

输入灰度图复制为 RGB，使用各模型的预训练归一化参数。放大图片用于匹配模型输入，并不会增加原始图片的真实细节；高分辨率阶段是否保留由验证表现决定。模型规模也不保证一定高于较小模型，最终以实测验证和竞赛测试结果为准。

默认每个模型进行五折训练，共 10 个独立模型，每折 40 轮常规训练加 15 轮精调。增强包括水平翻转、小角度旋转、轻度裁剪和平移、亮度/对比度扰动、少量 Random Erasing；前阶段 50% 的 batch 使用低强度 Mixup，后阶段关闭 Mixup/擦除并减弱增强。选择这些操作是为了保留眼睛、眉毛、嘴部表情信号；不使用垂直翻转或大面积 CutMix。

优化使用 AdamW、分类头更高学习率、早期骨干更低学习率、warmup + cosine decay、轻度 label smoothing、梯度裁剪、EMA。每轮比较原始参数和 EMA 参数，在验证集上保存最佳模型。第二阶段从第一阶段最佳参数开始，以 15% 基础学习率精调；最终可回退到第一阶段。每折比较原图、水平翻转平均、原图/轻度中心裁剪及翻转的四视图平均，再用所选方式预测测试集。

五折测试概率取平均；两种模型的融合权重通过 OOF 验证在小网格中选择，网格含单模型选项，因此不会强行融合表现更差的模型。不会用公开榜单成绩反复拟合权重。由于验证标签用于选择权重、TTA 和最佳轮次，输出的 OOF accuracy 是模型选择指标，不能称为完全独立测试精度，也不能保证最终榜单分数。

## 在 Kaggle 运行

1. 新建 Notebook，加入截图里的 `task2forml` 数据集。默认根目录为 `/kaggle/input/task2forml`，脚本会自动定位实际嵌套的 `fer_data/fer_data/train` 和 `test`。也兼容根目录下一层直接是 train/test 的组织方式。
2. Settings 选择 GPU T4 ×2，并打开 Internet，首次需要安装/更新 timm 及下载预训练权重。不会替换 Kaggle 自带的 CUDA torch。若当前竞赛环境不能联网，需要提前将 timm 安装包与两模型的预训练权重作为 Input 加入，并配置本地权重加载；默认脚本使用联网下载。
3. 最方便：导入随附 `task2_kaggle.ipynb` 后 Run All。或把 `task2_kaggle.py` 整个文件复制进一个代码单元格运行。
4. 如果把 Python 文件上传为 Kaggle Input，则执行：

```python
!python /kaggle/input/你的代码数据集名称/task2_kaggle.py
```

全部完成后，提交 `/kaggle/working/task2_output/submission.csv`。文件严格保持原模板 ID 顺序，只有 `ID,Emotion` 两列，没有 pandas 索引。每日提交上限按你提供的规则为 5 次，组名按要求设为个人学号。脚本不会自动提交。

## 多会话与断点恢复

这一方案可能超过单次 Kaggle 会话时限或每周 GPU 配额，不能因为忽略训练成本就绕过平台限制。建议分模型/分折运行；下面的命令保留默认训练参数，只改变调度范围：

```python
# 第一批：ConvNeXt 的第 0 折。
!python /kaggle/input/你的代码数据集名称/task2_kaggle.py --models convnext --folds 0
# 下一批：指定其他折，或同一命令继续未完成折。
!python /kaggle/input/你的代码数据集名称/task2_kaggle.py --models convnext --folds 1,2
!python /kaggle/input/你的代码数据集名称/task2_kaggle.py --models eva02 --folds 0
# 全部两模型五折完成后，仅重新融合。
!python /kaggle/input/你的代码数据集名称/task2_kaggle.py --ensemble-only
```

默认每张 GPU 的微批量为 1，累积 32 次，在 T4×2 上有效 batch=64。启用 fp16 AMP 和梯度检查点以适配 Large 模型。若仍出现显存不足，可把 `--accum` 调高并检查没有其他 Notebook 进程占用 GPU；微批量已为 1，累积次数本身不会进一步降低单步激活显存。必要时改用显存更大的 GPU，或将文件顶端 SPECS 改成相应 Base 模型，并使用新输出目录。实际 GPU 训练未在本地 CPU 环境执行。

每个 epoch 后保存 `.latest.pt`，含模型、EMA、优化器、AMP scaler、随机数状态。恢复粒度为 epoch，若会话在某个 epoch 中间结束则会重跑该 epoch；多进程增强的恢复不承诺逐位一致。已完成折保留 `.best.pt` 和 `.predictions.npz`，跳过已完成折。运行配置和输入指纹必须一致，否则脚本拒绝混用结果；改变模型参数、训练轮数、batch 等时，请使用新的 `--output` 目录。

**跨会话必须自行持久化工作目录**：Kaggle 新会话不保证保留上次 `/kaggle/working`。保存上次 Notebook 输出版本，或把 `task2_output` 上传为私有 Dataset，在下一会话中加入为 Input，再复制到工作目录。例如：

```python
import shutil
shutil.copytree('/kaggle/input/你的已保存结果/task2_output',
                '/kaggle/working/task2_output', dirs_exist_ok=True)
```

不要混入不同配置的旧预测文件。若只想先跑一个模型完整五折，用 `--models convnext` 即可输出该模型提交；后续两模型均完成后再用 `--models both --ensemble-only` 融合。部分折未完成时，代码会说明缺少哪些结果，并不生成新的最终提交。

## 输出与检查

- `submission.csv`：可提交文件。
- `data_audit.json`、`fold_manifest.csv`：数据审查和分折记录。
- `run_config.json`：训练配置与输入指纹。
- `validation_metrics.json`：各模型 OOF accuracy、融合权重、按竞赛标签顺序的混淆矩阵。
- `oof_probabilities.npy`、`test_probabilities.npy`：用于后续检查的概率。
- `exact_train_test_matches.json`：标签无冲突的训练/测试完全相同图像记录。
- 每折 `.best.pt`：最终采用的推理权重、分辨率和 TTA 模式。

本地快速检查无需 torch 或 GPU，只需 Pillow：

```powershell
python task2_kaggle.py --audit-only --data-root "C:/Users/lenovo/Desktop/dataset-for-task/task2" --output ./task2_local_audit
```

本地已经完成全量图片解码、数据/模板对应检查。模型 API、张量尺寸与训练/输出逻辑的本地检查结果另见 `task2_local_audit/verification.json`。这些检查不能替代在 Kaggle 上的真实 GPU 训练。
