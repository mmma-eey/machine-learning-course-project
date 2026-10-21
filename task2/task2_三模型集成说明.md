# 三模型集成：保留0.714基准，新增两次Tiny训练

本版保持已经成功运行的单卡、图片缓存、workers=0、轮内恢复配置。采用同一种 ImageNet-22K 预训练 ConvNeXt-Tiny，复用第0折，再训练第1、2折。每折用原来五份分组数据中的四份训练、一份验证，因此每个模型仍使用约80%的训练数据。这是“原五折划分中的三个模型”，不是重新做三等份划分。

三个模型对测试集的概率按1/3等权平均，再对标签无冲突的训练/测试完全相同图片使用训练标签。先不改变增强和学习率，以便单独检验集成能否改善0.714的成绩；实际提分没有保证。

## 你需要的文件

- `task2_三模型集成_直接复制.txt`：把全部内容粘贴进一个 Kaggle Code 单元格。
- `task2_ensemble3_copy.py`：相同代码的 Python 文件。
- `task2_baseline_0714.zip`：约1.86 MB，包含你已完成的第0折原始预测、配置、冻结分折以及0.714基准提交。不需要上传55 MB模型权重，因为第0折无需再次推理。

原本输出目录、模型权重和0.714提交均未修改。

## Kaggle 操作

1. 将 `task2_baseline_0714.zip` 上传为一个私有 Kaggle Dataset，例如命名 `task2-baseline-0714`。上传时解压 ZIP，确保 Dataset 文件树中能看到 `run_config.json`、`fold_manifest.csv` 和 `convnext_fold0.predictions.npz`。它们可以在同一个子文件夹里。
2. 在训练 Notebook 的 Input 中同时加入原数据集 `2026task2` 和这个基准 Dataset。若原 Notebook 的 `/kaggle/working/task2_copy_output` 仍完整存在，代码也能直接从该目录复用。
3. 将文本文件全部内容粘贴到一个 Code 单元格，开启 GPU 和 Internet，运行该单元格。先检查启动和复用日志，再安排后台执行。
4. 默认 `REUSE_DIR = 'auto'` 自动寻找基准；`TRAIN_FOLDS = '0,1,2'`，第0折会跳过，只新增第1、2折，各最多20+5轮，支持早停。

成功复用会打印：

```text
[分折] 已加载原始 fold_manifest；不会因 sklearn 版本变化重新划分
[复用完成] fold0 预测已校验，不需要重新训练或重新推理该模型
convnext_fold0: 已完成，跳过
```

如果提示找不到基准结果，确认 Dataset 已解压并 Add Input。也可把代码底部的 `REUSE_DIR` 改成含上述三个文件的实际目录。找不到基准时程序会停止，不会擅自从头训练三个模型。

本版会检查基准配置、训练图片名称/标签/像素指纹、冻结分折、测试ID顺序和测试图片指纹。请保持默认训练参数；改变参数会使本次严格复用检查失败。GPU型号可不同，但当前代码固定使用第0张卡。

## 输出与提交

最终提交：`/kaggle/working/task2_ensemble3_output/submission.csv`。

如果从提供的压缩包导入，输出目录还会保留 `submission_baseline_0714.csv`。提交新的 `submission.csv` 与原0.714结果比较，保留原提交记录。

`validation_metrics.json` 包含每折验证准确率、所选折的拼接验证分数和测试融合权重。这里的验证分数由每张图片对应的未参与训练模型预测拼接而成；它**不是三模型同时预测同一个独立验证集的融合分数**。不能拿已在某模型训练集中的图片评估该模型融合后的准确率。

## 分会话执行与恢复

需要拆开执行时，可先将 `TRAIN_FOLDS = '0,1'`，得到两模型提交；保存整个输出目录，再在下一会话恢复该目录并改为 `'0,1,2'`。第0、1折都会自动跳过，只训练第2折。

轮内恢复文件每约10分钟保存一次。跨会话必须把整个 `task2_ensemble3_output` 目录保存为输出或私有Dataset，下一次复制回 `/kaggle/working/task2_ensemble3_output` 后再运行。只添加基准包只能复用最初第0折，不能恢复后来第1、2折的训练进度。

使用Python文件执行时也可以：

```python
!python -u /kaggle/input/你的代码数据集/task2_ensemble3_copy.py --folds 0,1
# 下一次恢复工作目录后：
!python -u /kaggle/input/你的代码数据集/task2_ensemble3_copy.py --folds 0,1,2
```

仅检查基准兼容性、不训练时可加 `--prepare-only`，该模式允许CPU运行；三个模型完成后可用 `--ensemble-only` 重新生成融合提交。

本地已使用你真实的旧结果完成基准兼容性检查，并验证三折概率平均、提交ID顺序、验证索引和缺折保护。新模型尚未在Kaggle训练，无法预报提分幅度。
