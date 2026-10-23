# Task3：复用原五折最佳模型，关闭翻转增强

当前 Notebook 的 Input 已包含 `task3fordata` 原始数据集和 `task3` Notebook 输出。这个脚本用于当前目录结构，默认自动查找，不用手动修改 Kaggle 挂载路径。

## 怎么运行

1. 打开 `task3_只预测_直接复制.txt`，复制其中**全部内容**。
2. 将新 Notebook 的默认代码替换成这些代码，确认单元格类型为 **Code**。
3. 可开启 GPU 加速，执行单元格或 Save & Run All。代码仅做预测，没有 GPU 也能用 CPU 执行，只是可能更慢。
4. 看到 `[校验通过]` 后，会依次加载 `fold0.best.pt` 至 `fold4.best.pt`。每折打印原预测的验证 RMSE、本次关闭 TTA 后的 RMSE，以及权重保存时记录的 RMSE。
5. 完成后查看 `validation_comparison.json`。若验证误差下降，可尝试提交新生成的 **`task3_no_tta_output/submission.csv`**；验证改善仍不保证公开榜改善。

新结果目录是 `/kaggle/working/task3_no_tta_output`，与原 `task3_fovea_output` 分开。代码不训练、不下载 ImageNet 权重、不复制大型模型到 Output，也不使用旧的测试预测替代本次重新预测。若缺少 timm 等依赖，会尝试安装，此时需要 Internet。

## 改了什么

原训练程序保存最佳权重时使用单个原图视角验证，但导出预测时固定使用四视角翻转增强。你这次的记录中，第0折从保存时 RMSE 9.18 变成导出时 35.43，因此先验证关闭翻转，而不是重训大模型。

本版保持原数据、眼底裁剪、CLAHE、归一化、图像尺寸、固定分折、模型结构和原最佳权重对应的热力图解码方式。五个模型仍然等权融合，唯一主要变化是推理不再翻转。

它校验原配置的签名、实际图像与坐标/模板的数据指纹、权重签名、折号及旧预测ID。分折直接读取原配置，不重新随机划分。当前训练集没有不可见样本，仍按可见中心凹输出坐标，不新增未经训练的置零阈值。

## 提交哪个文件

- `submission.csv`：新的无翻转增强五模型融合结果。
- `submission_baseline.csv`：原Notebook提交的备份，仅供对照，不是本次新结果。
- `validation_comparison.json`：同一批验证样本上的新旧误差、各折误差及验证是否改善。
- `test_predictions_preview.jpg`：新的20张测试图定位预览。
- `foldN.no_tta_predictions.npz`、`final_predictions.npz`：新预测坐标，便于后续分析。

模型权重继续保留在原 Input。新版 Output 通常很小；如果原始 Input 只有未解压 ZIP，新 Output 还会包含解压后的 `input_cache` 数据目录。

公开榜只包含约一半测试图，OOF也不是五模型预测同一独立验证集的分数。优先看固定分折上的总体误差和异常样本，不要仅凭单次公开榜涨跌断定改动必然有效。

## 路径没有自动识别时

代码顶部可设置：

```python
DATA_ROOT = '/kaggle/input/你的原始数据集目录'
BASELINE_DIR = '/kaggle/input/原Notebook挂载目录/task3_fovea_output'
```

`BASELINE_DIR` 是**直接包含** `run_config.json` 和 `fold0.best.pt` 等文件的文件夹。Input 显示名不一定等于真实目录，默认自动查找最方便。若挂载了多个兼容的旧运行，代码会要求显式指定，避免加载错误权重。

## 已做的本地验证

用真实100张图像与本地流程测试权重跑通了数据核对、固定分折、权重加载、只预测、验证和40行提交导出；确认新路径与原单视角推理逐点一致，并检查模板重排、错误签名/ID拒绝和带用户名的嵌套挂载目录识别。代码不含训练反向或优化器更新。

本地没有你在Kaggle正式训练的五折权重，因此尚未生成新的正式竞赛坐标，也没有承诺新的Kaggle分数。运行完可先把很小的 `validation_comparison.json` 内容发来检查。
