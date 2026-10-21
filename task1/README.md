# 实验一 植物幼苗五分类

最佳方案为 `train_convnext_small.py`（原始文件为 `plant_classifier/kaggle_convnext_small.py`），320输入，原始500张训练图，分层五折，每折400训练100验证，先训头3轮再全量微调16轮，最终EMA，四视角TTA与五模型等权平均。

在Kaggle挂载含 `train/{类别名}/*.png` 与 `test/*.png` 的原始课设数据，开启GPU和Internet，复制脚本全部内容进入Code单元格运行。脚本自动搜索目录；也可设置 `PLANT_DATA_ROOT` 指向含train/test的父目录。

```python
import os
os.environ['PLANT_DATA_ROOT'] = '/kaggle/input/你的数据集/含train和test的目录'
```

输出 `/kaggle/working/submission_convnext_small.csv`。本地也支持设置上述环境变量后 `python train_convnext_small.py`。

`train_clean_384.py` 为1713张清洗数据的384对照，实际是十折划分取前五折，每个90%训练10%验证。分数未超过320版。其余伪标签、仲裁、合并等探索保留在 `legacy/`；历史脚本中的本地路径需要按实际情况调整，不应认为都能直接在当前电脑运行。

`results/` 存放可核实提交CSV与数据审计。训练数据、权重和完整GPU日志未打包，不能据此恢复历史训练全过程。`submission_pseudo.csv` 目前未在已给定task1目录找到，保留对应代码与截图分数，不虚构缺失文件。
