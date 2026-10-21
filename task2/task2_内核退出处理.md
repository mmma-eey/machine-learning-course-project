# 本次故障与下一次运行

提供的日志显示训练在 batch 625/1415 附近结束，loss 从 1.88 下降到 1.61，说明此前训练确实运行了。625 个 batch 用时 73.7 分钟，对应整轮约 167 分钟，不含验证、写入和后阶段更高分辨率。原实用版每轮才保存一次完整恢复状态；如果此次是第一轮且没有 `.latest.pt`，这些未保存的更新无法恢复。

`DeadKernelError` 是 Jupyter/papermill 发现内核退出后的异常，不提供根因。日志没有普通 CUDA OOM 的 Python 异常，不足以确认显存不足，也没有到日志设置的 43200 秒执行超时。主机内存耗尽导致进程被杀、原生库崩溃和平台终止均有可能，当前无法区分。调试器和 nbconvert 的 SyntaxWarning 也不能证明它们引发了本次退出。

## 已生成修正版

使用 `task2_kaggle_stable.ipynb`，默认仍为 ImageNet 预训练的 ConvNeXt-Base、单划分、最多20+5轮，保留竞赛标签编码、增强、EMA、TTA 和提交顺序。

- 固定 `cuda:0`，移除 DataParallel，即使选择 T4×2，也只使用第一张卡。这样避免每批复制模型及跨卡同步的额外开销，但不保证单卡在所有环境中都更快。
- 默认 workers=0，不创建 DataLoader 子进程，避免 Notebook 中多进程加载及主机内存复制风险。
- 全量检查时缓存约 81 MB 的灰度像素（另有对象和字典开销），训练中读取缓存而不反复打开 Kaggle Input 文件。
- 默认关闭梯度检查点，减少重计算；单卡 batch=8、累积8次，仍为有效 batch=64。若出现有明确 traceback 的 CUDA 显存不足，可以打开 `--grad-checkpoint` 或减小微批量。
- AdamW 禁用 foreach 临时张量批处理，并释放阶段结束时遗留的模型引用；加载恢复文件在 Linux 上使用 mmap，减少一次性加载带来的主机内存压力。
- 输出主进程 RSS、cgroup 可读的内存用量、GPU 已分配/预留显存、磁盘空间；训练日志分别累计数据读取/增强时间和计算时间。`compute` 是同步到 loss.item 的近似墙钟时间，不是精确 CUDA profiler。
- 每约10分钟在梯度累积完整优化步结束后保存轮内恢复文件，并在轮末保存；恢复时跳过已经完成的样本，不重跑整轮。实际间隔还受当前优化步和保存耗时影响。原子写入失败时已有完整文件保留，`.tmp` 不是恢复文件。
- Notebook 把训练脚本写入工作目录并通过独立子进程启动，转发日志并报告退出码。子进程被杀时可能留下负数退出码，`-9` 表示 SIGKILL，但并不单独证明 OOM。如果平台终止整个运行环境，Notebook 自身也可能退出。

PyTorch 官方文档说明多进程数据加载的主机内存影响，以及 DataParallel 在每次前向中复制模型的行为：

- https://docs.pytorch.org/docs/2.14/data.html
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.DataParallel.html

## 运行方式

先查看失败作业的 Output 是否可以取出 `.latest.pt` 或 `.best.pt`。如果有，保存它们；不要把它们混入新版目录。旧配置与新版运行方式不同，旧版恢复文件不能直接在新版中自动恢复，需要额外迁移。若是第一轮且没有模型权重，只能重新开始。

导入新版 Notebook，加入 `task2forml`，打开 Internet 并保留 GPU 配置。运行前确认没有旧训练进程同时占用资源。默认输出改为：

`/kaggle/working/task2_stable_output/submission.csv`

如果希望用更小模型来控制时间，在 Notebook 最后代码单元格的启动部分找到 `OPTIONS = []`，改为：

```python
OPTIONS = ['--backbone', 'tiny']
```

它使用 `convnext_tiny.fb_in22k_ft_in1k`，也有 ImageNet-22K 预训练，具体模型卡：

https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k

如果显存不足，使用：

```python
OPTIONS = ['--batch-per-gpu', '4', '--accum', '16', '--grad-checkpoint']
```

同一输出目录不允许混用不同训练配置；更改模型或 batch 后需要指定新的 `--output`。示例：

```python
OPTIONS = ['--backbone', 'tiny', '--output', '/kaggle/working/task2_tiny_output']
```

Python 文件独立运行方式：

```python
!python -u /kaggle/input/你的代码数据集名称/task2_kaggle_stable.py
```

## 第一次先观察运行速度

先查看前25～100批的 `data`、`compute` 和资源用量。这些数据才能判断主要时间花在读取/增强、模型计算还是其他开销，并估算剩余时间。此修正版未在真实 Kaggle GPU 上验证，不能承诺完成时长，也不能保证根因已解决。如果仍异常退出，提供最后资源日志及训练进程退出码即可进一步定位。

诊断文件：

- `/kaggle/working/task2_stable_console.log`：Notebook 子进程完整控制台日志。
- `/kaggle/working/task2_stable_output/resource_stats.jsonl`：资源记录。
- 每折 `.latest.pt`：最新轮内或轮末恢复状态。

保存输出后，跨会话把整个新版输出目录复制回工作目录，再使用相同参数运行即可恢复。不要只复制 `.latest.pt`，阶段最佳权重和 `run_config.json` 也应保留。

本地 CPU 小网络流程检查见 `task2_local_audit/stable_verification.json`；轮内恢复数值一致检查见 `task2_local_audit/stable_resume_verification.json`。这些检查不代表真实数据训练准确率，也不承诺不同 CUDA 软件环境中的逐位一致。
