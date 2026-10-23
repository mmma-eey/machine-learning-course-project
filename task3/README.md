# 任务三 中心凹检测定位

最终保留首版 `task3_kaggle_fovea.py` 与 `results/submission.csv`，公开分数13.64493，越低越好。输入80张训练图与20张测试图，512分辨率，ImageNet预训练ResNet34加U形热力图网络，五折分组划分，四视角翻转预测与五模型坐标等权平均。

Kaggle挂载2026task3数据，开启GPU与Internet，将脚本粘入Code单元格运行。输出 `/kaggle/working/task3_fovea_output/submission.csv`。原始ZIP、正式大权重不上传，已有权重应通过Kaggle Input复用。

对照：关闭翻转增强公开分数30.21800；无增强坐标中位数15.69068；局部精修18.26819。后三者均未超过首版。OOF分别为首版21.78956849、关闭增强15.06177797、精修18.10796794；这些是各折未见验证样本的拼接分数，且参与模型选择，不是独立集五模型融合指标。

结果和代码来自本地task3目录及授权引用对话01a12a2f-84f0-7842-a978-a7f8a26665e7。第一版、无增强和中位数成绩经提交截图核对，精修成绩来自用户实际提交记录。训练历史为第一版日志，不混用本地烟雾测试。results/no_tta_predictions.zip为用户下载的无增强输出小文件包。

Canva架构图：https://canva.link/dv1b88tss0rfj2v
