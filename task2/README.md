# 实验二 人脸表情六分类

最终方案：ImageNet预训练ConvNeXt-Tiny，像素指纹分组五折中的折0、1、2，三个测试概率等权平均。灰度48×48复制为RGB；224训练最多20轮，288精调最多5轮，验证选原始/EMA与TTA，patience=6。

- 单模型基线：`task2_2026task2_copy.py`
- 三模型融合：`task2_ensemble3_copy.py`
- 基线复用包：`task2_baseline_0714.zip`
- 结果校验：`python task2/verify_results.py`

在Kaggle挂载2026task2原始数据和解压后的复用包，开启GPU与Internet，把三模型脚本全部粘贴到Code单元格运行。保持 `TRAIN_FOLDS='0,1,2'`、`REUSE_DIR='auto'`。程序校验并复用第0折预测，训练第1、2折，输出 `/kaggle/working/task2_ensemble3_output/submission.csv`。无需为了复用第0折上传其模型权重。

结果目录保留配置、分折清单、每折验证/测试概率和最终CSV。原始数据及55MB以上权重不上传。第三方依赖具体历史版本未完整保存，复现应保留Kaggle原装CUDA torch，脚本只补装timm。部分历史说明末尾“尚未训练”是当时状态，当前已完成结果以results/evidence.md为准。

三折拼接验证70.7017%不是独立集上的三模型融合分数。两次公开提交均使用511张与训练图完全相同且标签无冲突测试图的匹配校正；无该校正的提交分数未知。
