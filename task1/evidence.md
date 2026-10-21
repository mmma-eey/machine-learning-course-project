# 实验一证据

用户提供Kaggle提交页面截图 `codex-clipboard-d7780477-5615-4bc2-badc-4489a178ed05.png`，截图所示分数：

- submission_convnext_small.csv 0.9259
- submission_pseudo.csv 0.9232
- submission_clean_384.csv 0.9206
- submission_advanced_ensemble.csv 0.7857
- submission_efficientnet_s.csv 0.8201
- submission_ensemble.csv 0.8015
- submission.csv 0.8174

最佳提交与 `kaggle_convnext_small.py` 的输出文件名一致，代码明确使用320、原始500张、完整分层五折及最终EMA。文件名和源码一致性支持方法对应关系；缺少原始完整训练日志时，不声称已重现历史全部过程。

本次审计日期2026-10-21。原始训练每类100，共500；清洗数据1713；测试378；清洗384提交ID与实际测试文件集合一致，无重复ID；RGB尺寸加像素哈希未发现清洗集与测试集完全相同图像。清洗版与320版38张预测不同，不据此判定正确与错误。

注意：最佳320脚本注释写8视角，但实际循环为2个裁剪尺度×2种翻转，共4视角。清洗384与二分类脚本注释写5折，实际为十折取前五折。
