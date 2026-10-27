# CCPD2020 车牌检测与 8 位识别

本工程依据课程课件实现 CCPD2020（新能源绿牌）专用任务。模型采用 RPNet 风格的共享骨干、多尺度 ROI 特征、车牌框回归和 8 个字符分类头；不混用 CCPD2019 的 7 位标签。

## 实验设计

| 实验 | 测试预处理 | 模型状态 | 目的 |
|---|---|---|---|
| Baseline | 无 | 原图训练 | 原始样本基线 |
| A | CLAHE | Baseline 权重，不微调 | 单独判断预处理作用 |
| B | CLAHE | 在 CLAHE 训练集上微调 | 判断训练域匹配后的收益 |

测试报告同时包含预测框整链路指标和 GT 框识别上限。测试预处理是确定性的，不使用随机增强。

## 环境与运行

已验证环境：Windows、Python 3.9、PyTorch 2.3.1、CUDA、OpenCV。当前机器可直接使用 `D:\Anaconda\envs\yolo\python.exe`。

```powershell
# 数据检查与图表
D:\Anaconda\envs\yolo\python.exe analyze_dataset.py

# 完整 Baseline + A + B 训练和测试
D:\Anaconda\envs\yolo\python.exe run_experiments.py

# 快速冒烟测试
D:\Anaconda\envs\yolo\python.exe run_experiments.py --limit 64

# 单独评估权重
D:\Anaconda\envs\yolo\python.exe run_experiments.py --mode evaluate --checkpoint outputs/checkpoints/baseline_best.pt

# 单图演示（B 实验最佳权重）
D:\Anaconda\envs\yolo\python.exe predict.py path\to\image.jpg
```

所有路径和超参数集中在 `configs/ccpd2020.yaml`。输出包括最佳权重、训练历史、测试指标、逐图预测 CSV、数据清单和汇报材料。

## 指标

- `mean_iou`：预测框与标注框的平均 IoU。
- `detection_accuracy_iou50`：IoU >= 0.5 的检测成功率。
- `character_accuracy`：使用预测框的 8 个字符平均准确率。
- `plate_accuracy`：使用预测框时 8 位全部正确的整牌准确率。
- `gt_box_*`：使用标注框时的识别上限，用于区分定位误差和 OCR 误差。
- `fps`：包含检测和两次识别分支的评估吞吐量，实际单次整链路会更快。

## 提交内容

- 可运行代码和 `requirements.txt`
- `outputs/analysis/dataset_manifest.json` 数据清单
- `outputs/checkpoints/*_best.pt` 最佳权重
- `outputs/metrics/test_results.json` 三组测试结果

