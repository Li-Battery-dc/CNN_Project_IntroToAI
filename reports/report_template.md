# 基于 STL-10 数据集的图像分类实践报告

## 1. 实验目标

说明 STL-10 十分类任务、训练/验证/测试划分方式，以及本项目关注的模型设计、优化对比和可解释性分析。

## 2. 数据集与固定验证集

数据目录采用 `STL10/train` 与 `STL10/test`。从训练集按类别固定划分训练/验证集，每类 560 张训练、140 张验证；测试集每类 100 张，只用于最终评估。

## 3. 模型与训练方法

描述 `BasicCNN` 的卷积层、池化层和分类头设计。说明自定义 `BaseOptimizer`、`SGD`、`Adam`、`BaseLoss`、`CrossEntropyLoss` 与 `LabelSmoothingCrossEntropy`，并说明没有调用 PyTorch 内置优化器和交叉熵损失类。

## 4. 对比实验

引用 `reports/assets/experiment_summary.md` 或 `runs/*/history.png`，比较：

- `baseline_sgd_ce`
- `optimizer_adam`
- `loss_label_smooth`
- `data_aug`
- `bn_dropout`
- `combined_best`

## 5. 测试集评估

给出最终模型的 Precision、Recall、F1-score、混淆矩阵，并分析容易混淆的类别。

## 6. Grad-CAM 可解释性分析

展示 `runs/<timestamp>_combined_best/test_gradcam.png`，分析模型关注区域是否和目标物体一致，并对错误样本做解释。

## 7. AI 工具使用说明

本项目使用 ChatGPT/Codex 辅助完成项目结构设计、训练 pipeline 代码编写和报告提纲整理。实验运行、结果选择与分析结论由本人基于实际训练日志和评估结果完成。
