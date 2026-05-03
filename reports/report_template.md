# 基于 STL-10 数据集的图像分类实践报告

## 1. 实验目标

说明 STL-10 十分类任务、训练/验证/测试划分方式，以及本项目关注的模型设计、优化对比和可解释性分析。

## 2. 数据集与固定验证集

数据目录采用 `STL10/train` 与 `STL10/test`。从训练集按类别固定划分训练/验证集，每类 560 张训练、140 张验证；测试集每类 100 张，只用于最终评估。

## 3. 模型与训练方法

描述 `BasicCNN`、`RegularizedCNN` 和最终采用的 `VGGCNN`。说明 `VGGCNN` 采用 VGG 风格的多阶段卷积块，每个阶段包含若干 `Conv-BN-ReLU`，阶段之间通过 `MaxPool2d` 下采样，最后使用全局平均池化和线性分类头。

说明自定义 `BaseOptimizer`、`SGD`、`Adam`、`BaseLoss`、`CrossEntropyLoss` 与 `LabelSmoothingCrossEntropy`，并说明没有调用 PyTorch 内置优化器和交叉熵损失类。若使用 `scheduler: cosine`，补充说明学习率从初始值按余弦衰减到 `min_lr`。

## 4. 对比实验

引用 `reports/assets/experiment_summary.md` 或 `runs/*/history.png`，比较：

- `baseline_sgd_ce`
- `optimizer_adam`
- `loss_label_smooth`
- `data_aug`
- `bn_dropout`
- `baseline_strong`
- `vgg_medium_tuned`
- `combined_best`

报告中需要专门说明本次调参验证过程：

- 原始 `baseline_sgd_ce` 是保留的弱对照，远程小规模实验显示其在 5 epoch 后仍接近随机水平，因此不能作为最终强 baseline。
- `baseline_strong` 是经过验证的强监督基线：`regularized_cnn + dropout 0.1 + Adam + train_aug + label smoothing 0.05`，35 epoch 固定验证集 best accuracy 约 67.71%。
- `vgg_medium_tuned` 用于展示模型容量和 cosine 学习率调度带来的提升，远程验证约 75.14% valid / 74.00% test。
- `combined_best` 是最终推荐配置：`vgg_cnn` 四阶段 `[64, 128, 256, 512]`、dropout 0.05、Adam、cosine decay、label smoothing 0.05，远程验证约 79.50% valid / 79.80% test。
- 需要讨论为什么默认 `dropout=0.3`、`smoothing=0.1` 偏强，以及为什么 `batch_size=128` 比 `64` 更稳定。

## 5. 测试集评估

给出最终模型的 Precision、Recall、F1-score、混淆矩阵，并分析容易混淆的类别。

## 6. Grad-CAM 可解释性分析

展示 `runs/<timestamp>_combined_best/test_gradcam.png`，分析模型关注区域是否和目标物体一致，并对错误样本做解释。

## 7. AI 工具使用说明

本项目使用 ChatGPT/Codex 辅助完成项目结构设计、训练 pipeline 代码编写和报告提纲整理。实验运行、结果选择与分析结论由本人基于实际训练日志和评估结果完成。
