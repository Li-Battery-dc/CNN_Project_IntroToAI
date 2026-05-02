# STL-10 CNN 图像分类实验

本项目用于完成“人工智能原理课程项目 2”：基于 STL-10 数据集实现 CNN 图像分类、优化对比实验和 Grad-CAM 可解释性分析。

## 数据集

```text
STL10/
  train/
    airplane/
    ...
  test/
    airplane/
    ...
```

当前 pipeline 使用固定验证集划分文件 `splits/stl10_seed42_valid20.json`，避免对比实验时验证效果不好。

## 环境配置

服务器建议使用 Python 3.10/3.11 和 CUDA 12.1：

```bash
conda create -n "CNN" python=3.10
pip install -r requirements-cu121.txt
```

## 常用命令

生成固定训练/验证划分：

```bash
python -m scripts.prepare_splits --data-root STL10 --out splits/stl10_seed42_valid20.json
```

训练单个实验：

```bash
python -m scripts.train --config configs/experiments.yaml --experiment baseline_sgd_ce
python -m scripts.train --config configs/experiments.yaml --experiment combined_best
```

自监督预训练与全标签微调：

```bash
python -m scripts.pretrain_ssl --config configs/self_supervised.yaml --method rotation
python -m scripts.pretrain_ssl --config configs/self_supervised.yaml --method simclr

python -m scripts.finetune_ssl --config configs/self_supervised.yaml --experiment supervised_full_baseline
python -m scripts.finetune_ssl --config configs/self_supervised.yaml --experiment rotation_finetune --pretrained runs_ssl/<rotation_run>/encoder.pt
python -m scripts.finetune_ssl --config configs/self_supervised.yaml --experiment simclr_finetune --pretrained runs_ssl/<simclr_run>/encoder.pt
```

自监督阶段只扫描 `STL10/train` 下全部 7000 张训练图像，不使用固定验证划分，也不会读取 `STL10/test`。微调阶段使用 `STL10/train` 的全部标签，训练结束后直接在 `STL10/test` 输出分类报告和混淆矩阵。

快速冒烟测试：

```bash
python -m scripts.train --config configs/experiments.yaml --experiment baseline_sgd_ce --epochs 1 --limit-per-class 10
```

最终测试集评估：

```bash
python -m scripts.evaluate --run-dir runs/<timestamp>_combined_best --split test
```

生成 Grad-CAM：

```bash
python -m scripts.gradcam --run-dir runs/<timestamp>_combined_best --split test
```

汇总实验结果：

```bash
python -m scripts.summarize_results --runs runs --out reports/assets
```

批量运行所有实验并评估最新的 `combined_best`：

```bash
bash scripts/run_all.sh configs/experiments.yaml
```

## 实验配置

所有实验定义在 `configs/experiments.yaml`：

- `baseline_sgd_ce`：基础 CNN + 无增强 + 自定义 SGD + 自定义交叉熵。
- `optimizer_adam`：只替换为自定义 Adam。
- `loss_label_smooth`：只替换为 Label Smoothing 交叉熵。
- `data_aug`：只加入随机裁剪、水平翻转和轻量颜色扰动。
- `bn_dropout`：只替换为带 BatchNorm/Dropout 的 CNN。
- `combined_best`：组合增强、正则化模型、Adam 和 Label Smoothing。

新增对比实验时，优先复制一个 YAML 实验项；只有新增模型、优化器、损失或变换类型时，才需要在 `src/factory.py` 增加一行映射。

## 输出

- `runs/<timestamp>_<experiment>/history.csv`：每轮 train/valid loss 与 accuracy。
- `runs/<timestamp>_<experiment>/history.png`：训练曲线。
- `runs/<timestamp>_<experiment>/best.pt`：验证集 accuracy 最佳 checkpoint。
- `runs/<timestamp>_<experiment>/test_classification_report.*`：测试集分类报告。
- `runs/<timestamp>_<experiment>/test_confusion_matrix.*`：混淆矩阵。
- `runs/<timestamp>_<experiment>/test_gradcam.png`：Grad-CAM 可视化。
- `runs_ssl/<timestamp>_<method>/encoder.pt`：自监督训练过程中按训练损失滑动平均保存的最佳 encoder。
- `runs_ssl/<timestamp>_<experiment>/final.pt`：全标签微调最终 checkpoint。

`runs/`、`runs_ssl/`、`STL10/` 默认被 `.gitignore` 忽略。报告中需要引用的小图表可复制或汇总到 `reports/assets/` 后提交。
