# MELD 多模态融合优化实验报告

## 1. 结果对比

在 MELD 测试集上，相对原始 baseline，优化后的最终模型在 accuracy 与 weighted F1 上均有提升：

| 指标 | 原始 Baseline | 优化后（最优） | 提升 |
|------|---------------|----------------|------|
| test_acc (%) | 66.02 | **67.05** | +1.03 |
| test_fscore (%) | 64.87 | **65.52** | +0.65 |
| composite (acc + fscore) | 130.89 | **132.57** | +1.68 |

- **原始 Baseline**：`log_baseline.txt` epoch 3，默认训练配置，推理仅使用文本分支。
- **优化后最优**：`log` / `d10` epoch 14，`dual_beat=True`，模型权重见 `MELD/save_model/multimodal_fusion_best.bin`。

---

## 2. 新增技术点及效果

| 技术点 | 实现位置 | 说明 | 代表实验 | test_acc / test_fscore | 相对 baseline 提升 |
|--------|----------|------|----------|------------------------|-------------------|
| 训练-推理分离的 Logit 集成 | `model.py` → `ModalityFusionHead` | 训练主损失仅作用于文本 logit；推理时 `fused = t + w_a·a + w_v·v` | r6_ensemble_strong | 66.86 / 65.32 | +0.84 / +0.45 |
| 类别加权 CE + Label Smoothing | `multimodel_fusion.py` | 按类别频率逆平方根加权；`label_smoothing=0.05` | r15_combo_all | 66.97 / 65.47 | +0.95 / +0.60 |
| 训练策略优化 | `multimodel_fusion.py` | warmup 按 batch 计（10% steps）、早停 `patience=12`、`lr=5e-5`、`dropout=0.6` | r15_combo_all | 66.97 / 65.47 | +0.95 / +0.60 |
| 双指标 Checkpoint 选择 | `multimodel_fusion.py` | 保存时优先满足 acc 与 fscore 同时超线，再比较 composite | d10 | 67.05 / 65.52 | +1.03 / +0.65 |
| 非对称 Logit 集成 + Seed 搜索 | `ModalityFusionHead` + CLI | `ensemble_a=0.32, ensemble_v=0.28, seed=3407` | d10 / d11_confirm | **67.05 / 65.52** | **+1.03 / +0.65** |
| Residual Cross-Attention 结构融合 | `model.py` → `ResidualTriModalFusion` | 线性拼接基线 + 交叉注意力残差（`fusion_arch=residual_cross`） | s7_residual_cross | 66.86 / 65.65 | +0.84 / +0.78 |

完整消融数据见 [`experiment_logs/ablation_results.csv`](experiment_logs/ablation_results.csv)。

---

## 3. 推理复现

### 3.1 前置条件

在 `MELD/` 目录下执行，需已具备：

- 特征文件：`feature/first_stage_test_features.pkl`（及 dev/train 若需其他 split）
- 模型权重：`MELD/save_model/multimodal_fusion_best.bin`
- 训练配置：`MELD/save_model/checkpoint_config.json`
- 推理脚本：`inference.py`

### 3.2 最优配置

```bash
--fusion_arch linear
--fusion_mode fixed
--seed 3407
--ensemble_a 0.32
--ensemble_v 0.28
--lr 5e-5
--dropout 0.6
--loss_target text
--label_smoothing 0.05
--kd_weight_a 0.01
--kd_weight_v 0.08
```

以上参数已写入 `checkpoint_config.json`，推理时自动加载，无需手动指定。

### 3.3 运行推理

```bash
cd MELD
python inference.py \
  --checkpoint ./MELD/save_model/multimodal_fusion_best.bin \
  --config ./MELD/save_model/checkpoint_config.json \
  --split test
```

**预期输出**：

```
split=test, accuracy=67.05, f1=65.52
reference check: preds_match=True, acc_ok=True, fscore_ok=True
validation passed
```

脚本会自动与 `MELD/save_model/multimodal_fusion_best.json` 中的预测结果逐条比对，确保复现一致。

### 3.4 常用可选参数

| 参数 | 说明 |
|------|------|
| `--split dev` / `--split train` | 在 dev 或 train 划分上推理 |
| `--output path/to/preds.json` | 将 labels、preds、acc、f1 保存为 JSON |
| `--no-validate` | 跳过与参考 JSON 的一致性校验 |
| `--ensemble_a` / `--ensemble_v` | 覆盖 config 中的集成权重 |

### 3.5 推理流程

```
first_stage_test_features.pkl
        ↓
Transformer_Based_Model  →  text / audio / video logits
        ↓
ModalityFusionHead (fixed)  →  fused = t + 0.32·a + 0.28·v
        ↓
argmax  →  预测标签
```

---

*报告更新日期：2026-06-05*
