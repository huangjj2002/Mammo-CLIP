# Prototype + EDL 接入说明

本文档说明如何把 Prototype + Evidential Deep Learning (EDL) 头接到乳腺癌二分类模型上。当前项目中的独立实现入口是 `edl_proto_train.py` 和 `edl_proto_test.py`，不会改变原有 `main.py`、`edl_train.py`、`edl_test.py` 的行为。

## 目标

Prototype + EDL 用 prototype 提供可解释锚点，用 EDL 输出不确定性。它替代的是 DST prototype head 中的 mass / Dempster 组合逻辑，而不是替代整个 backbone。

核心流程：

```text
model embedding z
  -> class-wise prototype distance
  -> prototype similarity
  -> prototype evidence
  -> class evidence
  -> alpha / probability / uncertainty
```

## 模块接口

要把该头接到其他模型上，模型只需要暴露一个样本级 embedding：

```text
z: Tensor[B, D]
label: Tensor[B], values in {0, 1}
```

其中：

- `0` 是当前任务的 negative class，例如 benign / non-cancer / non-lesion。
- `1` 是当前任务的 positive class，例如 cancer / mass positive / calcification positive。
- 不需要额外的 benign 或 cancer 文本字段，类别完全由 CSV 中 `args.label` 对应的 0/1 标签决定。

Prototype 参数形状：

```text
P: Tensor[2, K, D]
```

`K` 是每类 prototype 数量。默认 `K=4`，所以二分类总共有 `2*K=8` 个 prototype。

## 输出字段

Prototype + EDL 保持当前 EDL 的核心输出：

```text
evidence: Tensor[B, 2]
alpha: Tensor[B, 2]
S: Tensor[B, 1]
prob: Tensor[B, 2]
uncertainty: Tensor[B]
```

CSV 中的基础列和现有 MIL/EDL 对齐：

```text
patient_id
image_id
split
cohort_num
label
prediction_score
predicted_class
evidence_0
evidence_1
alpha_0
alpha_1
uncertainty
fold
```

Prototype 解释列追加在后面：

```text
proto_c0_top1_idx
proto_c0_top1_evidence
proto_c0_top1_similarity
proto_c1_top1_idx
proto_c1_top1_evidence
proto_c1_top1_similarity
...
```

其中 `c0` 是 negative class，`c1` 是 positive class。

## 和 DST 输出的对应关系

| DST 输出 | Prototype + EDL 对应设计 |
| --- | --- |
| `BetP` | `prob`，CSV 中主要是 `prediction_score = prob[:, 1]` |
| `abstain` | 不直接输出 hard abstain；用 `uncertainty` 作为 abstain score，后续可按阈值派生 |
| `mass bar` | 改为 evidence / alpha / uncertainty bar |
| `top-k protos` | 保留为 `proto_c*_top*_idx/evidence/similarity` |
| `protos` | 可学习参数 `P: [2, K, D]` |
| `patch map` | 当前 v1 先做 bag-level prototype 解释；patch-level map 后续可用 patch embedding 到 prototype 的 similarity 派生 |

## 初始化方式

当前项目默认使用按类 KMeans 初始化：

1. 每个 fold 单独初始化。
2. 只使用当前 fold 的 `train_df`，不使用 `val_df` 或 `test_df`。
3. 用 MIL backbone 提取训练样本的 bag embedding `z`。
4. 按 `train_df[args.label] == 0/1` 分为 negative / positive 两组。
5. 每组做 KMeans，得到 `K` 个中心。
6. 用两个类别的中心初始化 `P: [2, K, D]`。

如果某一类样本少于 `K`，实现会重复已有中心补齐并打印 warning。这样可以避免初始化阶段因为少数类样本过少而中断。

KMeans 只是 prototype 的初始化方式，训练时 prototype 仍然是可学习参数。新增的 prototype regularization 会在训练过程中继续约束这些 prototype，二者可以同时使用。

## Prototype 正则

Prototype + EDL 路径默认增加三类 prototype 正则，只作用于 `edl_proto_train.py`，不会影响原 EDL baseline：

```text
total_loss = EDLCombinedLoss + prototype_regularization
```

正则项包括：

- 同类吸引：每个样本靠近自己标签对应类别的最近 prototype。
- 异类排斥：每个样本远离另一类的最近 prototype，使用 margin loss。
- prototype 多样性：同一类别内部的 prototype 彼此保持距离，避免塌缩成一个点。

相关参数：

```text
--edl_proto_attract_weight      default 0.1
--edl_proto_separation_weight   default 0.1
--edl_proto_diversity_weight    default 0.01
--edl_proto_margin              default 1.0
--edl_proto_balance_classes     default y
```

`--edl_proto_balance_classes y` 表示 attraction / separation 先在 batch 内按类别分别求均值，再对出现的类别平均。这个设置对阳性样本极少的任务很重要，可以避免 prototype 正则几乎完全被 negative class 主导。

如果想先观察纯 Prototype + EDL head 的效果，可以把三个 weight 都设为 `0`。如果 prototype 解释性弱或出现塌缩，可以优先提高 `--edl_proto_diversity_weight` 或 `--edl_proto_separation_weight`。

## 当前项目用法

训练 Prototype + EDL：

```bash
python edl_proto_train.py \
  --resume path/to/pretrained_mil_or_experiment \
  --data_dir datasets/Vindir-mammoclip \
  --csv_file grouped_df.csv \
  --dataset ViNDr \
  --label Mass \
  --mil_type embedding \
  --feature_extraction offline \
  --n_folds 5 \
  --epochs 20 \
  --edl_proto_k 4 \
  --edl_proto_topk 3 \
  --edl_proto_attract_weight 0.1 \
  --edl_proto_separation_weight 0.1 \
  --edl_proto_diversity_weight 0.01 \
  --edl_proto_margin 1.0 \
  --edl_proto_balance_classes y
```

单独测试已有 Prototype + EDL checkpoint：

```bash
python edl_proto_test.py \
  --checkpoint_dir path/to/EDL_PROTO/output \
  --data_dir datasets/Vindir-mammoclip \
  --csv_file grouped_df.csv \
  --dataset ViNDr \
  --label Mass \
  --mil_type embedding \
  --feature_extraction offline \
  --n_folds 5 \
  --edl_proto_k 4 \
  --edl_proto_topk 3
```

输出目录：

```text
{output_dir}/EDL_PROTO/{dataset}_{label}/fold_{n_folds}/{date}
```

主要文件：

```text
edl_proto_results_summary.csv
{dataset}_edl_proto_val_fold_assignments.csv
fold_{i}/{dataset}_edl_proto_predictions_fold_{i}.csv
edl_proto_test_results/{dataset}_edl_proto_dev_predictions.csv
edl_proto_test_results/{dataset}_edl_proto_test_all_folds.csv
edl_proto_test_results/{dataset}_edl_proto_test_ensemble.csv
edl_proto_test_results/{dataset}_edl_proto_all_predictions.csv
```

## 迁移到其他乳腺癌模型

给其他模型接入该模块时，按以下步骤做：

1. 在原模型中找到分类头之前的样本级 embedding `z: [B, D]`。
2. 用 `PrototypeEDLHead(in_features=D, prototypes_per_class=K)` 替换原二分类 head。
3. 训练前用训练集 embedding 和 0/1 标签做按类 KMeans 初始化。
4. 训练时使用 `EDLCombinedLoss(alpha, label, epoch)`。
5. 验证和测试时导出统一 CSV 字段，至少包含 `prediction_score`, `predicted_class`, `evidence_0/1`, `alpha_0/1`, `uncertainty`。
6. 如需解释，额外导出每类 top-k prototype 的 index、evidence 和 similarity。

注意：不要在 prototype 初始化时使用验证集或测试集 embedding，否则会造成数据泄露。
