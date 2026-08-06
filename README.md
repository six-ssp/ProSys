# ProSys: A Product-to-System Framework for Target-Product-Driven Reaction-System Recommendation

> **Current manuscript source:** `ProSys_8_9.docx`. The target product is the
> only molecular query; a predefined reaction family selects a separately
> trained expert, retrieval memory, and vocabulary outside all feature vectors.

> **Current verified result source:** [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md)
> and `outputs/stage23_mainline_gnn_temperature_gated_20260803/`. The retained
> direct-GNN-ranking snapshot is historical and must not be cited for headline metrics.

`ProSys` 是一个从目标产物出发，逐步推荐完整反应体系的框架。
当前维护的官方主线已经更新为：

```text
target product
-> Stage 1 EditRetro route generation
-> Stage 2 KNN wide recall + ReaFNN feasible-condition filtering
-> Stage 3 tabular XGB-LTR reranking + validation-gated R-GNN temperature prediction
-> Top-k reaction systems
```

当前正式口径只保留：

- `6-family`
- `Non-Oracle`
- `end-to-end`

`Oracle` 只作为历史分析和上限参考，不再作为主结果入口。

**输入边界。** 目标产物的结构是主线唯一的分子输入；Stage 1 之后使用的
reactants、候选条件和排序特征均由该输入在系统内部产生或检索得到。反应家族
不会作为类别特征拼接到任何模型输入中。当前结果按家族分别训练和报告，运行时
由外层评测分区选择相应的专家模型、条件记忆和词表，因此这是家族专属专家设置，
而不是 `product + reaction type` 的特征融合设置。

## 当前官方结果快照

本轮最新、已经跑通并核对过的主线结果位于：

- `outputs/stage23_mainline_gnn_temperature_gated_20260803/`
- `outputs/stage23_mainline/` 仅保留为历史快照，不再作为结果来源

当前这轮 `6-family` 官方快照已经直接使用了最新的 Stage 2 代码路径：

- `train/val` 构表仍只保留历史已见完整组合
- `Non-Oracle` 推理时允许极少量 token-consistent 新组合进入候选池
- 目的不是刷高分数，而是明确说明当前主线不是只在历史完整答案空间里做“选择题”

其中最常用的结果文件是：

- `outputs/stage23_mainline_gnn_temperature_gated_20260803/overview.md`
- `outputs/stage23_mainline_gnn_temperature_gated_20260803/overview.txt`
- `outputs/stage23_mainline_gnn_temperature_gated_20260803/results_flat.csv`
- `outputs/stage23_mainline_gnn_temperature_gated_20260803/gnn_temperature_gate_audit.tsv`

Stage 1 路线缓存位于：

- `outputs/stage1_routes/`

## 当前主线模块

### Stage 1

- 目录：`stage1_retrosynthesis/`
- 作用：从目标产物生成 top-k 候选路线
- 输出：`outputs/stage1_routes/<family>/route_cache.json`

### Stage 2

- 目录：核心实现 `stage2_ReaFNN/`，兼容导入 `stage2_KNN/`
- 当前模型：`KNN + ReaFNN`
- 作用：先做 family-specific 宽 KNN 召回，再用 `ReaFNN` 对试剂/溶剂 token 做二次筛选，并在 `Non-Oracle` 推理时尝试极少量受限新组合，形成更稳的可行条件候选池
- 细节文档：`stage2_ReaFNN/stage2_KNN_detail.md`

### Stage 3

- 目录：`stage3_XGBoost/`
- 当前模型：`tabular XGB-LTR ranker + validation-gated R-GNN temperature regressor`
- 作用：`XGBRanker` 在无 `route_gnn_feat_*` 的表上做分组重排并验证集选择 Stage 2 prior；独立温度回归器使用 64 维 R-GNN 特征，只有 validation MAE 至少改善 `0.25 C` 才按 family 启用
- 评估口径：Stage 3 只重排 Stage 2 已有候选，不增加新候选；温度指标只在每个样本最高排名的 full-match 候选上单独统计
- 细节文档：`stage3_XGBoost/stage3_XGBoost_detail.md`

## 当前主线结果摘要

更新时间：`2026-08-03`

当前 6 个保留家族分别是：

- `Beckmann`
- `Buchwald-Hartwig`
- `Chan-Lam`
- `Diels-Alder`
- `Friedel-Crafts Acyl.`
- `Friedel-Crafts Alkyl.`

### Mainline: Route + System

| Family | Route@10 | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 69.8 | 39.6 | 18.7 | 23.8 | 26.0 | 29.8 |
| Buchwald-Hartwig | 70.0 | 48.7 | 37.2 | 44.7 | 45.8 | 46.8 |
| Chan-Lam | 77.4 | 66.7 | 45.9 | 55.4 | 59.2 | 60.8 |
| Diels-Alder | 37.3 | 29.8 | 17.3 | 23.6 | 26.2 | 27.6 |
| Friedel-Crafts Acyl. | 70.3 | 60.6 | 32.0 | 41.7 | 46.9 | 53.1 |
| Friedel-Crafts Alkyl. | 54.4 | 49.7 | 29.5 | 39.0 | 42.6 | 45.5 |
| MACRO-AVG | 63.2 | 49.2 | 30.1 | 38.0 | 41.1 | 43.9 |

当前主线最核心的 headline 是：

- `Route@10 = 63.2`
- `candidate recall = 49.18`
- `full-system Top-10 accuracy = 43.91`

### Mainline: Temperature

温度指标是在“每个样本里最高排名的 full-match 候选”上单独统计的，不和 `full-system Top-k accuracy` 混成一个指标。

- `MAE (deg C) = 11.11`
- `Within +/-5 deg C = 39.2%`
- `Within +/-10 deg C = 62.6%`
- `Within +/-20 deg C = 82.9%`

这说明当前主线除了离散体系推荐外，也已经能给出一个可用的连续温度估计。

## 为什么现在的主线是这一版

当前主线不是简单的 `KNN+XGB` 旧口径，而是：

- Stage 2：`KNN + ReaFNN`
- Stage 3：`tabular XGB-LTR ranker + validation-gated R-GNN temperature`

选择这一版作为官方主线，主要因为它更符合这个任务的分工：

1. `KNN` 负责把 family 内历史上“像”的反应先找回来。
2. `ReaFNN` 负责在宽候选池上做 token 级二次筛选，并允许极少量受限的新组合尝试，避免主线退化成封闭答案空间里的检索题。
3. `R-GNN` 只服务独立温度头，避免在小 family 中把同一路线恒定的 64 维特征直接送入树排序器。
4. `XGB-LTR` 负责在可行候选池中把真正的完整反应体系尽量排到前面；groupwise 标准化后的 `xgb_score_raw` 与轻量 Stage 2 heuristic prior 的融合权重在 validation 上选择。
5. 对 novel combination 的开放是刻意保守的，因为真实可行的新组合本来就稀少，被最终选中的概率更低，这种低频现象本身就是符合实际场景的。

所以当前主线更准确的理解应该是：

- `Stage 1` 决定路线空间
- `Stage 2` 决定可行候选池
- `Stage 3` 决定最终系统排序和温度输出

## 当前推荐入口

### 1. 环境与数据检查

```bash
conda activate ProSys
bash scripts/setup_prosys_env.sh
python data_preprocess/audit_data_splits.py --strict
```

### 2. 如果还没有 Stage 1 route cache

```bash
conda activate ProSys
python stage1_retrosynthesis/build_route_cache.py --repo_root . --family Beckmann
```

### 3. 复现当前主线

如果只是查看当前已经核对过的 `6-family` 官方结果快照，直接读取：

- `outputs/stage23_mainline_gnn_temperature_gated_20260803/`
- `CURRENT_RESULTS.md`

如果要运行当前维护代码，建议不要覆盖旧快照，而是显式指定一个新的输出目录：

```bash
conda activate ProSys
OUTPUT_ROOT=outputs/stage23_mainline_current \
ROUTE_ROOT=outputs/stage1_routes \
REAFNN_DEVICE=cuda:0 \
GNN_DEVICE=cuda:0 \
bash scripts/run_stage23_non_oracle_suite.sh .
```

这条入口只做当前主线：

1. 读取 Stage 1 的 `route_cache.json`
2. 构建 `KNN + ReaFNN` candidate pool
3. 训练 tabular `XGBRanker`，并训练 no-GNN/R-GNN 两个温度模型
4. 用 validation MAE 门控选择温度头，再输出 markdown / csv / json 汇总

### 4. Baseline / Ablation

这些不属于当前主线本体，单独维护在：

- `baseline/`
- `ablation/`

## 文档阅读顺序

如果只想快速理解当前项目状态，建议按下面顺序阅读：

1. `README.md`
2. `NOMENCLATURE.md`
3. `workflow.md`
4. `performance.md`
5. `stage2_ReaFNN/stage2_KNN_detail.md`
6. `stage3_XGBoost/stage3_XGBoost_detail.md`
7. `outputs/stage23_mainline_gnn_temperature_gated_20260803/overview.md`

## 目录约定

### 当前主线长期保留

- `stage1_retrosynthesis/`
- `prosys_shared/`
- `stage2_ReaFNN/`
- `stage2_KNN/`
- `stage3_XGBoost/`
- `scripts/`
- `outputs/stage1_routes/`
- `outputs/stage23_mainline_gnn_temperature_gated_20260803/`

### 历史结果快照

- `outputs/stage23_mainline/` 与
  `outputs/stage23_mainline_reafnn_gnn_fused_20260723/`（保留用于追溯，不可作为当前论文结果来源）

### 历史参考但不属于当前主线

- 统一放到 `Experiment/`

### 非主线实验

- `baseline/`
- `ablation/`

## 一句话总结

当前仓库正式维护的主线已经确定为：

- `Stage 1 EditRetro`
- `Stage 2 KNN + ReaFNN`
- `Stage 3 tabular XGB-LTR ranker + validation-gated R-GNN temperature regressor`

其中 Stage 2 的当前正式口径还包括：

- `Non-Oracle` 推理时允许极少量、强约束、强惩罚的 novel context 尝试

当前官方 Non-Oracle 6-family 结果是：

- `Route@10 = 63.2`
- `candidate recall = 49.18`
- `full-system Top-10 accuracy = 43.91`
- `MAE (deg C) = 11.11`
