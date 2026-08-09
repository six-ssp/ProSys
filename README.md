# ProSys: A Product-to-System Framework for Target-Product-Driven Reaction-System Recommendation

> **Current manuscript source:** `ProSys_8_9.docx`. The target product is the
> only molecular query; a predefined reaction family selects a separately
> trained expert, retrieval memory, and vocabulary outside all feature vectors.

> **Current verified result source:** [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md)
> and `outputs/unified_rgnn_multiseed_20260809/`. The retained 2026-08-03 gated
> point snapshot and direct-GNN-ranking snapshot are historical only and must
> not be cited for headline metrics.

`ProSys` 是一个从目标产物出发，逐步推荐完整反应体系的框架。
当前维护的官方主线已经更新为：

```text
target product
-> Stage 1 EditRetro route generation
-> Stage 2 KNN wide recall + ReaFNN feasible-condition filtering
-> Stage 3 tabular XGB-LTR reranking + fixed R-GNN temperature prediction
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

当前主线是固定 Stage 1 路线缓存后，在六个家族、三个随机种子上独立重建 Stage 2/3
得到的等家族宏平均。完整逐家族、逐种子结果在 [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md)。

| Candidate coverage | Full-system Top-1 | Top-3 | Top-5 | Top-10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48.52 +/- 0.59 | 28.96 +/- 0.91 | 36.61 +/- 1.05 | 39.53 +/- 1.24 | 42.90 +/- 1.00 | 33.78 +/- 0.96 | 35.13 +/- 0.97 |

- 固定 Stage 1 Route@10：`63.20%`
- 条件温度 MAE：`10.75 +/- 0.14 C`
- 温度命中率（+/-5 / +/-10 / +/-20 C）：`41.66 +/- 1.39% / 64.14 +/- 1.65% / 84.74 +/- 0.38%`
- 结果目录：`outputs/unified_rgnn_multiseed_20260809/`

温度只在最高排名的完整 route-and-condition exact match 且存在有效温度标签时统计；
它是条件回归指标，不进入最终系统排序分数。

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
- 当前模型：`tabular XGB-LTR ranker + fixed R-GNN temperature regressor`
- 作用：`XGBRanker` 在无 `route_gnn_feat_*` 的表上做分组重排并验证集选择 Stage 2 prior；每个 family 的独立温度回归器固定使用 128 维、四层 R-GNN 特征，不设 no-GNN 回退或按 family 的温度模型选择
- 评估口径：Stage 3 只重排 Stage 2 已有候选，不增加新候选；温度指标只在每个样本最高排名的 full-match 候选上单独统计
- 细节文档：`stage3_XGBoost/stage3_XGBoost_detail.md`

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

- `outputs/unified_rgnn_multiseed_20260809/README.md`
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
3. 训练 tabular `XGBRanker`，并训练固定使用 R-GNN 特征的温度模型
4. 输出 markdown / csv / json 汇总；温度头没有 validation gate 或 no-GNN 回退

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
7. `outputs/unified_rgnn_multiseed_20260809/README.md`

## 目录约定

### 当前主线长期保留

- `stage1_retrosynthesis/`
- `prosys_shared/`
- `stage2_ReaFNN/`
- `stage2_KNN/`
- `stage3_XGBoost/`
- `scripts/`
- `outputs/stage1_routes/`
- `outputs/unified_rgnn_multiseed_20260809/`

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
- `Stage 3 tabular XGB-LTR ranker + fixed R-GNN temperature regressor`

其中 Stage 2 的当前正式口径还包括：

- `Non-Oracle` 推理时允许极少量、强约束、强惩罚的 novel context 尝试

当前官方 Non-Oracle 6-family 结果是：

- `Route@10 = 63.2`
- `candidate coverage = 48.52 +/- 0.59`
- `full-system Top-10 accuracy = 42.90 +/- 1.00`
- `MAE (deg C) = 10.75 +/- 0.14`
