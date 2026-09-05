# Current Parallel Stage-2 Result

> **Reportable paired control (2026-09-05):**
> [`current_parallel_stage2_ablation_results_20260905.md`](current_parallel_stage2_ablation_results_20260905.md) and
> [`Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/`](../Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/README.md)
> contain the completed three-seed ReaFNN-removal ablation for the maintained
> parallel KNN + ReaFNN post-fusion mainline. The fixed Stage-1 route caches,
> KNN retrieval settings, and split contracts match the current full mainline.
> The KNN-only arm necessarily rebuilds its candidate pool and retrains its
> 52D XGB-LTR on that pool; it therefore tests end-to-end candidate
> availability and composition, rather than a fixed-pool ranking effect.
> Removing ReaFNN changes macro candidate recall from `54.26 +/- 0.15%` to
> `53.39 +/- 0.00%` and macro Sys@10 from `43.77 +/- 0.60%` to
> `39.86 +/- 2.08%` (`-3.91 pp`). All six family-level mean Sys@10 changes
> favor the full pipeline.

# Current Parallel Stage-3 Result

> **Reportable paired control (2026-09-04):**
> [`current_parallel_stage3_ablation_results_20260904.md`](current_parallel_stage3_ablation_results_20260904.md) and
> [`Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/`](../Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/README.md)
> contain the completed three-seed Stage-3 ablation for the maintained parallel
> KNN + ReaFNN post-fusion mainline. The Stage-2 protocol and candidate
> availability match the current full mainline exactly; replacing XGB-LTR with a
> deterministic prior lowers macro Sys@10 from `43.77 +/- 0.60%` to
> `36.03 +/- 0.16%` (`-7.74 pp`). This is the only reportable Stage-3 component
> result for the current parallel candidate distribution.

# Current Parallel Temperature-Representation Result

> **Reportable paired control (2026-09-04):**
> [`current_parallel_temperature_ablation_results_20260904.md`](current_parallel_temperature_ablation_results_20260904.md) and
> [`Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/`](../Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/README.md)
> contain the completed three-seed temperature-only control for the maintained
> parallel mainline. It preserves Stage 1, Stage 2, XGB-LTR, all ranked systems,
> and the conditional temperature support exactly; it replaces only the 128D
> R-GNN route embedding in the temperature XGBoost regressor with no graph
> features. Macro MAE is `11.49 +/- 0.26 C` with R-GNN versus `13.93 +/- 0.38 C`
> without it; within-10 C accuracy improves from `55.56 +/- 1.23%` to
> `63.09 +/- 1.76%` (`+7.53 pp`). This is the reportable R-GNN contribution:
> temperature quality only, not a Sys@k claim.

# Archived Ablation Plan

> Historical serial evidence: see
> current_mainline_matched_ablation_results_20260830.md in this directory. It
> contains the completed three-seed KNN-only and deterministic no-XGB-LTR
> controls for the former serial candidate distribution.
>
> This document records an earlier KNN-XGB-LTR plan and is retained only for
> history. The direct-R-GNN historical protocol and result interpretation are
> in ablation_reafnn_gnn_protocol.md and
> current_mainline_ablation_results_20260727.md. Do not combine values in those
> archived records with the current tabular XGB-LTR results.

# Historical ProSys Ablation Plan

更新日期：`2026-07-14`

## 1. 这轮 ablation 的定位

这轮消融只围绕当前主线展开：

- Stage 1：`EditRetro base model -> family-specific finetuned model`
- Stage 2：`KNN` 可行条件筛选
- Stage 3：`XGB-LTR` 重排序

目标不是再去比较很多“外部方法”，而是把主线拆开，分别回答下面 3 个问题：

1. Stage 1 的家族微调是否真的提升了 route 推荐能力
2. Stage 2 的 `KNN` 候选池是否真的比简单高频条件池更有效
3. Stage 3 的 `XGB-LTR` 重排序是否真的有必要，还是只靠 Stage 1 和 Stage 2 的先验信息就够了

这样组织后，消融和主线是一一对应的，逻辑最干净。


## 2. 统一实验原则

这轮 ablation 统一采用 **Non-Oracle** 设定。

统一原则：

- 所有实验都使用真实 Stage 1 输出，不使用 oracle route
- 每个消融只改动当前要讨论的那个模块
- 其他模块尽量保持与主线一致
- 指标按家族分别统计，并给出宏平均

主线参考组定义为：

- `A0 = Finetuned Stage1 + KNN Stage2 + XGB-LTR Stage3`

后面的所有对比，都是相对 `A0` 来解释。


## 3. Ablation A1：Stage 1 微调是否有用

### 3.1 要回答的问题

- 同样面对各家族测试集，`base retrosynthesis model` 和 `family-tuned model` 的 route 命中率差多少

### 3.2 对比设置

比较两组 Stage 1 输出：

- `Base`
  - 使用全局基模直接在各家族测试集上推理
- `Finetuned`
  - 先用对应家族训练集微调，再在该家族测试集上推理

这里不往后接 Stage 2/3，只看 Stage 1 本身的 route 能力变化。

### 3.3 指标

- `Route@1`
- `Route@3`
- `Route@5`
- `Route@10`

### 3.4 已有数据支撑

当前仓库里已经有现成统计口径：

- `outputs/checklist_stats/07_stage1_base_vs_tuned.csv`
- `outputs/stage1_base_vs_tuned/overview.md`

因此这部分不是新造实验，只需要按现有结果整理成论文表格即可。

### 3.5 预期结论

如果 `Finetuned` 在大多数家族上都高于 `Base`，就可以说明：

- Stage 1 的家族适配是必要的
- 后续系统推荐提升的上游来源之一，确实来自更好的 route 候选


## 4. Ablation A2：Stage 2 的 KNN 是否有用

### 4.1 要回答的问题

- 在 Stage 1 和 Stage 3 都保持主线思路不变时，`KNN` 候选池是否真的比“直接取高频条件”更有效

### 4.2 对比设置

固定：

- Stage 1：使用 `Finetuned` route cache
- Stage 3：使用 `XGB-LTR` 排序与温度模型

比较两种 Stage 2 候选池：

- `KNN pool`
  - 对每条测试 route，从训练集检索局部近邻条件
- `Top-K frequency pool`
  - 不做近邻检索，直接从该家族训练集取全局频率最高的 `K` 个条件组合作为候选

其中：

- `K` 必须与 `KNN` 的候选池大小超参保持一致
- 也就是复用当前主线的 `max_contexts`
- 如果主线当前使用 `max_contexts = 20`，那么频率池也固定取 top-20

### 4.3 实现注意事项

为了保证公平，Stage 2 一旦变化，后续 Stage 3 不能直接沿用旧模型，而应该：

- 分别基于各自的候选池重建 `train / val / test table`
- 在对应 table 上各自训练一套 `XGB-LTR`
- 最终都在 Non-Oracle 测试集上评估 `full-system Top-k accuracy`

原因是：

- 候选池变了，候选分布和正负样本构成也变了
- 如果不重训 Stage 3，会把 Stage 2 的影响和 Stage 3 的失配混在一起

### 4.4 指标

主指标：

- `full-system Top-1 accuracy`
- `full-system Top-3 accuracy`
- `full-system Top-5 accuracy`
- `full-system Top-10 accuracy`

辅助解释指标：

- `candidate recall` (internal `pool_coverage`)

`candidate recall` (internal `pool_coverage`) 不是 headline metric，但它能帮助解释：

- `KNN` 是不是更容易把正确条件放进候选池

### 4.5 预期结论

如果 `KNN pool + XGB-LTR` 明显优于 `Top-K frequency pool + XGB-LTR`，就可以说明：

- Stage 2 的价值不只是“给一些常见条件”
- 它确实提供了 route-conditioned、局部化、可行性更高的候选筛选


## 5. Ablation A3：Stage 3 的 XGB-LTR 是否有用

### 5.1 要回答的问题

- 在候选池已经由 `KNN` 给出的前提下，是否还需要一个显式的学习式重排序器

### 5.2 对比设置

固定：

- Stage 1：使用 `Finetuned` route cache
- Stage 2：使用 `KNN` candidate pool

比较两种排序方式：

- `w/ Stage3`
  - 保留当前主线 `XGB-LTR` 重排序
- `w/o Stage3`
  - 去掉学习式 reranker，只使用 Stage 1 和 Stage 2 已经产生的先验信息做确定性排序

### 5.3 去掉 Stage 3 时的排序规则

`w/o Stage3` 不能再额外引入新的学习模型，否则就不叫“去掉 Stage 3”了。

因此建议采用固定的 heuristic ranking。候选按下面顺序排序：

1. `retro_rank` 升序
2. `retro_probability` 降序
3. `knn_similarity_sum` 降序
4. `knn_similarity_max` 降序
5. `knn_neighbor_count` 降序
6. `knn_weighted_mean_yield` 降序

这个定义的优点是：

- 完全不训练新模型
- 只依赖 Stage 1 和 Stage 2 已经生成的字段
- 可复现、可解释，也最符合“只去掉 Stage 3”的设定

### 5.4 指标

- `full-system Top-1 accuracy`
- `full-system Top-3 accuracy`
- `full-system Top-5 accuracy`
- `full-system Top-10 accuracy`

这组实验的重点是看：

- `full-system Top-1 accuracy` 能否明显提升
- `full-system Top-5 accuracy / full-system Top-10 accuracy` 能否进一步拉开

因为这最能体现 `XGB-LTR` 在候选重排上的价值。

### 5.5 预期结论

如果 `w/ Stage3` 稳定优于 `w/o Stage3`，就可以说明：

- Stage 2 负责“把可行候选找进来”
- Stage 3 负责“把真正正确的系统往前排”
- 两者功能不同，不能互相替代


## 6. 最终建议表格

建议论文里至少整理成 3 张表。

### 表 A：Stage 1 微调消融

列建议：

- `family`
- `base Route@1/3/5/10`
- `finetuned Route@1/3/5/10`
- `delta@1/3/5/10`

这张表回答：

- 家族微调是否真的改善了逆合成候选质量

### 表 B：Stage 2 候选池消融

列建议：

- `family`
- `KNN + XGB-LTR` 的 `full-system Top-1 accuracy/3/5/10`
- `Top-K frequency + XGB` 的 `full-system Top-1 accuracy/3/5/10`
- `candidate recall` (internal `pool_coverage`)

这张表回答：

- `KNN` 候选池是否真的优于简单高频条件池

### 表 C：Stage 3 重排序消融

列建议：

- `family`
- `w/o Stage3` 的 `full-system Top-1 accuracy/3/5/10`
- `w/ Stage3` 的 `full-system Top-1 accuracy/3/5/10`

这张表回答：

- `XGB-LTR` 学习式重排序是否真的必要


## 7. 这版规划的优点

这版消融逻辑有 3 个优点：

1. 每个实验只回答一个问题，因果关系清楚
2. 完全围绕当前主线，不再引入额外旧模型干扰叙事
3. 后续论文写作时，可以自然组织成“Stage 1 / Stage 2 / Stage 3 各自的必要性证明”

因此，这一版可以直接作为后续 ablation 实验的执行规范。
