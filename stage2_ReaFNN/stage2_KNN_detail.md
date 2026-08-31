# Stage 2 KNN + ReaFNN: 可行条件筛选

更新日期：`2026-08-30`

## Current Mainline: Product-Morgan KNN + ReaFNN Wide-Pool Refinement

> **Authoritative scope.** This section documents the promoted six-family,
> fixed-Stage-1, three-seed result retained in
> `Experiment/stage23_product_morgan_reafnn_multiseed_20260830/`. The
> lower fixed-core text is a historical B control, not the maintained default.

- KNN queries use only a 4,096-bit, radius-2 Morgan fingerprint of the target
  product; predicted reactants and reaction delta are intentionally omitted
  from retrieval.
- The memory contains only the current family's train split. KNN retrieves 64
  neighbors and aggregates a 64-context historical wide pool.
- ReaFNN receives the 8,218-dimensional route vector and scores every context
  in that retrieved historical pool using separate reagent-token and
  solvent-token heads.
- The first 12 KNN contexts are retained as anchors. ReaFNN's bounded,
  within-route rank correction fills the remaining positions from the same
  retrieved wide pool, with at most 20 contexts per proposed route.
- The official ReaFNN is a 512x2 ReLU MLP with dropout 0.10, correction weight
  0.65, correction clip 0.35, and no context augmentation. Therefore no
  generated or novel reagent-solvent combination enters the official pool.
- KNN retrieval, route caches, splits, vocabularies, and validation-only score
  calibration are fixed across seeds 0/1/2. ReaFNN, the R-GNN, and XGBoost are
  rebuilt independently.

The current Stage 2 candidate coverage is `54.44 +/- 0.14%`; the downstream
end-to-end macro Sys@1/3/5/10 is
`27.12 +/- 0.37% / 36.84 +/- 0.80% / 40.47 +/- 0.77% / 44.62 +/- 0.42%`.
The fixed Stage 1 macro Route@10 is `63.20%`. Test candidate slates use
persisted Stage 1 predictions; train/validation candidate tables use reference
split routes, as disclosed in `Experiment/stage23_legality_audit_20260830.md`.

The within-route initialization is
`stage2_initial_score = p_KNN + clip(0.65 * (q_ReaFNN - p_KNN), -0.35, 0.35)`.
The XGB-LTR schema excludes the bounded correction columns; validation can only
select the optional heuristic-prior fusion, never use test labels.

Detailed metrics and metadata are retained in
`Experiment/stage23_product_morgan_reafnn_multiseed_20260830/` and summarized
in `CURRENT_RESULTS.md`.

### Current matched ReaFNN ablation

Under the exact three-seed product-Morgan configuration, disabling ReaFNN while
retraining the 52-feature XGB-LTR on the resulting KNN-only candidate tables
reduces candidate coverage from 54.44 +/- 0.14% to 53.39 +/- 0.00% and macro
Sys@10 from 44.62 +/- 0.42% to 39.86 +/- 2.08%. The corresponding changes are
-1.06 pp coverage and -4.76 pp Sys@10. The full system has the higher mean
Sys@10 in all six families. This control isolates ReaFNN from ranker
distribution shift because each arm trains its own ranker rather than reusing a
ranker fitted on a different candidate pool.

The exact protocol, all rank cutoffs, per-family values, and audit contracts are
in ablation/current_mainline_matched_ablation_results_20260830.md.

## Historical Core-Only Policy (B Controlled Ablation)
> **Historical scope.** This section describes the B core-check control. Any later occurrence of "default" or "current" in this historical block refers to the `2026-08-28` configuration, not the Product-Morgan development mainline above.


> **范围优先级。** 本节定义当前默认实现，并覆盖后文仍保留的历史扩池描述。

- KNN 从当前 family 的 train memory 宽召回 `top_k = 64` 个相似路线，并聚合最多 `prefilter_contexts = 64` 个历史条件。
- KNN 的前 `max_contexts = 20` 个条件构成固定的 **KNN core**，即默认主线最终候选成员。
- ReaFNN 只对这 20 个已有 context 做 token-level compatibility check；默认不生成、补充或删除 `(reagent_norm, solvent_norm)`。
- 同一路线内的 KNN 次序和 ReaFNN 分数分别归一为 `p_KNN` 与 `q_ReaFNN`，再计算 `s_init = p_KNN + clip(alpha * (q_ReaFNN - p_KNN), -c, c)`。
- 生成时默认 `alpha = 0.20`、`c = 0.10`；该分数仅重新排列 KNN core 内部的顺序，再交给 Stage 3。

因此，默认路径的候选成员与纯 KNN 的前 20 个成员严格一致。ReaFNN 的职责是检查 KNN 先例是否与当前路线的试剂/溶剂 token 偏好一致，并做小幅、可追溯的次序校正；它不能因绝对分数尺度而重写 KNN 检索结论。

在 Stage 3 训练时，`alpha` 会在 family validation split 上从 `{0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40}` 中选择，优先最大化 Sys@10、再以 Sys@1 tie-break；测试标签不参与选择。校正字段不进入 `XGBRanker` 的学习特征，只通过验证集选择的 Stage 2 prior 与 XGB 原始分数融合，避免把同一条弱信号重复学习两次。

`--reafnn_enable_context_augmentation` 是显式实验开关。只有它被打开时，后文的历史组合补充和受限 novel-combination 逻辑才会执行；该分支不是默认主线，除非完成同协议多随机种子复现，不能用于论文 headline 结果。

### Historical core-check note

The material below documents the older fixed-KNN-core control and its archived
outputs. It is retained to explain the controlled ablation, not to define the
current method. The only official Stage 2 result source is the promoted
2026-08-30 artifact named at the top of this document.

## 1. 模块目标

`stage2_ReaFNN` 承接 Stage 1 给出的反应路线 `(reactants -> product)`：先从家族内训练集历史反应中检索相似路线，得到一个宽 KNN 池；再固定其前 20 个 context，并用轻量双头网络 `ReaFNN` 检查这些先例与路线 token 的一致性。
这个阶段仍然只做 **可行性筛选与初始优先级校正**，不做最终端到端排序；它向 Stage 3 交付固定候选成员及其 KNN/ReaFNN 支持信号。

当前实现文件：

- 核心实现：`stage2_ReaFNN/knn_condition_selector.py`
- 核心实现：`stage2_ReaFNN/reafnn_selector.py`
- 兼容入口：`stage2_KNN/__init__.py`

核心类：

- `KNNContextPoolBuilder`
- `ReaFNNSelector`


## 2. 输入与输出

输入有两种模式：

1. 监督候选表构建模式
   读取 family 的训练或验证 split，为 ReaFNN 与 Stage 3 组织训练候选。

2. 目标产物推理模式
   读取 Stage 1 的 `route_cache.json`，只对 Stage 1 预测出的路线生成候选池。

论文和正式评估使用第二种模式；下游不会接收测试样本的真实 reactants 或 conditions。

输出是一个 candidate table，每一行表示：

- 一个 Stage 1 路线
- 配上一组 `(reagent_norm, solvent_norm)` 候选条件
- 再带上一些 KNN 支持特征

与下游兼容的关键字段包括：

- `family`
- `sample_index`
- `reaction_id`
- `reactants`
- `product`
- `reagent_norm`
- `solvent_norm`
- `retro_rank`
- `retro_score`
- `retro_probability`
- `from_baseline_knn`
- `knn_similarity_sum`
- `knn_similarity_max`
- `knn_neighbor_count`
- `knn_weighted_mean_yield`
- `stage2_knn_rank`
- `stage2_knn_prior`
- `stage2_reafnn_check_score`
- `stage2_reafnn_residual`
- `stage2_reafnn_correction`
- `stage2_reafnn_correction_clip`
- `stage2_initial_score`
- `reafnn_reagent_score`
- `reafnn_solvent_score`
- `reafnn_context_score`
- `reafnn_context_count`
- `reafnn_context_support`
- `reafnn_mean_yield`
- `from_reafnn_generated`
- `from_reafnn_novel`
- `reafnn_is_historical`

为了和已有表结构统一，Stage 2 KNN 还会补齐 cluster 占位列：

- `cluster_id = -1`
- `cluster_context_count = 0`
- `cluster_context_support = 0`
- `cluster_context_mean_yield = 0`


## 3. 路线编码方式

这里需要明确区分两套输入：

- `KNN` 检索时使用的路线指纹
- `ReaFNN` 训练和推理时使用的路线特征

它们都来自同一条路线 `(reactants, product)`，但并不是完全同一个向量。

### 3.1 KNN 检索用的反应指纹

`KNN` 检索时，每条路线用 `reaction_morgan_fp(...)` 编码。
这个编码来自 `prosys_shared/features.py`，本质上是：

- 产品分子的 Morgan fingerprint
- `product_fp - reactant_fp` 对应的反应变化 fingerprint
- 最终拼接成一个定长向量

默认超参：

- `fpsize = 4096`
- `radius = 2`

因此 `KNN` 检索向量维度通常是：

- `2 * fpsize = 8192`


### 3.2 KNN 相似度

所有 `KNN` 指纹在进入检索前先经过 `normalize_fp(...)` 做 L2 归一化。
归一化后，相似度可直接用向量点积计算：

`similarity(query, memory_i) = normalized_fp(query) dot normalized_fp(memory_i)`

这等价于 cosine similarity，优点是实现简单、速度快，而且和后续矩阵乘法兼容。


### 3.3 ReaFNN 用的路线特征

`ReaFNN` 不只吃 `product_fp + delta_fp`，还会额外拼接轻量描述符。
它的实际输入是 `route_feature_vector(...)`，由三部分构成：

1. `reaction_morgan_fp`
   - 即上面 KNN 使用的 `8192` 维主指纹

2. `reaction_graph_descriptors`
   - 反应物描述符 `8` 维
   - 产物描述符 `8` 维
   - 二者差分描述符 `8` 维
   - 合计 `24` 维

3. `route_dense`
   - 反应物组分数 `1` 维
   - `reactants` 字符串长度 `1` 维
   - 合计 `2` 维

因此当前默认的 `ReaFNN` 总输入维度是：

- `8192 + 24 + 2 = 8218`

这部分特征在进入网络前还会做标准化：

- `x_std = (x - mean_train) / std_train`


### 3.4 KNN 和 ReaFNN 的关系

> **默认实现。** KNN 决定 final-core 成员，ReaFNN 只在其中产生有界初始分数校正；本小节下方关于历史补充和 novel 组合的内容仅适用于显式扩池实验。
当前默认主线里，这两者是 **串联**，不是两个彼此独立的并行筛选器。

顺序是：

1. `KNN` 先根据路线相似性做宽召回
2. KNN 的前 `max_contexts` 个 context 固定为 final core
3. `ReaFNN` 对 final core 内的每个 context 做 token 级一致性检查
4. 有界 residual 只调整 core 内部的 `stage2_initial_score` 次序

所以 `ReaFNN` 不是脱离 KNN 单独给出最终条件答案，而是：

- 默认只用于校正 KNN 已有候选的相对优先级
- 仅在显式扩池实验中才会补充历史或 novel 组合


## 4. 训练记忆库如何构建

KNN 的 memory **严格只由当前 family 的 train split 构建**，不会把 val/test 或 non-oracle test 数据混进去。

构建步骤：

1. 读取 train split 全部行
2. 按 `(reaction_id, reactants, product)` 分组
3. 每个唯一路线生成一个 fingerprint
4. 该路线历史上所有出现过的 `(reagent_norm, solvent_norm, yield)` 一起挂到这个 memory slot 上

内部保存两部分：

- `route_matrix`: 所有训练路线的归一化向量矩阵
- `route_contexts`: 与每条训练路线关联的历史条件记录

此外还会构造一个 `global_contexts` 作为全局回退池：

- 统计 train 中所有 `(reagent_norm, solvent_norm)` 组合
- 记录出现次数、支持率、平均 yield
- 当 query 找不到有效近邻时，直接返回这个全局高频条件池


## 5. 候选池如何生成

> **默认成员不变。** 宽 KNN 后先固定前 20 个 context；ReaFNN 只重排该固定集合，不再以 `reafnn_context_score` 截断或替换候选成员。
当前实现不再是“纯 KNN 直接截断”，而是两级流程：

1. 宽 KNN 召回
2. ReaFNN 二次筛选

### 5.1 单条 query 路线

对每条待筛选路线：

1. 用同样的方式编码成 reaction fingerprint
2. 与 `route_matrix` 做点积，得到和所有训练路线的相似度
3. 取 top-k 个近邻路线
4. 汇总这些近邻对应的历史条件
5. 对条件去重、打分、排序
6. 截断为 `max_contexts`

当前官方主线快照使用：

- `top_k = 64`
- `prefilter_contexts = 64`
- `max_contexts = 20`

也就是说，Stage 2 现在会先保留一个更“泛”的局部历史池，而不是过早把候选压得过窄。


### 5.2 条件聚合特征

对同一个 `(reagent_norm, solvent_norm)`，会从 top-k 邻居中累计出四个核心特征：

- `knn_similarity_sum`
  - 所有命中该条件的近邻相似度之和
  - 表示“这组条件被多少个相似反应支持，而且支持强度有多大”

- `knn_similarity_max`
  - 支持它的最相似近邻的相似度
  - 表示“是否存在一个特别像的历史反应给它背书”

- `knn_neighbor_count`
  - 支持它的近邻条数
  - 表示“这组条件在相似邻域中是否稳定出现”

- `knn_weighted_mean_yield`
  - 用相似度加权的历史平均产率
  - 表示“相似反应下，这组条件 historically 表现如何”


### 5.3 ReaFNN 二次筛选

宽 KNN 池建立后，会额外训练一个 `ReaFNN`：

- 主输入：`product_fp + delta_fp`
- 附加输入：`reaction_graph_descriptors` 与轻量 route dense 特征
- 主干：浅层 MLP
- 输出头 1：试剂 token 多标签概率
- 输出头 2：溶剂 token 多标签概率

这里的两个头都不是直接预测 `(reagent_norm, solvent_norm)` 整体字符串，而是分别预测：

- 试剂 token 集合
- 溶剂 token 集合

原因是当前数据清洗已经限制了：

- 试剂最多 3 个
- 溶剂最多 2 个

因此 token 级建模后，再回到历史组合空间里做受限组合，是可控的。

#### 5.3.0 ReaFNN 网络结构

当前 `ReaFNN` 是一个 **共享主干的浅层 MLP + 两个多标签输出头**。

默认结构参数：

- `hidden_dim = 384`
- `hidden_layers = 2`
- `dropout = 0.10`

因此默认网络结构可以写成：

```text
input route feature (8218)
 -> Linear(8218, 384)
 -> ReLU
 -> Dropout(0.10)
 -> Linear(384, 384)
 -> ReLU
 -> Dropout(0.10)
 -> shared hidden
 -> reagent_head: Linear(384, num_reagent_tokens)
 -> solvent_head: Linear(384, num_solvent_tokens)
```

这里有几个要点：

- 主干参数在两个任务之间共享
- `reagent_head` 负责试剂 token 多标签预测
- `solvent_head` 负责溶剂 token 多标签预测
- 不是 sequence generation，也不是逐 token 自回归生成
- 也不是直接把完整 `(reagent_norm, solvent_norm)` 当成一个超大类别做多分类

它更像是：

- 先判断“这条路线可能需要哪些试剂 token”
- 再判断“这条路线可能需要哪些溶剂 token”

然后 Stage 2 再把这些 token 概率映射回具体条件组合。



#### 5.3.0.1 训练标签如何构造

`ReaFNN` 的标签不是单条记录级别的完整 context 类别，而是先按路线聚合，再做 token 级多标签。

做法是：

1. 读取 family 的 `train/val` condition rows
2. 按 `(reaction_id, reactants, product)` 聚合为一路线一个样本
3. 把这条路线历史上出现过的全部试剂 token 取并集
4. 把这条路线历史上出现过的全部溶剂 token 取并集
5. 分别转成 reagent multi-hot 和 solvent multi-hot

因此一条路线的监督信号是：

- 哪些试剂 token 可能出现
- 哪些溶剂 token 可能出现

而不是：

- 只能出现某一个完整固定条件串

这对长尾数据更稳，也更适合后续和历史组合空间做结合。


#### 5.3.0.2 训练损失与优化

两个头都使用：

- `BCEWithLogitsLoss`

并且按 token 频率构造 `pos_weight`，缓解长尾类别极不平衡问题。
总损失就是：

- `loss = reagent_loss + solvent_loss`

默认训练配置：

- `optimizer = AdamW`
- `learning_rate = 1e-3`
- `weight_decay = 1e-5`
- `batch_size = 64`
- `max_epochs = 20`
- `patience = 5`

即如果验证集损失连续若干轮不再下降，就提前停止。

#### 5.3.1 ReaFNN 如何用在候选池上

> **默认检查路径。** `reafnn_context_score` 先转换为同路线内的 rank-normalized check score，再与 KNN prior 做截断残差校正。下方的生成步骤只在 `--reafnn_enable_context_augmentation` 打开时执行。
对每条 query 路线：

1. 对宽 KNN 池里的已有 context 逐个打 `reafnn_context_score`
2. 额外取 `ReaFNN` 预测分数最高的若干试剂 token / 溶剂 token
3. 先在训练集历史上**已出现过的 context 组合**中，筛出与这些高分 token 一致的组合
4. 再对高分试剂 token 子集与高分溶剂 token 子集做**受限组合**
5. 对训练集中从未出现过的完整新组合，加一个明显的 novelty penalty，并只保留极少量名额
6. 最后把“宽 KNN 候选 + 历史生成候选 + 少量 novel 候选”合并，做最终截断

这里的串联关系可以再明确写成一句：

- `KNN` 先负责“把可能可行的条件捞上来”
- `ReaFNN` 再负责“根据 token 语义重新排序，并小范围补池”

所以当前主线不是：

- `KNN` 给一份候选
- `ReaFNN` 再独立给另一份候选

而是：

- `KNN` 和 `ReaFNN` 共用同一个最终 candidate pool，只是承担的角色不同

其中当前主线明确区分：

- `Oracle/train/val` 构表：`allow_novel = False`
- `Non-Oracle` 路线推理：`allow_novel = True`

这样既满足：

- 能尝试 token 层面的新搭配

又满足：

- 最终尽量推荐历史上真实出现过的组合
- 避免完全自由组合带来的组合爆炸和噪声
- 即使允许 novel combination，也默认把它排在历史组合之后

#### 5.3.2 ReaFNN 的候选打分

对每个候选 context，会额外计算：

- `reafnn_reagent_score`
  - 该 context 中全部试剂 token 的平均预测概率

- `reafnn_solvent_score`
  - 该 context 中全部溶剂 token 的平均预测概率

- `reafnn_token_score`
  - 试剂头与溶剂头原始 token 分数的平均值

- `reafnn_prior_score`
  - token 边际历史先验

- `reafnn_historical_bonus`
  - 对历史已出现组合额外给的正向偏置

- `reafnn_novelty_penalty`
  - 对训练集中未出现过的新组合给的负向惩罚

- `reafnn_context_score`
  - 最终用于 Stage 2 排序的综合分数

这几个分数不是最终 end-to-end 排名，而是 Stage 2 内部的筛选优先级。

当前官方实现中的默认限制是：

- 试剂组合最多 `3` 个 token
- 溶剂组合最多 `2` 个 token
- novel context 默认最多生成 `8` 个
- 最终 candidate pool 中默认最多保留 `1` 个 novel context


#### 5.3.3 扩池实验中 Token 概率如何变成完整条件组合

这里最容易误解的一点是：

- `ReaFNN` 输出的是 token 概率表
- 不是直接输出完整的 `(reagent_norm, solvent_norm)` 条件串

因此从网络输出到最终条件候选，中间还要经历一个“受限组合”的步骤。

具体过程是：

1. 网络输出一组 `reagent_probs`
2. 网络输出一组 `solvent_probs`
3. 先取分数最高的一小批试剂 token
4. 再取分数最高的一小批溶剂 token
5. 在训练集历史 `context_library` 中寻找：
   - 这些高分 token 能覆盖的历史组合
6. 对高分试剂 token 的子集做受限组合
7. 对高分溶剂 token 的子集做受限组合
8. 试剂子集和溶剂子集再配对，形成 novel 候选
9. 对 novel 候选施加强惩罚，并严格限制数量

也就是说，确实存在“token -> 组合”的步骤，但不是无约束全排列，而是：

- 先优先回到历史组合空间
- 再对少量高分 token 子集做小规模组合
- 再用 `historical_bonus` 和 `novelty_penalty` 压住噪声

因此这个设计既能说明：

- 系统不是只会背历史完整组合

又不会让：

- 自由拼接组合导致搜索空间爆炸


#### 5.3.4 历史扩池分支的示意例子（非默认主线）

下面给一个简化后的示意例子，说明一条路线如何从 Stage 1 传到 Stage 2。

假设 Stage 1 给出一条路线：

- `reactants = A.B`
- `product = P`

并且这是该样本的 `retro_rank = 2` 路线。

第一步，`KNN` 宽召回：

- 用这条路线去 train memory 中找 `top_k = 64` 个相似历史反应
- 假设汇总后得到 30 个不重复的历史条件组合
- 再按 `knn_similarity_sum / knn_similarity_max / knn_neighbor_count` 排序
- 截出前 `prefilter_contexts = 64`，这里实际就是保留这 30 个

举例说，其中 3 个候选可能是：

- `Pd(PPh3)4 ; K2CO3` + `DMF`
- `Pd2(dba)3 ; XPhos ; Cs2CO3` + `toluene`
- `CuI ; Et3N` + `dioxane`

第二步，`ReaFNN` 看这条路线本身，输出 token 概率：

- 高分试剂 token 可能是：
  - `Pd(PPh3)4`
  - `K2CO3`
  - `XPhos`
  - `Cs2CO3`
- 高分溶剂 token 可能是：
  - `DMF`
  - `toluene`

第三步，`ReaFNN` 先给 KNN 已有候选逐个重打分：

- 若某个候选的试剂 token 和溶剂 token 都和高分 token 对得上
- 则它的 `reafnn_reagent_score`、`reafnn_solvent_score` 会更高
- 历史支持率高的组合还会得到额外 `historical_bonus`

第四步，`ReaFNN` 再尝试补充候选：

- 如果训练历史中存在：
  - `Pd(PPh3)4 ; K2CO3` + `toluene`
- 但这组组合没被当前 KNN 宽池捞到
- 那么它可以作为“历史补充组合”加入池子

第五步，少量尝试 novel 组合：

- 例如 token 层面看起来：
  - `XPhos ; K2CO3` + `DMF`
  很合理
- 但训练集中从未出现过这一整组
- 则它可以被生成为 novel candidate
- 但会带 `novelty_penalty`
- 而且默认最终池子里最多保留 `1` 条 novel 组合

第六步，Stage 2 最终输出：

- 把 “KNN 原始候选 + ReaFNN 历史补充候选 + 极少量 novel 候选” 合并
- 按 `reafnn_context_score` 为主排序
- 截断成 `max_contexts = 20`

最终这 20 条记录就会进入 Stage 3。

这个例子里最重要的理解是：

- `KNN` 决定“候选池的主体来自哪些相似历史反应”
- `ReaFNN` 决定“这些候选里哪些更像、还要不要补进少量额外组合”

因此当前默认 Stage 2 的本质不是简单投票，而是：

- 路线相似性检索
- token 级可行性检查
- 有界残差校正
- 固定候选成员输出

#### 5.3.5 历史扩池分支：为什么 novel context 很少

扩池实验保留 novel context，不是为了让它在候选池里大量出现，而是为了满足两个更重要的目标：

1. 明确说明系统并不被限制在历史完整组合空间里做封闭式选择。
2. 又不让完全自由组合带来的长尾噪声压垮扩池实验的 `full-system Top-k accuracy`。

因此该实验策略是一个刻意保守的折中：

- novel context 只在显式启用 `--reafnn_enable_context_augmentation` 的 `Non-Oracle` 推理时开放
- novel context 有显式 `novelty_penalty`
- 历史 context 有显式 `historical_bonus`
- 最终池子里最多保留 `1` 个 novel context

从 `2026-07-22` 的两家族 sanity check 看，这个现象非常明显：

- `Beckmann`：`44,060` 条 test candidate rows 中只出现 `1` 条 novel context
- `Buchwald-HartwigCross-Coupling`：`206,249` 条 test candidate rows 中只出现 `37` 条 novel context

这说明历史扩池分支中的新组合尝试本来就是低频事件。
这组数字仅是该实验分支的 sanity check，不代表当前默认 KNN-core 主线或新的 `6-family` 官方成绩表。

### 5.4 Stage 2 内部排序规则

Stage 2 内部只做一个轻量排序，不做最终决策。
默认候选按以下优先级排序：

1. `stage2_initial_score` 降序
2. `stage2_knn_rank` 升序，作为稳定 tie-break
3. `reafnn_is_historical` 降序
4. `knn_similarity_sum` 降序
5. `reagent_norm` 字典序
6. `solvent_norm` 字典序

默认路径中，排序前后的候选成员完全一致，都是 KNN final core 的前 `max_contexts` 个条件。扩池实验若被显式启用，也采用同一初始分数规则，但仅在保留 KNN anchor slot 后填补额外位置。


### 5.5 回退策略

以下情况会触发全局回退：

- train memory 为空
- query 和所有 train route 的相似度都接近 0
- top-k 邻居虽然存在，但没有汇聚出有效条件

这时返回 `global_contexts[:prefilter_contexts]` 再参与后续 ReaFNN 过程，保证候选池永远可用。


## 6. Oracle 与 Non-Oracle 的差别

### 6.1 Oracle

Oracle 模式直接从 gold split 中读路线：

- `build_table(split=...)`

用途：

- 评估“如果路线已知，KNN 候选池本身能覆盖多少正确条件”
- 给 Stage 3 XGBoost 训练 `train/val/test` 表


### 6.2 Non-Oracle

Non-Oracle 模式从 Stage 1 的 `route_cache.json` 中读取预测路线：

- `build_non_oracle_table(route_cache_file=...)`

用途：

- 做真正 end-to-end 评估
- 让候选池覆盖率受到 Stage 1 路线召回率的约束

这也是最后汇报时更重要的 setting。


## 7. 为什么这样编码

> **当前表述。** 默认逻辑是“局部 KNN 历史先验 + ReaFNN token 一致性检查 + 有界残差校正”，而不是自由组合生成。
当前 Stage 2 的逻辑可以概括成一句话：

“先按反应相似性做宽召回，再用 ReaFNN 对固定 KNN core 做 token 一致性检查和有界校正。”

这样做有几个优点：

- KNN 提供 family 内稳定的局部历史先验和候选召回
- ReaFNN 检查某个历史条件是否与当前路线的试剂/溶剂 token 偏好一致
- 固定成员避免长尾生成造成候选覆盖率和 Sys@k 的不稳定下降
- 输出的 `knn_* + reafnn_* + stage2_*` 支持信号可供 Stage 3 的先验融合使用

这里尤其要注意：

- 校正强度不是手工固定的最终结论，而是由 validation split 选择
- validation 可以选择 `alpha = 0`，从而完全退回 KNN prior
- 可选扩池实验与默认主线必须分开报告

也就是说，Stage 2 不只是“拿邻居投票”，而是：

- `KNN` 做宽候选召回并确定 final core
- `ReaFNN` 做 token 级一致性检查和有界残差校正
- 共同为 Stage 3 提供稳定的初始排序先验


## 8. 与主线的接口关系

当前主线里，Stage 2 KNN 的职责是：

1. 接收 Stage 1 路线
2. 生成 top-N 候选条件池
3. 把候选及其 KNN 支持特征写入 csv
4. 交给 `label_candidate_table(...)` 打标
5. 再交给 Stage 3 XGB-LTR 重排

所以它输出的是“候选池 + 支持特征”，不是最终答案。


## 8.1 Historical Core-Check Results

The fixed-core figures formerly reported in this section belong to the B
control, which retains the KNN top-20 members and only calibrates their order.
They are not current headline values. The official wide-pool refinement has 12
KNN anchors, a 20-context cap, and three-seed candidate coverage
`54.44 +/- 0.14%`; see `CURRENT_RESULTS.md` and
`Experiment/stage23_product_morgan_reafnn_multiseed_20260830/`.

## 9. 无泄露说明

当前实现里最关键的一点是：

- KNN memory 只使用当前 family 的 `train` split
- `val/test` 和 non-oracle test 的 query 只是去查这个 memory
- 没有把测试样本本身写回 memory

因此这里不存在 test leakage。

唯一需要知道的 caveat 是：

- 如果在 `train` split 上自己给自己构候选池，那么它可能会查到同一条训练路线的“自邻居”
- 这会让 train 候选池更乐观一些

但这只影响训练表的难度，不影响 test/non-oracle 泄露。


## 10. 运行方式

命令行入口在：

- `python -m stage2_KNN.knn_condition_selector`

Oracle 示例：

```bash
python -m stage2_KNN.knn_condition_selector \
  --repo_root /root/autodl-tmp/ProSys \
  --family Buchwald-HartwigCross-Coupling \
  --split test \
  --output_file /tmp/knn_test.csv
```

Non-Oracle 示例：

```bash
python -m stage2_KNN.knn_condition_selector \
  --repo_root /root/autodl-tmp/ProSys \
  --family Buchwald-HartwigCross-Coupling \
  --route_cache /root/autodl-tmp/ProSys/outputs/stage1_routes/Buchwald-HartwigCross-Coupling/route_cache.json \
  --output_file /tmp/knn_non_oracle.csv
```

可调参数：

- `--top_k`
- `--prefilter_contexts`
- `--max_contexts`
- `--fpsize`
- `--radius`
- `--max_routes`
- `--reafnn_device`
- `--reafnn_knn_anchor_contexts`（主线 runner 默认 `12`；低层 selector 的 `0` 仅保留全部 core 的历史安全模式）
- `--reafnn_correction_weight`（生成时的默认残差权重；正式实际值由 validation 校准）
- `--reafnn_correction_clip`
- `--reafnn_enable_context_augmentation`（显式实验开关，默认关闭）


## 11. 预期效果

Stage 2 KNN 的预期不是直接把 `full-system Top-k accuracy` 拉满，而是：

- 比全局高频条件更精准
- 比 cluster 分桶更细粒度
- 在不引入复杂 DL 筛选器的前提下，提供更高的候选覆盖率
- 为 Stage 3 XGB-LTR 提供强支持特征

因此它的主要观察指标应当是：

- `candidate recall`（内部字段为 `pool_coverage`）
- `Condition recall`
- `Route recall`
- 为 Stage 3 提供的最终 `full-system Top-k accuracy` 上限

如果 Stage 2 候选池没把正确条件放进来，Stage 3 再强也救不回来。
