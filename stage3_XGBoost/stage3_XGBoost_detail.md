# Stage 3 XGB-LTR 重排 + 统一 R-GNN 温度预测

更新日期：`2026-09-01`

## Current Mainline Interface: Parallel-Stage-2 XGB-LTR + Temperature-Only R-GNN

> **Authoritative scope.** Stage 3 now consumes the parallel KNN + ReaFNN
> post-fusion candidate distribution specified in
> `stage2_ReaFNN/stage2_KNN_detail.md`. Earlier serial Stage 2/3 figures and
> no-XGB ablations remain historical controls; they do not quantify the current
> candidate distribution.

- Stage 2 independently supplies up to 64 product-Morgan KNN contexts and 64
  ReaFNN-scored historical contexts, fuses their rank priors using a
  validation-selected family-specific mixture, and retains the top 20 contexts
  per Stage 1 route. Generated and novel contexts remain disabled.
- XGB-LTR ranks each route-context candidate with the fixed 52-column tabular,
  non-graph schema. It excludes `route_gnn_feat_*` and Stage 2 score-calibration
  columns so it cannot relearn a validation-selected mixture from a test label.
  The final score is the route-wise standardized XGB-LTR score plus a
  deterministic Stage 2 prior; its nonnegative fusion coefficient is selected
  on validation only and never changes candidate membership.
- Temperature remains a separate 180-column XGBoost regressor: the 52 tabular
  fields plus 128 R-GNN route features from a four-step message-passing encoder.
  It predicts temperature only and never changes candidate membership or rank.
- Test candidate slates always originate from persisted Stage 1 predictions.
  Training memory is family-train-only, and the ordinary ranked training table
  uses canonical leave-one-reaction-out retrieval where required.

The latest matched seed-0 parallel record is
`Experiment/stage2_parallel_post_fusion_20260901.md` (54.10% candidate recall,
43.44% macro Sys@10). Joint route-contrastive optimization and wrong-route
negative-sample supervision were retired after exploratory checks; neither is
part of the maintained XGB-LTR training or the reported parallel result.

### Historical serial XGB-LTR ablation

The serial 2026-08-30 system's 44.62 +/- 0.42% Sys@10 and its no-XGB-LTR
comparison are valid only for the former anchor-and-correction Stage 2 pool.
They remain in `ablation/current_mainline_matched_ablation_results_20260830.md`
for traceability and must not be reported as an ablation of the parallel pool.

## Historical Core-Check Interface (B Controlled Arm)

> **排序边界。** 此历史控制中 Stage 2 固定 KNN top-20 候选成员；ReaFNN 只提供有界的初始次序校正，不能增加或删除 Stage 3 的候选行。

- `stage2_knn_prior`、`stage2_reafnn_check_score`、`stage2_reafnn_residual` 和 `stage2_initial_score` 都是训练 memory 与 query 生成的推理特征，不含 gold label。
> **Historical scope.** The fixed-core interface below applies to the B core-check control. Its references to the default Stage 2 policy do not override the parallel post-fusion mainline above.

- `XGBRanker` 保持 52 维非图特征空间，并显式排除全部上述校正字段；R-GNN 也仍只服务温度回归。
- 对每个 family，Stage 3 仅在 validation split 上从 `alpha in {0, 0.05, ..., 0.40}` 选择 ReaFNN 校正强度，再选择 XGB 分数与 Stage 2 prior 的融合权重，优化顺序为 Sys@10 后 Sys@1。
- 如果校正无帮助，验证集可选择 `alpha = 0`；测试标签不参与 alpha 或融合权重选择。
- The following fixed-core material is retained as a historical B control. It does not override the maintained parallel post-fusion mainline.

### Historical Core-Check Note

The historical fixed-core configuration and the pre-hardening 2026-08-09 output
are retained below only for controlled-ablation traceability. Do not mix their
values, or the serial 2026-08-30 values, with the parallel post-fusion result
records.

## Archived Serial Reference Details

> The detailed serial correction and calibration material below is retained for
> legacy-control traceability. The maintained parallel interface is the one
> specified at the top of this document.

## 1. 模块目标

`stage3_XGBoost` 负责承接 Stage 2 给出的 candidate pool，对每个样本内的候选条件做 **重排序**，并额外预测命中条件的反应温度。

这个阶段分成两个并行子任务：

1. `XGBRanker`
   - 学习“同一个样本内，哪组候选条件应该排得更前”

2. `XGBRegressor`
   - 学习“如果这组条件是对的，它的温度大概是多少”

当前实现文件：

- `stage3_XGBoost/xgb_reranker.py`
- `stage3_XGBoost/reaction_gnn_features.py`
- `stage3_XGBoost/__init__.py`
- `stage3_XGBoost/condition_aware_gnn.py`（探索性 candidate-aware residual）
- `stage3_XGBoost/condition_aware_gnn_detail.md`（该分支的输入、训练、门控和结果状态）


## 2. 输入与输出

Stage 3 的输入不是原始反应，而是 **已经打好标的 candidate table**。
这个表通常由两步得到：

1. Stage 2 生成候选池 csv
2. `prosys_shared.mainline.label_candidate_table(...)` 对候选池做监督标注

每一行代表：

- 一个候选路线 + 条件组合
- 一组结构/统计特征
- 一组监督标签

输出则是在原表上新增：

- `xgb_score_raw`
- `xgb_score_z`
- `stage2_heuristic_prior`
- `xgb_score`
- `xgb_temperature_pred`（如果温度模型可训练）

这里要特别强调一件事：

- Stage 3 **不会增加新的候选条件**
- Stage 3 只会重新排列 Stage 2 已经给出的候选，并附带温度输出

因此：

- `candidate recall`（内部字段为 `pool_coverage`）主要由 Stage 2 决定
- `full-system Top-k accuracy` 前排质量主要由 Stage 3 决定


## 3. 监督标签是怎么来的

对每个 candidate row，会和 gold split 进行对齐，得到三层标签：

### 3.1 `route_match`

只看路线是否命中。
即 candidate 的 `reactants` 规范化后，是否与 gold 的反应路线一致。


### 3.2 `context_match`

只看条件组合是否命中。
即 candidate 的 `(reagent_norm, solvent_norm)` 是否在该样本的 gold 条件集合里出现过。


### 3.3 `label`

严格正例。
只有 `(route_key, reagent_norm, solvent_norm)` 三者同时命中，`label = 1`。

这也是最后 `full-system Top-k accuracy` 用的真正命中定义。


### 3.4 中间派生标签

为排序任务还会派生：

- `label_type`
  - `positive`
  - `route_only`
  - `context_only`
  - `negative`

- `sample_weight`
  - 目前主要是为不同难度样本保留权重信息

- `rank_relevance`
  - `positive = 3`
  - `route_only = 2`
  - `context_only = 1`
  - `negative = 0`

XGBRanker 训练时真正回归/排序的目标就是 `rank_relevance`。


## 4. 特征如何组织

当前 Stage 3 同时维护两套特征视图：

1. 无图 candidate-table 特征，仅供 `XGBRanker` 重排；
2. 上述特征加 R-GNN embedding，仅供温度回归候选头。

因此，`route_gnn_feat_*` 存在于表中不代表它会被排序器读取。

### 4.1 永久保留的主干特征

优先使用 `prosys_shared/constants.py` 里定义的标准特征组：

- `ROUTE_DENSE_COLUMNS_V2`
- `CONTEXT_DENSE_COLUMNS_V2`
- `PRODUCT_DESCRIPTOR_COLUMNS_V2`
- `SUPPORT_FEATURE_COLUMNS_V2`

无图排序主干特征覆盖：

- 路线相关密集特征
- 条件组合密集特征
- 产物分子描述符
- 产品记忆/支持度特征


### 4.2 R-GNN 特征

为温度回归提供反应结构信息，Stage 3 会额外训练一个轻量 R-GNN，用它为每条 `(reactants, product)` 路线抽取固定维度 embedding。

实现思路：

- 用 RDKit 将 `reactants` 和 `product` 各自转成分子图
- 用共享的 message passing encoder 分别编码反应物图与产物图
- 做全局池化后得到：
  - `h_reactant`
  - `h_product`
- 最终 reaction embedding 取：
  - `concat(h_reactant, h_product, h_product - h_reactant)`

然后再经过一个小投影层，得到固定长度的：

- `route_gnn_feat_0 ... route_gnn_feat_127`

这批列会写回 `train / val / test table`，但当前 `XGBRanker` 显式排除它们；只有候选的 GNN 温度回归器读取这些列。

### 4.3 额外自动纳入的数值特征

除了标准主干列，Stage 3 还会把表里新增的数值列自动收进来，例如：

- `retro_rank`
- `retro_score`
- `retro_probability`
- `num_reagents`
- `num_solvents`
- `route_component_count`
- `reactants_length`
- `knn_similarity_sum`
- `knn_similarity_max`
- `knn_neighbor_count`
- `knn_weighted_mean_yield`
- `reafnn_reagent_score`
- `reafnn_solvent_score`
- `reafnn_context_score`
- `reafnn_context_count`
- `reafnn_context_support`
- `reafnn_mean_yield`
- `cluster_context_count`
- `cluster_context_support`
- `cluster_context_mean_yield`

所以 KNN/Cluster Stage 2 产生的支持特征，会自然被 Stage 3 学进去。

> **校正字段例外。** `stage2_knn_rank`、`stage2_knn_prior`、`stage2_reafnn_check_score`、`stage2_reafnn_residual`、`stage2_reafnn_correction` 和 `stage2_initial_score` 显式排除在 ranker 特征外；它们只用于 validation-gated heuristic prior，不作为可学习的排序输入。


### 4.4 明确排除的列

以下列不会作为特征进入模型：

- 文本列
  - `family`
  - `reaction_id`
  - `reactants`
  - `product`
  - `reagent_norm`
  - `solvent_norm`
  - `route_canonical`
  - `product_canonical`
  - `label_type`

- 监督目标列
  - `label`
  - `route_match`
  - `context_match`
  - `rank_relevance`
  - `sample_weight`
  - `temperature_gold`
  - `yield_gold`

- 历史 legacy 排名器输出
  - 任何 `legacy_*` 列

这样可以避免显式标签泄露给 XGBoost。


## 5. R-GNN 如何训练

R-GNN 本身不是最终排序器，它只负责输出 reaction embedding。

### 5.1 R-GNN 的输入图

每条路线 `(reactants, product)` 会被拆成两张图：

- 反应物图
- 产物图

图是用 RDKit 从规范化后的 SMILES 生成的。
如果某条 SMILES 无法正常转图，则会回退成一个最小零图，保证整个 Stage 3 不因为个别异常样本崩掉。

每个原子会编码一组轻量原子特征，主要包括：

- 常见原子序数 one-hot
- degree one-hot
- formal charge
- aromatic / ring 标记
- 氢原子数
- hybridization
- mass 归一化值


### 5.2 R-GNN 的网络结构

当前实现使用一个共享图编码器分别编码反应物图和产物图。

默认超参：

- `hidden_dim = 128`
- `embedding_dim = 128`
- `message_passing_steps = 4`
- `dropout = 0.10`

可以把当前结构近似写成：

```text
reactant graph
 -> GraphEncoder(shared)
 -> h_reactant

product graph
 -> GraphEncoder(shared)
 -> h_product

concat(h_reactant, h_product, h_product - h_reactant)
 -> Linear + ReLU + Dropout
 -> reaction embedding (128-d)
 -> reagent_head
 -> solvent_head
```

其中图编码器内部是一个轻量 message-passing 结构：

- 先把原子特征投影到 `hidden_dim`
- 再做若干轮邻居聚合
- 每轮都把自身信息和邻居平均信息一起更新
- 最后对整张图做 mean pooling

这样设计的目的不是追求特别重的图模型，而是：

- 用一个足够轻的图网络，给 Stage 3 提供稳定的结构表示
- 避免让主线因为 GNN 过重而难复现、难维护


### 5.3 R-GNN 的训练任务

R-GNN 的训练任务是一个辅助多标签任务：

- 头 1：预测试剂 token 集合
- 头 2：预测溶剂 token 集合

输入只看反应本身：

- `reactants`
- `product`

监督来自 family `train/val` split 中真实出现过的条件 token。

也就是说，它不是直接学：

- “哪个完整 `(reagent_norm, solvent_norm)` 应该排第一”

而是先学：

- “这条反应从结构上更偏向哪些试剂 token”
- “更偏向哪些溶剂 token”

然后把这个结构 embedding 交给候选的 GNN 温度回归器利用，而不交给当前排序器。


### 5.4 R-GNN 的训练方式

两个辅助头都使用：

- `BCEWithLogitsLoss`

并按 token 频率计算 `pos_weight` 来对抗长尾不平衡。
总损失就是：

- `loss = reagent_loss + solvent_loss`

默认训练配置：

- `learning_rate = 1e-3`
- `weight_decay = 1e-5`
- `batch_size = 48`
- `max_epochs = 20`
- `patience = 5`

因此当前 R-GNN 的角色可以概括成一句：

- 它不是 Stage 3 的最终排序器，而是所有家族固定使用的温度结构特征抽取器

### 5.5 candidate-aware GNN residual（已实现，未纳入正式主线）

当前代码还实现了一个历史 candidate-specific GNN residual：它把冻结的图表征
与 reagent/solvent token embedding 做交互，从而让同一路线下
不同条件候选得到不同 GNN 分数。该分支不向 `XGBRanker` 回灌 in-sample GNN 分数，
而是仅在 XGBoost 已经固定后，用 validation-only 的 score fusion gate 决定是否启用。

当前已完成的 Beckmann interaction-model pilot 未达到预先规定的 validation full-system Top-10 accuracy
提升阈值，因此选中 `alpha = 0`；相关 six-family auxiliary residual probe 也没有任何
family 通过非零 residual gate。故它不改变当前 `full-system Top-k accuracy`、不修改正式输出，不能被写成
已验证的主线增益。完整细节见
`stage3_XGBoost/condition_aware_gnn_detail.md`。


## 6. 排序模型如何训练

### 6.1 为什么用 Ranker 而不是普通分类器

这里每个样本对应一个 candidate slate，目标不是“单行二分类”，而是“同一个 slate 内谁更应该排前面”。
因此更合适的建模方式是 learning-to-rank。


### 6.2 训练单元

一个 `sample_index` 就是一组 query。
同一个 `sample_index` 下的所有 candidate rows 共同组成一个 ranking group。

实现里会先做稳定排序，再按 group 长度交给 `XGBRanker(group=...)`。
稳定排序优先参考：

- `sample_index`
- `reaction_id`
- `retro_rank`
- `retro_score`
- `retro_probability`
- `product`
- `reactants`
- `reagent_norm`
- `solvent_norm`

这样做的作用是：

- 保证训练和推理时 group 顺序稳定
- 避免同分候选由于 csv 读写顺序抖动而造成结果不可复现


### 6.3 排序目标到底是什么

XGBoost 看到的监督目标不是二值 `label`，而是：

- `rank_relevance`

对应关系是：

- `positive = 3`
- `route_only = 2`
- `context_only = 1`
- `negative = 0`

因此它学到的不是单纯“是不是正例”，而是一个更细粒度的优先级：

1. 路线和条件都对的，排最前
2. 至少路线对的，优先级高于完全负例
3. 只有条件对但路线不对的，也比纯负例更有信息

这让 Stage 3 在训练时能够利用更多弱监督结构，而不是只盯着稀少的严格正例。


### 6.4 目标函数

当前 ranker 使用：

- `objective = rank:ndcg`
- `eval_metric = ndcg@10`

含义是让模型更关注 top-10 位置的排序质量，这和我们最终看 `full-system Top-1 accuracy/5/10` 的评估目标是一致的。


### 6.5 默认超参

当前默认值：

- `n_estimators = 300`
- `learning_rate = 0.05`
- `max_depth = 6`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_lambda = 1.0`
- `tree_method = hist`
- `early_stopping_rounds = 30`

这些超参是一个稳健的主线设定，优先保证：

- 训练快
- 不容易过拟合得太离谱
- 在当前保留的 `6-family` 上比较稳定


## 7. 温度模型如何训练

### 7.1 训练样本选择

温度不是对所有 candidate 都有意义，只对真正命中的条件有意义。
因此温度回归只使用：

- `label = 1`
- 且 `temperature_gold` 有效

的样本行。

这一步很重要，因为如果把负例也塞进去，温度回归会学成“对错误条件也给出看似合理温度”，语义会被污染。


### 7.2 回归模型

当前使用：

- `XGBRegressor`
- `objective = reg:squarederror`
- `eval_metric = mae`

默认超参沿用 ranker 的主干超参，便于维护。


### 7.3 数据不足时怎么处理

某些 family 可能正例温度样本很少。
因此实现里做了容错：

- 如果 `train` 中没有可用正例温度样本
  - 不训练温度模型
  - 但仍然写出 `xgb_temperature_meta.json`
  - `trained = false`

这样：

- 主排序流程不会被卡住
- 调用方也能明确知道“这个 family 没有温度模型”


## 8. 推理时怎么工作

推理分两步：

### 8.1 先做重排

> **两步 validation-only 校准。** 先在 validation 表上以每个 `alpha` 重算 `stage2_initial_score = p_KNN + clip(alpha * (q_ReaFNN - p_KNN), -c, c)`，再由该 prior 选择 `heuristic_weight`；二者以 Sys@10、再 Sys@1 决定。候选行和测试标签均不参与这两个选择。
对每个 candidate row，当前主线会先计算：

- `xgb_score_raw`

然后在每个 `sample_index` 组内做标准化，得到：

- `xgb_score_z`

与此同时，代码还会根据 Stage 2 候选池的原始排序位置，生成一个轻量先验：

- `stage2_heuristic_prior`

最终正式排序分数定义为：

- `xgb_score = xgb_score_z + heuristic_weight * stage2_heuristic_prior`

其中 `heuristic_weight` 不是手工固定的，而是在 validation set 上做一个小网格搜索，优先最大化 `system_top10_all`（公开指标 `full-system Top-10 accuracy`），再用 `system_top1_all`（`full-system Top-1 accuracy`）作为 tie-break。
最后按 `xgb_score` 降序排序，最终 top-k 的定义只看这个融合后的排序分数。


### 8.2 再做温度预测

如果当前目录下存在可用的温度模型，则对所有 candidate row 额外计算：

- `xgb_temperature_pred`

注意：

- 温度预测不会参与排序
- 排序和温度是两个并行头
- 最终汇报时，`full-system Top-k accuracy` 还是只由 `xgb_score` 的 top-k 决定

这样设计的好处是：

- 不把温度误差引入排序目标
- 让“命中哪组条件”和“这组条件温度多少”分别优化


### 8.3 温度在评估时怎么统计

虽然温度模型会给所有 candidate rows 都输出一个 `xgb_temperature_pred`，
但最终评估时不会把所有候选都拿来统计温度误差。

当前正式口径是：

1. 对每个样本，先按 `xgb_score` 从高到低排序
2. 在这个排序里找到最高排名的 full-match candidate
3. 只在这条 full-match candidate 上统计温度误差

因此当前温度指标的含义是：

- “如果主线最终把这组完整体系排到最前，它给出的温度有多准”

而不是：

- “对所有候选行随便做一个回归误差平均”


### 8.4 一个具体例子

假设某个 `sample_index = 42` 在 Stage 2 之后保留了 5 条候选：

1. 路线对，条件错
2. 路线对，条件对
3. 路线错，条件看起来像
4. 路线错，条件也错
5. 路线对，条件错

那么打标后，这 5 行的 `rank_relevance` 可能是：

- `[2, 3, 1, 0, 2]`

Stage 3 的目标就是学习把第 2 行尽量排到最前，而不是只学一个“哪几行是 1，哪几行是 0”的分类器。

如果推理后 `xgb_score` 排序结果变成：

1. 第 2 行
2. 第 1 行
3. 第 5 行
4. 第 3 行
5. 第 4 行

那么：

- `full-system Top-1 accuracy` 记一次命中
- 如果第 2 行温度真值是 `80 C`，预测是 `73 C`
- 则该样本会给温度统计贡献一个 `7 C` 的绝对误差

这个例子说明：

- `XGBRanker` 决定的是完整体系排序
- `XGBRegressor` 只在真正 full-match 的候选上才有化学意义


## 9. 保存的模型文件

Stage 3 训练输出目录下会保存：

- `xgb_ranker.json`
- `xgb_ranker_meta.json`
- `xgb_temperature.json`（若可训练）
- `xgb_temperature_meta.json`

其中：

- ranker meta 记录排序模型的特征列和超参
- temperature meta 记录温度模型是否训练成功、用了多少正例、特征列和超参

因此后续复现实验时，不需要重新猜当时到底用了哪些列。


## 10. 与主线 pipeline 的关系

在维护中的并行主线里：

1. Stage 1 提供路线候选
2. KNN 与 ReaFNN 独立提出 train-only historical contexts，并在 Stage 2 后融合
3. Stage 3 的无图 `XGBRanker` 对固定融合候选池重排
4. R-GNN 表征和温度 `XGBRegressor` 为每个 family 输出温度预测

所以当前主线更准确的真实含义是：

- `KNN` 负责从 product-only Morgan 近邻提出历史先例
- `ReaFNN` 独立从完整历史条件库提出 token-compatible contexts
- validation-only 后融合决定二者对每条路线初始候选池的相对权重
- tabular `XGB-LTR` 负责在固定候选池中“把最像真的排前面”
- `R-GNN` 固定提供反应结构表示，温度 `XGBRegressor` 将其与路线和候选条件特征联合建模

不是简单的两个模型串起来，而是一个典型的：

- retrieval / screening
- reranking

两阶段结构。


## 11. 无泄露说明

- ReaFNN 校正强度和 XGB/Stage 2 prior 融合权重只使用 family validation split 选择；test/non-oracle test 仅在模型与权重冻结后评分一次。
当前实现里，XGBoost 的训练数据来源是：

- train table 只用于训练
- val table 只用于 early stopping / 调参监控
- test 或 non-oracle test 只用于最终打分

同时特征推断时明确排除了：

- `label`
- `route_match`
- `context_match`
- `temperature_gold`
- `yield_gold`

因此不会把监督答案直接喂给 XGBoost。

- `route_gnn_feat_*`（排序头显式排除）

## 12. 运行方式

命令行入口：

- `python -m stage3_XGBoost.xgb_reranker`

训练示例：

```bash
python -m stage3_XGBoost.xgb_reranker train \
  --train_table /root/autodl-tmp/ProSys/outputs/stage23_mainline_reafnn_gnn_fused_20260723/Buchwald-HartwigCross-Coupling/_shared_reaction_gnn/training_tables/train.csv \
  --val_table /root/autodl-tmp/ProSys/outputs/stage23_mainline_reafnn_gnn_fused_20260723/Buchwald-HartwigCross-Coupling/_shared_reaction_gnn/training_tables/val.csv \
  --output_dir /tmp/xgb_stage3
```

打分示例：

```bash
python -m stage3_XGBoost.xgb_reranker score \
  --table_file /root/autodl-tmp/ProSys/outputs/stage23_mainline_reafnn_gnn_fused_20260723/Buchwald-HartwigCross-Coupling/_shared_reaction_gnn/training_tables/test.csv \
  --model_file /tmp/xgb_stage3/xgb_ranker.json \
  --metadata_file /tmp/xgb_stage3/xgb_ranker_meta.json \
  --output_file /tmp/xgb_stage3/test_scored.csv
```


## 13. 预期效果

Stage 3 XGB-LTR 的核心收益不是提升 candidate pool 覆盖率，而是：

- 在正确候选已入池的前提下，把真正命中的条件排到更靠前的位置
- 提升 `full-system Top-1 accuracy`
- 同时稳定拉升 `full-system Top-5 accuracy` 和 `full-system Top-10 accuracy`
- 输出一个可用的温度预测头

所以如果观察实验现象，一般应该这样理解：

- `candidate recall`（内部字段为 `pool_coverage`）主要由 Stage 2 决定
- `full-system Top-k accuracy` 的前端排序质量主要由 Stage 3 决定
- 温度误差只在命中条件的样本上单独分析

也就是说，Stage 3 是“把 Stage 2 已经找出来的对答案，尽量往前推”的模块。


## 14. 历史：图特征直接进入 ranker 的快照（非当前 headline）

在 `outputs/stage23_mainline_reafnn_gnn_fused_20260723/` 的历史 direct-GNN-ranking 快照里，Stage 3 的宏平均表现是：

- `full-system Top-1 accuracy = 27.64%`
- `full-system Top-3 accuracy = 35.55%`
- `full-system Top-5 accuracy = 38.47%`
- `full-system Top-10 accuracy = 42.68%`
- `nDCG@10 = 0.341`
- `MRR = 0.327`

该历史快照的温度头在“最高排名 full-match 候选”上的宏平均表现是：

- `MAE (deg C) = 11.28`
- `Within +/-5 deg C = 37.96%`
- `Within +/-10 deg C = 61.60%`
- `Within +/-20 deg C = 82.98%`

该历史快照说明，图特征直接进入排序器没有带来稳定的 full-system Top-k accuracy 增益。当前主线
保留结构 encoder，但只把它输入所有家族固定使用的温度分支；新的统一实现见下一节。


## 15. Historical Serial Temperature Implementation and Validation

历史串联主线在 `scripts/run_stage23_mainline_non_oracle.py` 中执行两个固定头：

1. 排序头：从显式排除全部 `route_gnn_feat_*` 的 candidate table 训练 `XGBRanker`，并在
   validation 上选择 Stage 2 heuristic 融合权重。
2. 温度头：每个 family 固定训练一个读取 128 个 `route_gnn_feat_*` 的
   `XGBRegressor`。没有 no-GNN 温度备用头、没有基于温度 MAE 的模型选择，也不按
   family 分支。

该设计的边界是清楚的：R-GNN 不改写 `xgb_score`、候选顺序或 `full-system Top-k accuracy`；
它只给温度回归器补充从 `(reactants, product)` 提取的结构表征。温度指标仍只在
最高排名的完整命中候选且温度标签有效时计算。

历史串联六家族、三随机种子复现已经完成。固定 Stage 1 的宏平均结果为：

- `full-system Top-1 accuracy = 27.12 +/- 0.37%`
- `full-system Top-3 accuracy = 36.84 +/- 0.80%`
- `full-system Top-5 accuracy = 40.47 +/- 0.77%`
- `full-system Top-10 accuracy = 44.62 +/- 0.42%`
- `MRR = 33.28 +/- 0.48%`，`nDCG@10 = 34.70 +/- 0.53%`
- 条件温度 `MAE = 11.73 +/- 0.54 C`，`+/-5/+/-10/+/-20 C` 命中率分别为
  `40.59 +/- 0.56% / 61.80 +/- 2.36% / 83.04 +/- 1.10%`

每轮均有 `3,833/3,860` 个测试产物获得 Stage 1 candidate slate；其余 27 个仍
保留在全系统分母中并记为失败。所有 18 个 family-seed 温度元数据均记录
固定启用的 R-GNN + XGBoost 温度分支；18 个排序器均为 52 个非图特征，
18 个温度回归器均为 180 个特征（52 个表格特征加 128 个 R-GNN 特征）。

此结果证明无门控方案可稳定运行且不依赖测试集选择模型。它不是“仅移除 gate”的
严格因果消融，因为 R-GNN 容量同时从旧快照升级为 128 维、四层 message passing；
若要单独量化 gate 的影响，仍需在完全相同结构下做有/无 gate 的受控实验。
