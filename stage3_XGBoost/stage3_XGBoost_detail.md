# Stage 3 XGBoost: 候选重排与温度预测

## 1. 模块目标

`stage3_XGBoost` 负责承接 Stage 2 给出的 candidate pool，对每个样本内的候选条件做 **重排序**，并额外预测命中条件的反应温度。

这个阶段分成两个并行子任务：

1. `XGBRanker`
   - 学习“同一个样本内，哪组候选条件应该排得更前”

2. `XGBRegressor`
   - 学习“如果这组条件是对的，它的温度大概是多少”

当前实现文件：

- `stage3_XGBoost/xgb_reranker.py`
- `stage3_XGBoost/__init__.py`


## 2. 输入与输出

Stage 3 的输入不是原始反应，而是 **已经打好标的 candidate table**。  
这个表通常由两步得到：

1. Stage 2 生成候选池 csv
2. `baseline.common.label_candidate_table(...)` 对候选池做监督标注

每一行代表：

- 一个候选路线 + 条件组合
- 一组结构/统计特征
- 一组监督标签

输出则是在原表上新增：

- `xgb_score`
- `xgb_temperature_pred`（如果温度模型可训练）


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

这也是最后 `sys@k` 用的真正命中定义。


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

Stage 3 会先从 candidate table 中自动推断可用于 XGBoost 的数值特征。

### 4.1 永久保留的主干特征

优先使用 `prosys_shared/constants.py` 里定义的标准特征组：

- `ROUTE_DENSE_COLUMNS_V2`
- `CONTEXT_DENSE_COLUMNS_V2`
- `PRODUCT_DESCRIPTOR_COLUMNS_V2`
- `SUPPORT_FEATURE_COLUMNS_V2`

这些特征覆盖了：

- 路线相关密集特征
- 条件组合密集特征
- 产物分子描述符
- 产品记忆/支持度特征


### 4.2 额外自动纳入的数值特征

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
- `cluster_context_count`
- `cluster_context_support`
- `cluster_context_mean_yield`

所以 KNN/Cluster Stage 2 产生的支持特征，会自然被 Stage 3 学进去。


### 4.3 明确排除的列

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


## 5. 排序模型如何训练

### 5.1 为什么用 Ranker 而不是普通分类器

这里每个样本对应一个 candidate slate，目标不是“单行二分类”，而是“同一个 slate 内谁更应该排前面”。  
因此更合适的建模方式是 learning-to-rank。


### 5.2 训练单元

一个 `sample_index` 就是一组 query。  
同一个 `sample_index` 下的所有 candidate rows 共同组成一个 ranking group。

实现里会先按：

- `sample_index`
- `reaction_id`

做稳定排序，然后计算每组长度，交给 `XGBRanker(group=...)`。


### 5.3 目标函数

当前 ranker 使用：

- `objective = rank:ndcg`
- `eval_metric = ndcg@10`

含义是让模型更关注 top-10 位置的排序质量，这和我们最终看 `sys@1/5/10` 的评估目标是一致的。


### 5.4 默认超参

当前默认值：

- `n_estimators = 300`
- `learning_rate = 0.05`
- `max_depth = 6`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_lambda = 1.0`
- `tree_method = hist`
- `early_stopping_rounds = 30`

这些超参是一个稳健的 baseline 设定，优先保证：

- 训练快
- 不容易过拟合得太离谱
- 在 10 个 family 上都比较稳定


## 6. 温度模型如何训练

### 6.1 训练样本选择

温度不是对所有 candidate 都有意义，只对真正命中的条件有意义。  
因此温度回归只使用：

- `label = 1`
- 且 `temperature_gold` 有效

的样本行。

这一步很重要，因为如果把负例也塞进去，温度回归会学成“对错误条件也给出看似合理温度”，语义会被污染。


### 6.2 回归模型

当前使用：

- `XGBRegressor`
- `objective = reg:squarederror`
- `eval_metric = mae`

默认超参沿用 ranker 的主干超参，便于维护。


### 6.3 数据不足时怎么处理

某些 family 可能正例温度样本很少。  
因此实现里做了容错：

- 如果 `train` 中没有可用正例温度样本
  - 不训练温度模型
  - 但仍然写出 `xgb_temperature_meta.json`
  - `trained = false`

这样：

- 主排序流程不会被卡住
- 调用方也能明确知道“这个 family 没有温度模型”


## 7. 推理时怎么工作

推理分两步：

### 7.1 先做重排

对每个 candidate row 计算：

- `xgb_score`

最后按 `xgb_score` 降序排序。  
最终 top-k 的定义只看这个排序分数。


### 7.2 再做温度预测

如果当前目录下存在可用的温度模型，则对所有 candidate row 额外计算：

- `xgb_temperature_pred`

注意：

- 温度预测不会参与排序
- 排序和温度是两个并行头
- 最终汇报时，`sys@k` 还是只由 `xgb_score` 的 top-k 决定

这样设计的好处是：

- 不把温度误差引入排序目标
- 让“命中哪组条件”和“这组条件温度多少”分别优化


## 8. 保存的模型文件

Stage 3 训练输出目录下会保存：

- `xgb_ranker.json`
- `xgb_ranker_meta.json`
- `xgb_temperature.json`（若可训练）
- `xgb_temperature_meta.json`

其中：

- ranker meta 记录排序模型的特征列和超参
- temperature meta 记录温度模型是否训练成功、用了多少正例、特征列和超参

因此后续复现实验时，不需要重新猜当时到底用了哪些列。


## 9. 与主线 pipeline 的关系

在新的主线里：

1. Stage 1 提供路线候选
2. Stage 2 KNN 提供可行条件候选池
3. Stage 3 XGBoost 对候选池重排
4. 输出 top-k 条件，并附带温度预测

所以 `KNN + XGBoost` 的真实含义是：

- `KNN` 负责把“可能对的条件”找出来
- `XGBoost` 负责在这些候选里“把最像真的排前面”

不是简单的两个模型串起来，而是一个典型的：

- retrieval / screening
- reranking

两阶段结构。


## 10. 无泄露说明

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


## 11. 运行方式

命令行入口：

- `python -m stage3_XGBoost.xgb_reranker`

训练示例：

```bash
python -m stage3_XGBoost.xgb_reranker train \
  --train_table /root/autodl-tmp/ProSys/outputs/stage2_v2/Buchwald-HartwigCross-Coupling/training_tables/train.csv \
  --val_table /root/autodl-tmp/ProSys/outputs/stage2_v2/Buchwald-HartwigCross-Coupling/training_tables/val.csv \
  --output_dir /tmp/xgb_stage3
```

打分示例：

```bash
python -m stage3_XGBoost.xgb_reranker score \
  --table_file /root/autodl-tmp/ProSys/outputs/stage2_v2/Buchwald-HartwigCross-Coupling/training_tables/test.csv \
  --model_file /tmp/xgb_stage3/xgb_ranker.json \
  --metadata_file /tmp/xgb_stage3/xgb_ranker_meta.json \
  --output_file /tmp/xgb_stage3/test_scored.csv
```


## 12. 预期效果

Stage 3 XGBoost 的核心收益不是提升 candidate pool 覆盖率，而是：

- 在正确候选已入池的前提下，把真正命中的条件排到更靠前的位置
- 提升 `sys@1`
- 同时稳定拉升 `sys@5` 和 `sys@10`
- 输出一个可用的温度预测头

所以如果观察实验现象，一般应该这样理解：

- `pool_coverage` 主要由 Stage 2 决定
- `sys@k` 的前端排序质量主要由 Stage 3 决定
- 温度误差只在命中条件的样本上单独分析

也就是说，Stage 3 是“把 Stage 2 已经找出来的对答案，尽量往前推”的模块。
