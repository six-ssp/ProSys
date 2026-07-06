# Stage 2 Ablation: 具体实现说明

更新日期：`2026-07-05`

## 1. 这份文档解释什么

这份文档只解释 Stage 2 消融：

- `KNN + XGBoost`
- `Cluster + XGBoost`
- `FNN-pool + XGBoost`

这里固定的是 Stage 3：

- 排序统一用 `XGBRanker`
- 温度统一用 `XGBRegressor`

变化的只有 Stage 2 候选池来源。

因此它回答的问题是：

- 在同一个 XGBoost reranker 下，哪种 candidate screening 更适合当前任务


## 2. 相关代码文件

Stage 2 消融主要依赖下面这些文件：

- `ablation/run_non_oracle_ablation.py`
  - ablation 的独立运行入口
- `baseline/run_non_oracle_stage23_experiments.py`
  - 真正执行 Stage 2 / Stage 3 ablation 的统一脚本
- `stage2_KNN/knn_condition_selector.py`
  - KNN 候选池实现
- `baseline/run_oracle_baselines.py`
  - `ClusterContextPoolBuilder`
  - cluster 候选池实现
- `baseline/legacy_models.py`
  - 历史 FNN 兼容包装
- `baseline/common.py`
  - candidate table 打标
  - 统一评估
- `stage3_XGBoost/xgb_reranker.py`
  - Stage 3 固定使用的 XGBoost 排序与温度模型


## 3. 运行入口

推荐直接用 ablation 目录下的新入口：

```bash
conda activate ProSys
python ablation/run_non_oracle_ablation.py \
  --repo_root . \
  --families all \
  --output_root outputs/stage23_non_oracle_all10 \
  --run_set stage2_ablation
```

这个脚本本质上只是一个 wrapper，它内部调用：

- `baseline.run_non_oracle_stage23_experiments.main(...)`

并把：

- `--run_set stage2_ablation`

作为默认行为传进去。


## 4. Stage 2 消融的统一原则

为了让结论干净，三条 Stage 2 路线共享下面这些设定：

### 4.1 同一个 Stage 1 输入

三者都读取：

- `outputs/stage1_routes/<family>/route_cache.json`

所以它们面对的是同一批 Stage 1 预测路线。


### 4.2 同一个 train / val / test gold split

打标统一使用：

- `train`
- `val`
- `test`

的官方 split 文件。


### 4.3 同一个 Stage 3 学习器

三者都调用：

- `baseline.common.train_xgb_ranker(...)`
- `baseline.common.score_table_with_xgb(...)`

也就是：

- 排序固定 `XGBRanker`
- 温度固定 `XGBRegressor`

因此最终差异只能来自 candidate pool 本身。


## 5. 统一的数据组织方式

在统一脚本里，每个 family 都会先创建三套共享缓存目录：

- `<output_root>/<family>/_shared_knn/`
- `<output_root>/<family>/_shared_cluster/`
- `<output_root>/<family>/_shared_fnnpool/`

每套目录下面又分成：

- `candidate_pool/train.csv`
- `candidate_pool/val.csv`
- `candidate_pool/test.csv`
- `training_tables/train.csv`
- `training_tables/val.csv`
- `training_tables/test.csv`

其中：

- `candidate_pool/*.csv` 是 Stage 2 原始候选池输出
- `training_tables/*.csv` 是打完统一监督标签后的表

这样做的好处是：

- Stage 2 和 Stage 3 明确解耦
- 重跑 Stage 3 时不用重复构 candidate pool
- 三种 Stage 2 方法的中间结果能单独检查


## 6. A0: KNN + XGBoost 是怎么实现的

### 6.1 候选池构建函数

KNN 路线由：

- `_ensure_knn_tables(...)`

负责。

它内部会：

1. 初始化 `KNNContextPoolBuilder`
2. 用 train split 构建 KNN memory
3. 生成 `train.csv / val.csv / test.csv` 三份 candidate pool

其中：

- `train / val` 使用 gold split 里的路线
- `test` 使用 `route_cache.json` 里的 Stage 1 预测路线


### 6.2 KNN memory

KNN 的详细算法见：

- `stage2_KNN/stage2_KNN_detail.md`

这里简述一下最关键的点：

1. route 编码用 `reaction_morgan_fp(...)`
2. 只用 family 内 train split 建 memory
3. 相似度用归一化后的点积
4. 每个 query 取 top-k 邻居
5. 聚合出 `(reagent_norm, solvent_norm)` 候选

输出附带 4 个关键支持特征：

- `knn_similarity_sum`
- `knn_similarity_max`
- `knn_neighbor_count`
- `knn_weighted_mean_yield`


### 6.3 为什么它是 Stage 2 主线

KNN 的设计目标不是直接给最终答案，而是：

- 尽量把真正可行的条件放进候选池
- 同时给 Stage 3 提供可学习的支持信号

也就是说它既做筛选，也给后面的 reranker 提供结构化证据。


## 7. A4: Cluster + XGBoost 是怎么实现的

### 7.1 候选池构建函数

Cluster 路线由：

- `_ensure_cluster_tables(...)`

负责。

训练 / 验证 split 先通过：

- `ClusterContextPoolBuilder.build_table(...)`

构建。

Non-Oracle test 则是手动读取 `route_cache.json` 后，对每条预测路线调用：

- `builder._candidate_rows(record)`

生成候选池。


### 7.2 Cluster memory 怎么建

`ClusterContextPoolBuilder` 定义在：

- `baseline/run_oracle_baselines.py`

它的主要步骤是：

1. 用 train split 构建 route-level fingerprint memory
2. 如果维度过高，先做 `TruncatedSVD`
3. 再用 `MiniBatchKMeans` 给训练路线聚类
4. 在每个 cluster 内汇总高频 `(reagent_norm, solvent_norm)` 组合

每个 cluster 候选会带上：

- `cluster_id`
- `cluster_context_count`
- `cluster_context_support`
- `cluster_context_mean_yield`


### 7.3 Non-Oracle 查询怎么做

对一条 query 路线：

1. 编码成 reaction fingerprint
2. 如果训练时用过 SVD，就先做同样的投影
3. 用 `kmeans.predict(...)` 预测它属于哪个 cluster
4. 直接返回该 cluster 里最常见的若干条件组

所以它和 KNN 的本质差别是：

- KNN 是细粒度最近邻检索
- Cluster 是先粗聚类，再用 cluster 原型池


### 7.4 这条线为什么是消融而不是 baseline

因为它不是历史系统，也不是最终主线。

它的意义纯粹是：

- 看“粗粒度簇记忆”能不能替代“细粒度近邻检索”


## 8. A5: FNN-pool + XGBoost 是怎么实现的

### 8.1 候选池构建函数

FNN-pool 路线由：

- `_ensure_fnnpool_tables(...)`

负责。

它内部会分别为：

- train split
- val split
- Non-Oracle test

构建候选池。


### 8.2 它只用历史 FNN 的 candidate generation 部分

这里最重要的一点是：

- 它不会用历史 ranking head 做最终排序

而是只做：

1. `_load_legacy_evaluators(..., with_ranker=False)`
2. `mt.make_input_rxn_condition(rxn_fp)`

也就是只取历史 multitask FNN 提供的：

- solvent 候选集合
- reagent 候选集合


### 8.3 条件组合是怎么枚举的

得到候选 solvent / reagent 集合后，代码调用：

- `LegacyRankingEvaluator(...).make_contexts(input_solvents, input_reagents)`

这里只是借用了历史 ranking evaluator 的“枚举 context 组合”能力，并没有使用它的排序分数。

然后对 `(reagent_norm, solvent_norm)` 做：

- 规范化
- 去重
- 截断到 `legacy_max_contexts`

输出列里只有：

- `legacy_rank`

但没有 `legacy_score`

因为后面真正排序完全交给 XGBoost。


### 8.4 这条线要回答什么

它回答的是：

- 如果 Stage 2 候选池回退到历史 FNN 风格，单靠现代 XGBoost 重排，能不能恢复主线表现

如果不能，就说明：

- KNN 的价值不只是排序器强，而是 candidate screening 本身更合适


## 9. 统一打标过程

无论候选池来自哪种 Stage 2，都会统一调用：

- `baseline.common.label_candidate_table(...)`

把 raw candidate pool 转成有监督训练表。

统一生成的字段包括：

- `route_match`
- `context_match`
- `label`
- `label_type`
- `sample_weight`
- `rank_relevance`
- `temperature_gold`
- `yield_gold`
- `num_reagents`
- `num_solvents`
- `route_component_count`
- `reactants_length`
- `product_feat_*`

这一层很关键，因为它保证：

- 三种 Stage 2 方法下游看到的是同一种监督格式


## 10. Stage 3 为什么固定成 XGBoost

对 Stage 2 消融来说，Stage 3 必须固定住，否则结论会混掉。

所以统一调用：

- `train_xgb_ranker(train_table, val_table, output_dir/model)`
- `score_table_with_xgb(test_table, model_file, metadata_file)`

训练得到两类模型：

- `xgb_ranker.json`
- `xgb_temperature.json`

然后在 test table 上产生：

- `xgb_score`
- `xgb_temperature_pred`


## 11. 评估是怎么做的

最终三条 Stage 2 路线都用：

- `baseline.common.evaluate_scored_frame(...)`

统一评估。

### 11.1 `cover`

一个 slate 里是否至少存在 exact-positive candidate。

### 11.2 `sys@1 / sys@5 / sys@10`

按 `xgb_score` 排序，看 top-k 是否命中 `label=1`。

### 11.3 `Temp@10C / Temp@20C`

只在：

- `sys@10` 已命中
- 且该 positive row 有有效 `temperature_gold`
- 且有 `xgb_temperature_pred`

时统计。

这保证温度评估不会被未命中样本稀释。


## 12. 结果文件怎么落盘

每个 family / baseline 的目录结构大致是：

- `<output_root>/<family>/knn_xgb/non_oracle/`
- `<output_root>/<family>/cluster_xgb/non_oracle/`
- `<output_root>/<family>/fnnpool_xgb/non_oracle/`

每个目录里至少包含：

- `test_scored.csv`
- `result.json`
- `model/` 下的 XGBoost 文件

全家族聚合后会写成：

- `outputs/stage23_non_oracle_all10/ablation_stage2.md`
- `outputs/stage23_non_oracle_all10/results_flat.csv`
- `outputs/stage23_non_oracle_all10/all_results.json`
- `outputs/stage23_non_oracle_all10/average_effect.md`


## 13. 这份消融最终要支持什么结论

这组实验最终希望回答的是：

1. `KNN + XGBoost` 是否优于 `Cluster + XGBoost`
2. `KNN + XGBoost` 是否优于 `FNN-pool + XGBoost`
3. 差距是否同时体现在：
   - `cover`
   - `sys@k`
   - 温度命中率

如果答案是肯定的，那么就可以比较有说服力地说：

- KNN 的价值主要来自更合适的可行条件筛选
- 不是单靠后面的 reranker 在“救火”
