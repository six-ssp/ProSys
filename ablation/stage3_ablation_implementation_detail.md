# Legacy Stage 3 Detail (Superseded)

> This document describes retired RF/SVM/Bayes comparisons. It is not used by the current
> ReaFNN + Reaction-GNN mainline. See [`ablation_reafnn_gnn_protocol.md`](ablation_reafnn_gnn_protocol.md)
> and `run_current_mainline_ablation.py` for the maintained Stage 3 study.

# Stage 3 Ablation: 具体实现说明

更新日期：`2026-07-05`

## 1. 这份文档解释什么

这份文档只解释 Stage 3 消融：

- `KNN + XGBoost`
- `KNN + RandomForest`
- `KNN + SVM`
- `KNN + Bayes`

这里固定的是 Stage 2：

- 候选池统一来自 `KNN`

变化的只有 Stage 3 排序器和温度回归器。

因此它回答的问题是：

- 在同一个 KNN candidate pool 上，为什么主线最终选择 `XGBoost`


## 2. 相关代码文件

Stage 3 消融主要依赖：

- `ablation/run_non_oracle_ablation.py`
  - ablation 的独立运行入口
- `baseline/run_non_oracle_stage23_experiments.py`
  - 统一执行 Stage 2 / Stage 3 ablation
- `baseline/tabular_models.py`
  - `RF / SVM / Bayes` 的排序与温度模型实现
- `stage3_XGBoost/xgb_reranker.py`
  - 主线 `XGBRanker + XGBRegressor`
- `stage2_KNN/knn_condition_selector.py`
  - 固定候选池来源
- `baseline/common.py`
  - 打标与统一评估


## 3. 运行入口

推荐命令：

```bash
conda activate ProSys
python ablation/run_non_oracle_ablation.py \
  --repo_root . \
  --families all \
  --output_root outputs/stage23_non_oracle_all10 \
  --run_set stage3_ablation
```

或者直接调用统一脚本：

```bash
python baseline/run_non_oracle_stage23_experiments.py \
  --repo_root . \
  --families all \
  --output_root outputs/stage23_non_oracle_all10 \
  --run_set stage3_ablation
```


## 4. Stage 3 消融的统一原则

为了避免混淆，这组实验固定三件事。

### 4.1 固定 Stage 1

都用同一个：

- `route_cache.json`

### 4.2 固定 Stage 2

都用同一个 KNN 候选池：

- `<output_root>/<family>/_shared_knn/candidate_pool/*.csv`
- `<output_root>/<family>/_shared_knn/training_tables/*.csv`

### 4.3 只替换 Stage 3 学习器

也就是说：

- 候选集完全相同
- label 完全相同
- 特征列完全从同一张表推断

因此谁更强，反映的就是 reranker 本身的能力。


## 5. 共享 KNN 候选池怎么来

这组实验的基础是：

- `_ensure_knn_tables(...)`

它先构建一套共享 KNN 表：

- `candidate_pool/train.csv`
- `candidate_pool/val.csv`
- `candidate_pool/test.csv`
- `training_tables/train.csv`
- `training_tables/val.csv`
- `training_tables/test.csv`

其中 `test.csv` 是 Non-Oracle：

- 路线来自 Stage 1 `route_cache.json`

而 `train/val` 用于训练 Stage 3 模型。


## 6. 训练表长什么样

所有 Stage 3 模型都吃同一类 training table。

这些表由：

- `baseline.common.label_candidate_table(...)`

生成。

里面关键字段包括：

- 路线信息
  - `sample_index`
  - `reaction_id`
  - `reactants`
  - `product`
  - `retro_rank`
  - `retro_score`
  - `retro_probability`
- Stage 2 支持特征
  - `knn_similarity_sum`
  - `knn_similarity_max`
  - `knn_neighbor_count`
  - `knn_weighted_mean_yield`
- 结构描述符
  - `product_feat_*`
- 监督标签
  - `route_match`
  - `context_match`
  - `label`
  - `label_type`
  - `rank_relevance`
  - `temperature_gold`


## 7. A0: KNN + XGBoost 是怎么实现的

### 7.1 训练函数

主线调用：

- `baseline.common.train_xgb_ranker(...)`

它内部再调用：

- `stage3_XGBoost.train_xgb_ranker_and_temperature(...)`


### 7.2 排序目标

XGBoost 不是做单行二分类，而是做分组排序。

具体设定：

- group = `sample_index`
- target = `rank_relevance`
- objective = `rank:ndcg`
- eval_metric = `ndcg@10`

其中：

- `positive = 3`
- `route_only = 2`
- `context_only = 1`
- `negative = 0`

这样模型会更关心：

- top-10 里谁应该被排前面


### 7.3 温度模型

XGBoost 温度头使用：

- `XGBRegressor`

训练样本只保留：

- `label = 1`
- `temperature_gold` 有效


## 8. A1: KNN + RandomForest 是怎么实现的

### 8.1 训练函数

RandomForest 路线调用：

- `baseline.tabular_models.train_tabular_ranker_and_temperature(..., kind='rf')`


### 8.2 排序模型

排序器是：

- `RandomForestClassifier`

默认参数包括：

- `n_estimators = 400`
- `class_weight = balanced_subsample`
- `n_jobs = -1`

训练目标是：

- 二分类 `label`

不是 listwise ranking。


### 8.3 打分方式

推理时使用：

- `predict_proba(x)[:, 1]`

作为排序分数，写入：

- `model_score`


### 8.4 温度模型

温度回归器是：

- `RandomForestRegressor`

训练样本同样只保留 exact-positive rows。


## 9. A2: KNN + SVM 是怎么实现的

### 9.1 排序模型

SVM 路线的排序器是：

- `Pipeline(StandardScaler -> LinearSVC)`

具体原因是：

- 线性模型在这类高维稠密特征上更稳
- 比核 SVM 更容易在当前数据规模下运行


### 9.2 训练目标

和 RF 一样，仍然是：

- 二分类 `label`

不是 group ranking。


### 9.3 打分方式

排序分数来自：

- `decision_function(x)`

也就是超平面距离，而不是概率。


### 9.4 温度模型

温度回归器是：

- `Pipeline(StandardScaler -> LinearSVR)`


## 10. A3: KNN + Bayes 是怎么实现的

### 10.1 排序模型

Bayes 路线使用：

- `GaussianNB`

训练目标仍然是：

- `label`


### 10.2 打分方式

如果模型支持 `predict_proba`，就使用：

- 正类概率 `proba[:, 1]`

作为排序分数。


### 10.3 温度模型

温度回归器使用：

- `Pipeline(StandardScaler -> BayesianRidge)`

所以这里的 “Bayes” 实际上是一个组合：

- 分类用 `GaussianNB`
- 回归用 `BayesianRidge`


## 11. 这三类 classical ML 模型为什么共用一个框架

`baseline/tabular_models.py` 把三类 classical 模型统一成了同一个接口。

统一点包括：

### 11.1 特征列推断

所有模型都先调用：

- `stage3_XGBoost.infer_xgb_feature_columns(...)`

也就是说：

- XGBoost 和 classical baseline 看到的是同一组数值特征

这样比较才公平。


### 11.2 缺失值处理

所有缺失数值列都统一补成：

- `0.0`

### 11.3 温度样本筛选

所有模型都只用：

- positive 且有温度标注

的样本训练温度头。

### 11.4 输出接口

所有模型最后都输出：

- `model_score`
- `model_temperature_pred`

这样统一评估函数就不需要为每个模型单独写一套逻辑。


## 12. 统一评估怎么做

所有 Stage 3 路线最终都走：

- `baseline.common.evaluate_scored_frame(...)`

### 12.1 排序评估

先按分数列排序：

- XGBoost 用 `xgb_score`
- RF / SVM / Bayes 用 `model_score`

然后在每个 `sample_index` 内统计：

- `sys@1`
- `sys@5`
- `sys@10`


### 12.2 候选覆盖

虽然这组实验固定 KNN 候选池，但仍然会保留：

- `pool_coverage`

主要是为了检查：

- 是否因为某些 family 的 KNN pool 本身没有正例，导致所有 Stage 3 模型都无能为力


### 12.3 温度指标

温度仍然只在：

- top-10 内存在 exact-positive system

的样本上统计。

最终汇报：

- `Temp@10C`
- `Temp@20C`


## 13. 输出文件怎么落盘

每个 family 对应四个 Stage 3 目录：

- `<output_root>/<family>/knn_xgb/non_oracle/`
- `<output_root>/<family>/knn_rf/non_oracle/`
- `<output_root>/<family>/knn_svm/non_oracle/`
- `<output_root>/<family>/knn_bayes/non_oracle/`

每个目录下典型文件有：

- `test_scored.csv`
- `result.json`
- `model/`

其中：

- XGBoost 模型保存成 `json + meta.json`
- RF / SVM / Bayes 模型保存成 `pkl + meta.json`

全家族聚合后写成：

- `outputs/stage23_non_oracle_all10/ablation_stage3.md`
- `outputs/stage23_non_oracle_all10/results_flat.csv`
- `outputs/stage23_non_oracle_all10/all_results.json`


## 14. 这组消融最终要支持什么结论

Stage 3 消融的核心问题只有一个：

- 在同样的 KNN candidate pool 上，XGBoost 是否比 RF / SVM / Bayes 更会排

如果最后看到：

1. `KNN + XGBoost` 的 `sys@k` 更高
2. 温度命中率也更稳
3. 差距是跨 family 比较稳定的

那么就可以比较有说服力地说：

- 当前主线把 XGBoost 放在 Stage 3，不是随便选的
- 它确实更适合当前 candidate table 的 reranking 任务
