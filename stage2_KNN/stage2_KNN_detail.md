# Stage 2 KNN: 可行条件筛选

## 1. 模块目标

`stage2_KNN` 负责承接 Stage 1 给出的反应路线 `(reactants -> product)`，从家族内训练集历史反应中检索相似路线，并生成一个“可行条件候选池”。  
这个阶段只做 **筛选**，不做最终排序，目标是把真正命中的 `(reagent, solvent)` 尽量放进候选池里，供 Stage 3 再重排。

当前实现文件：

- `stage2_KNN/knn_condition_selector.py`
- `stage2_KNN/__init__.py`

核心类：

- `KNNContextPoolBuilder`


## 2. 输入与输出

输入有两种模式：

1. Oracle 模式  
   直接读取某个 family 的 `train/val/test` split 文件，构造候选池。

2. Non-Oracle 模式  
   直接读取 Stage 1 的 `route_cache.json`，对 Stage 1 预测出的路线生成候选池。

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

为了和已有表结构统一，Stage 2 KNN 还会补齐 cluster 占位列：

- `cluster_id = -1`
- `cluster_context_count = 0`
- `cluster_context_support = 0`
- `cluster_context_mean_yield = 0`


## 3. 路线编码方式

### 3.1 反应指纹

每条路线 `(reactants, product)` 用 `reaction_morgan_fp(...)` 编码。  
这个编码现在来自 `prosys_shared/features.py`，本质上是：

- 产品分子的 Morgan fingerprint
- 反应变化信息对应的 fingerprint
- 最终拼接成一个定长向量

默认超参：

- `fpsize = 4096`
- `radius = 2`

因此最终路线向量维度通常是 `2 * fpsize`。


### 3.2 相似度

所有指纹在进入 KNN 检索前先经过 `normalize_fp(...)` 做 L2 归一化。  
归一化后，相似度可直接用向量点积计算：

`similarity(query, memory_i) = normalized_fp(query) dot normalized_fp(memory_i)`

这等价于 cosine similarity，优点是实现简单、速度快，而且和后续矩阵乘法兼容。


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

### 5.1 单条 query 路线

对每条待筛选路线：

1. 用同样的方式编码成 reaction fingerprint
2. 与 `route_matrix` 做点积，得到和所有训练路线的相似度
3. 取 top-k 个近邻路线
4. 汇总这些近邻对应的历史条件
5. 对条件去重、打分、排序
6. 截断为 `max_contexts`

默认超参：

- `top_k = 20`
- `max_contexts = 20` 或 50，按脚本传参决定


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


### 5.3 排序规则

KNN 筛选阶段内部只做一个轻量排序，不做最终决策。  
候选按以下优先级排序：

1. `knn_similarity_sum` 降序
2. `knn_similarity_max` 降序
3. `knn_neighbor_count` 降序
4. `reagent_norm` 字典序
5. `solvent_norm` 字典序

排序后的前 `max_contexts` 个条件进入 candidate pool。


### 5.4 回退策略

以下情况会触发全局回退：

- train memory 为空
- query 和所有 train route 的相似度都接近 0
- top-k 邻居虽然存在，但没有汇聚出有效条件

这时返回 `global_contexts[:max_contexts]`，保证候选池永远可用。


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

KNN Stage 2 的逻辑可以概括成一句话：

“先按反应相似性找历史邻居，再把邻居真正用过的条件组合搬过来。”

这样做有几个优点：

- 不需要重新训练一个复杂神经筛选器
- family 内小样本时更稳
- 候选池语义直观，可解释性强
- 输出的 `knn_*` 特征可以直接送给 Stage 3 XGBoost 继续学习

也就是说，Stage 2 不只是在“拿邻居投票”，还在为 Stage 3 提供结构化支持信号。


## 8. 与主线的接口关系

当前主线里，Stage 2 KNN 的职责是：

1. 接收 Stage 1 路线
2. 生成 top-N 候选条件池
3. 把候选及其 KNN 支持特征写入 csv
4. 交给 `label_candidate_table(...)` 打标
5. 再交给 Stage 3 XGBoost 重排

所以它输出的是“候选池 + 支持特征”，不是最终答案。


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
- `--max_contexts`
- `--fpsize`
- `--radius`
- `--max_routes`


## 11. 预期效果

Stage 2 KNN 的预期不是直接把 `sys@k` 拉满，而是：

- 比全局高频条件更精准
- 比 cluster 分桶更细粒度
- 在不引入复杂 DL 筛选器的前提下，提供更高的候选覆盖率
- 为 Stage 3 XGBoost 提供强支持特征

因此它的主要观察指标应当是：

- `pool_coverage`
- `context_topk`
- `route_topk`
- 为 Stage 3 提供的最终 `sys@k` 上限

如果 Stage 2 候选池没把正确条件放进来，Stage 3 再强也救不回来。
