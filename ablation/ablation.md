# ProSys Ablation 实验规划

更新日期：`2026-07-05`

实现细节文档：

- `ablation/stage2_ablation_implementation_detail.md`
- `ablation/stage3_ablation_implementation_detail.md`
- `ablation/run_non_oracle_ablation.py`

## 1. 这轮 ablation 的定位

这轮 ablation 只围绕当前主线：

- `KNN + XGBoost`

来做模块替换分析。

目标不是证明“哪个历史方法更强”，而是回答两个更直接的问题：

1. `XGBoost` 为什么比别的经典 ML 排序器更适合放在 Stage 3
2. `KNN` 为什么比别的候选池方案更适合放在 Stage 2

因此这轮 ablation 的逻辑是：

- 固定一半模块
- 替换另一半模块
- 看 Non-Oracle end-to-end 指标怎么变


## 2. 只做 Non-Oracle

这轮 ablation 只做 **Non-Oracle end-to-end**。

统一设定：

- 固定 Stage 1 已有的 `route_cache.json`
- 只比较 Stage 1 之后的模块替换
- 所有结果都在同一个真实误差传播条件下汇报

这样做的好处是：

- 结论会直接对应最终使用场景
- 不需要在正文里来回切换不同设定


## 3. 主线参考组

## 3.1 A0：`KNN + XGBoost`

定义：

- Stage 2：`KNN` candidate pool
- Stage 3 排序：`XGBRanker`
- 温度：`XGBRegressor`

作用：

- 当前主线
- 也是所有 ablation 的参考上界


## 4. Stage 3 消融：固定 KNN，只换排序器

这一组实验只回答一件事：

- 在同样的 `KNN` 候选池上，`XGBoost` 是否真的比常见 ML 排序器更合适


## 4.1 A1：`KNN + RandomForest`

定义：

- Stage 2：固定 `KNN`
- Stage 3 排序：`RandomForestClassifier`
- 温度：`RandomForestRegressor`

预期：

- 会是一个合理的 classical baseline
- 但大概率弱于 `KNN + XGBoost`


## 4.2 A2：`KNN + SVM`

定义：

- Stage 2：固定 `KNN`
- Stage 3 排序：线性 `SVM`
- 温度：`LinearSVR`

说明：

- 这里优先使用线性版本
- 不建议核 `SVM`

原因：

- Non-Oracle candidate table 规模不小
- 核方法成本高且不稳定
- 线性版更适合当高效基线

预期：

- 会比 Bayes 稍强
- 但通常仍难超过 `XGBoost`


## 4.3 A3：`KNN + Bayes`

定义：

- Stage 2：固定 `KNN`
- Stage 3 排序：`GaussianNB`
- 温度：`BayesianRidge`

说明：

- 这里的“Bayes”是一个组合定义
- 排序和温度分别用最自然的贝叶斯系模型

预期：

- 是一个偏弱但有代表性的 lower baseline
- 主要用于证明极简概率模型在这个任务上不够强


## 5. Stage 2 消融：固定 XGBoost，只换候选池

这一组实验只回答一件事：

- 在同样的 `XGBoost` reranker 下，`KNN` 是否真的比其他候选池更适合做可行条件筛选


## 5.1 A4：`Cluster + XGBoost`

定义：

- Stage 2：`cluster` candidate pool
- Stage 3 排序：固定 `XGBRanker`
- 温度：固定 `XGBRegressor`

它回答的是：

- 如果把局部近邻筛选换成更粗粒度的 cluster memory，结果会怎样

预期：

- `sys@k` 会弱于 `KNN + XGBoost`
- 差距通常会体现在：
  - 候选池不够细粒度
  - `sys@10` 和整体覆盖更弱


## 5.2 A5：`FNN pool + XGBoost`

定义：

- Stage 2：原始 FNN candidate generation
- Stage 3 排序：固定 `XGBRanker`
- 温度：固定 `XGBRegressor`

这里要特别说明：

- 在 `baseline` 里，`FNN` 表示原始整条流水线
- 在 `ablation` 里，`FNN pool + XGBoost` 只表示“拿 FNN 当 Stage 2 候选池生成器”

这样做的作用是把问题拆干净：

- baseline 负责比较“整条旧系统 vs 整条新系统”
- ablation 负责比较“只换 Stage 2 候选池，会发生什么”

这组实验回答的是：

- 如果候选池换回 FNN 风格，单靠 `XGBoost` 能不能把结果救回来

预期：

- 会弱于 `KNN + XGBoost`
- 如果差距明显，就能说明 `KNN` 的主要价值在于 candidate screening


## 6. 为什么这版逻辑更好

这版组织方式确实更顺。

原因在于现在三层关系很清楚：

1. `baseline`
   - 只保留 `Original Prototype-FNN`
   - 只回答“当前主线相对原始项目提升多少”

2. `Stage 3 ablation`
   - `KNN + RF / SVM / Bayes / XGBoost`
   - 只回答“为什么主线选 XGBoost”

3. `Stage 2 ablation`
   - `KNN / Cluster / FNN pool + XGBoost`
   - 只回答“为什么主线选 KNN”

这样不会再出现：

- baseline 和 ablation 混在一起
- 历史方法和模块替换同时出现，导致解释混乱


## 7. 指标口径

这轮 ablation 的主指标固定成：

- `sys@1`
- `sys@5`
- `sys@10`
- `Temp@10C`
- `Temp@20C`

对于 Stage 2 消融，再保留一个辅助解释指标：

- `pool_coverage`

说明：

- 它不是最后最主要的 headline metric
- 但对解释 `KNN` 为什么更好非常有帮助


## 8. 温度口径

温度评估和 baseline 完全统一：

- 只在 `sys@10` 已命中的样本上统计
- 只看 top-10 中最高排名的 exact-positive system
- 汇总 `Temp@10C` 和 `Temp@20C`

这样表格才可以直接横向比较。


## 9. 推荐的最终表格组织

建议最终至少整理成两张 Non-Oracle 表。

### 表 A：Stage 3 消融

- `KNN + XGBoost`
- `KNN + RandomForest`
- `KNN + SVM`
- `KNN + Bayes`

这张表主要说明：

- 为什么主线 Stage 3 选 `XGBoost`


### 表 B：Stage 2 消融

- `KNN + XGBoost`
- `Cluster + XGBoost`
- `FNN pool + XGBoost`

这张表主要说明：

- 为什么主线 Stage 2 选 `KNN`

## 10. 当前结果位置

- `outputs/stage23_non_oracle_all10/ablation_stage3.md`
- `outputs/stage23_non_oracle_all10/ablation_stage2.md`
- `outputs/stage23_non_oracle_all10/average_effect.md`


## 10. 最终希望支持的结论

如果结果符合预期，这轮 ablation 最终应支持下面这条叙事：

- `XGBoost` 比 `RF / SVM / Bayes` 更适合在当前 candidate table 上做 reranking
- `KNN` 比 `Cluster / FNN pool` 更适合做可行条件筛选
- 因此当前主线 `KNN + XGBoost` 不是随意拼出来的，而是一个在 Non-Oracle end-to-end 设定下更合理的两阶段组合
