# ProSys Baseline 实验规划

更新日期：`2026-07-05`

实现细节文档：

- `baseline/baseline_implementation_detail.md`

## 1. 这轮 baseline 的定位

这轮 baseline 只保留一个对象：

- `Original Prototype-FNN`

也就是：

- 原项目最初版本的 FNN candidate generation
- 原项目最初的 ranking / temperature 头
- 不引入当前 `stage2_detail.md` 里的后续改造
- 不引入 `KNN`
- 不引入 `XGBoost`

这样定义的原因很直接：

- baseline 的任务不是做很多方法横向比较
- baseline 的任务是提供一个最硬的历史参照

所以这份文档只负责回答：

- 当前主线 `KNN + XGBoost` 相比原始项目，到底提升了多少


## 2. 只做 Non-Oracle

这轮 baseline 只做 **Non-Oracle end-to-end**。

也就是说：

- 固定使用 Stage 1 真实预测出来的 `route_cache.json`
- 所有方法都在同样的 Stage 1 误差条件下比较
- 不再把主要叙事建立在上限分析设定上

这样更符合最后真正要汇报的故事：

- 不是“如果路线已知会怎样”
- 而是“真实 end-to-end 跑起来会怎样”


## 3. baseline 方法定义

## 3.1 B0：`Original Prototype-FNN`

定义：

- Stage 1：固定当前已有 `route_cache.json`
- Stage 2：原始 FNN candidate generation
- Stage 3：原始 ranking / temperature 头

这里的关键点是：

- 它保留的是原始项目的整套后半段逻辑
- 不把它拆成现代模块组合
- 不用来做局部部件替换分析

所以它是一个真正的 **historical end-to-end baseline**。


## 4. baseline 为什么只保留 FNN

这样反而更清楚。

原因有三点：

1. `RF / SVM / Bayes` 不是历史方法
2. `Cluster / FNN pool + XGBoost` 本质上属于主线模块替换，更适合放到 ablation
3. 如果 baseline 塞太多方法，最后会和 ablation 的逻辑重叠

收缩之后，分工会很清楚：

- `baseline` 只负责历史参照
- `ablation` 只负责解释主线为什么这样设计


## 5. baseline 评估口径

这轮 baseline 的主指标现在统一成：

- `sys@1`
- `sys@5`
- `sys@10`
- `Temp MAE`
- `Temp±5C`
- `Temp±10C`
- `Temp±20C`

辅助解释指标：

- `cover`
- `rr@10`

其中：

- `cover` 用来解释候选池是否把正例放进来了
- `rr@10` 只作为 Stage 1 背景，不作为 baseline headline metric


## 6. 温度指标的统一定义

为了让 baseline 和主线可比，温度指标统一成：

- 对每个样本，找到最高排名的 exact-positive system
- 要求这个 system 有有效 `temperature_gold` 和预测温度
- 在这些样本上单独统计 `Temp MAE`
- 同时统计温度误差是否落在 `±5℃ / ±10℃ / ±20℃`

这样做的好处是：

- 保留了温度回归本身的解释性
- 不会和 `sys@k` 的 end-to-end 排序命中混在一起
- 口径能和当前主线、ablation 保持一致


## 7. 最终 baseline 结果应该怎么用

这份 baseline 不单独讲一个大故事，它主要服务于最后的主对比表。

最核心的用法是：

- 把 `Original Prototype-FNN`
- 和主线 `KNN + XGBoost`

放在同一张 Non-Oracle end-to-end 表里做比较。

当前结果根目录：

- `outputs/stage23_non_oracle_all10/baseline_historical.md`
- `outputs/stage23_non_oracle_all10/average_effect.md`

要回答的问题只有一个：

- 当前主线相对原始项目到底有没有清晰而稳定的收益


## 8. 最终希望支持的结论

如果结果符合预期，这份 baseline 最终应支持下面这条叙事：

- `Original Prototype-FNN` 是必要的历史参照
- 但当前主线 `KNN + XGBoost` 在真实 Non-Oracle end-to-end 设定下更强
- 因此主线改造不是“换汤不换药”，而是确实带来了系统级收益
