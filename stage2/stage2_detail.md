# ProSys Stage 2 改造实施说明

## 1. 文档范围

这份文档只写四件事：

1. Stage 2 新框架是什么
2. 每一层输入输出是什么
3. 代码具体要怎么改
4. 按什么顺序实施

这里默认：

- Stage 1（EditRetro）保持不变
- Stage 2 仍然按 family-specific 训练和评估

---

## 2. Stage 2 新框架

Stage 2 改成两层主结构：

```text
Stage 2A: 候选池生成
  = route-conditioned FNN pool
  ∪ product memory pool

Stage 2B: 神经排序 + 温度预测
  输入: product + route + context + support/source features
  输出: candidate score + temperature
```

完整链路如下：

```text
目标产物
-> Stage 1 EditRetro routes
-> Stage 2A 候选 context 池
-> Stage 2B 神经模型打分
-> 输出 Top-k route-context systems
```

这里的 `family-specific` 只表示：

- 每个 family 单独训练 Stage 2 模型
- 每个 family 使用自己的 train / validate / test 数据

---

## 3. Stage 2A：候选池生成

### 3.1 目标

给定一个 `(product, route)`，生成一个尽量覆盖真实条件的 candidate context pool。

这里的 `context` 指：

- `reagent set`
- `solvent set`

---

### 3.2 输入

Stage 2A 的输入是：

1. `product`
2. `reactants`
3. `family`
4. Stage 1 route metadata
   - `retro_rank`
   - `retro_score`
   - `retro_probability`

---

### 3.3 输出

Stage 2A 的输出是一个 candidate table，每一行表示一个候选 context。

建议输出字段：

- `family`
- `sample_index`
- `reaction_id`
- `product`
- `reactants`
- `retro_rank`
- `retro_score`
- `retro_probability`
- `reagent_norm`
- `solvent_norm`
- `from_fnn`
- `from_product_exact`
- `from_product_scaffold`
- `from_product_knn`
- `product_exact_pair_support`
- `product_exact_reagent_support`
- `product_exact_solvent_support`
- `product_scaffold_pair_support`
- `product_scaffold_reagent_support`
- `product_scaffold_solvent_support`
- `product_knn_pair_support`
- `product_knn_reagent_support`
- `product_knn_solvent_support`
- `product_pair_freq`
- `product_pair_mean_yield`

落地时建议按 `product` 或 `sample_index` 分块保存，避免单表过大。

---

### 3.4 候选池来源

Stage 2A 的候选来自两路并集。

### 路径 A：FNN 候选

输入：

- `reactants + product`

输出：

- FNN 预测得到的 `reagent set / solvent set`

保留现有流程：

- 预测 reagent 标签
- 预测 solvent 标签
- 按阈值筛选
- 枚举组合

### 路径 B：product memory 候选

从训练集里按 product 检索历史 context。

三种来源：

1. exact product
2. product scaffold
3. product Morgan KNN

输出是整套 context：

- `reagent_norm`
- `solvent_norm`

### 3.5 候选池合并规则

两路候选合并后，按下面规则处理：

1. 用 `(reagent_norm, solvent_norm)` 去重
2. 保留每个候选的来源标记
3. 保留每个候选的支持特征
4. 输出为一个统一 candidate table

---

## 4. product memory

### 4.1 目标

为 Stage 2A 和 Stage 2B 提供 product-aware 先验。

---

### 4.2 输入

输入数据是：

- 某个 family 的 Stage 2 train split
- 每条记录至少包含 `product`、`reagent set`、`solvent set`、`yield`、`temperature`

---

### 4.3 输出

建议新建一个 `product memory` 产物先验模块，并生成三类 artifact：

1. exact product memory 表
2. scaffold memory 表
3. product KNN 索引

---

### 4.4 各文件输入输出

### exact product memory 表

输入：

- train split 中所有 `(product, reagent, solvent, yield, temperature)`

输出字段：

- `product_canonical`
- `reagent_norm`
- `solvent_norm`
- `count`
- `mean_yield`
- `temperature_mean`
- `temperature_std`

### scaffold memory 表

输入：

- train split 中所有产物

输出字段：

- `product_scaffold`
- `reagent_norm`
- `solvent_norm`
- `count`
- `mean_yield`

### product KNN 索引

输入：

- train split 中所有 product

输出内容：

- product Morgan FP matrix
- product list
- product 到 context 的索引映射

### 4.5 product memory 的构建方式

product memory 不是单一 KNN，而是三层：

1. exact product
2. scaffold
3. product Morgan KNN

#### exact product

做法：

1. 对 train split 中每条记录取 `product`
2. 对 `product` canonicalize
3. 聚合同一个 `product` 下出现过的 `(reagent_norm, solvent_norm)`
4. 统计：
   - `count`
   - `mean_yield`
   - `temperature_mean`
   - `temperature_std`

输出：

- exact product memory 表

用途：

- 如果测试样本的 product 在训练集里 exact 命中，则直接提供 exact product context 候选
- 同时为每个候选提供 exact-support 特征

#### scaffold

做法：

1. 对 train split 中每个 product 提取 scaffold
2. 将同 scaffold 的 product 聚合
3. 统计 scaffold 下出现过的 `(reagent_norm, solvent_norm)` 频次和平均 yield

输出：

- scaffold memory 表

用途：

- exact product 没有命中时，提供 scaffold 级候选
- 同时提供 scaffold-support 特征

#### product Morgan KNN

做法：

1. 对 train split 中每个 product 计算 Morgan FP
2. 将所有 train product 的 Morgan FP 存成一个索引矩阵
3. 对测试 product 计算 Morgan FP
4. 在 train product 里找 top-k 相似邻居
5. 将这些邻居历史出现过的 `(reagent_norm, solvent_norm)` 聚合成候选

输出：

- product KNN 索引

用途：

- exact 和 scaffold 都不够时，提供近邻 product 的 context 候选
- 同时提供 knn-support 特征

### 4.6 product memory 支持分数怎么计算

对一个候选 `c = (reagent, solvent)`，定义三类支持：

#### exact product 支持

如果 exact product 命中：

```text
product_exact_pair_support(c)
= count(product = P, context = c) / count(product = P, any context)
```

单边支持：

```text
product_exact_reagent_support(r)
= count(product = P, reagent contains r) / count(product = P, any context)

product_exact_solvent_support(s)
= count(product = P, solvent contains s) / count(product = P, any context)
```

#### scaffold 支持

如果 scaffold 命中：

```text
product_scaffold_pair_support(c)
= count(scaffold = Scaf(P), context = c) / count(scaffold = Scaf(P), any context)
```

单边支持同理：

- `product_scaffold_reagent_support`
- `product_scaffold_solvent_support`

#### KNN 支持

如果用 product Morgan KNN：

设测试 product 为 `P`，找到邻居 `N1...Nk`，相似度为 `sim_i`。

则：

```text
product_knn_pair_support(c)
= Σ_i sim_i * 1[c in contexts(Ni)] / Σ_i sim_i
```

单边支持同理：

```text
product_knn_reagent_support(r)
= Σ_i sim_i * 1[r in reagent_set(Ni)] / Σ_i sim_i

product_knn_solvent_support(s)
= Σ_i sim_i * 1[s in solvent_set(Ni)] / Σ_i sim_i
```

#### 频次和产率特征

此外还保留两个简单统计：

- `product_pair_freq`
  - exact product 下该 pair 的出现次数
- `product_pair_mean_yield`
  - exact product 下该 pair 的平均 yield

如果 exact product 没命中，这两个值默认记为 `0`。

## 5. Stage 2B：纯神经排序 + 温度预测

### 5.1 目标

给定 `(product, route, context)` 候选，直接输出：

1. `candidate score`
2. `temperature`

不依赖后续 XGBoost。

---

### 5.2 输入

Stage 2B 的单个样本单位是：

```text
(product, route, context)
```

每个样本建议包含四类输入。

### A. route 输入

- reaction Morgan FP
- reaction graph descriptors（可选）
- `retro_rank`
- `retro_score`
- `retro_probability`

#### reaction Morgan FP 的提取方式

反应主特征建议沿用旧版口径：

- 用 reactants 和 product 分别做 Morgan 指纹
- 再做 `product - reactants` 的差分
- 最后拼成 reaction fingerprint

提取步骤：

1. 对 `reactants` 计算 Morgan FP，记为 `rfp`
2. 对 `product` 计算 Morgan FP，记为 `pfp`
3. 计算差分：

```text
delta_fp = pfp - rfp
```

4. 最终拼接：

```text
reaction_fp = concat(pfp, delta_fp)
```

默认参数：

- `radius = 2`

如果要和旧版结果对齐，这里也要区分两层口径：

- 旧版底层特征函数常见默认长度是 `16384`
- 但旧版训练配置真正常用的是 `4096`

因此按当前训练代码默认配置，实际常用的是：

- `fpsize = 4096`
- `radius = 2`

因此默认维度为：

```text
reaction_fp dimension = 4096 + 4096 = 8192
```

也就是说，当前 Stage 2 实际喂给模型的是：

```text
route input
= [product Morgan 4096 bits ; (product-reactant) delta 4096 bits]
```

#### reaction graph descriptors 的提取方式

轻量图特征建议保留为一个独立辅助分支。

步骤：

1. 提取 reactant 8 维分子图特征
2. 提取 product 8 维分子图特征
3. 计算 8 维差分
4. 拼成：

```text
rxn_graph_features = concat(reactant_feat, product_feat, product_feat - reactant_feat)
```

总维度：

```text
24
```

这 8 维单分子特征包括：

- 原子数
- 键数
- 环数
- 芳香原子数
- 杂原子数
- 精确分子量
- TPSA
- fraction sp3

### B. product 输入

- product Morgan FP
- product graph descriptors

#### product Morgan FP 的提取方式

建议新增：

- `product_morgan_fp(product, n_bits=2048, radius=2)`

提取步骤：

1. 对 `product` canonicalize
2. 用 RDKit 生成 Morgan fingerprint
3. 转成定长 bit / float 向量

用途：

- 作为 ProductEncoder 输入
- 作为 product KNN 检索索引

这里的 KNN 是：

- 只看 `product` 分子本身
- 不看 `reactants`
- 不看整条 `reaction_fp`

也就是说，product memory 的近邻检索不是“反应近邻”，而是“产物近邻”。

#### product graph descriptors

建议同时保留一组轻量 product descriptors。

当前维度：

```text
8
```

### C. context 输入

- reagent multi-hot 或 reagent token ids
- solvent multi-hot 或 solvent token ids

### D. support / source 输入

- `from_fnn`
- `from_product_exact`
- `from_product_scaffold`
- `from_product_knn`
- `product_exact_pair_support`
- `product_exact_reagent_support`
- `product_exact_solvent_support`
- `product_scaffold_pair_support`
- `product_scaffold_reagent_support`
- `product_scaffold_solvent_support`
- `product_knn_pair_support`
- `product_knn_reagent_support`
- `product_knn_solvent_support`
- `product_pair_freq`
- `product_pair_mean_yield`

#### support/source 特征的来源

这些特征全部来自 Stage 2A 生成的 candidate table。

其中：

- `from_fnn`
  - 该候选是否来自 route-conditioned FNN
- `from_product_exact`
  - 该候选是否来自 exact product memory
- `from_product_scaffold`
  - 该候选是否来自 scaffold memory
- `from_product_knn`
  - 该候选是否来自 product KNN memory

支持分数：

- 来自第 4 节定义的 exact / scaffold / knn 支持计算

统计特征：

- `product_pair_freq`
  - exact product 下该候选的出现次数
- `product_pair_mean_yield`
  - exact product 下该候选的平均 yield

---

### 5.3 输出

Stage 2B 每个 candidate 输出：

- `score_logit`
- `score_prob`
- `temperature_pred`

最终按 `score_logit` 或 `score_prob` 排序，取 top-k。

---

### 5.4 网络结构

建议新建一个 Stage 2 多头模型，用于同时做排序和温度预测。

结构如下：

```text
RouteEncoder
ProductEncoder
ContextEncoder
SupportEncoder
-> FusionBlock
-> RankingTrunk -> RankingHead
-> TemperatureTrunk -> TemperatureHead
```

---

### 5.5 各编码器输入输出

### RouteEncoder

输入：

- reaction FP
- 可选 reaction graph features
- route dense features

输出：

- `h_route`

### ProductEncoder

输入：

- product FP
- product graph features

输出：

- `h_product`

### ContextEncoder

输入：

- reagent 表示
- solvent 表示

输出：

- `h_context`

第一版推荐两种实现方式：

1. 最小改动版
   - reagent multi-hot
   - solvent multi-hot
   - 两个线性层编码
2. 升级版
   - reagent token ids -> embedding -> pooling
   - solvent token ids -> embedding -> pooling

### SupportEncoder

输入：

- 所有 dense support/source features

输出：

- `h_support`

建议 SupportEncoder 的输入顺序固定为：

```text
[
  from_fnn,
  from_product_exact,
  from_product_scaffold,
  from_product_knn,
  product_exact_pair_support,
  product_exact_reagent_support,
  product_exact_solvent_support,
  product_scaffold_pair_support,
  product_scaffold_reagent_support,
  product_scaffold_solvent_support,
  product_knn_pair_support,
  product_knn_reagent_support,
  product_knn_solvent_support,
  product_pair_freq,
  product_pair_mean_yield
]
```

### FusionBlock

输入：

- `h_route`
- `h_product`
- `h_context`
- `h_support`

建议额外构造交互项：

- `h_route * h_context`
- `h_product * h_context`
- `|h_route - h_context|`
- `|h_product - h_context|`

输出：

- `h_fused`

### RankingTrunk / RankingHead

输入：

- `h_fused`

输出：

- `score_logit`

### TemperatureTrunk / TemperatureHead

输入：

- `h_fused`

输出：

- `temperature_pred`

---

## 6. 训练样本表格式

### 6.1 样本单位

训练表中的每一行表示一个 candidate：

```text
(product, route, context)
```

建议输出三份拆分后的 candidate table：

- train split
- validate split
- test split

---

### 6.2 建议字段

### 基础标识

- `family`
- `sample_index`
- `reaction_id`
- `product`
- `reactants`

### route 字段

- `retro_rank`
- `retro_score`
- `retro_probability`

### context 字段

- `reagent_norm`
- `solvent_norm`
- `num_reagents`
- `num_solvents`

### support/source 字段

- `from_fnn`
- `from_product_exact`
- `from_product_scaffold`
- `from_product_knn`
- `product_exact_pair_support`
- `product_exact_reagent_support`
- `product_exact_solvent_support`
- `product_scaffold_pair_support`
- `product_scaffold_reagent_support`
- `product_scaffold_solvent_support`
- `product_knn_pair_support`
- `product_knn_reagent_support`
- `product_knn_solvent_support`
- `product_pair_freq`
- `product_pair_mean_yield`

### product / route 结构字段

- `route_component_count`
- `reactants_length`
- `product_feat_0` 到 `product_feat_7`

### 标签字段

- `label`
- `label_type`
- `sample_weight`
- `route_match`
- `context_match`

### 温度字段

- `temperature_gold`

### 6.3 一条原始记录如何展开成训练样本

这里给一个明确的数据展开流程。

假设训练集里某条原始记录是：

```text
product = P
gold route = R*
gold context = C* = (reagent*, solvent*)
temperature = T*
```

同时，Stage 1 对这个 `product = P` 给出了 top-10 路径：

```text
R1, R2, ..., R10
```

其中只有一个可能和 gold route 对应，也可能一个都不完全对应。

然后对每个 `Ri`：

1. 用 `reactants(Ri) + product(P)` 进入 FNN，生成一批 route-conditioned contexts
2. 再用 `product(P)` 去 exact/scaffold/KNN memory 里补 context
3. 两路并集、去重，得到该路径下的候选池：

```text
Ci1, Ci2, ..., Cin_i
```

于是训练表不是一条记录对应一行，而是展开成很多行：

```text
(P, R1, C11)
(P, R1, C12)
...
(P, R1, C1n1)
(P, R2, C21)
...
(P, R10, C10n10)
```

每一行都要补上标签：

- `route_match`
  - `Ri` 是否和 gold route 对应
- `context_match`
  - `Cij` 是否和 gold context 对应
- `label`
  - 只在 `route_match = 1 且 context_match = 1` 时记为 1
- `temperature_gold`
  - 只对正样本填 `T*`
  - 其余行记为 `NaN` 或训练时 mask

因此同一个 `product` 的一整组训练候选，结构上是：

```text
一个 sample_index
-> 多条 route
-> 每条 route 下多条 context
-> 合并成一个大 slate
```

这个展开方式的作用是：

1. 让模型直接面对“同产物下多个错误路径 + 多个错误条件”的真实推理形态
2. 让 `route_match=1, context_match=0` 这类 hard negative 明确进入训练
3. 让温度头只在真正命中的 route-context 上回归，不被错误路径污染

### 6.4 旧版 Stage 2 的训练样本是怎么做的

当前旧版代码不是按 `(product, route, context)` 展开，而是按：

```text
(reaction, observed context list)
```

来做一个 listwise slate。

在旧版实现里，这一步通常是一个“反应 -> context slate”的数据集构造过程。

旧版逻辑是：

1. 一条反应记录自带若干历史观测 context
2. 把这些 context 作为同一个 slate
3. relevance 由 `Yield2Relevance(yield)` 给出
4. 再用 cutoff augmentation 补一些 fake contexts

也就是说，旧版的负样本主要是“错误条件”，而不是“错误路径 + 错误条件”的联合空间。

V2 要改成上面的 route-context 展开表，原因就是现在真正困难的部分已经变成：

- 同 family 内，多条 EditRetro 路径都很像
- 真正需要排开的，是“哪条路径配哪套条件”

---

## 7. loss 设计

### 7.1 总 loss

建议总 loss 写成：

```text
L_total
= λ_rank * L_rank
+ λ_temp * L_temp
```

其中：

```text
L_rank
= λ_listmle * L_listMLE
+ λ_bce * L_bce
+ λ_margin * L_margin
```

---

### 7.2 排序 loss

### `L_listMLE`

作用：

- 约束一个 slate 内的整体排序顺序

输入：

- candidate logits
- relevance labels

输出：

- listwise ranking loss

#### `L_listMLE` 的计算单元

计算单位不是整批所有候选混在一起，而是：

- 同一个 `sample_index`
- 同一个 product 的全部 route-context candidates

也就是：

```text
一个样本的所有候选共同组成一个 slate
```

在这个 slate 内做 listwise 排序。

### `L_bce`

作用：

- 对每个 candidate 做正负分类

输入：

- candidate logits
- `label`
- `sample_weight`

输出：

- binary classification loss

建议实现：

- `BCEWithLogitsLoss(reduction='none')`
- 再乘 `sample_weight`

#### `L_bce` 的标签

建议：

```text
label = 1  当且仅当 route_match = 1 且 context_match = 1
label = 0  其余情况
```

也就是只有真正的 gold triple 记为正样本。

### `L_margin`

作用：

- 拉开正样本和 hard negative 的分数间隔

形式：

```text
max(0, margin - s_pos + s_neg)
```

输入：

- 正样本分数
- 负样本分数
- margin

输出：

- pairwise margin loss

#### `L_margin` 的正负样本构造

建议每个样本内做：

1. 取所有 `label = 1` 的正样本
2. 取 hardest negatives
3. 对每个正样本和若干 hard negatives 计算 margin loss

hard negatives 优先级：

1. `route_match = 1, context_match = 0`
2. `route_match = 0, context_match = 1`
3. `route_match = 0, context_match = 0`

---

### 7.3 温度 loss

建议：

- `SmoothL1Loss`

输入：

- `temperature_pred`
- `temperature_gold`

输出：

- temperature regression loss

训练规则：

- 第一版只在 `label = 1` 的正样本上计算温度 loss

更具体地说：

```text
L_temp = SmoothL1(temperature_pred, temperature_gold)
```

只对：

- `route_match = 1`
- `context_match = 1`

的样本求值，其余样本温度 loss mask 掉。

可选预处理：

- family 内 z-score 标准化温度
- 推理时反标准化

---

### 7.4 初始权重建议

建议第一版固定权重，不使用自动不确定性加权。

推荐初始值：

- `λ_rank = 1.0`
- `λ_temp = 0.2`
- `λ_listmle = 1.0`
- `λ_bce = 0.5`
- `λ_margin = 0.5`

因此第一版总 loss 可以直接写成：

```text
L_rank = 1.0 * L_listMLE + 0.5 * L_bce + 0.5 * L_margin
L_total = 1.0 * L_rank + 0.2 * L_temp
```

### 7.5 当前旧版神经排序 loss 是怎么做的

为了后面改代码时不混淆，这里把旧版口径单独写清楚。

旧版第二阶段是一个“listwise ranking + temperature regression”的联合训练入口。

旧版排序 loss：

- `listNet_top_one`

旧版 relevance 标签：

- `Yield2Relevance(yield)`

旧版具体做法：

1. 一个 batch 里的 ranking 数据来自旧版的 context-slate 数据集
2. 每个样本是一个 context slate
3. gold context 的 relevance 由 yield 映射得到
4. fake context 的 relevance 置为 0
5. 用 `listNet_top_one(preds_rank, targets)` 做排序训练

旧版温度 loss：

- `MSELoss`
- 来自单独的温度回归数据集
- 只用真实存在温度标签的 gold context 训练

旧版多任务合并方式：

```text
L_total_old
= exp(-log_var_rank) * L_rank
+ exp(-log_var_temp) * L_temp
+ 2 * (log_var_rank + log_var_temp)
```

也就是用 homoscedastic uncertainty 做动态加权。

### 7.6 V2 为什么不用旧版那套 loss 直接平移

这里不是分析优劣，只写实现上的原因。

旧版 loss 直接平移会有三个结构性不匹配：

1. 旧版的 slate 只比较同一路径下的条件，不比较不同路径
2. 旧版 relevance 主要来自 yield，而 V2 首先要判定 route-context 是否命中
3. 旧版温度回归和排序样本是两张表，V2 更适合统一到同一个 candidate table

所以 V2 里：

- `L_listMLE` 负责整组排序
- `L_bce` 负责候选命中判别
- `L_margin` 负责强化 hard negative 间隔
- `L_temp` 只在真正正样本上回归温度

---

## 8. 训练流程

建议训练分两阶段。

### 第一阶段：Oracle 预训练

输入：

- gold route
- gold context
- wrong context

输出：

- 初始神经排序模型

### 第二阶段：Non-Oracle 微调

输入：

- EditRetro 生成的 top-k routes
- candidate pool
- hard negatives

输出：

- 端到端可用的 Stage 2 neural model

---

## 9. 代码修改清单

下面只写“需要改什么”，不写分析。

### 9.1 配置层

新增参数：

- `use_product_branch`
- `use_support_features`
- `use_context_embedding`
- `product_fpsize`
- `product_feat_size`
- `support_feature_size`
- `context_embedding_dim`
- `lambda_rank`
- `lambda_temp`
- `lambda_listmle`
- `lambda_bce`
- `lambda_margin`
- `temp_loss_type`
- `temp_positive_only`
- `temperature_zscore_by_family`
- `hard_negative_refresh_epochs`

### 9.2 特征工程层

新增函数：

- `product_morgan_fp(product, n_bits=2048, radius=2)`
- `product_scaffold_smiles(product)`
- `normalize_fp(vec)`
- `tanimoto_similarity_from_bitvect(...)`

### 9.3 字段定义层

新增常量：

- Stage 2A/2B v2 用到的字段名
- `FEATURE_COLUMNS_V2`
- `LEARNED_FEATURE_COLUMNS_V2`

### 9.4 loss 层

新增函数：

- `weighted_listMLE`
- `binary_candidate_loss`
- `pairwise_margin_loss`

### 9.5 数据集与 DataLoader 层

新增一套 v2 数据结构：

- `Stage2CandidateDatapointV2`
- `Stage2CandidateDatasetV2`
- `Stage2CandidateDataLoaderV2`

负责输出：

- route 输入
- product 输入
- context 输入
- support/source 输入
- label
- sample_weight
- temperature

### 9.6 模型层

建议新建一个 Stage 2 排序+温度联合模型。

内部新增模块：

- `RouteEncoder`
- `ProductEncoder`
- `ContextEncoder`
- `SupportEncoder`
- `FusionBlock`
- `RankingTrunk`
- `RankingHead`
- `TemperatureTrunk`
- `TemperatureHead`

---

### 9.7 新建模块或脚本

#### 9.7.1 product memory 构建脚本

输入：

- family train split

输出：

- exact product memory 表
- scaffold memory 表
- product KNN 索引

#### 9.7.2 Stage 2A 候选池构建脚本

输入：

- route cache
- FNN checkpoint
- product memory

输出：

- candidate table

#### 9.7.3 route-context 训练表构建脚本

输入：

- train / validate / test split
- route cache
- candidate pool

输出：

- train candidate table
- validate candidate table
- test candidate table

#### 9.7.4 Stage 2 神经模型训练入口

输入：

- v2 candidate table
- family config
- model args

输出：

- neural checkpoint
- train / validate log

#### 9.7.5 Oracle 评估入口

输入：

- family test split
- v2 model checkpoint

输出：

- Oracle top-k 结果
- Oracle 温度结果

#### 9.7.6 Non-Oracle 评估入口

输入：

- product-only / route cache
- v2 candidate pool
- v2 model checkpoint

输出：

- Non-Oracle top-k 结果
- Non-Oracle 温度结果

---

## 10. 输入输出总结

### 10.1 product memory 构建

输入：

- Stage 2 train split

输出：

- exact product memory 表
- scaffold memory 表
- product KNN 索引

### 10.2 候选池构建

输入：

- route candidates
- FNN model
- product memory

输出：

- candidate table

### 10.3 route-context 训练表构建

输入：

- split 数据
- route cache
- candidate pool

输出：

- train/validate/test candidate tables

### 10.4 Stage 2 神经训练

输入：

- candidate tables

输出：

- neural model checkpoint

### 10.5 Oracle 评估

输入：

- gold route
- candidate table
- neural model

输出：

- Oracle top-k metrics

### 10.6 Non-Oracle 评估

输入：

- predicted routes
- candidate table
- neural model

输出：

- Non-Oracle top-k metrics

---

## 11. 实施顺序

推荐按下面顺序做。

### Step 1

先做：

- product memory 构建模块

### Step 2

再做：

- Stage 2A 候选池构建模块

### Step 3

再做：

- route-context 训练表构建模块

### Step 4

再做：

- Stage 2 排序+温度联合模型
- `weighted_listMLE`
- `binary_candidate_loss`
- `pairwise_margin_loss`
- 训练入口

### Step 5

最后做：

- Oracle 评估入口
- Non-Oracle 评估入口

---

## 12. 当前执行口径

当前 Stage 2 的执行口径固定为：

1. Stage 2A 做两路并集 candidate pool
2. Stage 2B 用纯神经模型直接输出最终 score 和 temperature
3. 先把网络结构、输入表示、训练 loss 和训练样本表做完整
