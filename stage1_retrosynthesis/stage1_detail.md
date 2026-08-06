# ProSys Stage 1 实施说明

## 1. 文档范围

这份文档只写四件事：

1. Stage 1 要解决什么问题
2. Stage 1 的数据怎么准备
3. 基模型怎么来、各家族怎么微调
4. Stage 1 最终输出什么给 Stage 2

这里默认：

- Stage 1 只负责路线生成
- 输入是目标产物
- 输出是 `top-k` 候选 `reactants >> product`
- Stage 1 按 family-specific 评估，但基模型可以共享

---

## 2. Stage 1 新框架

Stage 1 可以拆成四个部分：

```text
Stage 1A: 路线数据构建
Stage 1B: 基模型准备
Stage 1C: family-specific 微调
Stage 1D: 路线生成与 route ranking
```

完整链路如下：

```text
目标产物
-> EditRetro 路线生成
-> route aggregation / ranking
-> 输出 Top-k reactant routes
```

Stage 1 的最终目标不是只生成一条反应，而是输出一个可交给 Stage 2 的候选路线池。

---

## 3. Stage 1A：路线数据构建

### 3.1 目标

把原始反应记录整理成可用于 EditRetro 训练和评估的路线数据。

Stage 1 只关心：

- `reactants`
- `product`

因此它不直接依赖：

- reagent
- solvent
- temperature
- yield

这意味着：

- 某些记录即使因为条件字段缺失而不能用于 Stage 2
- 只要 `reactants` 和 `product` 有效，仍然可以进入 Stage 1 的训练集

### 3.2 输入

Stage 1A 需要两类输入：

1. 已经按 train / validate / test 划分好的标准化反应数据
2. 可选的原始反应表

第一类数据用于：

- 保证 Stage 1 与 Stage 2 的评估边界一致
- 固定 validate / test 集

第二类数据用于：

- 给 train split 做 route-only augmentation
- 补回那些对 Stage 2 无效、但对 Stage 1 仍然有效的反应

### 3.3 输出

Stage 1A 建议输出四类 artifact：

1. route raw train / validate / test 表
2. atom-mapped route 表
3. EditRetro 预处理后的文本数据
4. binarized training data

### 3.4 一条路线训练记录的基本单位

Stage 1 的原始训练单位是一条反应：

```text
(reactants, product)
```

建议保留这些字段：

- `pair_id`
- `reaction_id`
- `product_smiles`
- `mapped_reaction_smiles`
- `raw_reaction_smiles`
- `canonical_reaction_smiles`
- `mapping_confidence`

其中：

- `raw_reaction_smiles = reactants >> product`
- `mapped_reaction_smiles` 是 atom mapping 后的反应
- `canonical_reaction_smiles` 用于去重和防泄露

### 3.5 过滤规则

建议 Stage 1A 至少执行下面几步过滤。

#### 基础有效性过滤

删除：

- `product` 为空的记录
- RDKit 无法解析的记录
- 无法构造 canonical reaction key 的记录

#### 单产物约束

对于最终评估口径，推荐只保留：

- single-product reactions

因为整个 ProSys 是 target-product-driven 框架，Stage 1 的输入就是单个目标产物。

#### 训练增强数据的特殊规则

如果使用原始反应表给 train 做 augmentation，则：

1. 只增广 train
2. validate / test 不增广
3. 所有增广记录都要先和已选 train / validate / test 做 canonical reaction key 去重

也就是说：

- 允许把“Stage 2 筛掉但 route 有效”的反应补进 Stage 1 train
- 但绝不能让它和 validate / test 发生泄露

### 3.6 canonical reaction key

建议统一构造：

```text
canonical_reaction_smiles
= sorted_canonical_reactants + ">>" + canonical_product
```

作用有三个：

1. train augmentation 去重
2. 数据泄露控制
3. route-level 评估对齐

### 3.7 atom mapping

EditRetro 训练前需要把反应转成 atom-mapped reaction。

因此 Stage 1A 要有一个单独的 mapping 步骤：

1. 读取 `reactants >> product`
2. 调用 atom mapper
3. 对成功映射的记录保留：
   - `mapped_reaction_smiles`
   - `mapping_confidence`
4. 对映射失败的记录删除

最终训练使用的是：

- 成功 atom-mapped 的路线数据

### 3.8 EditRetro 预处理与 binarize

atom-mapped route 表还不能直接训练，需要继续变成 EditRetro 可读格式。

建议再做两步：

#### 预处理

把 mapped reactions 转成：

- source side
- target side
- augmentation 后的文本样本

#### binarize

把文本样本转成：

- 词表
- fairseq 可直接读取的二进制数据

因此 Stage 1A 的最终训练输入，不是原始 CSV，而是：

- 预处理后的 `src/tgt`
- 对应的 binarized data

---

## 4. Stage 1B：基模型准备

> **当前维护口径（`ProSys_8_9.docx`）。** 论文主线只使用
> `USPTO_STAGE2_FILTERED` 这一份由 USPTO-FULL 构建的 benchmark-safe 共享基模：
> 去原子映射、去除与 Reaxys 验证/测试锚点重叠的反应并按 canonical reaction 去重后，
> 共 934,575 条路线，按 8:2 固定为 747,660 条训练和 186,915 条验证样本。
> `run_base_train.sh` 与 family fine-tuning 脚本默认使用该数据集和其 checkpoint。
> 以下 `USPTO-50K` 内容仅保留为上游 EditRetro 的历史备选方案，不是当前报告结果的
> 初始化步骤。

### 4.1 目标

Stage 1 不建议从零开始重训一个超大路线模型。

历史上可以准备两层基线：

1. `USPTO-50K base`
2. `USPTO-full-safe base`
对于基模的数据就是只有训练和验证就好，没必要有测试集合

### 4.2 基模型 1：USPTO-50K base

这是最基础的路线基模。

它的作用是：

- 提供一个已经会做 retrosynthesis 的初始模型
- 作为后续所有 family 微调的最小起点

如果已有稳定 checkpoint，可以直接使用，不必重复从零训练。

### 4.3 基模型 2：USPTO-full-safe base

这是当前维护的共享基模。

构造逻辑是：

1. 从 USPTO-full 路线数据出发
2. 先过滤掉与当前 benchmark test products 重叠的样本
3. 得到 benchmark-safe 的大规模路线数据
4. 用 EditRetro 训练配置在该 benchmark-safe 数据集上训练共享基模

这个中间基模的作用是：

- 比纯 50K 基模拥有更广的路线覆盖能力
- 又避免 benchmark test product 泄露

### 4.4 当前推荐基线

当前 Stage 1 更推荐的共享起点是：

```text
USPTO_STAGE2_FILTERED benchmark-safe base
-> family-specific finetune
```
当前训练好的共享基模通过 `checkpoint_USPTO_STAGE2_FILTERED_best.pt` 引用；旧的 `checkpoint_UPSTO_full_best.pt` 仅保留为兼容别名。

也就是说，最终汇报时更建议把：

- `USPTO-full-safe base`

作为后续 family 微调的统一初始化模型。

### 4.5 基模型训练输入

基模型训练输入是：

- binarized route dataset
- 可选 restore checkpoint

对于当前维护的 `USPTO_STAGE2_FILTERED` base，默认训练入口不要求
`USPTO-50K` restore checkpoint；如显式提供 checkpoint，则其来源必须另行记录。

### 4.6 基模型训练超参数口径

建议保持一套统一的 EditRetro 训练配置。

当前可沿用的核心口径包括：

- architecture: `editretro_nat`
- task: `translation_retro`
- criterion: `nat_loss`
- noise: `random_delete_shuffle`
- optimizer: Adam
- learning rate scheduler: `inverse_sqrt`
- label smoothing: `0.1`
- dropout: `0.2`
- attention dropout: `0.2`
- weight decay: `0.01`
- share all embeddings
- learned positional embeddings
- mixed precision training

对于大数据中间基模，建议：

- epoch 数较少
- update budget 足够大

因为它的目标不是过度拟合某一类反应，而是提供共享路线先验。

### 4.7 基模型输出

基模型训练结束后，建议固定输出：

1. best checkpoint
2. last checkpoint
3. 若干周期性 checkpoint
4. 训练日志
5. 一个稳定的 latest alias

这样后续 family 微调和评估都不需要再猜路径。

---

## 5. Stage 1C：family-specific 微调

### 5.1 目标

基模型只提供通用路线能力。

family 微调的目标是：

- 让模型更适应某一类反应的局部分布
- 提高该 family 下的 route recall

### 5.2 family 微调输入

每个 family 都需要自己的：

1. train split
2. validate split
3. test split
4. route raw / mapped / binarized dataset

这里要强调：

- Stage 1 微调虽然只看 route
- 但 validate / test 仍然要和全项目数据划分保持一致

### 5.3 两种微调起点

family 微调可以有两条线：

1. `USPTO-50K base -> family finetune`
2. `USPTO-full-safe base -> family finetune`

当前推荐第二条线作为主结果。

### 5.4 family 微调训练规则

建议每个 family 独立训练一个模型。

训练时建议：

1. 第一次从共享基模开始时，重置 optimizer / lr scheduler / dataloader 状态
2. 如果训练中断，再次续跑时直接从该 family 的 latest checkpoint 恢复
3. 保留一个稳定的 latest alias，始终指向当前可用的最佳 checkpoint

### 5.5 family 微调输出

每个 family 最终都应输出：

1. family-specific best checkpoint
2. family-specific last checkpoint
3. 训练日志
4. 稳定的 latest alias

也就是说，后续任何 route-only evaluation、route cache generation、Stage 2 coupling，都只需要读取：

- family-specific latest checkpoint

### 5.6 catmerge 对 Stage 1 的影响

Stage 1 的预测目标始终是：

```text
product -> reactants
```

因此：

- `catmerge` 不改变 Stage 1 的预测空间
- 它只影响你拿哪些 Reaxys 记录来构造 family route dataset

换句话说，Stage 1 里看到的仍然只是反应本身，不预测 catalyst / reagent / solvent。

---

## 6. Stage 1D：路线生成与 route ranking

### 6.1 输入

推理时，Stage 1 的输入是：

- 一个或多个目标产物

如果输入是完整反应式，也只取 product 侧进入 Stage 1。

### 6.2 原始生成

EditRetro 生成时，通常会结合：

- test-time augmentation
- reposition beam
- token beam

因此一个产物在原始层面会得到多组候选 reactants。

设：

- augmentation 数为 `A`
- reposition beam 为 `B_r`
- token beam 为 `B_t`

则原始候选规模近似为：

```text
A * B_r * B_t
```

### 6.3 route aggregation

原始生成结果不能直接拿来给 Stage 2，用前需要做 route aggregation。

建议流程：

1. 对所有候选 reactants 做 canonicalization
2. 跨 augmentation 和 beam 合并相同 reactants
3. 用 EditRetro 自带的聚合打分逻辑得到每条 route 的总分
4. 按分数排序，保留 top-k unique reactant routes

### 6.4 route score 与 route probability

聚合后，每条路线有一个：

- `retro_score`

这个值主要用于排序，不一定是概率。

为了给 Stage 2 一个更稳定的数值输入，建议再做归一化：

```text
retro_probability_i
= retro_score_i / Σ_j retro_score_j
```

如果总分为 0，则退化为平均分配。

### 6.5 Stage 1 输出给 Stage 2 的字段

Stage 1 最终输出给 Stage 2 的每条 route，建议至少包含：

- `product`
- `retro_rank`
- `reactants`
- `retro_score`
- `retro_probability`
- `reaction_smiles`

其中：

```text
reaction_smiles = reactants >> product
```

### 6.6 推荐输出形式

建议同时输出两层结果：

#### 路线表

每个产物一张 route table，保存该产物的全部 top-k 路线。

#### 路线缓存

把所有产物的路线合并成一个 route cache，供 Stage 2A 反复读取。

这样做的作用是：

1. Stage 2 训练时不必反复调用 EditRetro
2. Oracle / Non-Oracle 评估可以共用同一套 route cache
3. 后续 reranking 实验可以直接复用路线结果

---

## 7. Stage 1 评估

### 7.1 route-only 评估目标

Stage 1 的核心评估不是条件命中，而是：

- 真实 reactants 是否出现在预测 top-k 路线里

### 7.2 route hit 定义

先对 gold reactants 和 predicted reactants 都做：

1. 去 atom map
2. canonicalize
3. reactant components 排序

然后定义：

```text
route_hit@k = 1
当且仅当 gold reactants 出现在前 k 条预测路线中
否则为 0
```

总体指标：

```text
route_top@k = Σ route_hit@k / N
```

常用 `k`：

- Top-1
- Top-3
- Top-5
- Top-10

### 7.3 分母口径

建议同时支持两种分母：

1. 全部测试样本
2. 仅有效可评估样本

这里的“不可评估样本”通常包括：

- 空 product
- 多产物
- 空 reactants
- 无法 canonicalize 的记录

### 7.4 评估输出

建议 route-only evaluation 固定输出：

- `N`
- `valid_eval_items`
- `route_top@1`
- `route_top@3`
- `route_top@5`
- `route_top@10`
- 若干不可评估样本计数

---

## 8. 新项目需要的模块

下面只写“需要什么模块”，不写旧仓库路径。

### 8.1 配置层

建议新建一套 Stage 1 配置，至少包含：

- dataset name
- augmentation
- beam settings
- top-k
- restore checkpoint
- max epoch
- max update
- learning rate
- warmup updates
- patience

### 8.2 数据构建层

建议新建：

1. route dataset builder
2. raw train augmentation builder
3. canonical reaction key 工具
4. atom mapping 模块
5. EditRetro preprocess / binarize wrapper

### 8.3 训练层

建议新建：

1. base model trainer
2. family finetune trainer
3. family finetune queue
4. latest checkpoint resolver

### 8.4 推理层

建议新建：

1. product-only route generator
2. route aggregation / ranking 模块
3. route cache builder

### 8.5 评估层

建议新建：

1. route-only evaluator
2. per-family route summary
3. route cache audit 工具

---

## 9. 实施顺序

推荐按下面顺序做。

### Step 1

先做：

- route dataset builder
- atom mapping
- preprocess / binarize

### Step 2

再做：

- `USPTO-50K base`
- `USPTO-full-safe base`

### Step 3

再做：

- family-specific route datasets
- family-specific finetune

### Step 4

再做：

- product-only route generation
- route aggregation
- route cache

### Step 5

最后做：

- route-only evaluation
- per-family summary
- Stage 2 handoff

---

## 10. 当前执行口径

当前 Stage 1 建议固定为：

1. Stage 1 只做 `product -> top-k reactant routes`
2. train 允许使用 route-only augmentation
3. validate / test 严格绑定正式 split
4. 推荐共享基模使用 `USPTO-full-safe base`
5. 推荐最终路线模型使用 `family-specific finetune`
6. 推荐对每个产物输出 top-10 route cache 给 Stage 2
