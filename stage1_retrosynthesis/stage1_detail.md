# ProSys Stage 1 实施说明

## 2026-09-25 当前方案与训练边界

当前只使用过滤后的 USPTO-50K 从随机权重训练共享基模，不继承 FULL 神经权重。
数据准备后为 39,683 条训练反应、4,988 条验证反应，对应 396,830/49,880 对增强输入。
基模完成 50 epochs、16,500 updates，验证集选择 epoch 49（loss 3.573）。
原始条件划分及全部 3,860 个测试查询保持不变；增强后的基模/专家输入通过
受保护留出集交叉与分子身份检查。精确反应不交叉不等于产物完全不交叉。

六家族分别从同一基模微调 seeds 0/1/2；固定 expert seed 1 为所有下游重复提供
路线，不按测试成绩选择专家。固定 seed-1 的宏平均 Route@10 为 55.86%，
对应基模 10.08%；三专家种子宏平均 Route@10 为 55.42±0.39%，
不能把固定 seed-1 数值当作三专家种子均值。18 组结果均已核查，详见
`../Experiment/stage1_50k_fidelity_v2_expert_multiseed_20260924/SUMMARY.md`。

最后一组 Diels-Alder seed 2 在 epoch 101 正常早停，但测试时一个候选
膨胀为 3,078 token，超过解码器 1,024 个位置的硬容量。独立保护仅将该候选
记为空的无效生成槽位，不截断拼接化学结构、不补抽候选，也不删除查询；
762 个查询与 76,200 个生成槽位完整保留。同家族 seed 1 全量 GPU 对照的
候选字符串、最终路线顺序/分数/概率均一致；仅两条原本无效 SMILES 的 EOS
分数不同，已记录，不能声称所有 token 分数逐位一致。其他 17 组原始解码
保持不变。恢复来源与准入记录位于
`../Experiment/stage1_decode_diagnostic_20260925/expert_recovery_admission.json`。

历史 FULL 审计曾发现与条件验证集重叠 98 条及增强后的留出反应交叉，相关队列
已停止且不再续训。旧模型仅作诊断记录，不是本轮 50K 结果。

增强文本与二进制核对还发现家族数据中的空反应物目标：训练 14610 对、验证 80 对。
这些是实际 EOS-only 训练目标，不是文件显示问题。USPTO 基模增强输入没有空样本。
`preprocess_data.py` 已加入增强完成后的非空成对过滤，并记录被排除的增强行号；
跨点环闭合也改为按 RDKit 完整连通分量处理。新的专家训练须使用版本化清理输入，
不能静默改写旧输入或从测试查询分母中删除失败样本。详细数量、证据和方案见
[验证方案说明](../Experiment/project_completion_20260913/VALIDATION_DECISION.md)。

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

这里的增广指补充新的原始反应记录，不是 SMILES 表示增强；模型准备阶段对
训练与验证反应均生成多种表示，并检查实际增强后的反应身份和划分边界。

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

> **2026-09-24 用户确认的新方案。** 共享基模改为
> `USPTO_50K_FILTERED`：仅使用过滤后的 USPTO-50K，从随机权重开始训练。
> 不继承旧 FULL 基模，也不采用可能含 FULL 预训练的上游 50K 权重。
> 新的六家族 50K 下游结果已完成并独立核验，根目录正文和 SI 已于
> 2026-09-25 完成本地合稿发布。旧 FULL 记录分版本保留，不能改名当作新结果。
> 新实验入口为 `scripts/run_stage1_50k_from_scratch.py --train`；具体状态、
> 过滤数及超参数写入 `Experiment/stage1_50k_from_scratch_20260924/`。

### 4.1 目标

用规模较小的 USPTO-50K 从头学习通用逆合成表示，再分别微调六个家族。
Stage 2/3 架构不变；基模、专家与依赖其路径的结果必须分版本管理。

### 4.2 基模型 1：USPTO-50K base

这是最基础的路线基模。

它的作用是：

- 提供一个已经会做 retrosynthesis 的初始模型
- 作为后续所有 family 微调的最小起点

当前要求从头训练，禁止加载任何预训练 checkpoint。沿用固定 SPE/ChEMBL
分词资源和共享词表，这不等于加载 FULL 训练的神经网络权重。

数据取自 RetroSim 的 Schneider 50K 文件，按其 `get_data.py` 对每个类别
保留原顺序做 80/10/10 划分。源文件实际为 50,016 条，原始 train/val/test
分别为 40,008/5,001/5,007 条；类别字段仅用于复现划分，不作为模型输入。
只将过滤后的 train/val 送入基模；原 USPTO 测试部分不训练、不调参。

过滤同时检查去映射的完整反应和产物拆分、映射反应物选择后的反应身份，
排除 Reaxys 条件与专家验证/测试集合的重叠、USPTO 自身留出集交叉及组内重复。
保留官方划分方向，不重新打散记录。随后做 10 倍增强并逐条核验增强文本、
实际二进制张量和跨集合成员关系，通过后才进入 GPU 训练。

### 4.3 基模型 2：USPTO-full-safe base

这是已经停用的历史方案，不是新训练的默认基模。

构造逻辑是：

历史上从 USPTO-full 路线数据构建 `USPTO_STAGE2_FILTERED`。后续审计发现，
原始划分无交叉不保证增强后无交叉，所以不能再将旧基模笼统称为
benchmark-safe。严格修复副本及其审计保留为证据，但不再安排 FULL 重训。

旧 FULL 模型和数值仅作历史记录，不作为新 50K 模型的初始化或结果。

### 4.4 当前推荐基线

当前 Stage 1 更推荐的共享起点是：

```text
USPTO_50K_FILTERED random initialization -> shared base training
-> family-specific finetune
```
默认新别名为 `checkpoint_USPTO_50K_FILTERED_best.pt`，但训练完成和来源核验前
不创建或晋升该别名。正式专家实验优先显式传入已核验的新 checkpoint。
若新基模不存在，入口报错，不自动回退到 FULL。

### 4.5 基模型训练输入

基模型训练输入是：

- binarized route dataset
- 随机初始化，不允许 restore checkpoint

`run_base_train.sh` 默认数据集为 `USPTO_50K_FILTERED`；该模式下设置
`RESTORE_CKPT` 会直接报错。仅保留 best/last，结果写入新目录，不覆盖旧模型。

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

本轮基模配置为学习率 0.0003、warmup 10,000、max tokens 16,384，
最多 50 epochs / 200,000 updates，验证 loss patience 10，seed 1。
实际以 50 epochs 正常结束；这些是保留日志中的设置，不是后续调参建议。

### 4.7 基模型输出

基模型仅保留 best 与 last checkpoint，以及训练日志、配置和哈希核验记录。
本轮不保存周期性模型，也不依赖容易漂移的 latest 别名；专家入口显式绑定
`Experiment/stage1_50k_from_scratch_20260924/` 下已核验的 best checkpoint。

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

### 5.3 唯一的本轮微调起点

`USPTO_50K_FILTERED scratch base -> family finetune` 是本轮唯一主线。
六家族、三个专家种子均从同一个已核验的 50K best checkpoint 开始。
历史 FULL 起点已停用，不作为备用初始化，更不作为当前主结果。

### 5.4 family 微调训练规则

建议每个 family 独立训练一个模型。

本轮独立种子训练遵循：

1. 第一次从共享基模开始时，重置 optimizer / lr scheduler / dataloader 状态
2. 固定当前种子的输入、超参数和停止条件；不能因日志安静而中断或重启
3. 保留 best/last 与配置哈希；确需恢复时必须另行核验同一实验，不使用漂移的 latest 别名

### 5.5 family 微调输出

每个 family 最终都应输出：

1. family-specific best checkpoint
2. family-specific last checkpoint
3. 训练日志
4. 配置、输入/模型哈希及完成核验记录

也就是说，后续任何 route-only evaluation、route cache generation、Stage 2 coupling，都只需要读取：

- 已明确绑定路径与 SHA256 的 family-specific best checkpoint

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
