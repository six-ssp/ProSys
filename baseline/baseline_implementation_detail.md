# ProSys Baseline Implementation Detail

> **Status updated: 2026-07-27.** This document defines the current four
> citable baselines for the `6-family / strict Non-Oracle /
> target-product-driven` task. Obsolete direct-retrieval artifacts and
> diagnostic records have been removed. This does not affect the mainline Stage 2 KNN,
> which is part of ProSys rather than a baseline.
>
> Exact direct-model implementation and results are maintained in
> [`product_condition_baselines_detail.md`](product_condition_baselines_detail.md)
> and [`current_baseline_results_20260727.md`](current_baseline_results_20260727.md).

---

## 1. 当前主线与 Baseline 设计目标

### 1.1 当前 ProSys 主线

当前维护的正式主线为：

```text
Target product
→ Stage 1: family-specific EditRetro
→ Stage 2: widened KNN retrieval + ReaFNN filtering
→ Stage 3: reaction-GNN features + XGBoost reranking
→ temperature prediction
→ ranked complete reaction systems
```

当前正式评价口径为：

- 6 个 Reaxys reaction families
- strict Non-Oracle
- target-product-driven
- Stage 2 和 Stage 3 只能读取 Stage 1 的预测路线
- 不允许在测试阶段访问 gold reactants 或 gold conditions

### 1.2 Baseline 设计原则

四个 baseline 覆盖两类输入设定：

1. **直接 Product-to-Condition 基线**：Baseline 1 和 Baseline 2 只读取
   target product；在条件预测完成后才与冻结的 Stage 1 路线组合。
2. **路线条件化基线**：Baseline 3 和 Baseline 4 读取同一条 predicted
   route 与 product，与主线共享冻结的 Stage 1 路线缓存。

这样可以分别检验低容量传统 ML、直接分子图模型、顺序条件网络和反应图模型，
同时保持最终严格 system-level 指标可比。

---

## 2. Current Baseline Mapping

| ID | Method | Stage 1 use | Condition-model input | Output | Temperature |
|---|---|---|---|---|---|
| Baseline 1 | Product-Bernoulli Naive Bayes | paired after condition prediction | target product Morgan fingerprint | ranked historical reagent-solvent contexts | N/A |
| Baseline 2 | Product-GNN | paired after condition prediction | target product molecular graph | ranked historical reagent-solvent contexts | N/A |
| Baseline 3 | EditRetro + Sequential FNN | predicted route supplied to condition model | predicted route plus product | reagent set, solvent set, temperature | FNN regressor |
| Baseline 4 | EditRetro + Reaction-GCNN | predicted route supplied to condition model | predicted route plus product | reagent and solvent sets | N/A |
| Mainline | ProSys | family-specific EditRetro | route-local KNN + ReaFNN + Reaction-GNN/XGBoost | ranked complete systems | XGBoost regressor |

---

# 3. 统一实验规范

## 3.1 数据划分

所有方法必须使用相同的：

- reaction families
- train / validation / test split
- canonical reaction grouping
- reagent / solvent normalization
- catalyst-to-reagent merge
- low-frequency label filtering
- reagent / solvent count constraints
- full-system exact-match definition

严禁：

- baseline 使用随机划分，而 ProSys 使用 canonical reaction grouped split
- baseline 使用 test 数据构建标签空间或候选库
- baseline 根据 test 结果调整阈值、beam size、融合权重
- Baseline 3、4 使用 gold route，而 ProSys 使用 predicted route

## 3.2 六个反应家族

当前正式家族为：

1. Beckmann
2. Buchwald–Hartwig
3. Chan–Lam
4. Diels–Alder
5. Friedel–Crafts Acylation
6. Friedel–Crafts Alkylation

所有 baseline 均按 family-specific 模式训练与测试。

## 3.3 统一 Stage 1 路线缓存

Baseline 1、Baseline 2 在条件预测完成后，以及 Baseline 3、Baseline 4
在条件预测输入阶段，均使用同一份冻结的 Stage 1 输出：

```text
outputs/stage1_routes/<family>/route_cache.json
```

建议统一字段：

```json
{
  "sample_index": 0,
  "product": "...",
  "routes": [
    {
      "reactants": "...",
      "retro_rank": 1,
      "retro_score": -1.23,
      "retro_probability": 0.37
    }
  ]
}
```

要求：

- 每个测试样本读取相同 top-10 路线
- canonicalization 规则一致
- 相同 route 去重策略
- 相同无效 SMILES 处理方式
- 相同 route score / probability 归一化方式

## 3.4 统一候选输出格式

所有方法最终应转成同一种 candidate table：

```text
sample_index
family
product
reactants
reagents
solvents
route_rank
route_score
condition_score
system_score
temperature_pred
```

推荐保存为：

```text
outputs/baselines/<baseline_name>/<family>/candidates.csv
```

最终评价脚本只读取统一 candidate table，不直接读取各模型内部输出。

## 3.5 统一完整体系命中定义

完整 system 命中要求同一个候选同时满足：

```text
canonical predicted route == canonical gold route
AND
normalized reagent set == normalized gold reagent set
AND
normalized solvent set == normalized gold solvent set
```

不能出现：

- route 和 context 分别由两个不同候选命中，却记为 full-system hit
- reagent / solvent 只命中部分标签就算完整命中
- 忽略分子角色后进行模糊匹配

---

## Archived Planning Notes

> Sections 4 onward preserve pre-revision implementation planning for the
> direct Transformer, Sequential FNN, and Reaction-GCNN. Their historical
> labels "Baseline 1/2/3" are superseded by the current Baseline 1–4 mapping
> in Section 2 and must not be used in reporting current results.

# 4. Archived Direct Transformer Design (not a current baseline)

## 4.1 方法定位

This archived design represents:

> 不显式拆分 retrosynthesis 与 condition recommendation，而是从 target product 直接生成完整 precursor sequence。

推荐名称：

```text
Product-to-Precursors Transformer
```

或：

```text
Direct Transformer
```

## 4.2 输入与输出

输入：

```text
<PRODUCT> product_smiles
```

推荐输出格式：

```text
<REACTANT> reactant_smiles
<REAGENT> reagent_smiles
<SOLVENT> solvent_smiles
```

实际序列可写为：

```text
<REACTANT> R1.R2 <REAGENT> G1.G2 <SOLVENT> S1
```

不建议直接生成无角色标记的混合 precursor string，因为后续无法稳定区分：

- reactants
- reagents
- solvents

## 4.3 数据构建

### 训练样本格式

```text
source:
<PRODUCT> product_smiles

target:
<REACTANT> reactants <REAGENT> reagents <SOLVENT> solvents
```

### 处理规则

- 所有 SMILES 先 canonicalize
- reactants 内部排序固定
- reagent set 内部排序固定
- solvent set 内部排序固定
- 相同 system 重复记录需要去重
- 保留角色标记作为特殊 token
- 不加入 temperature token，避免连续值生成带来的额外不稳定

## 4.4 模型实现建议

建议采用标准 encoder-decoder Transformer：

- encoder layers：6
- decoder layers：6
- hidden size：512
- attention heads：8
- dropout：0.1
- label smoothing：0.1

以上为推荐初始配置，最终可根据显存和 validation performance 调整。

优先复用现有分子 Transformer 代码框架，例如：

- OpenNMT-py
- fairseq Transformer
- 现有 reaction Transformer 仓库

## 4.5 解码策略

推荐：

- beam size：10 或 20
- 每个 product 输出 top-10 systems
- canonicalize 后去重
- 无效 SMILES 候选直接删除
- 若角色段缺失，则该候选判为格式无效

候选分数：

```text
system_score = normalized sequence log-probability
```

需要对输出长度做 normalization，避免模型偏好过短序列。

## 4.6 评价

Baseline 1 直接生成 route 和 conditions，因此可评价：

- Route@1 / 3 / 5 / 10
- System@1 / 3 / 5 / 10
- nDCG@10
- MRR
- validity
- role-format validity

建议不评价：

- Stage 2 pool coverage
- Stage 1 shared route recall
- temperature metrics

因为该 baseline 没有独立 candidate-pool stage，也没有温度头。

## 4.7 主要风险

### 风险 1：角色混淆

解决：

- 使用显式 `<REACTANT> / <REAGENT> / <SOLVENT>` token
- 在后处理时严格验证角色段完整性

### 风险 2：输出无效 SMILES

解决：

- 统一 RDKit parse
- canonicalization
- 统计 validity 作为额外指标

### 风险 3：生成序列过长

解决：

- 限制最大输出长度
- 过滤超长 reagent / solvent 组合
- 训练时按长度分桶

### 风险 4：数据量较小

解决：

- 可使用 USPTO 或通用 reaction data 预训练
- 再对 6 个家族分别微调
- 若预训练工作量过大，可先做 family-specific from-scratch 版本，并在论文中明确说明

---

# 5. Archived Sequential FNN Design (current Baseline 3)

## 5.1 方法定位

Baseline 2 代表经典的：

> reaction fingerprint → sequential condition prediction

推荐名称：

```text
EditRetro + Sequential FNN
```

## 5.2 输入

对 Stage 1 每条预测路线构造 reaction representation。

建议沿用当前 ProSys 指纹定义：

```text
x_rxn = [FP(product); FP(product) - FP(reactants)]
```

推荐参数：

- Morgan fingerprint
- radius = 2
- fp size = 4096

这样可减少“输入表示不同”造成的不公平。

## 5.3 输出顺序

由于当前数据中 catalyst 已合并到 reagent，建议顺序为：

```text
reagent → solvent → temperature
```

也可测试：

```text
solvent → reagent → temperature
```

最终顺序必须只根据 validation set 选择。

## 5.4 模型结构

### Reagent head

输入：

```text
reaction fingerprint
```

输出：

```text
multi-label reagent probabilities
```

推荐：

- 2–3 层 MLP
- hidden size：512 / 256
- ReLU
- dropout：0.1–0.3
- sigmoid output
- BCEWithLogitsLoss

### Solvent head

输入：

```text
reaction fingerprint + predicted reagent representation
```

输出：

```text
multi-label solvent probabilities
```

reagent representation 可用：

- predicted multi-hot vector
- top-k probability vector
- reagent embedding weighted sum

优先使用概率向量，避免训练与测试分布差异过大。

### Temperature head

输入：

```text
reaction fingerprint
+ reagent probability vector
+ solvent probability vector
```

输出：

```text
temperature
```

损失：

- SmoothL1Loss 或 MSELoss
- 推荐先使用 SmoothL1Loss

## 5.5 多标签集合解码

直接对单个标签取 top-k 不足以产生完整 reagent set / solvent set。

推荐两种方案。

### 方案 A：历史集合打分

先统计每个 family 的历史 reagent sets 与 solvent sets。

对完整集合打分：

```text
score(set) =
sum(log p(label in set))
+ sum(log(1-p(label not in set))) * beta
```

其中 beta 可取较小值，避免标签空间过大导致未选标签项占主导。

优点：

- 输出完整集合
- 与历史标签格式一致
- 组合空间可控

### 方案 B：beam search

依次扩展高概率标签，直到：

- reagent 数不超过 3
- solvent 数不超过 2

缺点：

- 容易产生训练集中从未出现的组合
- 不同 beam 策略可能影响比较公平性

建议主实验使用方案 A。

## 5.6 route 与 condition 分数融合

每个完整候选的最终分数：

```text
system_score =
alpha * normalized_route_score
+ beta * reagent_set_score
+ gamma * solvent_set_score
```

权重仅在 validation set 上搜索。

建议搜索范围：

```text
alpha ∈ {0.5, 1.0, 1.5, 2.0}
beta  ∈ {0.5, 1.0, 1.5}
gamma ∈ {0.5, 1.0, 1.5}
```

选择顺序：

1. validation sys@10 最大
2. 若相同，选择 sys@1 更高
3. 若仍相同，选择参数更简单的组合

## 5.7 输出候选数量

建议：

- 每条 route 生成 top-20 contexts
- 每个 product 最多 200 candidate systems
- 所有 route-condition 候选合并后统一排序

不能按每条路线单独计算 top-k。

## 5.8 评价

- shared Route@10
- candidate full-system cover
- System@1 / 3 / 5 / 10
- nDCG@10
- MRR
- Temp MAE
- Temp ±5 / ±10 / ±20 °C

---

# 6. Archived Reaction-GCNN Design (current Baseline 4)

## 6.1 方法定位

Baseline 3 代表：

> graph-based multilabel condition prediction

推荐名称：

```text
EditRetro + Reaction-GCNN
```

该 baseline 用于验证：

- 使用 GNN 表示是否已经足够
- ProSys 的提升是否来自 retrieval、filtering 和 reranking，而不只是 Reaction-GNN 特征

## 6.2 输入图构建

对每条预测 route：

- reactants：多个分子图
- product：单个分子图

每个分子图包含：

### atom features

- atomic number
- degree
- formal charge
- hybridization
- aromatic flag
- number of hydrogens
- chirality

### bond features

- bond type
- conjugation
- ring flag
- stereochemistry

## 6.3 图编码方式

推荐分别编码 reactants 和 product：

```text
h_R = Pool(GNN(reactant graphs))
h_P = Pool(GNN(product graph))
```

reactants 中多个分子可先逐分子编码，再做：

- sum pooling
- mean pooling
- attention pooling

初始实现优先使用 sum 或 mean pooling。

构造 reaction representation：

```text
h_rxn = concat(h_R, h_P, h_P - h_R, h_P * h_R)
```

## 6.4 模型结构

推荐：

- GNN layers：3–5
- hidden dim：256
- graph readout：mean + sum concatenation
- MLP hidden dim：512 → 256
- dropout：0.1–0.3

两个输出头：

```text
reagent_logits = MLP_reagent(h_rxn)
solvent_logits = MLP_solvent(h_rxn)
```

损失：

```text
L = L_reagent_BCE + lambda * L_solvent_BCE
```

lambda 初始取 1.0，并在 validation 上调整。

## 6.5 类别不平衡处理

由于 reagent / solvent 标签长尾明显，推荐依次测试：

1. 普通 BCE
2. positive class weighting
3. focal loss

主结果优先选 validation 表现最好的一个，但必须对所有 family 使用同一套选择规则。

## 6.6 完整 context 生成

建议不要自由组合所有高概率标签，而是对 family-specific 历史 contexts 做全局打分。

对一个历史 context：

```text
context = (reagent_set, solvent_set)
```

打分：

```text
context_score =
sum reagent log-probabilities
+ sum solvent log-probabilities
```

然后从该 family 全部历史 contexts 中选择 top-k。

该 baseline 不使用：

- KNN neighbor retrieval
- neighbor support count
- similarity-weighted yield
- ReaFNN filtering
- XGBoost reranking

因此它仍然是纯粹的全局图预测 baseline。

## 6.7 route 与 context 融合

```text
system_score =
alpha * normalized_route_score
+ beta * normalized_context_score
```

参数仅使用 validation set 选择。

推荐搜索：

```text
alpha ∈ {0.5, 1.0, 1.5, 2.0}
beta  ∈ {0.5, 1.0, 1.5, 2.0}
```

## 6.8 温度处理

有两种公平方案。

### 方案 A：统一温度回归器

Baseline 2、Baseline 3 与 ProSys 使用相同温度 regressor 输入格式。

优点：

- 温度结果更可比较
- 不把条件推荐与温度架构混在一起

缺点：

- baseline 不再完全对应原始方法

### 方案 B：Baseline 3 不报告温度

主表只比较 system metrics，温度放独立表，标记 N/A。

建议：

- 主文以 system metrics 为核心
- Baseline 3 温度可先使用 N/A
- 若审稿人要求完整输出，再补统一温度回归器

---

# 7. 统一评价脚本

已由 `baseline/external_adapters/evaluate_candidates.py` 实现。它复用
`prosys_shared.mainline` 的 gold matching 和排序指标，但以
`test_manifest.jsonl` 的所有正式测试样本作为分母；没有 Stage 1 route 的样本
必须作为 zero-hit slate 保留，不能因 candidate table 中没有行而被省略。

实际 CLI 参数为 `--candidates`、`--gold-split`、`--test-manifest`、
`--labeled-output`、`--output`、`--score-column` 和可选的
`--temperature-column`、`--topks`。

以下为早期接口草案，仅保留作历史说明：

```text
evaluation/evaluate_system_candidates.py
```

输入：

```text
--candidate_csv
--gold_csv
--family
--topk 1 3 5 10
```

输出：

```json
{
  "route_at_1": 0.0,
  "route_at_3": 0.0,
  "route_at_5": 0.0,
  "route_at_10": 0.0,
  "cover": 0.0,
  "system_at_1": 0.0,
  "system_at_3": 0.0,
  "system_at_5": 0.0,
  "system_at_10": 0.0,
  "ndcg_at_10": 0.0,
  "mrr": 0.0
}
```

`evaluate_candidates.py` 额外输出 `candidate_slates`、
`missing_candidate_slates` 和 `denominator=all_test_manifest_samples`，
使候选池缺失与评价分母都可审计。

## 7.1 温度评价

建议单独新建：

```text
evaluation/evaluate_temperature.py
```

统计对象：

- 每个样本最高排名的 full-match candidate
- candidate 具有有效 temperature prediction
- gold record 具有有效 temperature

输出：

- MAE
- RMSE
- Temp ±5 °C
- Temp ±10 °C
- Temp ±20 °C
- evaluated sample count

必须同时报告 evaluated sample count，避免不同方法支持集大小差异被隐藏。

---

# 8. 已落地代码目录与外部实现对齐

```text
baseline/
├── MolecularTransformer/                  # 上游仓库，只作 direct baseline 参考
├── Reaction_condition_recommendation/     # 上游仓库，只作 Sequential FNN 参考
├── reaction-gcnn/                         # 上游仓库，只作 Reaction-GCNN 参考
├── external_adapters/
│   ├── contracts.py                       # 统一标签、SMILES、route 和 JSONL/CSV 协议
│   ├── build_datasets.py                  # 从正式 split 构建三个 baseline 输入包
│   ├── model_common.py                    # 训练集词表、FP 与历史 context 库
│   ├── run_sequential_fnn.py              # PyTorch Sequential FNN 的训练与预测
│   ├── run_reaction_gcnn.py               # PyTorch Reaction-GCNN 的训练与预测
│   ├── export_candidates.py               # 统一转为 ProSys candidate table
│   ├── evaluate_candidates.py             # 固定全测试集分母的 Non-Oracle 评价
│   ├── fetch_molecular_transformer_weights.py
│   ├── source_manifest.json               # 来源、提交版本与权重检查记录
│   └── README.md                           # 可执行命令和输入输出约定
├── baseline.md                             # 精简实验概览
└── baseline_implementation_detail.md      # 本文件，实验设计与实现细节
```

三个上游仓库均保持原样，不在其内部修改数据处理或评价逻辑。原因是上游
Sequential FNN 使用旧版 Keras/Python 2 约定，而上游 Reaction-GCNN 依赖
Chainer 和固定的交叉偶联数据字典；它们不能直接接受本项目的六家族划分、
合并后的 catalyst/reagent 定义和 family-specific 标签空间。对应的模型思想
在 `external_adapters/` 中以 PyTorch 重实现，并明确使用本项目的训练集词表和
Stage 1 Non-Oracle route cache。

具体而言，`run_sequential_fnn.py` 实现 reaction fingerprint -> reagent ->
solvent -> temperature 的顺序结构；`run_reaction_gcnn.py` 使用本项目现有
`stage3_XGBoost/reaction_gnn_features.py` 的分子图编码器，并只接多标签
reagent/solvent 头，不使用 KNN、ReaFNN 或 XGBoost 信号。两者的预测均先生成
训练集见过的完整 condition context，再由 `export_candidates.py` 写入同一
candidate schema，以便使用同一 Non-Oracle evaluator。
其中 `test_manifest.jsonl` 来自正式 test split 的样本标识，而非 Stage 1 route
列表，因此 evaluator 会将 route cache 中没有候选的测试样本按零命中计入总分母。


Direct Transformer 已实现数据打包、beam 输出解析和候选表转换。上游公开链接
仅指向 IBM Box 模型集合，该链接在 2026-07-25 返回 404，因此未以不明来源权重
替代。即使恢复该公开 checkpoint，它也是 forward reaction-prediction 初始化，
仍须在本项目 product-to-system 序列上按家族微调后才可作为 Baseline 1。

---

# 9. 实施顺序

## Phase 0：统一基础设施

### 任务

- 固定 canonicalization
- 固定 label vocab
- 固定 candidate schema
- 固定 gold matching
- 固定 evaluation scripts
- 检查 Stage 1 route cache 完整性

### 验收条件

- ProSys 当前结果可以通过新 evaluator 复算
- 复算指标与官方结果一致
- 所有 family 的样本数一致
- route / context / system match 单元测试通过

---

## Phase 1：Baseline 2

优先实现 Baseline 2，因为：

- 工作量最小
- 反应指纹代码可复用
- 能快速验证统一 candidate pipeline
- 有温度输出

### 验收条件

- 六个 family 均可完成训练与推理
- top-k context 能稳定生成
- candidate CSV 格式通过 evaluator
- validation 上的权重搜索可复现
- test 全流程不读取 gold route

---

## Phase 2：Baseline 3

### 任务

- 建图
- Reaction-GCNN 编码
- reagent / solvent 多标签训练
- 全局历史 context 打分
- product-level candidate merge

### 验收条件

- 图构建失败率可统计
- 训练 loss 正常下降
- reagent / solvent label recall 高于频率先验
- 所有 family 均能输出 top-k systems

---

## Phase 3：Baseline 1

Baseline 1 工作量和风险最高，最后实现。

### 任务

- 构造角色标记序列
- 训练 Transformer
- beam decoding
- 解析和合法性过滤
- 统一系统评价

### 验收条件

- 输出格式有效率可接受
- route / reagent / solvent 可稳定解析
- beam 去重后每个样本有足够候选
- 至少完成 top-10 system 评价

---

# 10. 推荐时间安排

## 第 1–2 天：统一评价框架

- evaluator
- candidate schema
- matching tests
- ProSys 复算

## 第 3–5 天：Baseline 2

- 特征构建
- FNN 训练
- set decoding
- temperature
- 六家族测试

## 第 6–10 天：Baseline 3

- 图特征
- GCNN
- context scoring
- 六家族测试

## 第 11–16 天：Baseline 1

- 数据构建
- tokenizer
- Transformer 训练
- beam inference
- 解析与评估

## 第 17–18 天：统一调参与复跑

- validation-only tuning
- fixed seed rerun
- 汇总表
- error analysis

---

# 11. 当前可报告结果与主表设计

## 11.1 完整体系主结果表

| Method | Route@10 | Cover | System@1 | System@3 | System@5 | System@10 | nDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Product-to-System Transformer | Not reported | Not reported | Not reported | Not reported | Not reported | Not reported | Not reported | Not reported |
| EditRetro + Sequential FNN | 63.2 | 46.0 | 17.1 | 24.6 | 28.0 | 31.7 | 0.238 | 0.223 |
| EditRetro + Reaction-GCNN | 63.2 | 38.0 | 7.9 | 13.2 | 16.4 | 21.0 | 0.135 | 0.122 |
| ProSys current mainline | 63.2 | 49.2 | 27.6 | 35.6 | 38.5 | 42.7 | 0.341 | 0.327 |

注意：

- Direct Transformer remains unreported until a compatible checkpoint is validated.
- The first three reported methods share the same frozen Stage 1 route cache; their Route@10 is therefore identical by construction.

## 11.2 温度结果表

| Method | Temp MAE | Temp±5 °C | Temp±10 °C | Temp±20 °C | Support N |
|---|---:|---:|---:|---:|---:|
| EditRetro + Sequential FNN | 11.87 | 49.5 | 67.3 | 81.6 | 1,464 |
| EditRetro + Reaction-GCNN | N/A | N/A | N/A | N/A | N/A |
| ProSys current mainline | 11.28 | 38.0 | 61.6 | 83.0 | 1,603 |

---

# 12. 必做的公平性检查

## 12.1 数据泄漏检查

- train / valid / test canonical route overlap
- baseline context vocabulary 是否包含 test-only labels
- Direct Transformer tokenizer 是否由 test 数据构建
- 历史 context 库是否只来自 train
- 调参是否只使用 validation

## 12.2 输入一致性检查

Baseline 2、3 与 ProSys：

- 相同 product
- 相同 predicted route
- 相同 route rank
- 相同 route score
- 相同 route cache

## 12.3 输出一致性检查

- reagent set 顺序无关
- solvent set 顺序无关
- SMILES canonicalization
- 重复候选去重
- system score 从高到低排序
- 同分候选使用固定 tie-break

## 12.4 随机种子

每个模型至少：

```text
seed = 0, 1, 2
```

建议主表报告：

```text
mean ± standard deviation
```

若计算资源不足，至少主结果使用固定 seed，并对关键 baseline 做 3 次重复。

---

# 13. 错误分析计划

## 13.1 按阶段分解

对 Baseline 2、3 与 ProSys 分别统计：

- gold route absent
- gold context absent
- full system absent
- full system present but ranked below top-10

## 13.2 按反应家族分析

重点比较：

- Diels–Alder：Stage 1 route bottleneck
- Beckmann：condition candidate 与 ranking bottleneck
- Friedel–Crafts Alkylation：浅层 top-k 与深层 top-k 差异

## 13.3 Direct Transformer 特有错误

- invalid SMILES
- role tag missing
- reactant/reagent role confusion
- duplicated molecules
- incomplete condition segment
- correct route but wrong context
- correct context but wrong route

---

# 14. 论文中的写法

## 14.1 Baseline 段落

> We compared ProSys with three representative baselines covering distinct prediction paradigms. Product-to-Precursors Transformer directly generated reactants and condition species from the target product. EditRetro + Sequential FNN applied a reaction-fingerprint-based sequential condition predictor to the same Stage 1 routes used by ProSys. EditRetro + Reaction-GCNN used graph-based multilabel classification to predict reagent and solvent labels for each predicted route.

## 14.2 公平性说明

> For EditRetro + Sequential FNN, EditRetro + Reaction-GCNN, and ProSys, the same family-specific Stage 1 route caches were used. All models were trained and evaluated using identical reaction-family splits, label normalization rules, and complete-system matching criteria. Hyperparameters and score-fusion weights were selected exclusively on the validation sets.

## 14.3 方法差异说明

> The three baselines represent direct sequence generation, reaction-fingerprint-based sequential prediction, and graph-based multilabel prediction, respectively. In contrast, ProSys explicitly constructs a precedent-supported condition pool and performs system-level reranking using retrieval evidence, token-level feasibility filtering, structural representations, and learning-to-rank signals.

---

# 15. 最终交付物

必须产出：

```text
outputs/baselines/
├── baseline1_direct_transformer/
├── baseline2_sequential_fnn/
├── baseline3_reaction_gcnn/
├── summary/
│   ├── main_results.csv
│   ├── temperature_results.csv
│   ├── family_results.csv
│   ├── error_breakdown.csv
│   └── baseline_overview.md
```

论文层面交付：

- 1 张 baseline 主表
- 1 张温度表
- 1 张 family-level comparison 图
- 1 张 stage-wise error decomposition 图
- Supporting Information 中的超参数表
- Supporting Information 中的每家族完整结果

---

# 16. 最终验收标准

只有满足以下条件，baseline 才能进入论文主表：

- 使用相同数据划分
- strict Non-Oracle
- test 阶段不读取 gold route
- 输出可转换成统一 candidate table
- 使用相同 system match 定义
- 所有阈值和融合权重仅在 validation 上选择
- 六个 family 均有结果
- 结果可由统一 evaluator 复现
- 训练配置、随机种子、checkpoint 路径完整记录

---

# 17. 推荐优先级总结

推荐实施顺序：

```text
统一 evaluator
→ Baseline 2: Sequential FNN
→ Baseline 3: Reaction-GCNN
→ Baseline 1: Direct Transformer
→ 统一复跑与结果汇总
```

其中：

- Baseline 2 最容易落地
- Baseline 3 最适合验证 GNN 与 retrieval 的区别
- Baseline 1 最能形成方法范式上的对照，但实现风险最高
- 所有正式对比必须围绕 `System@k`，而不是只比较 reagent 或 solvent 的独立标签准确率
