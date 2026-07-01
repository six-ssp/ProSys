# ProSys 项目总目标

## 1. 这个项目要解决什么问题

ProSys 要解决的是一个比传统条件推荐更完整的问题：

**只给定目标产物 SMILES，自动推荐一套可执行的反应体系。**

这里的“完整反应体系”不是只给一个条件标签，而是尽量覆盖化学家真正关心的整条决策链，包括：

- 可行反应路线
- 对应反应物 `reactants`
- 试剂集合 `reagent set`
- 溶剂集合 `solvent set`
- 反应温度 `temperature`

也就是说，项目的目标不是做一个单点分类器，而是做一个从 **product 到 system** 的完整推荐框架。

## 2. 最终要实现什么

最终希望实现一个统一框架：

```text
目标产物
  -> 逆合成路线生成
  -> 候选条件池生成
  -> 候选条件打分与温度预测
  -> 候选级最终重排
  -> 输出 Top-k 完整反应体系
```

最终输出给用户的每个候选结果，至少应包含：

- `product`
- `predicted reactants`
- `predicted reagent set`
- `predicted solvent set`
- `predicted temperature`
- `overall score / rank`

项目的最终形态应当支持：

1. **Oracle 评估**
   - 已知真实反应，只评估后半段条件推荐能力
2. **Non-Oracle 评估**
   - 从目标产物直接出发，评估端到端完整链路
3. **按 family 拆分建模**
   - 每个反应家族使用自己的 Stage-2 pool / ranker / reranker

## 3. 核心方法是什么

当前统一方法主线是：

```text
EditRetro
  + family-specific FNN candidate pool
  + ReactionModel_LWTemp
  + XGBoost reranking
```

各部分职责如下：

### Stage 1：EditRetro

- 输入：目标产物
- 输出：`top-k` 候选反应路线
- 作用：把 product-only 问题转换为若干候选 `reactants >> product`

### Stage 2A：FNN 候选池生成

- 输入：反应 `reactants + product`
- 输出：候选 `reagent set / solvent set`
- 作用：尽量让真实高质量条件进入候选池

### Stage 2B：ReactionModel_LWTemp

- 输入：反应指纹 + 候选条件
- 输出：
  - 每个候选的基础分数
  - 每个候选的温度预测
- 作用：完成候选级排序建模，并承担温度预测

## 3. 为什么要这样做

这样设计不是为了堆模块，而是因为问题本身天然分层：

1. **先决定反应怎么做**
   - 这是路线层问题
2. **再决定这条反应用什么条件做**
   - 这是条件层问题
3. **最后在所有路线-条件组合里选最优体系**
   - 这是全局重排问题

如果把这三层强行压成一个单模型，问题会同时变得：

- 数据稀疏
- 标签组合爆炸
- 可解释性差
- 很难定位误差来源

因此，ProSys 的本质是一个 **分阶段、可诊断、可替换模块的反应体系推荐框架**。

## 4. 整个项目要怎么做

项目推进应按下面五步走，不要混着做。

### 第一步：把数据处理做干净

目标是把 Reaxys 原始记录整理成两类标准化数据：

1. **Stage 2 条件建模数据**
   - `(reactants, product, reagent_set, solvent_set, temperature, yield)`
2. **Stage 1 路线微调数据**
   - `reactants >> product`

这一步的关键不是“尽可能多保留数据”，而是：

- 字段统一
- 标签统一
- `Catalyst -> Reagent` 合并
- 泄露控制
- train/valid/test 按 canonical reaction 分组

### 第二步：分别训练路线模型和条件模型

### 路线模型

- 用 USPTO + SAFE 过滤数据做基础训练
- 再做 family-specific 微调
- 目标是提高各家族的 route recall

### 条件模型

- 每个 family 独立训练：
  - FNN 候选池模型
  - ReactionModel_LWTemp 排序 + 温度模型

目标不是单看 Top-1，而是同时兼顾：

- 候选池覆盖率
- Top-1 / 3 / 5 / 10
- 温度误差

### 第三步：把路线和条件真正接起来

把 Stage 1 和 Stage 2 接成完整链路：

```text
product
-> top-k routes
-> per-route context candidates
-> route-context joint reranking
-> final top-k systems
```

这一层是 ProSys 和传统条件推荐工作最核心的区别。

### 第四步：做两套评估

### Oracle

- 用真实反应输入后半段
- 评估条件模块上限

### Non-Oracle

- 用 EditRetro 预测路线
- 评估端到端真实性能

同时必须支持：

- overall 指标
- per-family 指标
- top-1 / 3 / 5 / 10
- candidate pool coverage
- 温度指标

### 第五步：形成可汇报、可复现、可维护的项目

最终不只是得到一批实验结果，而是要形成：

1. 一条清晰的方法主线
2. 一套能复跑的数据处理流程
3. 一套能复跑的训练与评估脚本
4. 一份逻辑自洽的 summary / paper narrative

## 6. 成功标准

这个项目最终至少要满足下面几个标准：

### 方法层

- 明确证明 product-only 设定下可以做完整反应体系推荐
- 明确证明 family-specific Stage-2 比混合通用池更合理
- 明确证明对比其他baseline（后续自建），当前项目效果更好

### 工程层

- 数据处理有固定入口
- 训练有固定入口
- Oracle / Non-Oracle 评估有固定入口
- 关键结果能从代码和数据复现出来
- 写好指令的.sh脚本，不要硬编码本地路径


### 结果层

- 能稳定输出端到端 `Top-k` 完整反应体系
- 能对 10 个 family 分别给出结果
- 能同时汇报路线命中、条件命中、端到端命中、温度表现

## 7. 当前项目边界

当前项目先聚焦在以下范围：

- 输入是目标产物 SMILES
- 输出是单步反应体系推荐
- 核心数据来自 USPTO + Reaxys family 数据
- 主实验围绕 10 个 reaction family 展开

当前**不优先**解决的问题包括：

- 多步全路线规划
- 实验成本 / 安全性 / 绿色化学多目标优化
- 大模型自然语言解释
- 在线交互式 synthesis planning 系统

这些可以作为后续扩展，但不应干扰当前主线。

## 8. 一句话版本

**ProSys 的总目标，是构建一个从目标产物直接到完整反应体系的 family-aware 推荐框架，并用可复现的数据、训练和评估流程证明这条路线是成立的。**
