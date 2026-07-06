# ProSys

`ProSys` 是一个从 **target product** 出发，推荐完整反应体系的框架。  
当前维护的主线已经固定为：

```text
product
-> Stage 1 EditRetro route generation
-> Stage 2 KNN candidate screening
-> Stage 3 XGBoost reranking + temperature prediction
-> Top-k reaction systems
```

当前项目的主要叙事是 **Non-Oracle end-to-end**。  
`Oracle` 只保留作历史分析和上限参考，不再是主结果入口。

## 当前主线

### Stage 1

- 目录：`stage1_retrosynthesis/`
- 作用：从目标产物生成候选路线，产出 `outputs/stage1_routes/<family>/route_cache.json`

### Stage 2

- 主线目录：`stage2_KNN/`
- 核心实现：`stage2_KNN/knn_condition_selector.py`
- 作用：对 Stage 1 预测路线做 family-specific KNN 检索，生成可行条件候选池

### Stage 3

- 主线目录：`stage3_XGBoost/`
- 核心实现：`stage3_XGBoost/xgb_reranker.py`
- 作用：对候选池重排，并预测温度

### 实验与报表

- 目录：`baseline/`、`ablation/`
- 当前主实验入口：
  - `baseline/run_non_oracle_stage23_experiments.py`
  - `baseline/render_stage23_nonoracle_reports.py`
  - `ablation/run_non_oracle_ablation.py`
  - `scripts/run_stage23_non_oracle_suite.sh`

## 当前保留的历史分支

下面这些内容仍然保留，但不再是主线：

- `Experiment/legacy_stage2/`
  - 原始 neural V2 分支归档
  - 现在主要为历史 baseline、旧模型包装、旧特征与评估工具提供兼容支持
- `prosys_shared/`
  - 从旧 `stage2/v2` 中抽出的当前共享工具层
  - 供 `stage2_KNN/`、`stage3_XGBoost/`、`baseline/` 复用
- `save_models/`
  - 原始 FNN / ReactionModel_LWTemp checkpoint
  - `Original FNN` baseline 仍依赖它们
- `scripts/run_full_pipeline.sh`
  - 旧的 neural-V2 Oracle 全流程入口
- `scripts/run_non_oracle_pipeline.sh`
  - 旧的 neural-V2 Non-Oracle 入口

## 当前推荐入口

### 1. 环境检查

```bash
conda activate ProSys
bash scripts/setup_prosys_env.sh
python scripts/audit_data_splits.py --strict
```

### 2. Stage 1 route cache

如果还没有 `route_cache.json`：

```bash
conda activate ProSys
python stage1_retrosynthesis/build_route_cache.py --repo_root . --family Beckmann
```

全家族 route cache 生成后，主线实验默认从 `outputs/stage1_routes/` 读取。

### 3. 当前主线实验套件

```bash
conda activate ProSys
bash scripts/run_stage23_non_oracle_suite.sh .
```

这条入口会完成：

1. `KNN + XGBoost` 主线
2. `Original FNN` historical baseline
3. Stage 3 ablation
   - `KNN + RF`
   - `KNN + SVM`
   - `KNN + Bayes`
4. Stage 2 ablation
   - `Cluster + XGBoost`
   - `FNN-pool + XGBoost`
5. 最终 markdown / csv / json 报表渲染

### 4. 只重渲染报表

```bash
conda activate ProSys
python baseline/render_stage23_nonoracle_reports.py \
  --output_root outputs/stage23_non_oracle_all10
```

## 当前主要结果位置

### 主线 + baseline + ablation

- 结果根目录：`outputs/stage23_non_oracle_all10/`
- 核心报表：
  - `overview.md`
  - `baseline_historical.md`
  - `ablation_stage2.md`
  - `ablation_stage3.md`
  - `average_effect.md`

## 当前结果摘要

更新时间：`2026-07-05`

当前正式结果以 `outputs/stage23_non_oracle_all10/` 为单一可信入口。  
本轮结果已经过以下处理：

- 重新执行了 Stage 1 之后的 Non-Oracle 全流程
- 重新渲染了 baseline / ablation / average-effect 报表
- 通过 `scripts/audit_data_splits.py --strict` 确认当前 split 无明显泄露
- 统一了温度指标口径：`Temp@10C` / `Temp@20C` 表示 top-10 内存在完整 system hit 且温度误差落在 `+/-10C` / `+/-20C`

### 10-family 全量主线概览

`KNN+XGB` 在全部 10 个家族上的宏平均为：

- `rr@10 = 41.9`
- `cover = 36.7`
- `sys@1 = 15.4`
- `sys@5 = 23.6`
- `sys@10 = 27.0`
- `Temp@10C = 9.2`
- `Temp@20C = 14.5`

与 `Original FNN` 相比，当前主线在全量 10 家族上更强的部分是：

- 候选池覆盖率更高
- end-to-end `sys@k` 更高

### 过滤后的主结果子集

为了与最终主表保持一致，当前 markdown 报表默认过滤为：

- `KNN+XGB sys@10 > 20%`

保留的 6 个家族为：

- `Beckmann`
- `Buchwald-Hartwig`
- `Chan-Lam`
- `Diels-Alder`
- `Friedel-Crafts Acyl.`
- `Friedel-Crafts Alkyl.`

在这 6 个家族上的宏平均：

- `Original FNN`: `cover 38.0`, `sys@1 10.5`, `sys@5 19.6`, `sys@10 28.3`, `Temp@10C 20.0`, `Temp@20C 24.1`
- `KNN+XGB`: `cover 52.2`, `sys@1 22.0`, `sys@5 33.7`, `sys@10 38.6`, `Temp@10C 13.1`, `Temp@20C 20.9`

也就是当前主线相对原始项目 baseline 的提升大致为：

- `cover +14.2 pp`
- `sys@1 +11.5 pp`
- `sys@5 +14.1 pp`
- `sys@10 +10.2 pp`

### 为什么主线定为 `KNN+XGB`

当前结论不是简单地“谁的某个 `sys@k` 最高就选谁”，而是更强调：

1. `KNN` 在 Stage 2 里承担 family-specific feasible-condition screening，解释最直接。
2. `XGBoost` 在 Stage 3 里承担 reranking + temperature prediction，工程成本低、可解释特征清晰、复现稳定。
3. Stage 2 ablation 中，`KNN+XGB` 明显强于 `Cluster+XGB` 和 `FNN-pool+XGB`，说明 `KNN` 这步筛选确实有用。
4. Stage 3 ablation 中，`KNN+SVM` 在过滤子集上的纯 `sys@k` 更高，但温度命中几乎为零，不适合作为完整反应体系推荐主线。

因此当前保留的叙事是：

- `KNN` 负责“找可行候选”
- `XGBoost` 负责“在可行候选里重排并给温度”
- `Original FNN` 作为历史 baseline

### 推荐阅读顺序

如果只想快速理解当前项目状态，建议按下面顺序看：

1. `README.md`
2. `outputs/stage23_non_oracle_all10/average_effect.md`
3. `outputs/stage23_non_oracle_all10/baseline_historical.md`
4. `outputs/stage23_non_oracle_all10/ablation_stage2.md`
5. `outputs/stage23_non_oracle_all10/ablation_stage3.md`
6. `baseline/non_oracle_reaudit_20260705.md`

### Stage 1 route cache

- `outputs/stage1_routes/`

### 旧 neural-V2 结果

- `outputs/stage2_v2/`

## 当前温度指标口径

当前 `Temp@10C` 和 `Temp@20C` 已统一改成：

- 在 top-10 内，存在一个候选同时满足：
- `route + reagent + solvent` 全命中
- 且该候选有有效温度标注
- 且 `|T_pred - T_gold| <= 10C` 或 `20C`

也就是说，温度命中率现在是**真正的 top-10 end-to-end 命中率**，分母是全部样本，不再是先条件化到 `sys@10` 命中的子集。

## 目录整理约定

### 主线长期保留

- `stage1_retrosynthesis/`
- `prosys_shared/`
- `stage2_KNN/`
- `stage3_XGBoost/`
- `baseline/`
- `ablation/`
- `scripts/`
- `outputs/stage1_routes/`
- `outputs/stage23_non_oracle_all10/`

### 历史参考但不属于当前主线

- 统一放到 `Experiment/`
- 例如：
  - notebook
  - 旧 baseline/oracle 输出
  - 归档的 neural-V2 代码
  - route-budget 分析
  - 旧渲染脚本和零散小工具

### 不保留的内容

- `__pycache__/`
- smoke 输出
- 重复中间产物

## 维护建议

- 新实验优先复用现有入口，不再新增零散脚本。
- 主线代码只围绕 `stage1 -> stage2_KNN -> stage3_XGBoost` 扩展。
- 旧 neural-V2 分支只做兼容维护，不再继续扩展成主实验。
- 新的分析性 notebook、临时脚本、旧结果，统一放 `Experiment/`，不要再堆在仓库根目录。
