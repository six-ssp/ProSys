# Historical Baseline: 具体实现说明

更新日期：`2026-07-05`

## 1. 这份文档解释什么

这份文档只解释一个对象：

- `Original Prototype-FNN`

也就是当前项目里保留下来的历史后半段系统：

- 候选池来自原始 FNN candidate generation
- 排序来自原始 ranking head
- 温度来自原始 temperature head
- Stage 1 固定使用现在已经生成好的 `route_cache.json`

它不是模块替换实验，不和 `RF / SVM / Bayes / Cluster / FNN-pool+XGB` 混在一起讲。


## 2. 相关代码文件

核心代码在下面几个文件里：

- `baseline/run_non_oracle_stage23_experiments.py`
  - 当前主实验总入口
  - 其中 `_run_prototype_fnn(...)` 是 historical baseline 的真正执行函数
- `baseline/run_oracle_baselines.py`
  - 提供 `_legacy_reaction_fp(...)`
  - 提供 `_load_legacy_evaluators(...)`
  - 负责把历史 checkpoint 包装成当前可调用对象
- `baseline/legacy_models.py`
  - 解决 `torch>=2.6` 下旧 checkpoint 的加载兼容问题
- `baseline/common.py`
  - 负责 candidate table 打标
  - 负责最终 `sys@k / Temp@10C / Temp@20C` 评估
- `Experiment/legacy_stage2/`
  - 归档的原始 neural-V2 / FNN 代码


## 3. 输入依赖

Historical baseline 运行时依赖下面几类输入。

### 3.1 Stage 1 路线缓存

每个 family 需要：

- `outputs/stage1_routes/<family>/route_cache.json`

它提供：

- `sample_index`
- `reaction_id`
- `product`
- Stage 1 预测的多条 `reactants`
- `retro_rank / retro_score / retro_probability`

这保证 baseline 和主线在完全同一个 Non-Oracle 设定下比较。


### 3.2 历史 checkpoint

当前默认 checkpoint 路径写死在 `baseline/run_oracle_baselines.py`：

- `save_models/test_10R_first_local_10/multitask_model_epoch-80.checkpoint`
- `save_models/test_10R_second_7/rxn_model_relevance_listwise_morgan_epoch-80.checkpoint`

前者负责：

- FNN 多标签条件预测

后者负责：

- 历史 ranking + temperature 预测


### 3.3 历史类别字典

还依赖：

- `data/reaxys_output/label_processed/class_names_solvent_labels_processed.pkl`
- `data/reaxys_output/label_processed/class_names_reagent_labels_processed.pkl`

它们定义了历史模型输出空间里的：

- solvent classes
- reagent classes


### 3.4 Test gold split

为了最后打标和评估，需要：

- `data/reaction_processed_<family>_catmerge/For_second_part_model/Splitted_second_test_labels_processed.txt`

它提供真实：

- 反应路线
- reagent / solvent
- yield
- temperature


## 4. 历史模型是怎么加载的

### 4.1 兼容包装

`baseline/legacy_models.py` 提供两个包装类：

- `LegacyMultiTaskEvaluator`
- `LegacyRankingEvaluator`

它们本质上继承自归档代码里的：

- `MultiTask_Evaluator`
- `Ranking_Evaluator`

但把 `torch.load(..., weights_only=False)` 和旧参数恢复过程重新封装了一层，保证旧 checkpoint 在当前环境还能直接用。


### 4.2 加载流程

`_load_legacy_evaluators(repo_root, with_ranker=True)` 会做下面几件事：

1. 读取 solvent / reagent class 名称
2. 构建 `LegacyMultiTaskEvaluator`
3. 恢复 multitask FNN checkpoint
4. 如果 `with_ranker=True`
   - 再构建 `LegacyRankingEvaluator`
   - 恢复 ranking checkpoint

其中历史 cutoff 也被显式固定为旧版本设定：

- `cutoff_solv = 0.25`
- `cutoff_reag = 0.3`
- `max_solv = 11`
- `max_reag = 11`

这样做的目的是尽可能忠实复现原型系统，而不是让它偷偷混入当前主线的后续调参。


## 5. Historical baseline 的完整数据流

真正的 Non-Oracle 执行函数是：

- `baseline/run_non_oracle_stage23_experiments.py::_run_prototype_fnn(...)`

它的流程可以拆成 6 步。

### 5.1 读取 Stage 1 预测路线

先用：

- `prosys_shared.route_cache.load_route_records_from_cache(...)`

把 `route_cache.json` 展开成一组 route records。

这里有一个重要约定：

- 同一个测试样本下的多条路线共用同一个 `sample_index`

后面排序评估就是按这个 `sample_index` 分组做 slate ranking。


### 5.2 把每条路线编码成历史模型输入

对每条 `(reactants, product)`：

1. 调用 `_legacy_reaction_fp(...)`
2. 优先使用历史函数 `create_rxn_Morgan2FP_concatenate(...)`
3. 如果历史函数失败，再回退到当前共享实现 `reaction_morgan_fp(...)`

这样可以尽量贴近原始 FNN 的输入定义，同时保留当前代码对异常 SMILES 的容错。


### 5.3 用 multitask FNN 生成候选条件空间

接着调用：

- `mt.make_input_rxn_condition(rxn_fp)`

得到：

- 候选 solvent 集合
- 候选 reagent 集合

这一步相当于 historical Stage 2。


### 5.4 用历史 ranking head 给条件排序并预测温度

然后调用：

- `rk.rank_top_contexts(rxn_fp, input_solvents, input_reagents)`

它会输出一个排序后的列表，每一项包含：

- `solvent_norm`
- `reagent_norm`
- `temp_pred`
- `score`

最后只保留前 `legacy_max_contexts` 个，默认是：

- `200`


### 5.5 写出原始 scored candidate table

每个候选 row 会写入：

- 路线信息
- `reagent_norm`
- `solvent_norm`
- `legacy_score`
- `legacy_temperature_pred`
- `legacy_rank`
- `from_fnn = 1`

输出文件位置：

- `<output_root>/<family>/prototype_fnn/non_oracle/candidate_pool_test_scored.csv`


### 5.6 对 candidate table 打标并评估

然后用：

- `baseline.common.label_candidate_table(...)`

把上一步候选表和真实 test split 对齐，生成：

- `route_match`
- `context_match`
- `label`
- `label_type`
- `rank_relevance`
- `temperature_gold`
- `yield_gold`

输出文件：

- `<output_root>/<family>/prototype_fnn/non_oracle/test_scored_labeled.csv`

最后再调用：

- `baseline.common.evaluate_scored_frame(...)`

使用：

- `score_column = legacy_score`
- `temperature_column = legacy_temperature_pred`

得到最终指标。


## 6. 打标逻辑是什么

为了和主线完全一致，historical baseline 不是沿用旧项目自己的评估脚本，而是走当前统一打标逻辑。

### 6.1 `route_match`

候选路线规范化后是否命中真实路线。

### 6.2 `context_match`

候选 `(reagent_norm, solvent_norm)` 是否命中真实条件对。

### 6.3 `label`

只有路线和条件同时命中才算 exact positive：

- `label = 1`

这也是最终 `sys@k` 的命中定义。

所以现在 historical baseline 和主线在评价口径上是严格统一的，差别只来自模型本身，不来自评估规则。


## 7. 指标是怎么统计的

### 7.1 Stage 1 背景指标

先统计：

- `rr@1`
- `rr@3`
- `rr@5`
- `rr@10`

实现函数：

- `baseline.run_non_oracle_baselines.stage1_route_recall(...)`

它只回答：

- Stage 1 的真实路线有没有出现在 top-k 预测里

它不是后半段模型的能力指标，只是 end-to-end 的背景上限。


### 7.2 候选池覆盖

`cover` 现在对应：

- `pool_coverage`

定义是：

- 一个 slate 里是否至少存在一条 exact-positive candidate row

也就是说：

- 不是“只命中路线”
- 不是“只命中条件”
- 必须 route + condition 同时命中


### 7.3 `sys@k`

对每个 `sample_index`：

1. 按 `legacy_score` 降序排序
2. 看 top-k 里是否有 `label=1`

统计得到：

- `system_top1_all`
- `system_top3_all`
- `system_top5_all`
- `system_top10_all`


### 7.4 温度指标

温度指标现在统一成：

- `Temp@10C`
- `Temp@20C`

并且只在下面条件同时满足时才统计：

1. 该样本 top-10 内已经出现 exact-positive system
2. 该 positive row 有有效 `temperature_gold`
3. 该 row 也有有效 `legacy_temperature_pred`

统计对象是：

- top-10 里排名最高的 exact-positive row

因此温度口径现在是：

- conditioned on `sys@10` hit

这和主线 XGBoost 以及所有 ablation 完全一致。


## 8. 输出目录结构

对每个 family，historical baseline 结果写到：

- `outputs/stage23_non_oracle_all10/<family>/prototype_fnn/non_oracle/`

里面核心文件有：

- `candidate_pool_test_scored.csv`
- `test_scored_labeled.csv`
- `result.json`

全家族聚合后，会进入：

- `outputs/stage23_non_oracle_all10/all_results.json`
- `outputs/stage23_non_oracle_all10/results_flat.csv`
- `outputs/stage23_non_oracle_all10/baseline_historical.md`
- `outputs/stage23_non_oracle_all10/average_effect.md`


## 9. 怎么运行

### 9.1 跑全套主实验

最常用的是直接跑：

```bash
conda activate ProSys
bash scripts/run_stage23_non_oracle_suite.sh .
```

这会同时跑：

- historical baseline
- 主线 KNN+XGB
- Stage 2 ablation
- Stage 3 ablation


### 9.2 只跑 historical baseline

可以直接调用统一实验脚本：

```bash
conda activate ProSys
python baseline/run_non_oracle_stage23_experiments.py \
  --repo_root . \
  --families all \
  --output_root outputs/stage23_non_oracle_all10 \
  --run_set historical
```


## 10. 这份 baseline 的作用边界

这份 baseline 只服务于一个问题：

- 当前主线相对于历史原型系统提升了多少

它不用于解释：

- 为什么选 KNN 而不是 Cluster
- 为什么选 XGBoost 而不是 RF / SVM / Bayes

这些问题都属于 ablation。

所以在最终汇报里，historical baseline 最合适的角色是：

- 一条强历史参照线
- 一张 “Original FNN vs KNN+XGB” 的主对比表
