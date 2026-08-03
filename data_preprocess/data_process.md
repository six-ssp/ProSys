# Reaxys 数据处理与划分

下面是对 **10 个 Reaxys 反应族**数据处理的完整说明。

## 快速使用

```bash
# 首次运行（含原始文件清理 + Stage 2 条件数据）
bash data_preprocess/run.sh

# 含 Stage 1 逆合成数据（需安装 rxnmapper）
bash data_preprocess/run.sh --stage1

# 数据已清理过，跳过原始清理
bash data_preprocess/run.sh --skip-clean --stage1

# 检查 Stage 1 / Stage 2 划分是否干净
python data_preprocess/audit_data_splits.py --strict
```

核心预处理逻辑在 `data_preprocess/preprocess.py`，切分泄露审计在 `data_preprocess/audit_data_splits.py`。
旧顶层 `preprocess_data/` 目录里仍有价值的标签清洗规则，现已并入这里的
`normalize_label()` 与 `build_name_to_smiles_table.py`；旧的 OPSIN / ChemSpider /
手工改名流水线不再作为当前主线依赖。

---

## 一、目标：产出两类数据

从 Reaxys 下载的原始 xlsx 出发，最终得到：

### 1. Stage 2 条件建模数据（用于 FNN、ReactionModel_LWTemp、XGBoost 重排）

- 每条记录为 `(reactants, product, reagent_set, solvent_set, temperature, yield)`
- 条件字段完整、标签干净
- 按 `canonical_reaction_smiles` 分组做 `8:1:1` 划分
- 输出到 `data/reaction_processed_{family}_catmerge`

### 2. Stage 1 逆合成路线微调数据（用于 EditRetro finetune）

- 每条记录为 `reactants >> product`（带原子映射）
- `val/test` 由 Stage 2 的 split 锚定
- `train` 额外吸收被 Stage 2 筛掉但反应本身合法的记录
- **任一路线微调数据都不能与 Stage 2 的测试反应重合**
- 输出到 `editretro/datasets/REAXYS_{FAMILY}_SINGLE_CATMERGE/raw/`

---

## 二、原始数据预清理（Step 0）

在进入 Stage 2 之前，先对从 Reaxys 下载的原始 `.xlsx` / `.csv` 文件做预清理。

### 执行

这一步已经内置到当前的 `data_preprocess/preprocess.py` / `data_preprocess/run.sh`
里；正常跑主预处理时会自动先执行原始文件清理，不再需要单独调用旧脚本。

### 清理规则

以下规则按顺序执行：

#### (1) 删除 Excel 临时锁定文件

删除所有 `~$*.xlsx`（Windows 下打开 Excel 时产生的临时文件），避免后续读取报错。

#### (2) 删除版权声明尾部行

Reaxys 导出的每个文件末尾有 2 行版权声明：

```
Disclaimer: please refer to our Terms and Conditions on authorized use of Reaxys data.
Copyright © 2026 Elsevier Life Sciences IP Limited ...
```

通过检查 `Reaction ID` 列是否包含 `Disclaimer` / `Copyright` / `Terms and Conditions` 关键词来定位并删除。

#### (3) 删除无用列

以下 4 列在建模中完全用不到，统一删除：

| 列名 | 说明 |
|---|---|
| `Reaction: Links to Reaxys` | Reaxys 页面链接（与 `Links to Reaxys` 不同列） |
| `Data Count` | 该反应在 Reaxys 中的记录总数 |
| `References` | 参考文献信息 |
| `Links to Reaxys` | 另一个 Reaxys 链接列 |

删除时只删实际存在的列（若某列已不存在则跳过），避免 `KeyError`。

#### (4) 仅保留单步反应

检查 `Number of Reaction Steps` 列，只保留值为 `1` 的行，删除所有多步反应。

> **原因**：多步反应的条件（试剂、溶剂、温度）通常是分阶段记录的，结构复杂且与单步反应不可比。Stage 2 模型只处理一步到位的反应。

### 输入/输出

```
输入：data/reaxys_input/<family>/*.xlsx (或 .csv)  ← Reaxys 原始导出
输出：同路径原地覆盖（已清理）
```

---

## 三、Stage 2 数据处理

### Step 1：从原始 xlsx 构建 Merged Reaxys Root

```
输入：data/reaction/<family>/*.xlsx  （10 个 family 的原始 Reaxys 导出文件）
输出：data/reaction_processed_reaxys_catmerge/  （merged root）
     data/reaction_processed_{family}_catmerge/  （family-specific root）
```

执行：

```bash
bash scripts/run_reaxys_catalystmerge_rebuild.sh
```

脚本内部会调用：

1. `scripts/rebuild_reaxys_processed_dataset.py` — 合并 10 个 family 的 xlsx，清洗后得到 merged root
2. `scripts/build_reaxys_family_processed_dataset.py` — 按 `data/reaxys_input/<family>/*.txt` 中的 ID 列表拆出 family-specific root

**xlsx 列名统一**：读取时自动处理以下别名，统一为内部字段：

| 内部字段 | 接受的别名 |
|---|---|
| `reaction_id` | `Reaction ID`, `CID` |
| `reaction_smiles` | `Reaction` |
| `reactants_smiles` | `reactants` |
| `product_smiles` | `products`, `product` |
| `yield` | `Yield (numerical)`, `Yield` |
| `solvent` | `Solvent (Reaction Details)`, `Solvent` |
| `reagent` | `Reagent` |
| `catalyst` | `Catalyst` |

### Step 2：数据清洗规则

以下规则按顺序执行，被筛掉的记录不会再进入 Stage 2。

#### (a) 反应合法性检查

1. `reactants_smiles` 或 `product_smiles` 缺失 → 尝试从 `reaction_smiles` 按 `>>` 拆分恢复；无法恢复则删除
2. RDKit 无法解析 `reactants_smiles` 或 `product_smiles` → 删除

> **原因**：Stage 2 的输入是反应指纹（Morgan fingerprint），反应本身必须可解析。

#### (b) yield、solvent 缺失

- 缺 `yield` → 删除
- `yield < 25` → 删除
- 缺 `solvent` → 删除

> **原因**：排序训练需要 yield 生成 relevance；条件推荐显式预测 solvent。另一个经验性处理是删除低产率（`yield < 25`）记录，以减少明显低可行性条件对主训练分布的干扰。

#### (c) 温度

- `temperature` 缺失 → 保留为 NaN
- 多阶段温度 → 取**最高温度**作为统一标签

#### (d) Catalyst 并入 Reagent

```text
Reagent_final = 原 Reagent + 原 Catalyst（合并后不再单独预测 catalyst）
```

> **原因**：原始 Reaxys 中 `Catalyst` 列不稳定，且很多金属催化剂本就写在 `Reagent` 中。合并后与 AHO 的条件定义一致。

#### (e) reagent / solvent 角色重分配

统计全局每个标签作为 reagent 和 solvent 的出现次数。若同时出现在两类中，统一归到频次更高的角色。重分配后同一记录中同一标签不能同时出现在 `reagent_set` 和 `solvent_set`。

> **原因**：避免模型在两个任务中重复学习同一实体，减少候选池歧义。

#### (f) 标签标准化

1. 大小写归一化，清理多余空格
2. 过滤无效值：`nan`, `none`, `not given`, `unknown`, `-`
3. 默认加载本地 `data_preprocess/name_to_smiles.tsv`，把俗名、缩写和拼写变体映射到统一 canonical token；该 token 可以是 canonical SMILES，也可以是规范标签（如 `neat (no solvent)`）
4. 多个名称映射到同一 canonical token 的视为同一标签
5. hydrate 后缀按无水形式归并（如 `sodium carbonate monohydrate` → `sodium carbonate`）

#### (g) 条件复杂度过滤

仅保留 `reagent_set 大小 ≤ 3` 且 `solvent_set 大小 ≤ 2` 的记录，删除条件过于复杂的。

#### (h) 低频标签过滤

默认按**各 family 内部**分别统计 reagent / solvent 标签频次，删除频次 `< 6` 的标签。若某记录的 reagent_set 或 solvent_set 因此变空，删除该记录。

> 说明：这些阈值可配置，但当前主线默认使用 `min_yield=25`、`scope=family`、`min_freq=6`。

#### (i) 条件记录去重

对 `(reaction_id, canonical_reaction_smiles, reagent_joined, solvent_joined)` 四元组去重，只保留一条。

### Step 3：按 canonical reaction 分组做 8:1:1 划分

这是**最关键的泄露控制**。

**构造 canonical reaction key**：

1. `reactants_smiles` 按 `.` 拆分 → 每个分子 canonicalize → 排序 → `.` 拼接
2. `product_smiles` 同样按 `.` 拆分 → 每个分子 canonicalize → 排序 → `.` 拼接（支持 A+B→C+D 等多产物反应）
3. 拼接为：`sorted_canonical_reactants >> sorted_canonical_products`

示例：`CCO.CC>>CCOCC.Br` → `CC.CCO>>Br.CCOCC`

**划分规则**：

- 同一个 `canonical_reaction_smiles` 的所有条件记录进入**同一个 split**
- 比例：`train : validate : test = 8 : 1 : 1`

> **后果**：若不按 canonical reaction 分组，同一反应不同条件的记录可能散落在 train 和 test，导致 Stage 2 条件推荐结果被高估。

### Step 4：输出目录结构

每个 family-specific root 最终包含：

```
data/reaction_processed_{family}_catmerge/
├── For_first_part_model/          # Stage 2A (FNN) 输入
│   ├── Splitted_first_train_labels_processed.txt
│   ├── Splitted_first_validate_labels_processed.txt
│   ├── Splitted_first_test_labels_processed.txt
│   └── Splitted_first_total_labels_processed.txt
├── For_second_part_model/         # Stage 2B (排序+温度) 输入
│   ├── Splitted_second_train_labels_processed.txt
│   ├── Splitted_second_validate_labels_processed.txt
│   ├── Splitted_second_test_labels_processed.txt
│   └── Splitted_second_test_labels_processed_valid_single_product.txt
├── label_processed/               # 标签类别
│   ├── class_names_reagent_labels_processed.pkl
│   └── class_names_solvent_labels_processed.pkl
└── summary.txt                    # 数据统计摘要
```

---

## 四、Stage 1 逆合成路线微调数据

### Step 5：构建 EditRetro 路线数据集

Stage 1 只建模 `product → reactants`，不需要 yield、solvent、温度。

**核心策略**：

1. **`val/test` 由 Stage 2 的 split 锚定** — 从 `processed_root/For_second_part_model/` 读取，确保 val/test 中 product→reactants 与 Stage 2 一致
2. **`train` 额外吸收被 Stage 2 筛掉的合法反应** — 从 `data/reaction/<family>/*.xlsx` 原始文件重新读取所有合法反应，即使缺 yield、缺 solvent、条件过复杂，只要 reactants 和 product 可解析，就可以加入 Stage 1 的 train
3. **零泄露约束** — 所有额外 train 反应必须与 selected `train/val/test` 的 `canonical_reaction_smiles` 做去重；与 selected `val/test` 重合的直接删除

> **一句话**：Stage 1 可以用更多反应做微调，但 val/test 仍由筛后的基准 split 锚定，额外训练反应绝不会和测试反应重合。

执行：

```bash
bash scripts/run_prepare_editretro_catmerge_route_datasets.sh
```

脚本内部流程：

1. 读取 `processed_root/For_second_part_model/` 的 `train/validate/test`
2. 提取 grouped unique reactions，构造 `reactants>>product`
3. 用 RXNMapper 做 atom mapping（batch_size=64）
4. 从原始 xlsx 抽取合法但被 Stage 2 筛掉的反应，只补到 train
5. 去重：排除与已选 train/val/test 重叠的 canonical reaction

输出到：

```
editretro/datasets/REAXYS_{FAMILY}_SINGLE_CATMERGE/raw/
├── raw_train.csv   # Stage 2 train + 额外 route-only 增广
├── raw_val.csv     # 与 Stage 2 val 一致
└── raw_test.csv    # 与 Stage 2 test 一致
```

row CSV 格式（EditRetro 可直接训练）：

| 列名 | 说明 |
|---|---|
| `pair_id` | 唯一标识 |
| `reaction_id` | 原始 Reaxys ID |
| `product_smiles` | 产物 SMILES（无映射） |
| `reactants>reagents>production` | 带原子映射的完整反应 |
| `mapped_reaction_smiles` | 原子映射后的反应 |
| `mapping_confidence` | RXNMapper 置信度 |

---

## 五、完整流程总结

```text
                       Reaxys xlsx/csv (10 families)
                              │
                              ▼
              ┌──────────────────────────────┐
              │  clean_reaxys_input.py        │  ← Step 0: 预清理
              │  → 删除版权尾部行              │
              │  → 删除无用列                  │
              │  → 仅保留单步反应              │
              └──────────┬───────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │  preprocess.py               │  ← Stage 2 预处理
              │  → (a) 合法性检查             │
              │  → (b) 温度处理               │
              │  → (c) Catalyst 并入 Reagent │
              │  → (d) 试剂/溶剂角色重分配     │
              │  → (e) 标签标准化             │
              │  → (f) 条件复杂度过滤          │
              │  → (g) 低频标签过滤           │
              │  → (h) 条件记录去重           │
              │  → (i) 按 canonical rxn 8:1:1│
              └──────────┬───────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    data/reaction_processed    editretro/datasets/
    _{family}_catmerge/        REAXYS_{FAMILY}_SINGLE
    (Stage 2 训练数据)            _CATMERGE/raw/
    ├── For_first_part        (Stage 1 训练数据)
    ├── For_second_part         ├── raw_train.csv
    ├── label_processed/        ├── raw_val.csv
    └── summary.txt             └── raw_test.csv
              │                     │
              ▼                     ▼
    ┌─────────────────┐   ┌──────────────────┐
    │  FNN 候选池训练   │   │  EditRetro 微调    │
    │  ReactionModel_  │   │  (product→reactants)│
    │  LWTemp 排序训练  │   └──────────────────┘
    │  XGBoost 重排    │            │
    └─────────────────┘            │
              │                     │
              ▼                     ▼
    ┌──────────────────────────────────────┐
    │         端到端评估                      │
    │  product → EditRetro → 条件预测 → 重排  │
    └──────────────────────────────────────┘
```

## 六、泄露控制清单

| 检查项 | 规则 |
|---|---|
| Stage 2 split | 按 `canonical_reaction_smiles` 分组，同一反应的所有条件记录在同一 split |
| Stage 1 val/test | 由 Stage 2 split 锚定，不额外引入 |
| Stage 1 train 增广 | 与 selected train/val/test 严格去重 |
| USPTO SAFE | 按产品级别过滤，移除与 Stage 2 test 产品重叠的记录 |

**验收**：Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0, USPTO train ∩ Stage 2 test products = 0
