# ProSys

**ProSys: A Product-to-System Framework for Target Product-Driven Reaction System Recommendation**

## 项目目标

ProSys 的目标不是只预测单个条件标签，而是从目标产物直接推荐完整反应体系：

```text
(product, reactants, reagent set, solvent set, temperature)
```

当前主线固定为两阶段：

```text
target product
-> Stage 1 逆合成路线生成
-> Stage 2A 候选条件池生成
-> Stage 2B 路线-条件联合排序与温度预测
-> Top-k 完整反应体系
```

## 当前目录约定

统一只维护 ProSys 主线，新增运行入口集中到下面几处：

```text
scripts/
  check_runtime.py              # 运行环境预检
  setup_prosys_env.sh           # ProSys 环境检查 / torch 修复 / fairseq 扩展检查
  prepare_prosys_cuda121_overlay.sh
                               # 用本机缓存离线构建 workspace-local CUDA overlay
  run_in_prosys_cuda121_overlay.sh
                               # 在 ProSys + overlay 组合运行时中执行命令
  libittnotify_stub.c          # PyTorch ITT profiling no-op stub
  audit_data_splits.py          # Stage 1 / Stage 2 数据划分审计
  run_stage2_v2_family_batch.py # Stage 2 V2 全家族批处理入口

stage1/scripts/
  ensure_fairseq_extensions.sh  # fairseq 扩展构建与检查
  run_family_finetune_one.sh    # Stage 1 单家族训练入口
  run_family_finetune_batch.sh  # Stage 1 多家族并行训练入口

stage2/
  stage2_detail.md              # Stage 2 需求文档
  train_stage2_v2.py            # Stage 2 V2 单实验训练入口
  v2/                           # Stage 2 V2 核心实现
```

日常开发尽量不要再新增零散脚本；重复操作优先归并到上述入口。

## 环境准备

默认直接使用已有 `ProSys` 虚拟环境。

```bash
conda activate ProSys
bash scripts/setup_prosys_env.sh
```

如果当前 `torch` 装错成 CPU 版或包状态混乱，直接在 `ProSys` 环境里强制重装：

```bash
conda activate ProSys
FORCE_TORCH_REINSTALL=1 bash scripts/setup_prosys_env.sh
```

这个入口会统一做三件事：

1. 运行 `scripts/check_runtime.py` 检查依赖、CUDA、`fairseq` 扩展状态。
2. 尝试解析 `CUDA_HOME`。
3. 调用 `stage1/scripts/ensure_fairseq_extensions.sh` 构建并检查 `fairseq.libnat`。

如果当前会话不能直接写 `/home/six_ssp/miniconda3/envs/ProSys`，或者原环境里 `torch` 已经被 CPU 版覆盖，可以改用仓库内的 workspace-local CUDA overlay：

```bash
bash scripts/prepare_prosys_cuda121_overlay.sh
scripts/run_in_prosys_cuda121_overlay.sh "$(pwd)" \
  /home/six_ssp/miniconda3/envs/ProSys/bin/python scripts/check_runtime.py --repo_root .
```

这条路径的设计目标是：

- 保留 `ProSys` 环境里的其他依赖不变
- 只在仓库内覆盖 `torch / torchvision / torchaudio` 和必要 CUDA 运行库
- 避免直接污染或修改不可写的原始 conda env

当前已验证这套 overlay 能导入：

- `torch 2.5.1`
- `torchvision 0.20.1`
- `torchaudio 2.5.1`

并且 `torch.backends.cuda.is_built()` 为 `True`、`torch.version.cuda` 为 `12.1`。

注意：

- 如果当前 shell / 沙箱本身屏蔽了 GPU 设备，`torch.cuda.is_available()` 仍然会是 `False`。
- 这时问题不在包本身，而在运行时对 `/dev/nvidia*` 的访问权限。

## 数据划分审计

开始训练前先审计一次数据划分，避免 Stage 1 / Stage 2 泄漏：

```bash
conda activate ProSys
python scripts/audit_data_splits.py --strict
```

## Stage 1 训练

### 单家族

```bash
conda activate ProSys
PYTHON_BIN="$(which python)" \
GPU_ID=0 \
stage1/scripts/run_family_finetune_one.sh "$(pwd)" REAXYS_Beckmann_SINGLE_CATMERGE
```

如果需要强制通过 overlay 运行：

```bash
scripts/run_in_prosys_cuda121_overlay.sh "$(pwd)" \
  bash stage1/scripts/run_family_finetune_one.sh "$(pwd)" REAXYS_Beckmann_SINGLE_CATMERGE
```

### 全家族并行

```bash
conda activate ProSys
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
RESULTS_ROOT="$(pwd)/stage1/results/family_finetune" \
stage1/scripts/run_family_finetune_batch.sh "$(pwd)"
```

如果需要强制通过 overlay 运行：

```bash
scripts/run_in_prosys_cuda121_overlay.sh "$(pwd)" \
  bash stage1/scripts/run_family_finetune_batch.sh "$(pwd)"
```

说明：

- `GPU_IDS` 控制家族训练并发槽位，一个槽位对应一个 family 训练进程。
- 默认会先检查 `torch.cuda.is_available()`；若没有 CUDA，会拒绝启动全量训练。
- 只做 CPU 烟雾测试时，显式设置 `ALLOW_CPU_STAGE1=1`。
- 每个 family 的日志和 checkpoint 会落到 `stage1/results/family_finetune/<dataset>/<timestamp>/`。

## Stage 2 V2 训练

统一使用 `scripts/run_stage2_v2_family_batch.py`，该入口已支持：

- family 过滤
- candidate pool 预处理并行
- 按设备槽位并行训练多个 family
- 已生成 artifact 自动复用
- 按需 `force rebuild / retrain`
- 每个 family 单独日志

示例：

```bash
conda activate ProSys
python scripts/run_stage2_v2_family_batch.py \
  --repo_root . \
  --families all \
  --output_root outputs/stage2_v2 \
  --candidate_device cpu \
  --parallel_preprocess 8 \
  --train_devices cuda:0,cuda:1 \
  --parallel_train 2 \
  --epochs 30 \
  --slates_per_batch 8 \
  --train_num_workers 4
```

如果需要强制通过 overlay 运行：

```bash
scripts/run_in_prosys_cuda121_overlay.sh "$(pwd)" \
  /home/six_ssp/miniconda3/envs/ProSys/bin/python scripts/run_stage2_v2_family_batch.py \
    --repo_root . \
    --families all \
    --output_root outputs/stage2_v2 \
    --candidate_device cpu \
    --parallel_preprocess 8 \
    --train_devices cuda:0 \
    --parallel_train 1
```

常用补充参数：

- `--fnn_checkpoint_pattern outputs/fnn/{family}/best_model.pt`
- `--force_rebuild_memory`
- `--force_rebuild_candidates`
- `--force_rebuild_tables`
- `--force_retrain`
- `--max_train_slates N`
- `--max_val_slates N`

输出位置：

- `outputs/stage2_v2/<family>/memory/`
- `outputs/stage2_v2/<family>/candidate_pool/`
- `outputs/stage2_v2/<family>/training_tables/`
- `outputs/stage2_v2/<family>/train/`

其中训练日志固定写到：

```text
outputs/stage2_v2/<family>/train/train.log
```

## 文件管理约定

- 重要决策、环境问题、实际运行结论记到 `log.md`。
- 仅保留仍需推进的目标在 `todo.md`，完成项及时标记或移到已完成区。
- 临时缓存、`__pycache__`、一次性 smoke 输出应及时清理。
- 稳定产物统一放到 `outputs/`、`stage1/results/` 等固定目录，不要散落到仓库根目录。

## 建议阅读顺序

1. `ProSys_goal.md`
2. `data_preprocess/data_process.md`
3. `stage1/stage1_detail.md`
4. `stage2/stage2_detail.md`
5. `README.md`
6. `log.md`
7. `todo.md`
