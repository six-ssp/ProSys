# -*- coding: utf-8 -*-
"""
Reaxys 反应数据全流程预处理 —— 按 data_process.md 规范执行。

══════════════════════════════════════════════
  Step 0：原始文件预清理（删除版权行/无用列/多步反应）
  Stage 2：条件建模数据
    (a) 合法性检查：反应物/产物 SMILES 必须可解析；缺失产率/溶剂 → 删除
    (b) 产率过滤：缺失删除；默认删除 yield < 25 的记录
    (c) 温度处理：多阶段 → 取最高温；缺失 → 保留为 NaN
    (d) 催化剂并入试剂列
    (e) 试剂/溶剂角色重分配（按全局频次）
    (f) 标签标准化：大小写归一化、去空格、过滤哨兵值、
        水合物后缀归并、可选 name→SMILES 合并
    (g) 条件复杂度过滤：试剂 ≤ 3，溶剂 ≤ 2
    (h) 低频标签过滤：全局频次 < 10 → 删除该条记录
    (i) 按四元组去重：(reaction_id, canonical_reaction_smiles, reagent, solvent)
    (j) 按 canonical_reaction_smiles 分组做 8:1:1 划分
  Stage 1：逆合成路线数据（--do_stage1）
    → train/val 由 Stage 2 split 锚定
    → train 额外吸收被 Stage 2 筛掉但反应合法的记录
    → RXNMapper 原子映射（仅 train + val，不需要 test）
══════════════════════════════════════════════

Stage 2 输出（每个反应族）：
  data/reaction_processed_{family}_catmerge/
  ├── For_first_part_model/          # FNN 候选生成模型
  ├── For_second_part_model/         # 排序 + 温度回归
  ├── label_processed/               # 标准化标签类别文件
  └── summary.txt

Stage 1 输出（每个反应族）：
  data/editretro/datasets/REAXYS_{family}_SINGLE_CATMERGE/raw/
  ├── raw_train.csv                  # train + 额外增广
  └── raw_val.csv                    # 与 Stage 2 val 一致

用法：
  # 仅 Stage 2
  python preprocess.py --input_dir ../data/reaxys_input --output_dir ../data

  # Stage 2 + Stage 1
  python preprocess.py --input_dir ../data/reaxys_input --output_dir ../data --do_stage1

  # 带 name→SMILES 映射
  python preprocess.py --input_dir ../data/reaxys_input --output_dir ../data \\
      --name_to_smiles mapping.tsv

  # 数据已清理，跳过 Step 0
  python preprocess.py --input_dir ../data/reaxys_input --output_dir ../data \\
      --skip_raw_clean
"""

import os
import re
import math
import json
import pickle
import argparse
import itertools
import multiprocessing
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from sklearn.utils import shuffle

# 关闭 RDKit 的警告日志，避免刷屏
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────────────────

# reaxys_input/ 下的反应族子目录名列表
REACTION_TYPES = [
    'Beckmann',                          # Beckmann 重排
    'Buchwald-HartwigCross-Coupling',   # Buchwald-Hartwig 交叉偶联
    'Chan_LamCoupling',                  # Chan-Lam 偶联
    'DielsAlder',                        # Diels-Alder 反应
    'Friedel-CraftsAcylation',           # Friedel-Crafts 酰基化
    'Friedel-CraftsAlkylation',          # Friedel-Crafts 烷基化
]

# Stage 2 从原始数据中加载的列（其余列全部丢弃）
KEEP_COLUMNS = [
    'Reaction ID',                              # Reaxys 反应编号
    'Reaction',                                  # 反应 SMILES (reactants>>products)
    'Temperature (Reaction Details) [C]',         # 反应温度（摄氏度）
    'Yield (numerical)',                          # 产率（数值）
    'Reagent',                                    # 试剂
    'Catalyst',                                   # 催化剂
    'Solvent (Reaction Details)',                 # 溶剂
    'Number of Reaction Steps',                   # 反应步数（用于过滤单步反应）
]

# 视为缺失值的哨兵字符串（大小写不敏感）
SENTINEL_VALUES = {'nan', 'none', 'not given', 'unknown', '-'}

DEFAULT_NAME_TO_SMILES_PATH = Path(__file__).resolve().with_name('name_to_smiles.tsv')
DEFAULT_MIN_LABEL_FREQ = 6
DEFAULT_LABEL_FREQ_SCOPE = 'family'
DEFAULT_MIN_YIELD = 25.0
VALID_LABEL_FREQ_SCOPES = {'global', 'family'}

# ──────────────────────────────────────────────────────
# 1. 文件读写
# ──────────────────────────────────────────────────────

def load_file(file_path: str) -> pd.DataFrame:
    """
    加载 .xlsx 或 .csv 文件，只保留需要的列。
    CSV 文件自动尝试多种编码（utf-8 → cp1252 → latin-1 → ISO-8859-1）。
    """
    if file_path.endswith('.csv'):
        df = None
        for enc in ['utf-8', 'cp1252', 'latin-1', 'ISO-8859-1']:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if df is None:
            raise ValueError(f'无法解码文件 {file_path}')
    else:
        df = pd.read_excel(file_path, engine='openpyxl')
    # 只保留 KEEP_COLUMNS 中实际存在的列
    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    return df[keep]


def load_name_to_smiles(path: str) -> dict:
    """
    加载 name→canonical token 映射表（可选，用于标签标准化）。
    文件格式：每行  name<TAB>canonical_token
    其中 canonical_token 可以是 canonical SMILES，也可以是归并后的规范标签。
    返回 {小写名称: canonical_token} 字典。
    """
    mapping = {}
    if not path or not os.path.exists(path):
        return mapping
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                name, smi = parts[0].strip(), parts[1].strip()
                if smi and smi.lower() != 'none':
                    mapping[name.lower()] = smi
    return mapping


def resolve_name_to_smiles_path(path: Optional[str] = None) -> Optional[Path]:
    """解析映射表路径；未显式传入时自动查找 data_preprocess/name_to_smiles.tsv。"""
    if path:
        candidate = Path(path).expanduser().resolve()
        return candidate if candidate.exists() else None
    return DEFAULT_NAME_TO_SMILES_PATH if DEFAULT_NAME_TO_SMILES_PATH.exists() else None


# ──────────────────────────────────────────────────────
# 2. 原始文件预清理（Step 0：原地修改 xlsx/csv 文件）
# ──────────────────────────────────────────────────────

# 需要删除的无用列
RAW_USELESS_COLUMNS = [
    'Reaction: Links to Reaxys',
    'Data Count',
    'References',
    'Links to Reaxys',
]

# 版权声明行的识别关键词（检查 Reaction ID 列）
RAW_FOOTER_KEYWORDS = ['Disclaimer', 'Copyright', 'Terms and Conditions']


def raw_is_footer_row(row) -> bool:
    """检查一行是否为版权声明尾部行。"""
    val = str(row.get('Reaction ID', ''))
    return any(kw in val for kw in RAW_FOOTER_KEYWORDS)


def raw_load_file(file_path: str) -> pd.DataFrame:
    """加载原始 xlsx/csv，保留所有列。编码自动尝试。"""
    if file_path.endswith('.csv'):
        for enc in ['utf-8', 'cp1252', 'latin-1', 'ISO-8859-1']:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f'无法解码 {file_path}')
    else:
        return pd.read_excel(file_path, engine='openpyxl')


def raw_save_file(df: pd.DataFrame, file_path: str):
    """按原格式写回文件。"""
    if file_path.endswith('.csv'):
        df.to_csv(file_path, index=False, encoding='utf-8')
    else:
        df.to_excel(file_path, index=False)


def clean_raw_file(file_path: str) -> dict:
    """
    清理单个原始文件：
      (1) 删除版权尾部行
      (2) 删除无用列
      （注意：不删除多步反应 — 多步反应在 Stage 2 加载时过滤，
       Stage 1 逆合成数据仍可回收使用）
    返回统计信息。
    """
    df = raw_load_file(file_path)
    stats = {'before': len(df)}

    # (1) 删除版权尾部行
    mask = df.apply(raw_is_footer_row, axis=1)
    if mask.any():
        df = df[~mask].reset_index(drop=True)

    # (2) 删除无用列（只删实际存在的）
    drop_cols = [c for c in RAW_USELESS_COLUMNS if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    stats['after'] = len(df)
    stats['removed'] = stats['before'] - stats['after']

    raw_save_file(df, file_path)
    return stats


def clean_raw_directory(input_dir: str) -> None:
    """
    对整个 input_dir 下所有 .xlsx/.csv 文件执行预清理。
    先删除 Excel 临时文件（~$*），再逐个清理。
    """
    # 删除 Excel 临时锁定文件
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.startswith('~$'):
                os.remove(os.path.join(root, f))

    # 收集所有数据文件
    data_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if (f.endswith('.xlsx') or f.endswith('.csv')) and not f.startswith('~$'):
                data_files.append(os.path.join(root, f))

    total_before = 0
    total_after = 0

    for fp in data_files:
        try:
            stats = clean_raw_file(fp)
            total_before += stats['before']
            total_after += stats['after']
            rel = os.path.relpath(fp, input_dir)
            if stats['removed'] > 0:
                print(f'  {rel}: {stats["before"]} → {stats["after"]} '
                      f'(删除 {stats["removed"]} 条)')
        except Exception as e:
            print(f'  错误 {os.path.relpath(fp, input_dir)}: {e}')

    print(f'  合计: {total_before} → {total_after} '
          f'(删除 {total_before - total_after} 条)')


# ──────────────────────────────────────────────────────
# 3. 基础验证与处理函数
# ──────────────────────────────────────────────────────

def is_nan(val) -> bool:
    """
    判断值是否"缺失"。
    条件：None / 浮点 NaN / 空字符串 / 哨兵值(nan, none, not given, unknown, -)
    """
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.lower().strip() in SENTINEL_VALUES:
        return True
    return False


def is_valid_smiles(smi: str) -> bool:
    """用 RDKit 检查 SMILES 是否可解析。"""
    if is_nan(smi) or not smi:
        return False
    return Chem.MolFromSmiles(str(smi)) is not None


def canonical_smiles(smi: str) -> str:
    """将 SMILES 转为 canonical 形式；失败返回空字符串。"""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return ''
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ''


def highest_temperature(temp) -> str:
    """
    多阶段温度处理：取各阶段中的最高温度。
    示例：'80; 100' → '100'  （两个阶段分别80°C和100°C，取100）
          '20 - 30'  → '25.0' （一个阶段的温度范围，取均值）
    缺失 → 返回 'nan'
    """
    if is_nan(temp):
        return 'nan'
    temp_str = str(temp)
    parts = temp_str.split('; ')   # 按阶段拆分
    values = []
    for p in parts:
        sub = p.split(' - ')        # 按范围拆分
        try:
            values.append(np.mean([float(x) for x in sub]))
        except ValueError:
            return 'nan'
    return str(max(values)) if values else 'nan'


def remove_duplicates_in_string(s: str) -> str:
    """去除 '; ' 分隔字符串中的重复 token，保留首次出现顺序。"""
    if is_nan(s) or not s:
        return s if s is not None else 'nan'
    tokens = str(s).split('; ')
    return '; '.join(list(dict.fromkeys(tokens)))


# ──────────────────────────────────────────────────────
# 4. Canonical Reaction Key（划分分组依据）
# ──────────────────────────────────────────────────────

def make_canonical_reaction_key(reactants: str, product: str) -> str:
    """
    生成 canonical reaction key，用于 split 时的分组。
    步骤：
      1. 反应物按 '.' 拆分 → 每个分子 canonicalize → 排序 → '.' 拼接
      2. 产物同样按 '.' 拆分 → 每个分子 canonicalize → 排序 → '.' 拼接
      3. 拼接为 'sorted_reactants>>sorted_products'

    示例：'CCO.CC' + 'CCOCC' → 'CC.CCO>>CCOCC'
          'Br.CCO' + 'CCOCC.Br'  → 'Br.CCO>>Br.CCOCC'
          （多产物 A+B→C+D 正确拆为多个产物分子分别标准化）

    设计目的：同一反应的不同条件记录共享同一个 canonical key，
    确保它们全部进入同一个 split（train/val/test），避免数据泄露。
    """
    # 反应物：拆分 → 逐个标准化 → 排序 → 拼接
    r_parts = str(reactants).split('.')
    r_canon = sorted(
        c for p in r_parts if (c := canonical_smiles(p.strip()))
    )
    # 产物：同样处理，支持 A+B→C+D 等多产物反应
    p_parts = str(product).split('.')
    p_canon = sorted(
        c for p in p_parts if (c := canonical_smiles(p.strip()))
    )
    if not r_canon or not p_canon:
        return ''
    return f"{'.'.join(r_canon)}>>{'.'.join(p_canon)}"


# ──────────────────────────────────────────────────────
# 5. 标签标准化（Label Standardization）
# ──────────────────────────────────────────────────────

def hydrate_strip(name: str) -> str:
    """
    去除水合物后缀。
    例如：'sodium carbonate monohydrate' → 'sodium carbonate'
          'copper sulfate pentahydrate'  → 'copper sulfate'
    覆盖 mono/di/tri/tetra/penta/hexa/hepta/octa/nona/deca hydrate。
    """
    for pat in [r'\s+monohydrate', r'\s+dihydrate', r'\s+trihydrate',
                r'\s+tetrahydrate', r'\s+pentahydrate', r'\s+hexahydrate',
                r'\s+heptahydrate', r'\s+octahydrate', r'\s+nonahydrate',
                r'\s+decahydrate', r'\s+hydrate']:
        name = re.sub(pat, '', name, flags=re.IGNORECASE)
    return name.strip()


def normalize_legacy_name_artifacts(name: str) -> str:
    """
    修正旧 Reaxys 标签里常见的历史书写问题。

    当前处理：
      1. 将 ``?`` 统一替换为 ``-``，如 ``beta?cyclodextrin`` → ``beta-cyclodextrin``
      2. 将括号中的小写罗马数字 / ``l``-型误写统一为标准大写形式，
         如 ``copper(l) iodide`` → ``copper(I) iodide``，``iron(iii)`` → ``iron(III)``
    """
    if not name:
        return name

    normalized = name.replace('?', '-')

    def _fix_parenthetical_oxidation(match: re.Match) -> str:
        token = match.group(1)
        if not token:
            return match.group(0)
        converted = token.replace('l', 'I').replace('L', 'I').upper()
        if re.fullmatch(r'[IVX]+', converted):
            return f'({converted})'
        return match.group(0)

    normalized = re.sub(r'\(([ivxlIVXL]+)\)', _fix_parenthetical_oxidation, normalized)
    return normalized


def normalize_label(name: str) -> Optional[str]:
    """
    标签规范化：去首尾空格 → 修正旧标签书写问题 → 去水合物后缀 → 合并多余空格。
    返回 None 表示标签无效。
    """
    if is_nan(name) or not name:
        return None
    name = str(name).strip()
    name = normalize_legacy_name_artifacts(name)
    name = hydrate_strip(name)
    name = re.sub(r'\s+', ' ', name)   # 合并连续空格
    return name


def standardize_labels_in_series(series: pd.Series,
                                  name_to_smiles: dict) -> pd.Series:
    """
    对整列 '; '-分隔的标签做第一轮标准化：
      1. normalize_label 规范化每个 token
      2. 过滤哨兵值
      3. 可选：通过 name_to_smiles 映射替换为 canonical SMILES
      4. 去行内重复
    """
    def _standardize_cell(val):
        if is_nan(val) or not val:
            return 'nan'
        tokens = str(val).split('; ')
        cleaned = []
        for t in tokens:
            t_norm = normalize_label(t)
            if t_norm is None or t_norm.lower() in SENTINEL_VALUES:
                continue

            # 如果有 name→SMILES 映射，尝试替换
            if name_to_smiles:
                smi = name_to_smiles.get(t_norm.lower(), None)
                if smi:
                    t_norm = smi

            if t_norm not in cleaned:
                cleaned.append(t_norm)
        return '; '.join(cleaned) if cleaned else 'nan'

    return series.apply(_standardize_cell)


def build_label_name_map(series: pd.Series, name_to_smiles: dict) -> dict:
    """
    第二轮标准化：构建名称归并映射。

    核心逻辑：多个不同名称映射到同一个 canonical SMILES →
    取其中出现频次最高的名称作为标准名，其余全部替换为标准名。

    例如：'AcOH' (出现50次) 和 'acetic acid' (出现200次)
    都映射到 SMILES 'CC(=O)O' → 统一使用 'acetic acid'。

    返回 {原始名称: 标准名称} 映射字典。
    """
    global_freq = Counter()
    canonical_map = {}   # canonical键 → [(原始名, 频次), ...]

    for val in series:
        if is_nan(val) or not val:
            continue
        for t in str(val).split('; '):
            t_norm = normalize_label(t)
            if t_norm is None or t_norm.lower() in SENTINEL_VALUES:
                continue
            # 确定 canonical 标识（SMILES 或 小写名称）
            if name_to_smiles and t_norm.lower() in name_to_smiles:
                canonical = name_to_smiles[t_norm.lower()]
            else:
                canonical = t_norm.lower()

            if canonical not in canonical_map:
                canonical_map[canonical] = []
            canonical_map[canonical].append(t)
            global_freq[t] += 1

    # 每个 canonical 组取频次最高的原始名作为代表
    name_map = {}
    for canon, originals in canonical_map.items():
        best = sorted(originals, key=lambda x: global_freq[x], reverse=True)[0]
        for orig in set(originals):
            name_map[orig] = best

    return name_map


def apply_label_map(series: pd.Series, name_map: dict) -> pd.Series:
    """
    应用名称归并映射：把每个 token 替换为 name_map 中的标准名。
    不存在的保持原样。
    """
    def _remap(val):
        if is_nan(val) or not val:
            return 'nan'
        tokens = str(val).split('; ')
        mapped = []
        for t in tokens:
            t_norm = normalize_label(t)
            if t_norm is None or t_norm.lower() in SENTINEL_VALUES:
                continue
            new = name_map.get(t, t)   # 有映射则替换，无则保留
            if new not in mapped:
                mapped.append(new)
        return '; '.join(mapped) if mapped else 'nan'

    return series.apply(_remap)


# ──────────────────────────────────────────────────────
# 6. 频次统计工具
# ──────────────────────────────────────────────────────

def build_frequency_dict(series: pd.Series) -> dict:
    """
    统计整列中每个 '; '-分隔 token 的出现次数。
    用于低频过滤和类别文件生成。
    """
    freq = Counter()
    for val in series:
        if is_nan(val) or not val:
            continue
        for token in str(val).split('; '):
            if token.lower() not in SENTINEL_VALUES:
                freq[token] += 1
    return dict(freq)


# ──────────────────────────────────────────────────────
# 7. 逐项清洗步骤
# ──────────────────────────────────────────────────────

def split_reaction_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """拆分 Reaction 列 → 'reactants' + 'products'。"""
    df[['reactants', 'products']] = df['Reaction'].str.split('>>', expand=True)
    df.drop(columns=['Reaction'], inplace=True)
    return df


def merge_reagent_catalyst(df: pd.DataFrame) -> pd.DataFrame:
    """
    (c) 将 Catalyst 列并入 Reagent 列。
    四种情况：
      - 两者都缺失 → NaN
      - 仅试剂有值 → 保留试剂
      - 仅催化剂有值 → 催化剂提升为试剂
      - 两者都有 → 用 '; ' 拼接
    """
    def _combine(row):
        r, c = row.get('Reagent'), row.get('Catalyst')
        r_nan, c_nan = is_nan(r), is_nan(c)
        if r_nan and c_nan:
            return float('nan')
        elif not r_nan and c_nan:
            return r
        elif r_nan and not c_nan:
            return c
        else:
            return f"{r}; {c}"

    df['Reagent'] = df.apply(_combine, axis=1)
    df.drop(columns=['Catalyst'], inplace=True, errors='ignore')
    return df


def filter_valid_entries(df: pd.DataFrame, min_yield: float = DEFAULT_MIN_YIELD) -> pd.DataFrame:
    """
    (a) 合法性检查，依次删除以下记录：
      - 反应物或产物 SMILES 无法被 RDKit 解析
      - 缺失产率（无法计算 relevance score）
      - 低于最小产率阈值的记录（默认 yield < 25）
      - 缺失溶剂（溶剂是预测目标之一）
    """
    df = split_reaction_smiles(df)

    # 反应物和产物 SMILES 必须可解析
    for col in ['reactants', 'products']:
        bad = df[col].apply(lambda x: is_nan(x) or not is_valid_smiles(x))
        df = df[~bad].reset_index(drop=True)

    # 产率必须存在
    bad = df['Yield (numerical)'].apply(is_nan)
    df = df[~bad].reset_index(drop=True)

    # 过滤低产率记录
    if min_yield is not None:
        def _keep_yield(value) -> bool:
            try:
                return float(value) >= float(min_yield)
            except (TypeError, ValueError):
                return False
        good = df['Yield (numerical)'].apply(_keep_yield)
        df = df[good].reset_index(drop=True)

    # 溶剂必须存在
    bad = df['Solvent (Reaction Details)'].apply(is_nan)
    df = df[~bad].reset_index(drop=True)

    return df


def process_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    (b) 温度处理：
    - 多阶段取最高温
    - 缺失保留 'nan'
    - 异常高温（>500°C）视为录入错误，置为 'nan'
      （如 Reaxys 中 1110.0 应为 110.0 的小数点错位）
    """
    df['Temperature (Reaction Details) [C]'] = (
        df['Temperature (Reaction Details) [C]'].apply(highest_temperature))

    # 过滤异常值：>500°C 大概率是录入错误
    def _clamp(val):
        try:
            v = float(val)
            if v > 500:
                return 'nan'
            return val
        except (ValueError, TypeError):
            return val

    df['Temperature (Reaction Details) [C]'] = (
        df['Temperature (Reaction Details) [C]'].apply(_clamp))
    return df


def deduplicate_condition_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    行内去重：同一行的 reagent/solvent 字符串内部
    不应出现重复 token（如 'Pd; Pd; K2CO3' → 'Pd; K2CO3'）。
    """
    df = df.copy()
    for i in df.index:
        df.at[i, 'Solvent (Reaction Details)'] = remove_duplicates_in_string(
            df.at[i, 'Solvent (Reaction Details)'])
        df.at[i, 'Reagent'] = remove_duplicates_in_string(
            df.at[i, 'Reagent'])
    return df


def reassign_roles_all_families(raw_data: dict) -> dict:
    """
    (d) 试剂/溶剂角色重分配（跨所有反应族全局执行）。

    问题：Reaxys 中同一化学名可能同时出现在 Reagent 和 Solvent 列。
    例如 'acetic acid' 有时当作试剂、有时当作溶剂。

    解决：统计该化学名在 Reagent 列和 Solvent 列中的全局频次，
    将其统一归入频次更高的角色，并从另一个角色中移除。

    重分配后：逐行检查，将放错位置的 token 移到正确的列。
    若某条记录的溶剂因此变空 → 删除该记录。
    """
    # 合并所有 family 统计全局频次
    all_data = pd.concat(raw_data.values(), ignore_index=True)
    reagent_freq = build_frequency_dict(all_data['Reagent'])
    solvent_freq = build_frequency_dict(all_data['Solvent (Reaction Details)'])

    # 找出重叠的化学名，保留频次更高的角色
    overlaps = set(reagent_freq.keys()) & set(solvent_freq.keys())
    for chem in overlaps:
        if solvent_freq.get(chem, 0) >= reagent_freq.get(chem, 0):
            reagent_freq.pop(chem, None)    # 归入溶剂
        else:
            solvent_freq.pop(chem, None)    # 归入试剂

    # 逐行修正
    for rtype, df in raw_data.items():
        for i in df.index:
            # 解析当前行的试剂和溶剂列表
            reagents = (
                [r for r in str(df.at[i, 'Reagent']).split('; ')
                 if r and r.lower() not in SENTINEL_VALUES]
                if not is_nan(df.at[i, 'Reagent']) else []
            )
            solvents = (
                [s for s in str(df.at[i, 'Solvent (Reaction Details)']).split('; ')
                 if s and s.lower() not in SENTINEL_VALUES]
                if not is_nan(df.at[i, 'Solvent (Reaction Details)']) else []
            )

            # 按字典归属重新分配
            new_reagents = [r for r in reagents if r not in solvent_freq]
            new_solvents = [s for s in solvents if s not in reagent_freq]

            for r in reagents:
                if r in solvent_freq and r not in new_solvents:
                    new_solvents.append(r)
            for s in solvents:
                if s in reagent_freq and s not in new_reagents:
                    new_reagents.append(s)

            df.at[i, 'Reagent'] = (
                'nan' if not new_reagents else '; '.join(new_reagents))
            df.at[i, 'Solvent (Reaction Details)'] = (
                'nan' if not new_solvents else '; '.join(new_solvents))

        # 删除溶剂变空的记录
        bad = df['Solvent (Reaction Details)'].apply(is_nan)
        df = df[~bad].reset_index(drop=True)
        raw_data[rtype] = df

    return raw_data


def filter_rare_labels_all_families(raw_data: dict,
                                    min_freq: int = DEFAULT_MIN_LABEL_FREQ,
                                    scope: str = DEFAULT_LABEL_FREQ_SCOPE) -> dict:
    """
    (g) 低频标签过滤。

    - scope='global'：跨所有 family 共享频次统计
    - scope='family'：每个 family 单独统计标签频次

    频次 < min_freq 的标签视为不可靠，删除包含它的整条记录。
    """
    if scope not in VALID_LABEL_FREQ_SCOPES:
        raise ValueError(f'unsupported label frequency scope: {scope}')

    def _has_rare(val, rare_set):
        """检查当前行的标签是否包含低频 token。"""
        if is_nan(val) or not val:
            return False
        return any(t in rare_set for t in str(val).split('; '))

    if scope == 'global':
        all_data = pd.concat(raw_data.values(), ignore_index=True)
        shared_reagent_freq = build_frequency_dict(all_data['Reagent'])
        shared_solvent_freq = build_frequency_dict(all_data['Solvent (Reaction Details)'])
        shared_rare_r = {k for k, v in shared_reagent_freq.items() if v < min_freq}
        shared_rare_s = {k for k, v in shared_solvent_freq.items() if v < min_freq}
    else:
        shared_rare_r = None
        shared_rare_s = None

    filtered: dict = {}
    for rtype, df in raw_data.items():
        if scope == 'family':
            reagent_freq = build_frequency_dict(df['Reagent'])
            solvent_freq = build_frequency_dict(df['Solvent (Reaction Details)'])
            rare_r = {k for k, v in reagent_freq.items() if v < min_freq}
            rare_s = {k for k, v in solvent_freq.items() if v < min_freq}
        else:
            rare_r = shared_rare_r
            rare_s = shared_rare_s

        bad_r = df['Reagent'].apply(lambda x: _has_rare(x, rare_r))
        bad_s = df['Solvent (Reaction Details)'].apply(lambda x: _has_rare(x, rare_s))
        filtered[rtype] = df[~(bad_r | bad_s)].reset_index(drop=True)
    return filtered


def deduplicate_condition_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    (h) 条件记录去重。

    去重键：(Reaction ID, canonical_reaction_smiles, Reagent, Solvent)
    同一反应ID + 同一反应式 + 同一试剂组合 + 同一溶剂组合 → 只保留一条。
    """
    if 'canonical_key' not in df.columns:
        df['canonical_key'] = df.apply(
            lambda row: make_canonical_reaction_key(row['reactants'],
                                                     row['products']), axis=1)

    key_cols = ['Reaction ID', 'canonical_key', 'Reagent',
                'Solvent (Reaction Details)']
    df = df.drop_duplicates(subset=key_cols, keep='first').reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────
# 8. 数据划分与文件输出
# ──────────────────────────────────────────────────────

def split_by_canonical_reaction(df: pd.DataFrame, train_pct=0.8,
                                 val_pct=0.1, seed=45):
    """
    (i) 按 canonical_reaction_smiles 分组做 8:1:1 划分。

    关键约束：同一 canonical reaction 的所有条件记录
    必须全部进入同一个 split（train/val/test），防止数据泄露。

    流程：
      1. 计算每行的 canonical_key
      2. 按 key 分组，收集每组的所有行索引
      3. 对 canonical key 列表 shuffle
      4. 按 8:1:1 切分 key → 收集对应的行 → 返回三个子 DataFrame
    """
    if 'canonical_key' not in df.columns:
        df['canonical_key'] = df.apply(
            lambda row: make_canonical_reaction_key(row['reactants'],
                                                     row['products']), axis=1)

    # 将每行按 canonical key 分组
    key_to_indices = {}
    for i, key in enumerate(df['canonical_key']):
        if not key:
            continue
        key_to_indices.setdefault(key, []).append(i)

    unique_keys = list(key_to_indices.keys())
    unique_keys = shuffle(unique_keys, random_state=seed)

    n_train = int(len(unique_keys) * train_pct)
    n_val = int(len(unique_keys) * (train_pct + val_pct))

    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train:n_val])
    test_keys = set(unique_keys[n_val:])

    def _gather(key_set):
        """收集指定 key 集合对应的所有行。"""
        indices = []
        for k in key_set:
            indices.extend(key_to_indices[k])
        return df.iloc[sorted(indices)].reset_index(drop=True)

    return _gather(train_keys), _gather(val_keys), _gather(test_keys)


def make_class_list(values: list) -> str:
    """从 '; '-分隔字符串列表中提取所有不重复 token，返回 '; ' 拼接。"""
    seen = []
    for v in values:
        if is_nan(v) or not v:
            continue
        for token in str(v).split('; '):
            if token and token.lower() not in SENTINEL_VALUES and token not in seen:
                seen.append(token)
    return '; '.join(seen)


def write_first_part(df: pd.DataFrame, path: str):
    """
    输出 Stage 2A (FNN 候选生成模型) 格式：
    每个 canonical reaction 一行。
    列：Reaction ID \t reactants \t product \t reagents \t solvents
    reagents/solvents 列出该反应所有条件中出现的全部标签（去重）。
    """
    # 按 canonical key 分组
    key_order = []
    key_to_rows = {}
    for i in range(len(df)):
        key = df.at[i, 'canonical_key']
        if not key:
            continue
        if key not in key_to_rows:
            key_order.append(key)
            key_to_rows[key] = []
        key_to_rows[key].append(i)

    with open(path, 'w', encoding='utf-8') as f:
        for key in key_order:
            indices = key_to_rows[key]
            chunk = df.iloc[indices]
            rid = str(chunk['Reaction ID'].iloc[0])
            reactant = str(chunk['reactants'].iloc[0])
            product = str(chunk['products'].iloc[0])
            # 聚合该反应所有条件的标签
            solvents = make_class_list(
                chunk['Solvent (Reaction Details)'].tolist())
            reagents = make_class_list(chunk['Reagent'].tolist())
            f.write(f'{rid}\t{reactant}\t{product}\t{reagents}\t{solvents}\n')


def write_second_part(df: pd.DataFrame, path: str):
    """
    输出 Stage 2B (排序 + 温度回归模型) 格式：
    每个条件记录一行。
    列：Reaction ID \t reactants \t product \t yield \t reagent \t solvent \t temperature
    """
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(df)):
            row = df.iloc[i]
            line = '\t'.join([
                str(row['Reaction ID']),
                str(row['reactants']),
                str(row['products']),
                str(row['Yield (numerical)']),
                str(row['Reagent']),
                str(row['Solvent (Reaction Details)']),
                str(row['Temperature (Reaction Details) [C]']),
            ]) + '\n'
            f.write(line)


def save_dict_pkl(d: dict, path: str):
    """保存 Python 字典为 pickle 文件。"""
    with open(path, 'wb') as f:
        pickle.dump(d, f)


def write_summary(summary: dict, path: str):
    """输出统计摘要到文本文件。"""
    with open(path, 'w', encoding='utf-8') as f:
        for key, value in summary.items():
            f.write(f'{key}: {value}\n')


# ──────────────────────────────────────────────────────
# 9. 流水线编排
# ──────────────────────────────────────────────────────

def cleanup_to_complexity(rtype: str, df: pd.DataFrame,
                           name_to_smiles: dict,
                           min_yield: float = DEFAULT_MIN_YIELD) -> pd.DataFrame:
    """
    单个反应族的前半段清洗：执行步骤 (a) 到 (g)。

    输入：原始 DataFrame（已加载并做了角色重分配）
    输出：清洗后的 DataFrame（未去重、未划分）
    """
    # (a) 合法性检查
    df = filter_valid_entries(df, min_yield=min_yield)

    # (b) 温度处理
    df = process_temperature(df)

    # (c) 催化剂并入试剂
    df = merge_reagent_catalyst(df)

    # 行内去重（去除同一行内重复的 reagent/solvent token）
    df = deduplicate_condition_strings(df)

    # (e) 标签标准化：两轮处理
    # 第一轮：规范化 + name→SMILES 映射
    df['Reagent'] = standardize_labels_in_series(df['Reagent'], name_to_smiles)
    df['Solvent (Reaction Details)'] = standardize_labels_in_series(
        df['Solvent (Reaction Details)'], name_to_smiles)
    # 第二轮：同 SMILES 的不同名称归并为最频名称
    reagent_name_map = build_label_name_map(df['Reagent'], name_to_smiles)
    solvent_name_map = build_label_name_map(
        df['Solvent (Reaction Details)'], name_to_smiles)
    df['Reagent'] = apply_label_map(df['Reagent'], reagent_name_map)
    df['Solvent (Reaction Details)'] = apply_label_map(
        df['Solvent (Reaction Details)'], solvent_name_map)

    # (f) 条件复杂度过滤：溶剂 ≤ 2，试剂 ≤ 3
    def _count(val):
        if is_nan(val) or not val:
            return 0
        return len([t for t in str(val).split('; ')
                    if t.lower() not in SENTINEL_VALUES])
    bad_s = df['Solvent (Reaction Details)'].apply(lambda x: _count(x) > 2)
    bad_r = df['Reagent'].apply(lambda x: _count(x) > 3)
    df = df[~(bad_s | bad_r)].reset_index(drop=True)

    return df


def finalize_and_output(rtype: str, df: pd.DataFrame, output_base: str,
                         stats: dict):
    """
    单个反应族的后半段处理：执行步骤 (h)(i) 并写出全部输出文件。

    输入：已清洗到 (g) 之后的 DataFrame
    输出目录：data/reaction_processed_{family}_catmerge/
    """
    out_root = os.path.join(output_base, f'reaction_processed_{rtype}_catmerge')
    stats['family'] = rtype
    stats['after_complexity'] = len(df)

    # (h) 条件记录去重
    df = deduplicate_condition_records(df)
    stats['after_dedup'] = len(df)

    # 生成 canonical reaction key（用于划分和输出）
    df['canonical_key'] = df.apply(
        lambda row: make_canonical_reaction_key(row['reactants'],
                                                 row['products']), axis=1)

    # (i) 按 canonical reaction 做 8:1:1 划分
    train, val, test = split_by_canonical_reaction(df)
    stats['train'] = len(train)
    stats['val'] = len(val)
    stats['test'] = len(test)
    stats['train_reactions'] = len(set(train['canonical_key']))
    stats['val_reactions'] = len(set(val['canonical_key']))
    stats['test_reactions'] = len(set(test['canonical_key']))
    stats['total_reactions'] = len(set(df['canonical_key']))

    # 从训练集生成标签频次字典（仅基于 train，避免用到 val/test 信息）
    reagent_freq = build_frequency_dict(train['Reagent'])
    solvent_freq = build_frequency_dict(train['Solvent (Reaction Details)'])

    # ── 创建输出目录 ──
    first_dir = os.path.join(out_root, 'For_first_part_model')
    second_dir = os.path.join(out_root, 'For_second_part_model')
    label_dir = os.path.join(out_root, 'label_processed')
    for d in [first_dir, second_dir, label_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 写出数据文件 ──
    for name, sub in [('total', df), ('train', train),
                       ('validate', val), ('test', test)]:
        write_first_part(sub, os.path.join(first_dir,
                          f'Splitted_first_{name}_labels_processed.txt'))
        write_second_part(sub, os.path.join(second_dir,
                           f'Splitted_second_{name}_labels_processed.txt'))

    # ── 写出标签类别文件（pickle + 文本） ──
    save_dict_pkl(reagent_freq, os.path.join(label_dir,
                  'class_names_reagent_labels_processed.pkl'))
    save_dict_pkl(solvent_freq, os.path.join(label_dir,
                  'class_names_solvent_labels_processed.pkl'))

    with open(os.path.join(label_dir,
              'class_names_reagent_labels_processed.txt'), 'w',
              encoding='utf-8') as f:
        for key in sorted(reagent_freq.keys()):
            f.write(f'{key}\t{reagent_freq[key]}\n')
    with open(os.path.join(label_dir,
              'class_names_solvent_labels_processed.txt'), 'w',
              encoding='utf-8') as f:
        for key in sorted(solvent_freq.keys()):
            f.write(f'{key}\t{solvent_freq[key]}\n')

    # ── 写出统计摘要 ──
    write_summary(stats, os.path.join(out_root, 'summary.txt'))

    return stats


# ──────────────────────────────────────────────────────
# 10. Stage 1 逆合成路线数据（EditRetro 微调用）
# ──────────────────────────────────────────────────────

def collect_stage1_route_data(rtype: str, input_dir: str, output_dir: str,
                               stage2_root: str, rxn_mapper_batch_size: int = 64):
    """
    为一个反应族生成 Stage 1 逆合成路线数据（train + val + test）。

    核心策略（按 data_process.md 四节）：
      1. train/val/test 由 Stage 2 的 split 锚定 — 从 processed_root 读取
      2. train 额外吸收被 Stage 2 筛掉但反应本身合法的记录 —
         回原始 xlsx 捞取，补入 train
      3. 零泄露约束 — 额外 train 与已有 train/val/test 的 canonical reaction 去重

    输出目录：
      editretro/datasets/REAXYS_{FAMILY}_SINGLE_CATMERGE/raw/
    """
    stage2_second_dir = os.path.join(stage2_root, 'For_second_part_model')

    # ── 1. 从 Stage 2 输出读取 train/validate/test，按 canonical key 去重 ──
    split_data = {}
    for split_name in ['train', 'validate', 'test']:
        path = os.path.join(stage2_second_dir,
                            f'Splitted_second_{split_name}_labels_processed.txt')
        if not os.path.exists(path):
            print(f'    警告: 找不到 {path}，跳过 Stage 1')
            return

        seen_keys = set()
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    rid, reactants, product = parts[0], parts[1], parts[2]
                    ckey = make_canonical_reaction_key(reactants, product)
                    # 同一 canonical reaction 的多个条件只保留一条
                    if ckey and ckey not in seen_keys:
                        seen_keys.add(ckey)
                        rows.append({
                            'reaction_id': rid,
                            'reactants': reactants,
                            'product': product,
                            'canonical_key': ckey,
                        })
        split_data[split_name] = rows
        print(f'    Stage 2 {split_name}: {len(rows)} 条独特反应')

    # ── 2. 从原始文件捞取被 Stage 2 筛掉的合法反应（含多步反应，补入 train） ──
    train_canonical_keys = set(r['canonical_key'] for r in split_data['train'])
    val_canonical_keys = set(r['canonical_key'] for r in split_data['validate'])
    test_canonical_keys = set(r['canonical_key'] for r in split_data['test'])
    selected_keys = train_canonical_keys | val_canonical_keys | test_canonical_keys

    extra_train = []
    rdir = os.path.join(input_dir, rtype)
    if os.path.isdir(rdir):
        files = sorted([f for f in os.listdir(rdir)
                        if (f.endswith('.xlsx') or f.endswith('.csv'))
                        and not f.startswith('~$')])
        for f in files:
            # 用 raw_load_file 读取所有行（含多步反应），而非 load_file
            df = raw_load_file(os.path.join(rdir, f))
            # 只需要 Reaction, Reaction ID 列（不需要 yield/solvent 等）
            if 'Reaction' not in df.columns:
                continue
            df[['reactants', 'products']] = df['Reaction'].str.split(
                '>>', expand=True)
            for _, row in df.iterrows():
                rid = str(row.get('Reaction ID', ''))
                reactants = str(row.get('reactants', ''))
                product = str(row.get('products', ''))
                # 反应式必须合法
                if not is_valid_smiles(reactants) or not is_valid_smiles(product):
                    continue
                if '.' in reactants:
                    parts_ok = all(is_valid_smiles(p.strip())
                                   for p in reactants.split('.'))
                    if not parts_ok:
                        continue
                ckey = make_canonical_reaction_key(reactants, product)
                if not ckey:
                    continue
                # 只加入 train，与 val/test 严格去重
                if ckey in selected_keys:
                    continue
                if ckey not in train_canonical_keys:
                    train_canonical_keys.add(ckey)
                    extra_train.append({
                        'reaction_id': rid,
                        'reactants': reactants,
                        'product': product,
                        'canonical_key': ckey,
                    })
    print(f'    额外 train: {len(extra_train)} 条（被 Stage 2 筛掉但反应合法）')

    # 合并 train
    all_train = split_data['train'] + extra_train

    # ── 3. 原子映射（RXNMapper） ──
    def run_atom_mapping(records, desc='mapping'):
        """对一批反应运行 RXNMapper，返回带映射结果的列表。"""
        results = []
        for i in tqdm(range(0, len(records), rxn_mapper_batch_size),
                       desc=desc):
            batch = records[i:i + rxn_mapper_batch_size]
            rxn_smis = [f"{r['reactants']}>>{r['product']}" for r in batch]
            try:
                mapped = rxn_mapper.get_attention_guided_atom_maps(rxn_smis)
                for j, m in enumerate(mapped):
                    results.append({
                        **batch[j],
                        'mapped_reaction_smiles': m.get('mapped_rxn', ''),
                        'mapping_confidence': m.get('confidence', 0.0),
                    })
            except Exception as e:
                # 映射失败时保留原始反应，置信度标记为 0
                for r in batch:
                    results.append({
                        **r,
                        'mapped_reaction_smiles': '',
                        'mapping_confidence': 0.0,
                    })
        return results

    # 尝试导入 RXNMapper
    rxn_mapper = None
    try:
        from rxnmapper import RXNMapper
        rxn_mapper = RXNMapper()
        print('    RXNMapper 已加载')
    except ImportError:
        print('    警告: RXNMapper 未安装，跳过原子映射（产物不含映射信息）')

    for split_name, records in [
        ('train', all_train),
        ('val', split_data['validate']),
        ('test', split_data['test']),
    ]:
        if rxn_mapper and len(records) > 0:
            records = run_atom_mapping(records, desc=f'    atom mapping {split_name}')
        else:
            for r in records:
                r['mapped_reaction_smiles'] = ''
                r['mapping_confidence'] = 0.0

        # ── 4. 写出 CSV ──
        out_dir = os.path.join(
            output_dir, 'editretro', 'datasets',
            f'REAXYS_{rtype}_SINGLE_CATMERGE', 'raw')
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f'raw_{split_name}.csv')
        df_out = pd.DataFrame([{
            'pair_id': f"{rtype}_{split_name}_{i:06d}",
            'reaction_id': r['reaction_id'],
            'product_smiles': r['product'],
            'reactants>reagents>production': (
                f"{r['reactants']}>>{r['product']}"),
            'mapped_reaction_smiles': r['mapped_reaction_smiles'],
            'mapping_confidence': r.get('mapping_confidence', 0.0),
        } for i, r in enumerate(records)])
        df_out.to_csv(out_path, index=False)
        print(f'    {split_name}: {len(df_out)} 条 → {out_path}')


# ──────────────────────────────────────────────────────
# 11. USPTO 数据处理（通用逆合成数据，去原子映射 + 去重泄露）
# ──────────────────────────────────────────────────────

# 原子映射正则：匹配 [元素:数字] 中的 :数字 部分
_ATOM_MAP_RE = re.compile(r':\d+')


def strip_atom_mapping(mapped_smiles: str) -> str:
    """去除原子映射，[C:1] → [C], [CH2:13] → [CH2], c:5 → c。"""
    return _ATOM_MAP_RE.sub('', str(mapped_smiles))


def canonical_key_from_unmapped_reaction(rxn_smiles: str) -> str:
    """把去原子映射后的 reactants>>product 反应式转成 canonical key。"""
    if '>>' not in str(rxn_smiles):
        return ''
    reactants, product = str(rxn_smiles).split('>>', 1)
    return make_canonical_reaction_key(reactants, product)


def process_uspto_data(output_dir: str, seed: int = 45):
    """
    处理 USPTO_full 数据：
      1. 合并 raw_train/raw_val/raw_test.csv
      2. 用去原子映射后的反应式构造 canonical key（仅用于去重/防泄露）
      3. 与所有 Stage 2 测试集对比，删除重叠反应（防泄露）
      4. 按 8:2 划分为训练/验证集
      5. 保留原始 mapped reaction 作为 EditRetro 训练输入，输出到
         editretro/datasets/USPTO_STAGE2_FILTERED/raw/
    """
    uspto_dir = os.path.join(output_dir, 'editretro', 'datasets', 'USPTO_full')
    if not os.path.isdir(uspto_dir):
        print('  跳过 USPTO 处理：目录不存在')
        return {}

    print(f'\n{"=" * 60}')
    print('USPTO 数据处理')
    print(f'{"=" * 60}')

    # ── 1. 合并 ──
    dfs = []
    input_counts = {}
    for fname in ['raw_train.csv', 'raw_val.csv', 'raw_test.csv']:
        fp = os.path.join(uspto_dir, fname)
        if os.path.exists(fp):
            current = pd.read_csv(fp)
            dfs.append(current)
            input_counts[fname] = int(len(current))
            print(f'  {fname}: {len(current)} 条')
    uspto = pd.concat(dfs, ignore_index=True)
    print(f'  合并后: {len(uspto)} 条')
    stats = {
        'input_counts': input_counts,
        'merged_raw': int(len(uspto)),
    }

    # ── 2. 基于去映射反应式构造 canonical key（训练仍保留原始 mapped reaction） ──
    print('  基于去原子映射反应式构造 canonical keys...')
    rxns = uspto['reactants>reagents>production'].fillna('').astype(str).apply(
        strip_atom_mapping)

    # 向量化拆分为 reactants 和 product
    split = rxns.str.split('>>', n=1, expand=True)
    if split.shape[1] < 2:
        split[1] = ''
    uspto['reactants'] = split[0].fillna('')
    uspto['product'] = split[1].fillna('')

    # ── 3. 生成 canonical key ──
    unique_rxns = pd.Index(rxns.unique())
    n_proc = min(max(multiprocessing.cpu_count() // 2, 1), 16)
    print(f'  并行计算 canonical keys: {len(unique_rxns)} 条唯一反应, processes={n_proc}')
    with multiprocessing.Pool(processes=n_proc) as pool:
        canonical_keys = list(
            tqdm(
                pool.imap(canonical_key_from_unmapped_reaction, unique_rxns),
                total=len(unique_rxns),
                desc='  canonical keys',
            )
        )
    key_map = dict(zip(unique_rxns, canonical_keys))
    uspto['canonical_key'] = rxns.map(key_map)
    # 删掉 key 为空的
    before = len(uspto)
    uspto = uspto[uspto['canonical_key'] != ''].reset_index(drop=True)
    print(f'  合法反应: {len(uspto)} 条（丢弃 {before - len(uspto)} 条无效）')
    stats['valid_after_canonicalization'] = int(len(uspto))
    stats['removed_invalid'] = int(before - len(uspto))

    # ── 4. 收集所有 Stage 2 测试集 canonical key ──
    print('  收集 Stage 2 测试集 reaction keys...')
    stage2_test_keys = set()
    for fam_dir_name in sorted(os.listdir(output_dir)):
        if not fam_dir_name.startswith('reaction_processed_'):
            continue
        second_dir = os.path.join(output_dir, fam_dir_name,
                                  'For_second_part_model')
        test_path = os.path.join(second_dir,
                                 'Splitted_second_test_labels_processed.txt')
        if not os.path.exists(test_path):
            continue
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    k = make_canonical_reaction_key(parts[1], parts[2])
                    if k:
                        stage2_test_keys.add(k)
    print(f'  Stage 2 测试集共有 {len(stage2_test_keys)} 个独特反应')
    stats['stage2_test_unique_reactions'] = int(len(stage2_test_keys))

    # 去重
    before = len(uspto)
    uspto = uspto[~uspto['canonical_key'].isin(stage2_test_keys)]
    uspto = uspto.reset_index(drop=True)
    print(f'  过滤后: {len(uspto)} 条（移除 {before - len(uspto)} 条重叠）')
    stats['after_overlap_filter'] = int(len(uspto))
    stats['removed_overlap'] = int(before - len(uspto))

    # ── 5. 按 canonical key 去重（USPTO 可能有重复反应） ──
    before = len(uspto)
    uspto = uspto.drop_duplicates(subset='canonical_key', keep='first')
    uspto = uspto.reset_index(drop=True)
    print(f'  去重独特反应: {len(uspto)} 条')
    stats['after_dedup'] = int(len(uspto))
    stats['removed_duplicates'] = int(before - len(uspto))

    # ── 6. 8:2 划分 ──
    keys = uspto['canonical_key'].unique()
    keys = list(keys)
    from sklearn.utils import shuffle
    keys = shuffle(keys, random_state=seed)
    n_train = int(len(keys) * 0.8)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:])

    train_df = uspto[uspto['canonical_key'].isin(train_keys)]
    val_df = uspto[uspto['canonical_key'].isin(val_keys)]

    # ── 7. 输出 ──
    out_dir = os.path.join(output_dir, 'editretro', 'datasets',
                           'USPTO_STAGE2_FILTERED')
    raw_out_dir = os.path.join(out_dir, 'raw')
    os.makedirs(raw_out_dir, exist_ok=True)

    for name, sub in [('train', train_df), ('val', val_df)]:
        # EditRetro 训练需要带 atom mapping 的原始反应串；
        # 上面的 strip_atom_mapping 仅用于构造 canonical key。
        out = sub[['id', 'reactants>reagents>production']].copy()
        out_path = os.path.join(raw_out_dir, f'raw_{name}.csv')
        out.to_csv(out_path, index=False)
        print(f'  {name}: {len(out)} 条 → {out_path}')
        stats[name] = int(len(out))

    # 泄露验证
    train_key_set = set(train_df['canonical_key'])
    val_key_set = set(val_df['canonical_key'])
    print(f'  验证: train={len(train_key_set)}, val={len(val_key_set)}, '
          f'重叠={len(train_key_set & val_key_set)}, '
          f'与S2 test重叠={len((train_key_set | val_key_set) & stage2_test_keys)}')
    stats['train_unique_reactions'] = int(len(train_key_set))
    stats['val_unique_reactions'] = int(len(val_key_set))
    stats['train_val_overlap'] = int(len(train_key_set & val_key_set))
    stats['train_val_vs_stage2_test_overlap'] = int(len((train_key_set | val_key_set) & stage2_test_keys))
    stats['canonical_key_uses_unmapped_reaction'] = True
    stats['training_reaction_preserves_atom_mapping'] = True

    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'  摘要: {summary_path}')
    return stats


# ──────────────────────────────────────────────────────
# 12. 主入口
# ──────────────────────────────────────────────────────

def main(input_dir: str, output_dir: str, name_to_smiles_path: str = None,
         skip_raw_clean: bool = False, do_stage1: bool = False,
         process_uspto: bool = False,
         min_label_freq: int = DEFAULT_MIN_LABEL_FREQ,
         label_freq_scope: str = DEFAULT_LABEL_FREQ_SCOPE,
         min_yield: float = DEFAULT_MIN_YIELD):
    """
    总调度函数，按 data_process.md 规定的顺序执行全流程：

      Step 0：原始文件预清理
        → 加载全部反应族
        → (d) 全局角色重分配
        → 逐族 (a)(b)(c)(d)(f)(g) → (h) 全局低频过滤 → 逐族 (i)(j) + 输出
        → (可选) Stage 1 逆合成路线数据
        → (可选) USPTO 数据过滤处理
    """
    os.makedirs(output_dir, exist_ok=True)

    resolved_mapping_path = resolve_name_to_smiles_path(name_to_smiles_path)
    name_to_smiles = (
        load_name_to_smiles(str(resolved_mapping_path)) if resolved_mapping_path else {}
    )
    if name_to_smiles:
        print(f'已加载 {len(name_to_smiles)} 条 name→canonical token 映射: {resolved_mapping_path}')
    else:
        print('未加载 name→canonical token 映射表，使用纯字符串标准化')
    print(f'最小保留产率阈值: yield >= {min_yield}')

    # ═══════════════════════════════════════════
    # 步骤 0：原始文件预清理
    # ═══════════════════════════════════════════
    if not skip_raw_clean:
        print('=' * 60)
        print('Step 0: 原始文件预清理（删除尾部行/无用列/多步反应）')
        print('=' * 60)
        clean_raw_directory(input_dir)
    else:
        print('跳过 Step 0（--skip_raw_clean），假定数据已清理')

    # ═══════════════════════════════════════════
    # 步骤 1：加载全部反应族的原始数据
    # ═══════════════════════════════════════════
    print('=' * 60)
    print('加载原始数据')
    print('=' * 60)
    raw_data = {}
    for rtype in REACTION_TYPES:
        rdir = os.path.join(input_dir, rtype)
        if not os.path.isdir(rdir):
            print(f'  跳过 {rtype}：目录不存在')
            continue
        files = sorted([f for f in os.listdir(rdir)
                        if (f.endswith('.xlsx') or f.endswith('.csv'))
                        and not f.startswith('~$')])
        if not files:
            print(f'  跳过 {rtype}：无数据文件')
            continue
        dfs = [load_file(os.path.join(rdir, f)) for f in files]
        df = pd.concat(dfs, axis=0, ignore_index=True)
        # 在 Stage 2 加载时过滤单步反应（保留多步反应在原始文件中供 Stage 1 使用）
        if 'Number of Reaction Steps' in df.columns:
            df = df[df['Number of Reaction Steps'] == 1].reset_index(drop=True)
        raw_data[rtype] = df
        print(f'  {rtype}: {len(raw_data[rtype])} 条记录')

    if not raw_data:
        print('未找到任何反应数据！')
        return

    # ═══════════════════════════════════════════
    # 步骤 (d)：全局试剂/溶剂角色重分配
    # ═══════════════════════════════════════════
    print('\n' + '=' * 60)
    print('(d) 角色重分配（跨所有反应族）')
    print('=' * 60)
    raw_data = reassign_roles_all_families(raw_data)

    # ═══════════════════════════════════════════
    # 步骤 (a)-(f)：逐族清洗到复杂度过滤
    # ═══════════════════════════════════════════
    print('\n' + '=' * 60)
    print('(a-f) 逐族清洗：合法性 → 温度 → 催化剂合并 → '
          '标签标准化 → 复杂度过滤')
    print('=' * 60)
    for rtype in raw_data:
        raw_before = len(raw_data[rtype])
        raw_data[rtype] = cleanup_to_complexity(
            rtype, raw_data[rtype], name_to_smiles, min_yield=min_yield)
        print(f'  {rtype}: {raw_before} → {len(raw_data[rtype])} 条')

    # ═══════════════════════════════════════════
    # 步骤 (g)：全局低频标签过滤
    # ═══════════════════════════════════════════
    print('\n' + '=' * 60)
    print(f'(g) 低频标签过滤（scope={label_freq_scope}, 频次 < {min_label_freq}）')
    print('=' * 60)
    all_entries_before = sum(len(v) for v in raw_data.values())
    raw_data = filter_rare_labels_all_families(
        raw_data,
        min_freq=min_label_freq,
        scope=label_freq_scope,
    )
    all_entries_after = sum(len(v) for v in raw_data.values())
    print(f'  {all_entries_before} → {all_entries_after} 条')

    # ═══════════════════════════════════════════
    # 步骤 (h)-(i)：逐族去重 + 划分 + 输出
    # ═══════════════════════════════════════════
    all_stats = {}
    for rtype, df in raw_data.items():
        print(f'\n{"=" * 60}')
        print(f'(h-i) 收尾输出: {rtype}')
        print(f'{"=" * 60}')
        stats = {}
        stats = finalize_and_output(rtype, df, output_dir, stats)
        all_stats[rtype] = stats
        print(f'  记录数: → 复杂度过滤后={stats["after_complexity"]} → '
              f'去重后={stats["after_dedup"]}')
        print(f'  划分: train={stats["train"]}, val={stats["val"]}, '
              f'test={stats["test"]}')
        print(f'  反应数: total={stats["total_reactions"]}, '
              f'train={stats["train_reactions"]}, '
              f'val={stats["val_reactions"]}, '
              f'test={stats["test_reactions"]}')

    # ═══════════════════════════════════════════
    # 全局汇总
    # ═══════════════════════════════════════════
    total_train = sum(s['train'] for s in all_stats.values())
    total_val = sum(s['val'] for s in all_stats.values())
    total_test = sum(s['test'] for s in all_stats.values())
    print(f'\n{"=" * 60}')
    print(f'全族汇总: train={total_train}, val={total_val}, test={total_test}')
    print(f'输出路径: {output_dir}/reaction_processed_{{family}}_catmerge/')

    # ═══════════════════════════════════════════
    # Stage 1：逆合成路线数据（可选）
    # ═══════════════════════════════════════════
    if do_stage1:
        print(f'\n{"=" * 60}')
        print('Stage 1: 生成逆合成路线数据（EditRetro 微调用）')
        print(f'{"=" * 60}')
        for rtype in raw_data:
            print(f'\n  处理: {rtype}')
            stage2_root = os.path.join(
                output_dir, f'reaction_processed_{rtype}_catmerge')
            collect_stage1_route_data(
                rtype, input_dir, output_dir, stage2_root)
        print(f'\n  Stage 1 输出: {output_dir}/editretro/datasets/')
    else:
        print(f'\n  跳过 Stage 1（使用 --do_stage1 启用）')

    # ═══════════════════════════════════════════
    # USPTO：通用逆合成数据过滤（可选）
    # ═══════════════════════════════════════════
    if process_uspto:
        process_uspto_data(output_dir)
    else:
        print(f'\n  跳过 USPTO 处理（使用 --process_uspto 启用）')

    print(f'{"=" * 60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reaxys 反应数据全流程预处理（含 Stage 1 + Stage 2 + USPTO）')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='reaxys_input/ 目录路径（含各反应族子目录）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('--name_to_smiles', type=str, default=None,
                        help='可选 TSV 文件: name<TAB>canonical_token；不传则自动查找 data_preprocess/name_to_smiles.tsv')
    parser.add_argument('--skip_raw_clean', action='store_true',
                        help='跳过 Step 0 原始文件预清理（数据已清理时使用）')
    parser.add_argument('--do_stage1', action='store_true',
                        help='生成 Stage 1 逆合成路线数据（需安装 RXNMapper）')
    parser.add_argument('--process_uspto', action='store_true',
                        help='处理 USPTO 逆合成数据（去原子映射 + 防泄露过滤）')
    parser.add_argument('--min_label_freq', type=int, default=DEFAULT_MIN_LABEL_FREQ,
                        help=f'低频标签过滤阈值，默认 {DEFAULT_MIN_LABEL_FREQ}')
    parser.add_argument('--label_freq_scope', type=str, default=DEFAULT_LABEL_FREQ_SCOPE,
                        choices=sorted(VALID_LABEL_FREQ_SCOPES),
                        help=f'低频标签频次统计范围，默认 {DEFAULT_LABEL_FREQ_SCOPE}')
    parser.add_argument('--min_yield', type=float, default=DEFAULT_MIN_YIELD,
                        help=f'最小保留产率阈值，默认 {DEFAULT_MIN_YIELD}；低于该值的记录会被删除')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.name_to_smiles,
         args.skip_raw_clean, args.do_stage1, args.process_uspto,
         args.min_label_freq, args.label_freq_scope, args.min_yield)
