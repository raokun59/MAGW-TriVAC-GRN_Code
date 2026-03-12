# -*- coding: utf-8 -*-
"""
03_train_gam_mvatt.py — Unsupervised multi-view attention GRN

思路（完全无监督）：
  1) 使用 02_build_view.py 产生的三种视图：
       - view_expr         （原始表达，cells x genes，log1p + per-gene z-score）
       - view_axis_smooth  （沿轨迹平滑后的表达）
       - view_window_resid （按 state 去均值后的残差）
  2) 在给定细胞集合（全局 / 某个 state）上，对每个视图分别计算
       TF x Target 的 Pearson 相关矩阵 r_v，并做 Fisher z 变换：
           z_v = arctanh(r_v)
  3) 对每条边 (TF, Target)，在三个视图上的 z 作为 “多视图特征”，
     用一个解析的注意力机制（attention）做融合：
         α_v ∝ exp(β * |w_v * z_v|)
         z_fused = Σ_v α_v * z_v
     最终 score = |z_fused|，表示多视图上的综合关联强度。
  4) 每个 TF 仅保留 TopK 条边，输出：
       adaptive_windows_rl/grn/GRN_global.csv
       adaptive_windows_rl/grn_by_window/GRN_<state>.csv
     GRN 文件格式：TF, Target, score  （与后续验证脚本兼容）

备注：
  - 完全无监督，不依赖 TRRUST；TRRUST 只在 receive_good.py 里做外部验证。
"""

import os
import sys
import math
from glob import glob

import numpy as np
import pandas as pd

# ---------------- 路径配置 ----------------
DATA = "/home/lab501-1/WorkSpace/Rk_work_GSE131907/data/Grn_input"
RL_DIR = f"{DATA}/adaptive_windows_rl"
VIEWS_DIR = f"{RL_DIR}/views"

GRN_GLB = f"{RL_DIR}/grn";           os.makedirs(GRN_GLB, exist_ok=True)
GRN_WIN = f"{RL_DIR}/grn_by_window"; os.makedirs(GRN_WIN, exist_ok=True)
RES_DIR = f"{RL_DIR}/resources";     os.makedirs(RES_DIR, exist_ok=True)

# ---------------- 超参数（可按需微调） ----------------
USE_VIEWS = {"expr": True, "axis": True, "resid": True}
VIEW_WEIGHTS = {"expr": 1.0, "axis": 1.0, "resid": 1.0}   # 视图权重 w_v
ATTN_BETA = 2.0                # attention 温度 β，越大越偏向 |z| 最大的视图
TOPK_PER_TF = 200              # 每个 TF 输出的最大边数
MIN_CELLS_PER_STATE = 100      # 每个 state 至少多少细胞才导出 GRN

# TF 列表候选路径（优先使用前面的）
TF_CANDS = [
    f"{RES_DIR}/TFs_used.txt",                     # 若之前已跑过其它步骤，会生成
    f"{DATA}/TF2/Homo_sapiens_TF.txt",
    f"{DATA}/TF_all/Homo_sapiens_TF.txt",
    f"{DATA}/DatabaseExtract_v_1.01.csv",
]

# ---------------- 小工具函数 ----------------
def read_table_guess(fp: str, **kw) -> pd.DataFrame:
    if fp.endswith(".parquet"):
        return pd.read_parquet(fp)
    if fp.endswith(".feather"):
        return pd.read_feather(fp)
    return pd.read_csv(fp, **kw)

def load_view(name: str) -> pd.DataFrame:
    for fp in [f"{VIEWS_DIR}/view_{name}.parquet",
               f"{VIEWS_DIR}/view_{name}.feather",
               f"{VIEWS_DIR}/view_{name}.csv.gz",
               f"{VIEWS_DIR}/view_{name}.csv"]:
        if os.path.exists(fp):
            # parquet 已自带 index；csv 这里指定 index_col=0
            return read_table_guess(fp, index_col=0)
    raise FileNotFoundError(f"找不到视图 {name} 于 {VIEWS_DIR}")

def load_labels() -> pd.DataFrame:
    fp = f"{VIEWS_DIR}/train_labels.tsv"
    if not os.path.exists(fp):
        sys.exit(f"[Error] 未找到 train_labels.tsv 于 {VIEWS_DIR}")
    lab = pd.read_csv(fp, sep="\t", index_col=0)
    if "state" not in lab.columns:
        sys.exit("[Error] train_labels.tsv 需要包含 'state' 列")
    lab.index = lab.index.astype(str)
    return lab

def intersect_index(*dfs):
    idx = set(dfs[0].index)
    for d in dfs[1:]:
        idx &= set(d.index)
    idx = pd.Index(sorted(idx))
    return [d.loc[idx] for d in dfs]

def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def fast_corr_block(A: np.ndarray, tf_idx: np.ndarray, tgt_idx: np.ndarray) -> np.ndarray:
    """
    A: (n_cells x n_genes) 已大致标准化（来自 02_build_view：per-gene z-score）。
    返回 len(tf_idx) x len(tgt_idx) 的相关矩阵。
    """
    n = A.shape[0]
    if n < 3:
        return np.zeros((len(tf_idx), len(tgt_idx)), dtype=np.float32)
    # 若每列近似 mean=0, var=1，则协方差即相关系数
    R = (A[:, tf_idx].T @ A[:, tgt_idx]) / max(1, (n - 1))
    return np.clip(R.astype(np.float32), -0.999999, 0.999999)

def multi_view_attention_fuse(z_list, w_list, beta: float = 2.0) -> np.ndarray:
    """
    z_list: 每个视图的 Fisher z 矩阵列表，形状统一为 (n_TF, n_TG)
    w_list: 对应视图权重列表
    返回 fused_z: (n_TF, n_TG)
    """
    Z_stack = np.stack(z_list, axis=0)                    # (V, n_TF, n_TG)
    W = np.array(w_list, dtype=np.float32).reshape(-1, 1, 1)
    absZ_w = np.abs(W * Z_stack)
    # 数值稳定的 softmax：减掉 max
    m = absZ_w.max(axis=0, keepdims=True)
    exp_scores = np.exp(beta * (absZ_w - m))
    alpha = exp_scores / np.clip(exp_scores.sum(axis=0, keepdims=True), 1e-12, None)
    fused_z = np.sum(alpha * Z_stack, axis=0)
    return fused_z.astype(np.float32)

def load_TFs(gene_list):
    geneset = set(gene_list)
    for fp in TF_CANDS:
        if not os.path.exists(fp):
            continue
        try:
            if fp.endswith(".csv"):
                df = pd.read_csv(fp)
                cand_cols = [c for c in ["Symbol","Name","TF","Gene","HGNC symbol","Approved symbol"]
                             if c in df.columns]
                if cand_cols:
                    tfs = df[cand_cols[0]].dropna().astype(str).str.strip().unique().tolist()
                else:
                    tfs = df.iloc[:,0].dropna().astype(str).str.strip().unique().tolist()
            else:
                tfs = [l.strip() for l in open(fp) if l.strip()]
            tfs = [t for t in tfs if t in geneset]
            if tfs:
                print(f"[TF] 来自 {os.path.basename(fp)} 命中 {len(tfs)} / {len(gene_list)}")
                return sorted(set(tfs))
        except Exception as e:
            print(f"[TF] 读取 {fp} 时出错: {e}")
            continue
    print("[TF] 未找到外部 TF 列表，回退到 HVG（上限1000）")
    return sorted(list(geneset))[:1000]

def export_grn_for_cells(EXPR: pd.DataFrame,
                         AXIS: pd.DataFrame,
                         RESI: pd.DataFrame,
                         labels: pd.DataFrame,
                         tf_list,
                         out_fp: str,
                         cell_index=None):
    """
    对指定细胞集合（cell_index=None 表示全体细胞）：
      1) 计算三视图 TF x Target 相关矩阵
      2) Fisher z + 无监督 attention 融合，得到 fused_z
      3) 每个 TF 取 TopK 边，score = |fused_z|
    """
    if cell_index is None:
        cell_index = EXPR.index
    if len(cell_index) < 3:
        print(f"[Skip] {out_fp}: 细胞数不足 3，跳过")
        return

    # 子集三视图
    X = EXPR.loc[cell_index].values.astype(np.float32)
    A = AXIS.loc[cell_index].values.astype(np.float32) if AXIS is not None else None
    R = RESI.loc[cell_index].values.astype(np.float32) if RESI is not None else None

    genes = np.array(EXPR.columns)
    gene_to_idx = {g:i for i,g in enumerate(genes)}
    tf_names = [t for t in tf_list if t in gene_to_idx]
    if not tf_names:
        print(f"[Skip] {out_fp}: TF 列表与基因集无交集")
        return

    tf_idx = np.array([gene_to_idx[t] for t in tf_names], dtype=np.int64)
    tgt_idx = np.arange(len(genes), dtype=np.int64)

    z_list = []
    w_list = []

    # expr 视图
    if USE_VIEWS.get("expr", True):
        r_expr = fast_corr_block(X, tf_idx, tgt_idx)
        z_expr = fisher_z(r_expr)
        z_list.append(z_expr)
        w_list.append(VIEW_WEIGHTS.get("expr", 1.0))

    # axis_smooth 视图
    if USE_VIEWS.get("axis", True) and A is not None:
        r_axis = fast_corr_block(A, tf_idx, tgt_idx)
        z_axis = fisher_z(r_axis)
        z_list.append(z_axis)
        w_list.append(VIEW_WEIGHTS.get("axis", 1.0))

    # window_resid 视图
    if USE_VIEWS.get("resid", True) and R is not None:
        r_resi = fast_corr_block(R, tf_idx, tgt_idx)
        z_resi = fisher_z(r_resi)
        z_list.append(z_resi)
        w_list.append(VIEW_WEIGHTS.get("resid", 1.0))

    if not z_list:
        print(f"[Skip] {out_fp}: 未启用任何视图")
        return

    fused_z = multi_view_attention_fuse(z_list, w_list, beta=ATTN_BETA)   # (n_TF, n_TG)
    score_mat = np.abs(fused_z)                                           # 无符号强度

    rows = []
    for i, tf in enumerate(tf_names):
        scores = score_mat[i, :]
        # 不保留 self-loop
        tf_gene_idx = gene_to_idx[tf]
        scores[tf_gene_idx] = -1.0
        k = min(TOPK_PER_TF, len(genes) - 1)
        # 取 topk 索引
        top_idx = np.argpartition(-scores, k)[:k]
        for j in top_idx:
            rows.append((tf, genes[j], float(scores[j])))

    if not rows:
        print(f"[Warn] {out_fp}: 未得到任何边")
        return

    grn = pd.DataFrame(rows, columns=["TF","Target","score"])
    grn = grn.sort_values(["TF","score"], ascending=[True, False])
    grn.to_csv(out_fp, index=False)
    print(f"[Save] {out_fp} (TF={grn['TF'].nunique()}, edges={len(grn)})")

# ---------------- 主流程 ----------------
def main():
    # 读三视图 + 标签
    EXPR = load_view("expr")
    AXIS = load_view("axis_smooth") if USE_VIEWS.get("axis", True) else None
    RESI = load_view("window_resid") if USE_VIEWS.get("resid", True) else None
    LABELS = load_labels()

    # 对齐索引
    dfs = [EXPR, LABELS]
    if AXIS is not None:
        dfs.append(AXIS)
    if RESI is not None:
        dfs.append(RESI)

    aligned = intersect_index(*dfs)
    EXPR, LABELS, *rest = aligned
    rest_iter = iter(rest)
    AXIS = next(rest_iter) if AXIS is not None else None
    RESI = next(rest_iter) if RESI is not None else None

    genes = np.array(EXPR.columns)
    print(f"[Views] cells={EXPR.shape[0]}, genes={EXPR.shape[1]}")

    # 载入 TF 列表
    TF_LIST = load_TFs(genes.tolist())
    if not TF_LIST:
        sys.exit("[Error] 未能构建 TF 列表")
    print(f"[Vars] TF candidates={len(TF_LIST)}, Targets={len(genes)}")

    # 全局 GRN
    out_global = os.path.join(GRN_GLB, "GRN_global.csv")
    export_grn_for_cells(EXPR, AXIS, RESI, LABELS, TF_LIST, out_global, cell_index=EXPR.index)

    # 按 state 输出 GRN_W*
    if "state" not in LABELS.columns:
        sys.exit("[Error] train_labels.tsv 缺少 'state' 列")
    for st, sub in LABELS.groupby("state"):
        idx = sub.index
        if len(idx) < MIN_CELLS_PER_STATE:
            print(f"[Skip] {st}: 细胞数太少 ({len(idx)})，不导出 GRN")
            continue
        out_fp = os.path.join(GRN_WIN, f"GRN_{st}.csv")
        export_grn_for_cells(EXPR, AXIS, RESI, LABELS, TF_LIST, out_fp, cell_index=idx)

    print("[DONE] Unsupervised multi-view attention GRNs exported.")

if __name__ == "__main__":
    main()
