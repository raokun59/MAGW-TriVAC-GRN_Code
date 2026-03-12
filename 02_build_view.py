# =============================
# 02_build_view.py  (REVISED)
# =============================
# Purpose
#   Build three complementary views for GRN construction:
#     - view_expr:      log1p + per-gene z-score (cells x genes)
#     - view_axis_smooth: spline (B-spline) + Ridge vs. axis (GAM-style)
#     - view_window_resid: expression residual after per-state mean removal
#   Also writes train_labels.tsv (state, axis, optional rare_flag & train_w)
# Notes
#   * Robust readers for state_adapt_rl.csv (with/without header).
#   * Avoid chained-assignment warnings.
#   * Saves parquet (fallback to csv if parquet not available).
#
# References
#   - SplineTransformer for B-spline bases in scikit-learn (used for GAM-like smoothing)
#   - tradeSeq (NB-GAM) shows GAM is standard for pseudotime modeling in scRNA-seq
#
# Paths assumed
#   DATA=/home/.../GSE131907/data/Grn_input
#   V1  =DATA/adaptive_windows_rl_V1

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge

# ---------------------- paths ----------------------
DATA = "/home/lab501-1/WorkSpace/Rk_work_GSE131907/data/Grn_input"
V1   = f"{DATA}/adaptive_windows_rl"
VIEWS= f"{V1}/views"
Path(VIEWS).mkdir(parents=True, exist_ok=True)

# ---------------------- io helpers ----------------------
def _save_df(df: pd.DataFrame, fp_parquet: str):
    """Save DataFrame to parquet if possible, else csv."""
    try:
        df.to_parquet(fp_parquet)
    except Exception:
        df.to_csv(fp_parquet.replace('.parquet', '.csv'))

# ---------------------- 1) read expression ----------------------
expr_fp = f"{DATA}/epi_expression_all_NS.csv"
if not os.path.exists(expr_fp):
    sys.exit(f"[Error] missing expression matrix: {expr_fp}")

df = pd.read_csv(expr_fp)
assert "Gene" in df.columns, "The first column must be 'Gene'"
df = df.drop_duplicates(subset=["Gene"]).set_index("Gene")
# ensure unique gene index
if df.index.duplicated().any():
    df = df[~df.index.duplicated(keep="first")]
print(f"[Expr] genes x cells: {df.shape}")

# ---------------------- 2) read labels: state + axis + (optional) rare_flag/train_w ----------------------
# state
state_fp_pref = [
    f"{V1}/state_adapt_rl.csv",
    f"{DATA}/adaptive_windows_rl/state_adapt_rl.csv",
    f"{DATA}/state_adapt_rl.csv",
]
st = None
last_err = None
for fp in state_fp_pref:
    if not os.path.exists(fp):
        continue
    try:
        st = pd.read_csv(fp, header=None, names=["cell","state"]).set_index("cell")[["state"]]
        break
    except Exception as e:
        last_err = e
        try:
            tmp = pd.read_csv(fp)
            if {"cell","state"}.issubset(tmp.columns):
                st = tmp.set_index("cell")[ ["state"] ]
                break
        except Exception as e2:
            last_err = e2
if st is None:
    raise FileNotFoundError("state_adapt_rl.csv not found in expected locations")

# axis from meta (prefer *_plus)
# ===== robust meta reader: ensure we have a 'cell' column, then pick axis =====
meta_plus = f"{DATA}/epi_metadata_states_NS_plus.csv"
meta_base = f"{DATA}/epi_metadata_states_NS.csv"
meta_fp   = meta_plus if os.path.exists(meta_plus) else meta_base

meta = pd.read_csv(meta_fp)

# 你的文件第一列名为 "index"，就是细胞ID；改名为 'cell'
if "cell" not in meta.columns and "index" in meta.columns:
    meta = meta.rename(columns={"index": "cell"})

# 如果仍然没有 'cell'（比如有时第一列叫 Unnamed: 0），用首列兜底
if "cell" not in meta.columns:
    meta = pd.read_csv(meta_fp, index_col=0)   # 把第一列当索引读进来
    meta.index.name = "cell"
    meta = meta.reset_index()                  # 变成普通列 'cell'

# 规范、去重、设为索引
meta["cell"] = meta["cell"].astype(str).str.strip()
meta = meta.drop_duplicates("cell").set_index("cell")

# 选轴列（任意一个存在即可）
axis_col = next((k for k in ["NT_PT", "NT_axis", "NT_MCS", "MCS"] if k in meta.columns), None)
assert axis_col is not None, f"No axis column found in meta; available columns: {list(meta.columns)[:12]}"
meta = meta[[axis_col]].rename(columns={axis_col: "axis"})

# optional rare flag (from R step)
rare_fp = f"{V1}/rare_by_window.tsv"
if os.path.exists(rare_fp):
    rare = pd.read_csv(rare_fp, sep='\t')
    if {"cell","state","rare_flag"}.issubset(rare.columns):
        rare = rare.drop_duplicates("cell").set_index("cell")[ ["rare_flag"] ]
    else:
        rare = None
else:
    rare = None

# assemble labels
tmp = st.join(meta, how="inner")
if rare is not None:
    tmp = tmp.join(rare, how="left")
else:
    tmp["rare_flag"] = np.nan

# weights: by default 1.0; if you want weighting for rare cells, change here
train_w = pd.Series(1.0, index=tmp.index, name="train_w")
labels = pd.concat([tmp[["state","axis","rare_flag"]], train_w], axis=1)
labels = labels.dropna(subset=["state","axis"])  # keep labeled cells only
labels.index = labels.index.astype(str)

# ---------------------- 3) align cells with expression ----------------------
all_cells = [c for c in df.columns if c in labels.index]
X = df[all_cells].copy()                 # genes x cells
labels = labels.loc[all_cells].copy()    # align
print("[Align] cells after join:", X.shape[1])

# ---------------------- 4) choose HVG (variance topK) ----------------------
var = X.var(axis=1)
HVG = var.sort_values(ascending=False).head(3000).index
X = X.loc[HVG]
print("[HVG] n=", len(HVG))

# ---------------------- 5) build views ----------------------
# 5.1 EXPR: log1p then per-gene z-score across cells (scikit-learn StandardScaler)
Xlog = np.log1p(X)
scaler = StandardScaler(with_mean=True, with_std=True)
Xsc   = pd.DataFrame(
    scaler.fit_transform(Xlog.T),   # cells x genes
    index=Xlog.columns, columns=Xlog.index
)
EXPR = Xsc.copy()

# 5.2 AXIS_SMOOTH: B-spline basis over axis + Ridge per gene
ax = labels["axis"].astype(float).values.reshape(-1,1)
# normalize axis to [0,1] for numerical stability
ax_min, ax_max = float(np.nanmin(ax)), float(np.nanmax(ax))
ax = (ax - ax_min) / (ax_max - ax_min + 1e-12)

spline = SplineTransformer(n_knots=7, degree=3, include_bias=False)
B = spline.fit_transform(ax)  # (n_cells x n_basis)
R = Ridge(alpha=1.0, random_state=0)

pred = np.zeros_like(EXPR.values, dtype=float)
for j, g in enumerate(EXPR.columns):
    y = EXPR[g].values
    R.fit(B, y)
    pred[:, j] = R.predict(B)
AXIS_SMOOTH = pd.DataFrame(pred, index=EXPR.index, columns=EXPR.columns)

# 5.3 WINDOW_RESID: subtract per-state mean (emphasize within-state variation)
wm = EXPR.join(labels[["state"]]).groupby("state").transform("mean")
WINDOW_RESID = EXPR - wm

# ---------------------- 6) save outputs ----------------------
# labels file (keep only used cells)
labels.to_csv(f"{VIEWS}/train_labels.tsv", sep='\t')

# views
_save_df(EXPR,         f"{VIEWS}/view_expr.parquet")
_save_df(AXIS_SMOOTH,  f"{VIEWS}/view_axis_smooth.parquet")
_save_df(WINDOW_RESID, f"{VIEWS}/view_window_resid.parquet")

# bookkeeping
with open(f"{VIEWS}/HVG_list.txt", "w") as f:
    for g in EXPR.columns: f.write(str(g)+"\n")
with open(f"{VIEWS}/cells_kept.txt", "w") as f:
    for c in EXPR.index: f.write(str(c)+"\n")

print("[Save] views ->", VIEWS)




