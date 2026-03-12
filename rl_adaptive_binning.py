# -*- coding: utf-8 -*-
"""
MAGW 自适应分窗（上皮细胞恶性轴 / MCS）
- 输入：epi_expression_all_NS.csv (Gene x Cell, 第一列为 Gene)
- 轴：优先使用 epi_metadata_states_NS.csv 的 MCS / NT_MCS（找不到再尝试 NT_axis / NT_PT）
- 关键修正：
  1) 剔除 MCS 为 NaN 的细胞，仅用有效细胞训练与分窗；NaN 细胞最终标为 "NA"
  2) 状态增加 axis_delta 特征，提升“该不该切”的可分性
  3) 合理的动作屏蔽与奖励重塑，避免每回合机械切满
"""

import os, json, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path

# 线程限制（可选，避免过度并行导致卡顿）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ----------------- 路径配置 -----------------
DATA_DIR = "/home/lab501-1/WorkSpace/Rk_work_GSE131907/data/Grn_input"
FN_EXPR  = os.path.join(DATA_DIR, "epi_expression_all_NS.csv")
FN_META  = os.path.join(DATA_DIR, "epi_metadata_states_NS.csv")
OUT_DIR  = os.path.join(DATA_DIR, "adaptive_windows_rl")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- 依赖 -----------------
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =============== DQN 组件 ===============
class QNet(nn.Module):
    def __init__(self, in_dim, n_act=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_act)
        )
    def forward(self, x):
        return self.net(x)

class Replay:
    def __init__(self, cap=80000):
        self.buf = deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(args)
    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buf)

# =============== DQN 环境 ===============
class SegEnv:
    """
    状态 s = [
        pos/N, (pos-cur_start)/N, budget/Kmax,
        axis_delta,               # 新增：当前段的轴差 (axis[pos-1] - axis[cur_start])
        PCA_mean[use_dims], PCA_var[use_dims]
    ]
    动作 a: 0=继续, 1=切段（非法切段会被屏蔽）
    即时奖励(切刀时):
        r_cut = w_between*between - alpha*within
    终局奖励:
        R = w_between*Σbetween - alpha*Σwithin - lam*K
    另：对非法切刀屏蔽、末段过短惩罚、0 刀强惩罚
    """

    def __init__(self, Z, axis_sorted, Kmax=6, min_seg_ratio=0.02,
                 use_dims=5, w_between=4.0, alpha=0.05, lam=0.04,
                 zero_cut_penalty=25.0, illegal_penalty=0.0):
        self.Z = Z
        self.axis = axis_sorted
        self.N = Z.shape[0]
        self.Kmax = int(Kmax)
        self.min_seg = max(50, int(self.N * min_seg_ratio))
        self.use_dims = int(use_dims)

        self.w_between = float(w_between)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.zero_cut_penalty = float(zero_cut_penalty)
        self.illegal_penalty = float(illegal_penalty)

        self.reset()

    def reset(self):
        self.pos = self.min_seg            # 从留够一段起步
        self.cuts = [0]
        self.budget = self.Kmax
        self._cur_start = 0
        self._cache_stats()
        return self._state()

    def _cache_stats(self):
        a, b = self._cur_start, self.pos
        if b <= a:
            a = min(a, self.N - 1); b = min(a + 1, self.N)
        X = self.Z[a:b, :self.use_dims]
        self.cur_mean = X.mean(axis=0)
        self.cur_var  = X.var(axis=0)

    def _state(self):
        # 轴差：当前段起点到当前位置的轴值差（轴已升序）
        axis_delta = float(self.axis[max(0, self.pos-1)] - self.axis[self._cur_start])
        return np.array([
            self.pos / self.N,
            (self.pos - self._cur_start) / max(1, self.N),
            self.budget / max(1, self.Kmax),
            axis_delta,
            *self.cur_mean.tolist(),
            *self.cur_var .tolist(),
        ], dtype=np.float32)

    def _window_quality(self, a, b, prev_mean=None):
        if b <= a:
            a = min(a, self.N - 1); b = min(a + 1, self.N)
        X = self.Z[a:b, :self.use_dims]
        mean = X.mean(axis=0)
        var  = X.var(axis=0).mean()
        between = 0.0 if prev_mean is None else float(np.linalg.norm(mean - prev_mean))
        return between, var, mean

    def _is_cut_legal(self, cut_pos):
        valid_len = cut_pos - self._cur_start
        remaining = self.N - cut_pos
        return (
            self.budget > 0 and
            valid_len >= self.min_seg and
            remaining >= (self.min_seg + (self.budget - 1) * self.min_seg)
        )

    def step(self, act):
        # 非法切刀直接屏蔽为继续
        if act == 1 and not self._is_cut_legal(self.pos):
            act = 0

        done = False
        # 轻微步进成本，让 agent 没事别一直拖
        reward = -0.0003
        info = {}

        if act == 1:
            # 计算当前段质量；与上一段均值作对比
            prev_mean = None
            if len(self.cuts) > 1:
                ps, pe = self.cuts[-2], self._cur_start
                Xprev = self.Z[ps:pe, :self.use_dims]
                prev_mean = Xprev.mean(axis=0)
            between, within, _ = self._window_quality(self._cur_start, self.pos, prev_mean)
            reward += self.w_between * between - self.alpha * within

            self.cuts.append(self.pos)
            self.budget -= 1
            self._cur_start = self.pos

        self.pos += 1

        if self.pos >= self.N:
            done = True
            # 末段过短惩罚
            if (self.N - self._cur_start) < self.min_seg:
                reward -= 2.0

            cuts = self.cuts + [self.N]
            K = len(cuts) - 2
            between_sum, within_sum = 0.0, 0.0
            prev_mean = None
            for i in range(len(cuts) - 1):
                a, b = cuts[i], cuts[i + 1]
                between, within, mean = self._window_quality(a, b, prev_mean)
                between_sum += between
                within_sum  += within
                prev_mean = mean

            reward += (self.w_between * between_sum - self.alpha * within_sum - self.lam * K)
            if K == 0:
                reward -= self.zero_cut_penalty

            info["cuts"] = cuts[1:-1] if len(cuts) > 2 else []
            self.pos = self.N
            self._cache_stats()
        else:
            self._cache_stats()

        if not np.isfinite(reward):
            reward = -5.0
        return self._state(), float(reward), done, info

# =============== DQN 训练 & 推断 ===============
def dqn_train(env, episodes=100, gamma=0.99, lr=1e-3,
              eps_start=1.0, eps_end=0.05, eps_decay=0.99,
              bs=256, tgt_sync=1000):
    in_dim = env._state().shape[0]
    q, qt = QNet(in_dim), QNet(in_dim)
    qt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    mem = Replay()
    eps = eps_start
    loss_hist, rew_hist = [], []
    step = 0

    for ep in range(episodes):
        s = env.reset()
        ep_rew = 0.0
        done = False
        last_info = {}
        while not done:
            if random.random() < eps:
                a = np.random.randint(0, 2)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(q(torch.tensor(s).float().unsqueeze(0))).item())

            s2, r, done, info = env.step(a)
            last_info = info

            if not np.isfinite(r):
                r = -5.0

            mem.push(s, a, r, s2, float(done))
            s = s2
            ep_rew += r
            step += 1

            if len(mem) >= bs:
                S, A, R, S2, D = mem.sample(bs)
                S  = torch.tensor(S).float()
                A  = torch.tensor(A).long()
                R  = torch.tensor(R).float()
                S2 = torch.tensor(S2).float()
                D  = torch.tensor(D).float()

                qv = q(S).gather(1, A.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    next_q = torch.max(qt(S2), dim=1)[0]
                    tv = R + gamma * (1.0 - D) * next_q

                loss = nn.SmoothL1Loss()(qv, tv)
                if torch.isfinite(loss):
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss_hist.append(loss.item())

            if step % tgt_sync == 0:
                qt.load_state_dict(q.state_dict())

        rew_hist.append(ep_rew)
        eps = max(eps_end, eps * eps_decay)
        print(f"[EP {ep+1:03d}] reward={ep_rew:.3f}, eps={eps:.3f}, cuts_ep={len(last_info.get('cuts', []))}")

    # 可视化
    plt.figure()
    if loss_hist:
        import pandas as pd
        plt.plot(pd.Series(loss_hist).rolling(50).mean())
    plt.title("Loss")
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(rew_hist)
    plt.title("Episode Reward")
    plt.savefig(os.path.join(OUT_DIR, "reward_curve.png"), dpi=150)
    plt.close()

    return q

def dqn_infer(env, q, greedy=True):
    s = env.reset()
    done = False
    info = {}
    while not done:
        with torch.no_grad():
            if greedy:
                a = int(torch.argmax(q(torch.tensor(s).float().unsqueeze(0))).item())
            else:
                a = np.random.randint(0, 2)
        s, r, done, info = env.step(a)
    return info.get("cuts", [])

# =============== 主流程 ===============
def main():
    print("=== DQN Adaptive Windows (MCS axis, fixed) ===")

    # ---------- 读取表达 ----------
    if not os.path.exists(FN_EXPR):
        raise FileNotFoundError(f"表达矩阵不存在: {FN_EXPR}")
    df = pd.read_csv(FN_EXPR)
    assert "Gene" in df.columns, "表达矩阵第一列必须是 'Gene'"
    df = df.drop_duplicates(subset=["Gene"]).set_index("Gene")
    expr  = df.values  # G x C
    genes = df.index.values
    cells = df.columns.values
    G, C = expr.shape
    print(f"[Load] expr: genes={G} cells={C}")

    # ---------- 读取轴 ----------
    if not os.path.exists(FN_META):
        raise FileNotFoundError(f"找不到 meta: {FN_META}")
    meta = pd.read_csv(FN_META)
    axis = None
    for k in ["MCS", "NT_MCS", "NT_axis", "NT_PT"]:
        if k in meta.columns:
            axis = np.asarray(meta[k].values)
            print(f"[Axis] use {k} from {os.path.basename(FN_META)}")
            break
    if axis is None:
        raise RuntimeError("找不到 MCS/NT_MCS/NT_axis/NT_PT 任一列")

    assert len(axis) == C, f"轴长度({len(axis)})必须等于细胞数({C})"

    # ---------- 剔除 NaN 轴细胞 ----------
    mask_valid = ~np.isnan(axis)
    n_drop = int((~mask_valid).sum())
    if n_drop > 0:
        print(f"[Axis] 丢弃 {n_drop} 个 MCS 为 NaN 的细胞，保留 {int(mask_valid.sum())} 个用于 DQN 分窗")
    axis_valid  = axis[mask_valid]
    cells_valid = cells[mask_valid]
    expr_valid  = expr[:, mask_valid]

    # 归一化 0-1（仅对有效细胞）
    axis_valid = (axis_valid - np.nanmin(axis_valid)) / (np.nanmax(axis_valid) - np.nanmin(axis_valid) + 1e-8)

    # ---------- 排序 & PCA ----------
    ord_idx = np.argsort(axis_valid)
    axis_sorted = axis_valid[ord_idx]
    X = expr_valid[:, ord_idx].T  # cells_valid x genes

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xp = scaler.fit_transform(X)
    pca = PCA(n_components=20, random_state=0)
    Z = pca.fit_transform(Xp)  # cells_valid x 20
    N = Z.shape[0]
    print(f"[PCA] 降维到: {Z.shape}")

    # ---------- RL ----------
    print("[DQN] 开始训练...")
    env = SegEnv(
        Z=Z,
        axis_sorted=axis_sorted,
        Kmax=6,               # 放宽预算，让策略有探索空间（避免恒等=4）
        min_seg_ratio=0.02,
        use_dims=5,
        w_between=4.0,
        alpha=0.05,
        lam=0.04,             # 稍微提高刀数成本，减少“切满”倾向
        zero_cut_penalty=25.0
    )

    q = dqn_train(
        env,
        episodes=100,
        gamma=0.99,
        lr=1e-3,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.99,
        bs=256,
        tgt_sync=1000,
    )

    cuts = dqn_infer(env, q, greedy=True)
    print("[CUTS]", cuts)
    if len(cuts) == 0:
        raise RuntimeError("DQN 未找到切点。可增大 w_between 或降低 alpha/lam，或增加 episodes。")

    # ---------- 标签映射（仅 valid 细胞有窗口标签） ----------
    W = np.empty(N, dtype=object)
    edges = [0] + cuts + [N]
    for i in range(len(edges) - 1):
        W[edges[i]:edges[i+1]] = f"W{i+1}"

    state_rl_sorted_valid = pd.Series(W, index=cells_valid[ord_idx], name="state_adapt_rl")
    # 全体细胞：默认 NA，再用 valid 覆盖
    state_rl = pd.Series("NA", index=cells, name="state_adapt_rl")
    state_rl.update(state_rl_sorted_valid)
    state_rl.to_csv(os.path.join(OUT_DIR, "state_adapt_rl.csv"))

    # ---------- 可视化 ----------
    plt.figure(figsize=(9,4))
    plt.scatter(np.arange(N), axis_sorted, s=3, c=np.linspace(0,1,N), cmap="viridis", alpha=0.7)
    for c in cuts:
        plt.axvline(c, ls="--", c="red", alpha=0.7)
    plt.title("Adaptive Windows (DQN) on MCS Axis (valid cells)")
    plt.xlabel("Cells (sorted by MCS, valid only)")
    plt.ylabel("MCS (0-1)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "axis_rl.png"), dpi=150)
    plt.close()

    # ---------- 导出每窗表达（仅 valid 细胞） ----------
    expr_sorted_valid = expr_valid[:, ord_idx]
    expr_sorted_df_valid = pd.DataFrame(expr_sorted_valid, index=genes, columns=cells_valid[ord_idx])
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        cs = cells_valid[ord_idx][a:b]
        out = expr_sorted_df_valid.loc[:, cs]
        out.insert(0, "Gene", out.index)
        out.to_csv(os.path.join(OUT_DIR, f"epi_expression_W{i+1}.csv"), index=False)

    # 窗口尺寸报告
    sizes = [edges[i+1]-edges[i] for i in range(len(edges)-1)]
    print("[WINDOW_SIZES(valid)]", sizes)
    print("[DONE] outputs @", OUT_DIR)

if __name__ == '__main__':
    main()
