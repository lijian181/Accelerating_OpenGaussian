# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np

def _l2norm(x, eps=1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)

class RPQSidecar:
    """
    Runtime for 2-level Residual PQ on OPQ-rotated CLIP space.
    Files expected in rpq_dir (see meta.json):
      R, C1, C2, codes1, codes2, leaf_ind, leaf_score (optional), occu_count (opt)
    Metric: cosine_on_unit (query单位化；库向量用查表近似并做归一)
    """
    def __init__(self, rpq_dir: str):
        self.dir = Path(rpq_dir)
        with open(self.dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.keys = meta["keys"]
        self.D   = int(meta.get("dim", 512))
        self.M   = int(meta.get("M"))
        self.Ks  = int(meta.get("Ks"))
        self.metric = meta.get("metric", "cosine_on_unit")

        # load arrays
        self.R       = np.load(self.dir / self.keys["R"])          # (D,D)
        self.C1      = np.load(self.dir / self.keys["C1"])         # (M,Ks,dsub)
        self.C2      = np.load(self.dir / self.keys["C2"])         # (M,Ks,dsub)
        self.codes1  = np.load(self.dir / self.keys["codes1"])     # (N,M)
        self.codes2  = np.load(self.dir / self.keys["codes2"])     # (N,M)
        self.leaf_ind   = np.load(self.dir / self.keys["leaf_ind"])    if "leaf_ind"   in self.keys else None
        self.leaf_score = np.load(self.dir / self.keys["leaf_score"])  if "leaf_score" in self.keys else None
        self.occu_count = np.load(self.dir / self.keys["occu_count"])  if "occu_count" in self.keys else None

        self.N = self.codes1.shape[0]
        self.dsub = self.C1.shape[-1]
        assert self.D == self.M * self.dsub, "dim mismatch"
        # 预计算每个子空间内的平方范数和交叉项，便于余弦归一化
        # norms1[m,k] = ||C1[m,k]||^2 ; norms2[m,k] = ||C2[m,k]||^2
        self.norms1 = np.sum(self.C1 * self.C1, axis=-1).astype(np.float32)  # (M,Ks)
        self.norms2 = np.sum(self.C2 * self.C2, axis=-1).astype(np.float32)  # (M,Ks)
        # cross[m,k1,k2] = 2 * <C1[m,k1], C2[m,k2]>
        # Ks=64 时大小为 64*64*64 ≈ 262k float，完全可接受
        self.cross  = np.empty((self.M, self.Ks, self.Ks), dtype=np.float32)
        for m in range(self.M):
            self.cross[m] = 2.0 * (self.C1[m] @ self.C2[m].T)

        # 无效行（训练阶段我们对无效行 codes 置 0）
        self.invalid_mask = np.all(self.codes1 == 0, axis=1) & np.all(self.codes2 == 0, axis=1)

    def _split(self, Y):
        # 把 512D 切成 M 个 dsub 片段
        return [Y[:, m*self.dsub:(m+1)*self.dsub] for m in range(self.M)]

    def _query_to_scores_innerprod(self, q_rot: np.ndarray) -> np.ndarray:
        """
        内积打分（不做归一化），主要用于debug或你想完全追求速度时。
        q_rot: (D,) = q @ R
        return: scores (N,)
        """
        # 预计算查表：T1[m,k] = <q'_m, C1[m,k]> ; T2 同理
        scores = np.zeros((self.N,), dtype=np.float32)
        for m in range(self.M):
            sl = slice(m*self.dsub, (m+1)*self.dsub)
            q_m = q_rot[sl]                                 # (dsub,)
            T1 = self.C1[m] @ q_m                           # (Ks,)
            T2 = self.C2[m] @ q_m                           # (Ks,)
            # gather & 累加
            scores += T1[self.codes1[:, m]] + T2[self.codes2[:, m]]
        # 无效行保底为 0
        scores[self.invalid_mask] = 0.0
        return scores

    def _approx_norm_per_leaf(self) -> np.ndarray:
        """
        估计 ||Y_hat||，Y_hat = concat_m (C1[m,c1] + C2[m,c2])
        因为子空间拼接，无跨子空间交叉项；每个子空间内部考虑 cross。
        """
        # sum_m ( ||C1||^2 + ||C2||^2 + 2 <C1,C2> )
        s = np.zeros((self.N,), dtype=np.float32)
        for m in range(self.M):
            idx1 = self.codes1[:, m]
            idx2 = self.codes2[:, m]
            s += self.norms1[m, idx1]
            s += self.norms2[m, idx2]
            s += self.cross[m, idx1, idx2]
        s = np.maximum(s, 1e-12)
        return np.sqrt(s, dtype=np.float32)

    def query_scores(self, q: np.ndarray, use_cosine=True) -> np.ndarray:
        """
        q: (D,)  一般是文本或图像的 CLIP 向量（未旋转）
        return scores: (N,)
        """
        assert q.shape[-1] == self.D
        if self.metric == "cosine_on_unit" or use_cosine:
            q = _l2norm(q.reshape(1, -1))[0]
        q_rot = q @ self.R  # (D,)

        scores = self._query_to_scores_innerprod(q_rot)  # 先算内积分数
        if self.metric == "cosine_on_unit" and use_cosine:
            # 近似余弦：score / (||q|| * ||Yhat||) ; 其中 ||q||=1
            norms = self._approx_norm_per_leaf()
            scores = scores / norms
            scores[self.invalid_mask] = 0.0
        return scores

    def topk(self, q: np.ndarray, k: int = 100, use_cosine=True):
        """
        返回 topk 的 (indices, scores)，按分数降序
        """
        scores = self.query_scores(q, use_cosine=use_cosine)
        if k >= self.N:
            order = np.argsort(-scores)
        else:
            # partial topk
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return order, scores[order]

    # 可选：对一批 leaf 解码近似向量（便于 debug 或可视化）
    def reconstruct_subset(self, idxs: np.ndarray) -> np.ndarray:
        """
        返回近似的 Y_hat (OPQ域)，你也可以后续乘 R^T 回原域。
        """
        idxs = np.asarray(idxs, dtype=np.int64)
        out = np.zeros((idxs.shape[0], self.D), dtype=np.float32)
        for m in range(self.M):
            c1 = self.C1[m][self.codes1[idxs, m]]  # (B,dsub)
            c2 = self.C2[m][self.codes2[idxs, m]]  # (B,dsub)
            out[:, m*self.dsub:(m+1)*self.dsub] = c1 + c2
        return out

    # 可选：把 OPQ 域向量映射回原始 512D（便于做和原 leaf_feat 的余弦对比）
    def to_original_space(self, Y_opq: np.ndarray) -> np.ndarray:
        return Y_opq @ self.R.T
