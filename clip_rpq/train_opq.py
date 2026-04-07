import argparse, numpy as np, sys
import os, json
from pathlib import Path

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # norms: (N, 1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # 只对“非近零”的行做归一化；近零行保留为 0，避免被 eps 放大
    mask1d = (norms[:, 0] > eps).ravel()     # (N,)
    Xn = np.zeros_like(X, dtype=np.float32)  # (N, 512)
    Xn[mask1d] = (X[mask1d] / norms[mask1d]).astype(np.float32)   # norms[mask1d] -> (Nv, 1)
    return Xn


def load_leaf_feat_from_cluster_lang(npz_path: str) -> np.ndarray:
    """
    读取 cluster_lang.npz 的 leaf_feat 并在内存中做 L2 标准化（不回写 npz）
    返回 shape=(N_leaf,512), dtype=float32
    """
    data = np.load(npz_path, allow_pickle=True)
    if "leaf_feat" not in data:
        raise KeyError(f"{npz_path} 缺少 'leaf_feat'")
    X = data["leaf_feat"].astype(np.float32)
    if X.ndim != 2 or X.shape[1] != 512:
        raise ValueError(f"leaf_feat 期望形状 [N_leaf,512]，实际 {X.shape}")
    return l2_normalize_rows(X)

def update_rpq_meta(rpq_dir: str, M: int, R_key: str = "opq_R.npy") -> str:
    """
    在 <rpq_dir>/meta.json 写入/更新 OPQ 信息（M、R 路径等）。
    若 meta.json 已存在则 merge 更新 keys。
    """
    rpq = Path(rpq_dir); rpq.mkdir(parents=True, exist_ok=True)
    meta_path = rpq / "meta.json"
    # 侧车中 stable 字段名与路径
    keys = {
        "R": R_key,
        "C1": None,
        "C2": None,
        "codes1": None,
        "codes2": None,
        "leaf_ind": "leaf_ind.npy",
        "leaf_score": "leaf_score.npy" if (rpq / "leaf_score.npy").exists() else None,
        "occu_count": "occu_count.npy" if (rpq / "occu_count.npy").exists() else None,
    }
    base = {
        "dim": 512,
        "M": M,
        "Ks": None,                 # L1 训练后会填 256
        "normalized": True,         # 我们在单位球面上训练/查询
        "metric": "cosine_on_unit",
        "keys": keys
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        # 合并：顶层覆盖 M/normalized/metric；keys 逐项更新
        old.update({k: v for k, v in base.items() if k != "keys"})
        ok = old.get("keys", {})
        ok.update(keys)
        old["keys"] = ok
        meta = old
    else:
        meta = base
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(meta_path)

def train_opq_faiss(Zs, D=128, M=16, epochs=8):
    import faiss
    opq = faiss.OPQMatrix(D, M, D)   # 1 sub-quantizer per subvector
    opq.niter = epochs
    opq.verbose = True
    opq.train(Zs.astype(np.float32))
    R = faiss.vector_to_array(opq.A).reshape(D, D).T  # 注意转置对齐
    return R.astype(np.float32)

def train_opq_simple(Zs, D=128, M=16, epochs=8, seed=0):
    """
    简化退化版：每轮对旋转后的分段做PCA对齐，再用SVD强制正交化。
    效果不如FAISS，但可用。
    """
    np.random.seed(seed)
    R = np.eye(D, dtype=np.float32)
    d = D // M
    for ep in range(epochs):
        Zr = Zs @ R
        # 近似：让每段协方差尽量“去相关”
        R_update = np.zeros((D, D), dtype=np.float32)
        for m in range(M):
            Xm = Zr[:, m*d:(m+1)*d]
            # PCA: 协方差特征分解
            C = np.cov(Xm, rowvar=False)
            w, V = np.linalg.eigh(C)
            # 取V作为该段的旋转（从大到小）
            V = V[:, ::-1].astype(np.float32)
            R_update[m*d:(m+1)*d, m*d:(m+1)*d] = V
        R = (R @ R_update).astype(np.float32)
        # 再正交化（SVD投影）
        U, _, Vt = np.linalg.svd(R, full_matrices=False)
        R = (U @ Vt).astype(np.float32)
        print(f"[OPQ-simple] epoch {ep+1}/{epochs}")
    return R

def main():
    ap = argparse.ArgumentParser()

    # 二选一：旧管线 vs leaf 模式
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=str,
                      help="旧管线：OPQ 训练样本 .npy（已 L2 归一化）")
    mode.add_argument("--cluster-lang", type=str,
                      help="leaf 模式：cluster_lang.npz 路径，直接读取其中 leaf_feat 做 OPQ")

    # 输出位置：按模式分别要求
    ap.add_argument("--out", type=str,
                    help="旧管线：输出 R 的路径（.npy）")
    ap.add_argument("--rpq-dir", type=str,
                    help="leaf 模式：侧车输出目录（写 opq_R.npy 并更新 meta.json）")

    # 其余超参
    ap.add_argument("--M", type=int, default=64, help="OPQ 子空间数（建议 64）")
    ap.add_argument("--epochs", type=int, default=8, help="OPQ 训练迭代")

    args = ap.parse_args()

    # —— leaf 模式 —— #
    if args.cluster_lang is not None:
        if not args.rpq_dir:
            ap.error("--rpq-dir 是 leaf 模式必需参数（侧车输出目录）")
        # 读取 leaf_feat 并 L2 归一化
        X = load_leaf_feat_from_cluster_lang(args.cluster_lang)  # (N,512)

        # 统计范数分布
        row_norms = np.linalg.norm(X, axis=1)
        valid = row_norms > 1e-6
        print(f"[OPQ] valid rows for training: {valid.sum()}/{X.shape[0]} "
              f"(invalid/near-zero: {(~valid).sum()})")
        print(f"[OPQ] norms: min={row_norms.min():.4g}, median={np.median(row_norms):.4g}, "
              f"mean={row_norms.mean():.4f}, max={row_norms.max():.4g}")
        Zs = X[valid].astype(np.float32)
        D = Zs.shape[1]
        if D != 512:
            raise ValueError(f"期望 512 维，实际 {D}")
        if 512 % args.M != 0:
            ap.error(f"--M 必须整除 512（当前 M={args.M} -> 子维不是整数）")

        # 训练 OPQ（faiss 优先，失败 fallback）
        try:
            R = train_opq_faiss(Zs, D=D, M=args.M, epochs=args.epochs)
            print("[OPQ] Using FAISS OPQMatrix.")
        except Exception as e:
            print(f"[OPQ] FAISS failed: {e}\n[OPQ] Fallback to simple OPQ.")
            R = train_opq_simple(Zs, D=D, M=args.M, epochs=args.epochs)

        # 侧车落盘 + meta 更新
        rpq_dir = Path(args.rpq_dir);
        rpq_dir.mkdir(parents=True, exist_ok=True)
        out_path = rpq_dir / "opq_R.npy"
        np.save(out_path, R.astype(np.float32))
        print(f"[OPQ] Saved R to {out_path}, shape={R.shape}, dtype={R.dtype}")

        meta_path = update_rpq_meta(str(rpq_dir), M=args.M, R_key=out_path.name)
        print(f"[OPQ] meta.json updated: {meta_path}")

        # 可选自检
        try:
            frob = np.linalg.norm(R.T @ R - np.eye(D), ord='fro')
            print(f"[OPQ] ||R^T R - I||_F = {frob:.4e}")
            mean_norm = float(np.linalg.norm(Zs, axis=1).mean())
            mean_norm_train = float(np.linalg.norm(Zs, axis=1).mean())
            print(f"[OPQ] mean L2-norm on training rows: {mean_norm_train:.4f} (should≈1)")

        except Exception:
            pass

        return  # 结束 leaf 模式

    # —— 旧管线 —— #
    if not args.sample or not args.out:
        ap.error("--sample 与 --out 是旧管线必需参数（未使用 --cluster-lang 时）")

    Zs = np.load(args.sample)  # (S, D) 已 L2
    D = Zs.shape[1]
    if D % args.M != 0:
        ap.error(f"--M 必须整除特征维度（当前 D={D}, M={args.M}）")

    try:
        R = train_opq_faiss(Zs, D=D, M=args.M, epochs=args.epochs)
        print("[OPQ] Using FAISS OPQMatrix.")
    except Exception as e:
        print(f"[OPQ] FAISS failed: {e}\nFallback to simple OPQ.")
        R = train_opq_simple(Zs, D=D, M=args.M, epochs=args.epochs)

    np.save(args.out, R.astype(np.float32))
    print(f"[OPQ] Saved to {args.out}, shape={R.shape}, dtype={R.dtype}")


if __name__ == "__main__":
    main()
