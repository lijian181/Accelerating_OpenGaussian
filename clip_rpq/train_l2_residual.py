# -*- coding: utf-8 -*-
"""
train_l2_residual.py  (cluster_lang + RPQ sidecar version)

Usage (example):
  python clip_rpq/train_l2_residual.py ^
    --cluster-lang "E:\opengaussians\OpenGaussian-main\scripts\output_lexisplat\lerf\figurines\cluster_lang.npz" ^
    --rpq-dir      "E:\opengaussians\OpenGaussian-main\scripts\output_lexisplat\lerf\figurines\cluster_lang_rpq" ^
    --M 64 --Ks 64 --kmeans-iters 40

What it does:
- Load leaf_feat (N,512) from cluster_lang.npz
- Load OPQ rotation R, L1 codebook C1, and L1 codes codes1 from rpq sidecar
- Compute residual E = (X @ R) - recon_L1  in OPQ domain
- Train L2 PQ on E (per subspace), with zero-centroid convention
- Encode all leaves to codes2, set invalid rows' codes to 0
- Save codebook_l2.npy, codes2.npy, and update meta.json (keys.C2 / keys.codes2)
- Print validation: cosine(valid, Y vs Y1_hat) and cosine(valid, Y vs Y1_hat+Y2_hat)
"""

import argparse
import json
from pathlib import Path

import numpy as np

# --------- Optional: use faiss if available for kmeans ----------
_HAS_FAISS = False
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


def set_seed(seed: int):
    np.random.seed(seed)


def l2norm_rows(X: np.ndarray, eps: float = 1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def cosine_mean(A: np.ndarray, B: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is not None:
        A = A[mask]
        B = B[mask]
    if A.shape[0] == 0:
        return float("nan")
    An = l2norm_rows(A)
    Bn = l2norm_rows(B)
    return float(np.mean(np.sum(An * Bn, axis=1)))


def split_subspaces(X: np.ndarray, M: int):
    N, D = X.shape
    assert D % M == 0, f"D={D} not divisible by M={M}"
    dsub = D // M
    return [X[:, m * dsub:(m + 1) * dsub] for m in range(M)], dsub


def assign_codes_l2(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    X: (N, dsub), C: (Ks, dsub)
    return indices (N,)
    """
    # dist^2 = ||x||^2 + ||c||^2 - 2 x c^T
    x2 = np.sum(X * X, axis=1, keepdims=True)          # (N,1)
    c2 = np.sum(C * C, axis=1, keepdims=True).T        # (1,Ks)
    D2 = x2 + c2 - 2.0 * (X @ C.T)                     # (N,Ks)
    return np.argmin(D2, axis=1).astype(np.int64)


def kmeans_train(X: np.ndarray, Ks: int, niter: int, seed: int) -> np.ndarray:
    X = X.astype(np.float32)
    n, d = X.shape
    Ks_eff = int(Ks)
    if Ks_eff > n:
        Ks_eff = n  # 数据量小于聚类数时，先把 Ks 压到 n，避免 nx >= k 断言

    if _HAS_FAISS:
        km = faiss.Kmeans(
            d, Ks_eff, niter=niter, verbose=False, seed=seed,
            spherical=False, nredo=1, min_points_per_centroid=1
        )
        km.train(X)

        # --- 关键改动：健壮地取出质心 ---
        cent = getattr(km, "centroids", None)
        if isinstance(cent, np.ndarray):
            C = cent.reshape(Ks_eff, d).astype(np.float32)
        else:
            try:
                C = faiss.vector_to_array(cent).reshape(Ks_eff, d).astype(np.float32)
            except Exception:
                # 兼容某些 wheel 里的 FloatVector 类型
                C = faiss.vector_float_to_array(cent).reshape(Ks_eff, d).astype(np.float32)

        # 若前面把 Ks 压成 Ks_eff，这里补齐到 (Ks, d)，重复最后一个中心即可
        if Ks_eff < Ks:
            pad = np.tile(C[-1:], (Ks - Ks_eff, 1))
            C = np.concatenate([C, pad], axis=0)
        return C

    # ---- Numpy fallback (Lloyd) ----
    rng = np.random.default_rng(seed)
    # init centers with k-means++ (simplified)
    idx0 = rng.integers(low=0, high=n)
    centers = [X[idx0:idx0+1]]
    for k in range(1, Ks_eff):
        # distance to nearest chosen center
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / (d2.sum() + 1e-12)
        pick = rng.choice(n, p=probs)
        centers.append(X[pick:pick+1])
    C = np.concatenate(centers, axis=0)  # (Ks_eff,d)

    for _ in range(max(1, niter)):
        # assign
        codes = assign_codes_l2(X, C)  # (n,)
        # update
        for k in range(Ks_eff):
            mask = (codes == k)
            if np.any(mask):
                C[k] = X[mask].mean(axis=0)
        # small jitter for empty clusters
        for k in range(Ks_eff):
            if not np.isfinite(C[k]).all():
                C[k] = X[rng.integers(0, n)]
    if Ks_eff < Ks:
        pad = np.tile(C[-1:], (Ks - Ks_eff, 1))
        C = np.concatenate([C, pad], axis=0)
    return C.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-lang", type=str, required=True,
                        help="Path to cluster_lang.npz (contains leaf_feat).")
    parser.add_argument("--rpq-dir", type=str, required=True,
                        help="Sidecar dir created by prepare_scidecar.py / train_opq.py / train_l1_pq.py")
    parser.add_argument("--M", type=int, default=None,
                        help="Number of subspaces. If None, will infer from C1.shape[0].")
    parser.add_argument("--Ks", type=int, default=64, help="Codewords per subspace for L2.")
    parser.add_argument("--kmeans-iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)

    rpq = Path(args.rpq_dir)
    assert rpq.exists(), f"rpq-dir not found: {rpq}"

    # ---- load meta from sidecar ----
    meta_path = rpq / "meta.json"
    assert meta_path.exists(), f"meta.json not found in {rpq}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    keys = meta.get("keys", {})
    assert "R" in keys and "C1" in keys and "codes1" in keys, \
        "meta.keys must contain R, C1, codes1 (run step2/step3 first)."

    # ---- load leaf features ----
    cl = np.load(args.cluster_lang)
    X = cl["leaf_feat"].astype(np.float32)  # (N,512)
    N, D = X.shape
    norms = np.linalg.norm(X, axis=1)
    valid_mask = norms > 1e-6
    num_valid = int(valid_mask.sum())
    print(f"[L2] valid rows for training: {num_valid}/{N} (invalid: {N - num_valid})")
    X[~valid_mask] = 0.0  # sanitize invalid rows

    # ---- load OPQ rotation + L1 assets ----
    R = np.load(rpq / keys["R"])                 # (512,512)
    C1 = np.load(rpq / keys["C1"])               # (M1,K1,dsub)
    codes1 = np.load(rpq / keys["codes1"])       # (N,M1)

    M1, K1, dsub1 = C1.shape
    if args.M is None:
        M = M1
    else:
        M = int(args.M)
        assert M == M1, f"--M ({args.M}) mismatch C1.shape[0] ({M1})"

    assert D == M * dsub1, f"Feature dim D={D} not equal to M*dsub={M*dsub1}"
    assert codes1.shape == (N, M), "codes1 shape mismatch"

    # ---- rotate to OPQ domain ----
    Y = X @ R  # (N,512)

    # ---- L1 reconstruction in OPQ domain ----
    parts = []
    for m in range(M):
        Cm = C1[m]                 # (K1,dsub1)
        idx = codes1[:, m]         # (N,)
        parts.append(Cm[idx])      # (N,dsub1)
    Y1_hat = np.concatenate(parts, axis=1)  # (N,512)
    Y1_hat[~valid_mask] = 0.0

    print(f"[L2][Check] mean cosine(valid, L1) = {cosine_mean(Y, Y1_hat, valid_mask):.4f}")

    # ---- residual E ----
    E = Y - Y1_hat
    E[~valid_mask] = 0.0

    # ---- split residual into subspaces ----
    subs_E, dsub = split_subspaces(E, M)
    assert dsub == dsub1

    # ---- train L2 codebook per subspace ----
    Ks2 = int(args.Ks)
    C2_list = []
    print(f"[L2] Train: M={M}, Ks={Ks2}, dsub={dsub}, N_train_valid={num_valid}")
    for m, Xm in enumerate(subs_E):
        Xm_valid = Xm[valid_mask]  # train on valid rows only
        C = kmeans_train(Xm_valid, Ks=Ks2, niter=args.kmeans_iters, seed=args.seed + m)
        C2_list.append(C.astype(np.float32))
        if (m % 8) == 0:
            print(f"  - trained L2 subspace {m}/{M}, C.shape={C.shape}")
    C2 = np.stack(C2_list, axis=0).astype(np.float32)  # (M,Ks2,dsub)

    # zero-centroid convention
    C2[:, 0, :] = 0.0
    np.save(rpq / "codebook_l2.npy", C2)
    print(f"[L2] Saved codebook_l2.npy, shape={C2.shape}")
    print(f"[L2] zero-centroid check: ||C2[:,0,:]||_max = {np.abs(C2[:,0,:]).max():.2e}")

    # ---- encode residual ----
    codes_dtype = np.uint8 if Ks2 <= 256 else np.uint16
    codes2 = np.empty((N, M), dtype=codes_dtype)
    for m, Xm in enumerate(subs_E):
        idx = assign_codes_l2(Xm, C2[m]).astype(codes_dtype)
        codes2[:, m] = idx
        if (m % 8) == 0:
            print(f"  - encoded L2 subspace {m}/{M}")

    # invalid rows -> all zeros
    if (~valid_mask).any():
        codes2[~valid_mask, :] = 0

    np.save(rpq / "codes2.npy", codes2)
    print(f"[L2] Saved codes2.npy, shape={codes2.shape}, dtype={codes2.dtype}")

    # ---- quick reconstruction check (valid only) ----
    # Y2_hat from L2
    Y2_parts = []
    for m in range(M):
        Cm = C2[m]                  # (Ks2,dsub)
        idx = codes2[:, m]          # (N,)
        Y2_parts.append(Cm[idx])    # (N,dsub)
    Y2_hat = np.concatenate(Y2_parts, axis=1)
    Y2_hat[~valid_mask] = 0.0

    cos_L1_valid = cosine_mean(Y, Y1_hat, valid_mask)
    cos_L12_valid = cosine_mean(Y, Y1_hat + Y2_hat, valid_mask)
    print(f"[L2][Check] mean cosine(valid, L1)     = {cos_L1_valid:.4f}")
    print(f"[L2][Check] mean cosine(valid, L1+L2)  = {cos_L12_valid:.4f}")

    # ---- update meta.json ----
    keys["C2"] = "codebook_l2.npy"
    keys["codes2"] = "codes2.npy"
    meta["keys"] = keys
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[L2] meta.json updated: {meta_path}")


if __name__ == "__main__":
    main()
