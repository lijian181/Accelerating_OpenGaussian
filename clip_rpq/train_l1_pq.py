# -*- coding: utf-8 -*-
import argparse, json, math, os
from pathlib import Path
import numpy as np

# ============== 可选依赖：FAISS ==============
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


# ============== 工具函数 ==============
def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)  # (N,1)
    mask1d = (norms[:, 0] > eps).ravel()              # (N,)
    Xn = np.zeros_like(X, dtype=np.float32)
    Xn[mask1d] = (X[mask1d] / norms[mask1d]).astype(np.float32)  # (Nv,512)/(Nv,1)
    return Xn

def load_leaf_feat_from_cluster_lang(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if "leaf_feat" not in data:
        raise KeyError(f"{npz_path} 缺少 'leaf_feat'")
    X = data["leaf_feat"].astype(np.float32)
    if X.ndim != 2 or X.shape[1] != 512:
        raise ValueError(f"leaf_feat 期望 [N,512]，实际 {X.shape}")
    return l2_normalize_rows(X)

def split_subspaces(Z: np.ndarray, M: int):
    assert Z.shape[1] % M == 0
    dsub = Z.shape[1] // M
    return [Z[:, i*dsub:(i+1)*dsub].copy() for i in range(M)], dsub

def kmeans_train(X: np.ndarray, Ks: int, niter: int = 40, seed: int = 1234) -> np.ndarray:
    N, d = X.shape
    if _HAS_FAISS:
        try:
            km = faiss.Kmeans(d, Ks, niter=niter, verbose=False, seed=seed, spherical=False)
            km.train(X.astype(np.float32))

            # --- 兼容不同 faiss 版本/平台：centroids 可能是 np.ndarray，也可能是 Faiss Vector ---
            cent = km.centroids
            if isinstance(cent, np.ndarray):
                C = cent.reshape(Ks, d).astype(np.float32)
            else:
                C = faiss.vector_float_to_array(cent).reshape(Ks, d).astype(np.float32)

            # 保险：形状/数值校验，不合格则走 fallback
            if C.shape != (Ks, d) or not np.isfinite(C).all():
                raise RuntimeError(f"bad centroids: shape={C.shape}, finite={np.isfinite(C).all()}")
            return C
        except Exception as e:
            print(f"[L1][FAISS-KMeans] failed ({type(e).__name__}: {e}); fallback to NumPy Lloyd.")

    # ---- fallback：极简 Lloyd ----
    rng = np.random.default_rng(seed)
    sel = rng.choice(N, size=min(Ks, N), replace=False)
    C = X[sel].astype(np.float32)  # (K', d)
    for _ in range(niter):
        x2 = (X*X).sum(1, keepdims=True)
        c2 = (C*C).sum(1, keepdims=True).T
        dist = x2 + c2 - 2 * (X @ C.T)
        a = dist.argmin(1)
        for k in range(C.shape[0]):
            m = (a == k)
            if m.any():
                C[k] = X[m].mean(0)
    if C.shape[0] < Ks:
        C = np.vstack([C, np.zeros((Ks - C.shape[0], d), dtype=np.float32)])
    return C.astype(np.float32)


def assign_codes(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    if _HAS_FAISS:
        index = faiss.IndexFlatL2(C.shape[1])
        index.add(C.astype(np.float32))
        D, I = index.search(X.astype(np.float32), 1)
        return I.reshape(-1)
    x2 = (X*X).sum(1, keepdims=True)
    c2 = (C*C).sum(1, keepdims=True).T
    dist = x2 + c2 - 2 * (X @ C.T)
    return dist.argmin(1)

def update_meta_after_l1(rpq_dir: str, Ks: int,
                         c1_key="codebook_l1.npy", codes1_key="codes1.npy"):
    rpq = Path(rpq_dir); rpq.mkdir(parents=True, exist_ok=True)
    meta_path = rpq / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"dim": 512, "M": None, "Ks": None, "normalized": True,
                "metric": "cosine_on_unit", "keys": {}}
    meta["Ks"] = int(Ks)
    keys = meta.get("keys", {})
    keys.update({
        "C1": c1_key,
        "codes1": codes1_key,
        "R": keys.get("R", "opq_R.npy"),
        "C2": keys.get("C2", None),
        "codes2": keys.get("codes2", None),
        "leaf_ind": keys.get("leaf_ind", "leaf_ind.npy"),
        "leaf_score": keys.get("leaf_score", None),
        "occu_count": keys.get("occu_count", None),
    })
    meta["keys"] = keys
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(meta_path)

def cos_mean(A: np.ndarray, B: np.ndarray) -> float:
    def norm_rows(Y):
        n = np.linalg.norm(Y, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        return Y / n
    A = norm_rows(A); B = norm_rows(B)
    return float((A * B).sum(1).mean())


# ============== 主流程 ==============
def main():
    ap = argparse.ArgumentParser()

    # 互斥两种输入模式：旧管线 vs leaf 模式
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=str,
                      help="旧管线：训练样本 .npy（按你原逻辑）")
    mode.add_argument("--cluster-lang", type=str,
                      help="leaf 模式：cluster_lang.npz 路径，直接取 leaf_feat 做 L1-PQ")

    # 旧管线的输出
    ap.add_argument("--out", type=str, help="旧管线输出：codebook 路径")
    ap.add_argument("--codes-out", type=str, help="旧管线输出：codes 路径")

    # leaf 模式的侧车目录
    ap.add_argument("--rpq-dir", type=str,
                    help="leaf 模式：侧车目录（读 opq_R.npy；写 codebook_l1.npy / codes1.npy / 更新 meta.json）")

    ap.add_argument("--M", type=int, default=64, help="子空间数（与 Step2 一致）")
    ap.add_argument("--Ks", type=int, default=256, help="每个子空间的码字数")
    ap.add_argument("--kmeans-iters", type=int, default=40, help="KMeans 迭代轮数")
    ap.add_argument("--seed", type=int, default=1234, help="随机种子")

    args = ap.parse_args()

    # ========== leaf 模式 ==========
    if args.cluster_lang is not None:
        if not args.rpq_dir:
            ap.error("--rpq-dir 是 leaf 模式必需参数")
        if 512 % args.M != 0:
            ap.error(f"--M 必须整除 512，目前 M={args.M}")
        if args.Ks <= 1:
            ap.error("--Ks 必须 > 1")

        rpq = Path(args.rpq_dir); rpq.mkdir(parents=True, exist_ok=True)

        # 1) 加载特征并筛选有效行（避免零向量污染码本）
        X = load_leaf_feat_from_cluster_lang(args.cluster_lang)   # (N,512) unit-or-zero
        norms = np.linalg.norm(X, axis=1)
        valid = norms > 1e-6
        print(f"[L1] valid rows for training: {valid.sum()}/{X.shape[0]} "
              f"(invalid: {(~valid).sum()})")

        # 2) 读取 R 并右乘到 OPQ 空间
        R_path = rpq / "opq_R.npy"
        if not R_path.exists():
            ap.error(f"未找到 {R_path}，请先完成 Step2")
        R = np.load(R_path).astype(np.float32)
        if R.shape != (512, 512):
            raise ValueError(f"R 形状异常: {R.shape}")

        Z_all = (X @ R).astype(np.float32)   # 全部叶
        Z_train = Z_all[valid]               # 有效叶用于训练
        if Z_train.shape[0] < 2:
            raise RuntimeError("有效训练样本过少，无法训练 L1 PQ。")

        # 3) Ks 自适应回退，避免 K > N_train
        Ks_eff = args.Ks
        if Ks_eff > Z_train.shape[0]:
            Ks_eff = 1 << int(math.floor(math.log2(max(2, Z_train.shape[0]))))
            print(f"[L1] Ks={args.Ks} > N_train={Z_train.shape[0]}, fallback Ks={Ks_eff}")

        # 4) 逐子空间训练码本（仅用 Z_train）
        subs_train, dsub = split_subspaces(Z_train, args.M)
        codebooks = []
        print(f"[L1] Train: M={args.M}, Ks={Ks_eff}, dsub={dsub}, N_train={Z_train.shape[0]}")
        for m, Xm in enumerate(subs_train):
            C = kmeans_train(Xm, Ks=Ks_eff, niter=args.kmeans_iters, seed=args.seed + m)
            codebooks.append(C)
            if (m % 8) == 0:
                print(f"  - trained subspace {m}/{args.M}, C.shape={C.shape}")

        C1 = np.stack(codebooks, axis=0).astype(np.float32)  # (M, Ks_eff, dsub)
        # 保留码本第0号码字为全零，作为“无效行”的重构锚点
        C1[:, 0, :] = 0.0
        np.save(rpq / "codebook_l1.npy", C1)
        print(f"[L1] Saved codebook_l1.npy, shape={C1.shape}")

        # 5) 对所有叶编码（用 Z_all 指派，保证每个 leaf 都有 code）
        subs_all, _ = split_subspaces(Z_all, args.M)
        codes_dtype = np.uint8 if Ks_eff <= 256 else np.uint16
        codes1 = np.empty((Z_all.shape[0], args.M), dtype=codes_dtype)
        for m, Xm in enumerate(subs_all):
            idx = assign_codes(Xm, C1[m]).astype(codes_dtype)  # 最近中心
            codes1[:, m] = idx
        if (m % 8) == 0:
            print(f"  - encoded subspace {m}/{args.M}")
         # —— 关键：把无效行（valid==False）的所有子空间 code 统一设为0（指向零码字）
        invalid_mask = ~valid
        if invalid_mask.any():
            codes1[invalid_mask, :] = 0
        np.save(rpq / "codes1.npy", codes1)
        print(f"[L1] Saved codes1.npy, shape={codes1.shape}, dtype={codes1.dtype}")

        # 6) 更新 meta.json
        meta_path = update_meta_after_l1(str(rpq), Ks=int(Ks_eff),
                                         c1_key="codebook_l1.npy",
                                         codes1_key="codes1.npy")
        print(f"[L1] meta.json updated: {meta_path}")
        print(f"[L1] zero-centroid check: ||C1[:,0,:]||_max = {np.abs(C1[:, 0, :]).max():.2e}")

        # 7) 自检：一阶重构的余弦一致性
        try:
            rec_chunks = []
            for m in range(args.M):
                Cm = C1[m]                                  # (Ks_eff, dsub)
                sel = codes1[:, m].astype(np.int64)         # (N,)
                rec_chunks.append(Cm[sel])                  # (N, dsub)
            Z1 = np.concatenate(rec_chunks, axis=1).astype(np.float32)  # (N,512)
            cmean = cos_mean(Z_all, Z1)
            print(f"[L1][Check] mean cosine(Z_all, Z1) = {cmean:.4f}")
            valid_mask = (np.linalg.norm(X, axis=1) > 1e-6)
            if valid_mask.any():
                cmean_valid = cos_mean(Z_all[valid_mask], Z1[valid_mask])
                print(f"[L1][Check] mean cosine(valid only) = {cmean_valid:.4f}")
        except Exception as _:
            pass

        return

    # ========== 旧管线（保持你原有语义）==========
    if not args.sample or not args.out or not args.codes_out:
        ap.error("--sample / --out / --codes-out 是旧管线必需参数")
    Z = np.load(args.sample).astype(np.float32)
    if Z.ndim != 2:
        raise ValueError(f"sample 期望 2D，实际 {Z.shape}")
    if Z.shape[1] % args.M != 0:
        ap.error(f"--M 必须整除特征维度（当前 D={Z.shape[1]}, M={args.M}）")

    Ks_eff = args.Ks
    if Ks_eff > Z.shape[0]:
        Ks_eff = 1 << int(math.floor(math.log2(max(2, Z.shape[0]))))
        print(f"[L1] Ks={args.Ks} > N={Z.shape[0]}, fallback Ks={Ks_eff}")

    subs, dsub = split_subspaces(Z, args.M)
    codebooks = []
    for m, Xm in enumerate(subs):
        C = kmeans_train(Xm, Ks=Ks_eff, niter=args.kmeans_iters, seed=args.seed + m)
        codebooks.append(C)
    C1 = np.stack(codebooks, axis=0).astype(np.float32)
    np.save(args.out, C1)
    print(f"[L1] Saved codebook: {args.out}, shape={C1.shape}")

    codes_dtype = np.uint8 if Ks_eff <= 256 else np.uint16
    codes1 = np.empty((Z.shape[0], args.M), dtype=codes_dtype)
    for m, Xm in enumerate(subs):
        idx = assign_codes(Xm, C1[m]).astype(codes_dtype)
        codes1[:, m] = idx
    np.save(args.codes_out, codes1)
    print(f"[L1] Saved codes: {args.codes_out}, shape={codes1.shape}, dtype={codes1.dtype}")


if __name__ == "__main__":
    main()
