#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: prepare RPQ sidecar for OpenGaussians cluster_lang.npz (non-destructive).
- Read the whole cluster_lang.npz (do NOT modify it).
- Sanity-check keys/shapes.
- Create <model_path>/cluster_lang_rpq/ and copy stable fields (leaf_ind, leaf_score, occu_count).
- Write a meta.json skeleton for later RPQ training (OPQ/PQ).
"""

import argparse, json, os
from pathlib import Path
import numpy as np

REQ_KEYS = ["leaf_feat", "leaf_ind"]
OPT_KEYS = ["leaf_score", "occu_count"]

def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def sanity_check(d):
    missing = [k for k in REQ_KEYS if k not in d]
    if missing:
        raise KeyError(f"cluster_lang.npz 缺少必要键: {missing}")
    lf = d["leaf_feat"]; li = d["leaf_ind"]
    if lf.ndim != 2 or lf.shape[1] != 512:
        raise ValueError(f"leaf_feat 期望形状 [N_leaf,512]，实际 {lf.shape}")
    if li.ndim != 1:
        raise ValueError(f"leaf_ind 期望形状 [N_pts]，实际 {li.shape}")
    n_leaf = lf.shape[0]
    li_min, li_max = int(li.min()), int(li.max())
    if not (0 <= li_min <= li_max < n_leaf):
        raise ValueError(f"leaf_ind 越界: 范围 [{li_min},{li_max}] 但 N_leaf={n_leaf}")
    return {
        "N_leaf": int(n_leaf),
        "N_pts": int(li.shape[0]),
        "leaf_ind_range": [li_min, li_max],
        "leaf_feat_dtype": str(lf.dtype),
        "leaf_ind_dtype": str(li.dtype),
        "has_leaf_score": "leaf_score" in d,
        "has_occu_count": "occu_count" in d,
    }

def write_sidecar(out_dir: Path, d, check):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy stable fields (no change of dtype/values)
    np.save(out_dir / "leaf_ind.npy", d["leaf_ind"])
    if "leaf_score" in d:
        np.save(out_dir / "leaf_score.npy", d["leaf_score"])
    if "occu_count" in d:
        np.save(out_dir / "occu_count.npy", d["occu_count"])
    # Write meta skeleton (will be completed after OPQ/PQ training)
    meta = {
        "dim": 512,
        "M": None,         # to be filled (e.g., 64)
        "Ks": None,        # to be filled (e.g., 256)
        "normalized": True,      # we will train/query on L2-normalized vectors
        "metric": "cosine_on_unit",
        "keys": {
            "R": None,          # e.g., "opq_R.npy"
            "C1": None,         # e.g., "codebook_l1.npy"
            "C2": None,         # e.g., "codebook_l2.npy"
            "codes1": None,     # e.g., "codes1.npy"
            "codes2": None,     # e.g., "codes2.npy"
            "leaf_ind": "leaf_ind.npy",
            "leaf_score": "leaf_score.npy" if "leaf_score" in d else None,
            "occu_count": "occu_count.npy" if "occu_count" in d else None,
        }
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # Write inspection report
    with open(out_dir / "inspect.json", "w", encoding="utf-8") as f:
        json.dump(check, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Prepare RPQ sidecar for cluster_lang.npz (Step 1)")
    ap.add_argument("--cluster-lang", required=True, help="Path to cluster_lang.npz")
    ap.add_argument("--out-dir", required=True, help="Output sidecar dir: <model_path>/cluster_lang_rpq/")
    args = ap.parse_args()

    npz_path = Path(args.cluster_lang)
    out_dir = Path(args.out_dir)

    if not npz_path.is_file():
        raise FileNotFoundError(f"cluster_lang.npz 不存在: {npz_path}")

    d = load_npz(npz_path)
    check = sanity_check(d)
    print("[Step1] 检查通过：", json.dumps(check, ensure_ascii=False))
    write_sidecar(out_dir, d, check)
    print(f"[Step1] 已创建侧车目录并写入: {out_dir}")
    print("        meta.json 为骨架，后续 OPQ/PQ 训练后会补齐。")

if __name__ == "__main__":
    main()
