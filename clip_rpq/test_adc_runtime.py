# -*- coding: utf-8 -*-
import time, json, numpy as np
from pathlib import Path
from adc_runtime import RPQSidecar

# === 配置 ===
rpq_dir = r"E:\opengaussians\OpenGaussian-main\scripts\output_lexisplat\lerf\figurines\cluster_lang_rpq"
cluster_lang_path = r"E:\opengaussians\OpenGaussian-main\scripts\output_lexisplat\lerf\figurines\cluster_lang.npz"
num_queries = 50       # 随机抽多少个 leaf_feat 当做查询
topk_eval = 10         # 评估 Recall@K 中的 K
topk_error_bucket = 100# 在 Top-100 上算分数误差（可观察近似质量）

# === 载入侧车 & 原始特征 ===
rpq = RPQSidecar(rpq_dir)
cl = np.load(cluster_lang_path, allow_pickle=False)
leaf_feat = cl["leaf_feat"].astype(np.float32)  # (N,512)
N, D = leaf_feat.shape
assert D == rpq.D, f"dim mismatch: {D} vs {rpq.D}"

# 处理无效行（与训练一致：接近零范数的视为无效）
norms = np.linalg.norm(leaf_feat, axis=1)
valid_mask = norms > 1e-6
invalid_mask = ~valid_mask
leaf_feat_norm = np.zeros_like(leaf_feat, dtype=np.float32)
leaf_feat_norm[valid_mask] = (leaf_feat[valid_mask] / norms[valid_mask, None]).astype(np.float32)

# 随机挑选查询（仅从 valid 行里选，避免全零）
rng = np.random.default_rng(123)
valid_indices = np.flatnonzero(valid_mask)
q_idx = rng.choice(valid_indices, size=min(num_queries, valid_indices.size), replace=False)

def recall_at_k(ranklist, gt_idx, k):
    return 1.0 if (gt_idx in ranklist[:k]) else 0.0

rec1, reck = [], []
mae_topbucket = []

t0 = time.perf_counter()
for i, idx in enumerate(q_idx):
    q = leaf_feat_norm[idx]                 # 用库里一条向量当“查询”，应当最像自己
    # A) ADC 余弦分数（全库）
    s_adc = rpq.query_scores(q, use_cosine=True)             # (N,)
    # B) 原始 512D 余弦（全库）
    s_bf = (leaf_feat_norm @ q.astype(np.float32))           # (N,)
    s_bf[invalid_mask] = 0.0                                 # 与侧车一致：无效行为 0

    # TopK 排名
    top_adc = np.argsort(-s_adc)
    top_bf  = np.argsort(-s_bf)

    # 评估 Recall@1 / Recall@K
    rec1.append(recall_at_k(top_adc, top_bf[0], 1))
    reck.append(recall_at_k(top_adc, top_bf[0], topk_eval))

    # 误差对比：在 brute-force 的前 topk_error_bucket 集合上，比较分数差
    bucket = top_bf[:topk_error_bucket]
    mae = float(np.mean(np.abs(s_adc[bucket] - s_bf[bucket])))
    mae_topbucket.append(mae)

t1 = time.perf_counter()

print(f"[Test] queries: {len(q_idx)}, N={N}, D={D}")
print(f"[Test] Recall@1 = {np.mean(rec1):.4f}")
print(f"[Test] Recall@{topk_eval} = {np.mean(reck):.4f}")
print(f"[Test] MAE on BF-Top{topk_error_bucket}: mean={np.mean(mae_topbucket):.6f}, median={np.median(mae_topbucket):.6f}")
print(f"[Test] Elapsed = {(t1 - t0)*1000:.1f} ms")

# 粗略内存对比
bytes_raw = 4 * D                                       # 每条 512D float32
bytes_codes = rpq.codes1.shape[1] + rpq.codes2.shape[1] # 每级 uint8 * M
print(f"[Mem] Raw 512D per-leaf ≈ {bytes_raw} B")
print(f"[Mem] RPQ codes per-leaf ≈ {bytes_codes} B (L1 {rpq.codes1.shape[1]} + L2 {rpq.codes2.shape[1]})")
print(f"[Mem] Compression ≈ {bytes_raw/bytes_codes:.1f}x (不含 1MB 级别的小型R与码本开销)")
