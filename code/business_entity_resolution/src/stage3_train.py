"""Stage 3: Feature Extraction, Model Training & Threshold Calibration.
Loads preprocessed train cache, extracts features with tqdm progress tracking,
fits LightGBM, optimizes the decision threshold, and saves the trained model.
"""

import argparse
import gc
import pickle
import sys
from pathlib import Path

# Safe encoding configuration
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import load_ground_truth
from src.feature_engineering import extract_pair_features, FEATURE_NAMES
from src.model import EntityMatcherModel
from src.postprocessing import optimize_threshold


def run_stage3_train(
    cache_dir: Path = Path("processed_data"),
    candidates_cache_dir: Path = Path("cache"),
    models_dir: Path = Path("models"),
    gt_file: Path = Path("student_resource/dataset/train/train_ground_truth.tsv"),
    max_train_entities: int = 150000,
    n_estimators: int = 600,          # raised from 300 - 5M+ pairs need more trees to converge
    learning_rate: float = 0.05,      # lowered from 0.08 for better generalization
    hard_negative_multiplier: int = 3, # hard-negative mining: up-sample near-miss negatives
) -> None:
    """Trains the pairwise entity resolution model with live progress tracking."""
    cache_dir = Path(cache_dir)
    candidates_cache_dir = Path(candidates_cache_dir)
    models_dir = Path(models_dir)
    gt_file = Path(gt_file)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[STAGE] [STAGE 3] FEATURE EXTRACTION & LIGHTGBM MODEL TRAINING")
    print("=" * 60)

    print("\n[1/4] Loading Preprocessed Train Data & Candidates from Cache...")
    pool_train = pd.read_parquet(cache_dir / "pool_train_clean.parquet")
    s1_train = pd.read_parquet(cache_dir / "s1_train_clean.parquet")

    pool_dict = {
        eid: {
            "id": eid,
            "name": nm,
            "address": addr,
            "combined": comb,
            "country": ctry,
            "numbers": set(nums.split(",")) if nums else set(),
        }
        for eid, nm, addr, comb, nums, ctry in zip(
            pool_train["entity_id"].values,
            pool_train["clean_name"].values,
            pool_train["clean_address"].values,
            pool_train["combined_text"].values,
            pool_train["address_numbers_str"].values,
            pool_train["country"].values,
        )
    }
    del pool_train
    gc.collect()

    s1_dict = {
        eid: {
            "id": eid,
            "name": nm,
            "address": addr,
            "combined": comb,
            "country": ctry,
            "numbers": set(nums.split(",")) if nums else set(),
        }
        for eid, nm, addr, comb, nums, ctry in zip(
            s1_train["entity_id"].values,
            s1_train["clean_name"].values,
            s1_train["clean_address"].values,
            s1_train["combined_text"].values,
            s1_train["address_numbers_str"].values,
            s1_train["country"].values,
        )
    }
    del s1_train
    gc.collect()

    train_cand_file = candidates_cache_dir / "train_candidates.pkl"
    with open(train_cand_file, "rb") as f:
        train_candidates = pickle.load(f)

    ground_truth = load_ground_truth(gt_file)

    # 2. Build pairwise feature matrix
    print("\n[2/4] Mining Training Pairs & Computing C++ String Features...", flush=True)
    train_keys = list(train_candidates.keys())
    if len(train_keys) > max_train_entities:
        np.random.seed(42)
        sample_keys = list(np.random.choice(train_keys, size=max_train_entities, replace=False))
    else:
        sample_keys = train_keys

    X_train_list = []
    y_train_list = []

    total_keys = len(sample_keys)
    log_step = max(10000, total_keys // 10)
    import time
    start_time = time.time()

    for idx, s1_id in enumerate(sample_keys):
        if idx % log_step == 0 or idx == total_keys - 1:
            pct = (idx + 1) / total_keys * 100.0
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 0.001)
            eta = (total_keys - (idx + 1)) / max(rate, 0.001)
            print(
                f"  [{pct:5.1f}%] {idx + 1:,} / {total_keys:,} entities | "
                f"Extracted Pairs: {len(X_train_list):,} | Speed: {rate:,.0f} ent/s | ETA: {eta:.1f}s",
                flush=True,
            )

        cands = train_candidates[s1_id]
        if s1_id not in s1_dict:
            continue
        s1_rec = s1_dict[s1_id]
        true_matches = ground_truth.get(s1_id, set())

        for tm in true_matches:
            if tm in pool_dict:
                feats = extract_pair_features(s1_rec, pool_dict[tm])
                X_train_list.append(feats)
                y_train_list.append(1)

        for cand_id in cands:
            if cand_id not in true_matches and cand_id in pool_dict:
                feats = extract_pair_features(s1_rec, pool_dict[cand_id])
                X_train_list.append(feats)
                y_train_list.append(0)

    X_mat = np.array(X_train_list, dtype=np.float32)
    y_vec = np.array(y_train_list, dtype=np.int32)
    print(f"\n[METRICS] Training Matrix: {X_mat.shape[0]:,} pairs, {X_mat.shape[1]} features (Positives: {np.sum(y_vec):,}, Negatives: {len(y_vec) - np.sum(y_vec):,})", flush=True)

    # Hard-negative mining: use a held-out 20% split to train the quick model.
    # FIX (Bottleneck 1): Previously trained on ALL of X_mat then scored the same data,
    # causing tree models to deflate in-sample negative scores to <0.1, missing real
    # hard negatives. Now we split 80/20 and score the held-out portion only.
    if len(np.unique(y_vec)) > 1 and len(X_mat) >= 10:
        print(f"\n[HARD NEG] Mining hard negatives (x{hard_negative_multiplier} up-sampling for score 0.3-0.8)...", flush=True)
        from src.model import EntityMatcherModel as _TmpModel
        n_total = len(X_mat)
        np.random.seed(0)
        perm = np.random.permutation(n_total)
        split = max(1, int(0.8 * n_total))
        train_idx_hn, holdout_idx_hn = perm[:split], perm[split:]
        if len(np.unique(y_vec[train_idx_hn])) > 1:
            quick_model = _TmpModel(n_estimators=50, learning_rate=0.1, n_folds=1, enable_ensemble=False)
            quick_model.train_on_pairs(X_mat[train_idx_hn], y_vec[train_idx_hn])
            # Score only the held-out negatives — these scores are out-of-sample and unbiased
            holdout_neg_mask = y_vec[holdout_idx_hn] == 0
            holdout_neg_idx = holdout_idx_hn[holdout_neg_mask]
            if len(holdout_neg_idx) > 0:
                neg_scores = quick_model.predict_pair_proba(X_mat[holdout_neg_idx])
                hard_neg_mask = (neg_scores >= 0.3) & (neg_scores <= 0.8)
                hard_neg_idx = holdout_neg_idx[hard_neg_mask]
                print(f"  Hard negatives found: {len(hard_neg_idx):,} of {len(holdout_neg_idx):,} held-out negatives", flush=True)
                if len(hard_neg_idx) > 0:
                    extra_X = np.tile(X_mat[hard_neg_idx], (hard_negative_multiplier - 1, 1))
                    extra_y = np.zeros(len(extra_X), dtype=np.int32)
                    X_mat = np.vstack([X_mat, extra_X])
                    y_vec = np.concatenate([y_vec, extra_y])
                    print(f"  After hard-neg augmentation: {X_mat.shape[0]:,} total pairs", flush=True)
            del quick_model
            import gc as _gc
            _gc.collect()

    # X_train_list and y_train_list already consumed above
    gc.collect()

    # 3. Fit Model Ensemble
    print(f"\n[3/4] Fitting Dual Model Ensemble ({n_estimators} trees, lr={learning_rate})...", flush=True)
    matcher = EntityMatcherModel(n_estimators=n_estimators, learning_rate=learning_rate)
    matcher.train_on_pairs(X_mat, y_vec)
    del X_mat, y_vec
    gc.collect()

    # 4. Calibrate F_0.5 Threshold on HELD-OUT validation set with country partitioning.
    # FIX (Bug 2): val_keys must come from entities NOT in sample_keys to avoid
    # in-sample probability inflation skewing the threshold too conservative.
    print("\n[4/4] Calibrating Macro F_0.5 Decision Thresholds on Held-Out Validation Entities...", flush=True)
    sample_keys_set = set(sample_keys)
    held_out_keys = [k for k in train_keys if k not in sample_keys_set]
    if len(held_out_keys) >= 25000:
        val_keys = held_out_keys[:25000]
        print(f"  Using {len(val_keys):,} true held-out entities for threshold calibration.", flush=True)
    elif len(held_out_keys) >= 5000:
        val_keys = held_out_keys
        print(f"  Using {len(val_keys):,} available held-out entities (fewer than 25k but still unseen).", flush=True)
    else:
        # Fallback: no held-out entities available (dataset smaller than max_train_entities).
        # Use the LAST 25% of sample_keys — seen less often in later tree rounds.
        val_keys = sample_keys[-(len(sample_keys) // 4):]
        print(f"  WARN: No held-out entities available. Using last {len(val_keys):,} of sample_keys as approximate val.", flush=True)
    total_val = len(val_keys)
    val_log_step = max(5000, total_val // 5)
    val_start = time.time()
    val_candidate_scores = {}
    country_map = {}

    for idx, s1_id in enumerate(val_keys):
        if idx % val_log_step == 0 or idx == total_val - 1:
            pct = (idx + 1) / total_val * 100.0
            elapsed = time.time() - val_start
            rate = (idx + 1) / max(elapsed, 0.001)
            eta = (total_val - (idx + 1)) / max(rate, 0.001)
            print(
                f"  [{pct:5.1f}%] {idx + 1:,} / {total_val:,} val entities | "
                f"Speed: {rate:,.0f} ent/s | ETA: {eta:.1f}s",
                flush=True,
            )

        cands = train_candidates[s1_id]
        if s1_id not in s1_dict:
            continue
        s1_rec = s1_dict[s1_id]
        country_map[s1_id] = s1_rec.get("country", "US")
        c_recs = [pool_dict[cid] for cid in cands if cid in pool_dict]
        scores = matcher.score_candidates(s1_rec, c_recs)
        val_candidate_scores[s1_id] = scores

    val_ground_truth = {k: ground_truth.get(k, set()) for k in val_keys}
    
    from src.postprocessing import optimize_country_thresholds, filter_matches_with_barrier
    from src.local_evaluator import run_local_evaluation
    import json

    country_thresholds, train_f05 = optimize_country_thresholds(val_candidate_scores, val_ground_truth, country_map)
    best_global_thresh, _ = optimize_threshold(val_candidate_scores, val_ground_truth)

    val_predictions = {}
    for s1_id, score_list in val_candidate_scores.items():
        c = country_map.get(s1_id, "US")
        th = country_thresholds.get(c, best_global_thresh)
        val_predictions[s1_id] = set(filter_matches_with_barrier(score_list, threshold=th, margin=0.08, max_matches=12))

    print("\n")
    run_local_evaluation(val_predictions, val_ground_truth, country_map)

    # Save artifacts
    model_save_path = models_dir / "matcher_lgbm.pkl"
    joblib.dump(matcher, model_save_path)
    print(f"\n[OK] Saved Trained Model: {model_save_path}", flush=True)

    thresh_save_path = models_dir / "best_threshold.txt"
    thresh_save_path.write_text(f"{best_global_thresh}\n", encoding="utf-8")
    print(f"[OK] Saved Global Threshold: {thresh_save_path}", flush=True)

    country_thresh_path = models_dir / "country_thresholds.json"
    country_thresh_path.write_text(json.dumps(country_thresholds, indent=2), encoding="utf-8")
    print(f"[OK] Saved Country Thresholds: {country_thresh_path}", flush=True)

    print("\n[OK] [STAGE 3 COMPLETE] Model trained, evaluated, and saved successfully!", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Feature extraction and LightGBM training with progress tracking")
    parser.add_argument("--cache-dir", type=str, default="processed_data")
    parser.add_argument("--candidates-cache-dir", type=str, default="cache")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--gt-file", type=str, default="student_resource/dataset/train/train_ground_truth.tsv")
    parser.add_argument("--max-train-entities", type=int, default=150000)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--hard-negative-multiplier", type=int, default=3)
    args = parser.parse_args()

    run_stage3_train(
        cache_dir=Path(args.cache_dir),
        candidates_cache_dir=Path(args.candidates_cache_dir),
        models_dir=Path(args.models_dir),
        gt_file=Path(args.gt_file),
        max_train_entities=args.max_train_entities,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        hard_negative_multiplier=args.hard_negative_multiplier,
    )


if __name__ == "__main__":
    main()
