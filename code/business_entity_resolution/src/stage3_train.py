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
    n_estimators: int = 150,
    learning_rate: float = 0.08,
) -> None:
    """Trains the pairwise entity resolution model with live progress tracking."""
    cache_dir = Path(cache_dir)
    candidates_cache_dir = Path(candidates_cache_dir)
    models_dir = Path(models_dir)
    gt_file = Path(gt_file)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧠 [STAGE 3] FEATURE EXTRACTION & LIGHTGBM MODEL TRAINING")
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
            "numbers": set(nums.split(",")) if nums else set(),
        }
        for eid, nm, addr, comb, nums in zip(
            pool_train["entity_id"].values,
            pool_train["clean_name"].values,
            pool_train["clean_address"].values,
            pool_train["combined_text"].values,
            pool_train["address_numbers_str"].values,
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
            "numbers": set(nums.split(",")) if nums else set(),
        }
        for eid, nm, addr, comb, nums in zip(
            s1_train["entity_id"].values,
            s1_train["clean_name"].values,
            s1_train["clean_address"].values,
            s1_train["combined_text"].values,
            s1_train["address_numbers_str"].values,
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
    print(f"\n📊 Training Matrix: {X_mat.shape[0]:,} pairs, {X_mat.shape[1]} features (Positives: {np.sum(y_vec):,}, Negatives: {len(y_vec) - np.sum(y_vec):,})", flush=True)

    del X_train_list, y_train_list
    gc.collect()

    # 3. Fit LightGBM
    print(f"\n[3/4] Fitting LightGBM Match Classifier ({n_estimators} trees, lr={learning_rate})...", flush=True)
    matcher = EntityMatcherModel(n_estimators=n_estimators, learning_rate=learning_rate)
    matcher.train_on_pairs(X_mat, y_vec)
    del X_mat, y_vec
    gc.collect()

    # 4. Calibrate F_0.5 Threshold on validation set
    print("\n[4/4] Calibrating Macro F_0.5 Decision Threshold on 25k Validation Entities...", flush=True)
    val_keys = sample_keys[:25000]
    total_val = len(val_keys)
    val_log_step = max(5000, total_val // 5)
    val_start = time.time()
    val_candidate_scores = {}

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
        c_recs = [pool_dict[cid] for cid in cands if cid in pool_dict]
        scores = matcher.score_candidates(s1_rec, c_recs)
        val_candidate_scores[s1_id] = scores

    val_ground_truth = {k: ground_truth.get(k, set()) for k in val_keys}
    best_thresh, train_f05 = optimize_threshold(val_candidate_scores, val_ground_truth)
    print(f"\n🎯 [CALIBRATION RESULT] Optimal Threshold: {best_thresh:.3f} | Validation F_0.5 Score: {train_f05:.4f}", flush=True)

    # Save artifacts
    model_save_path = models_dir / "matcher_lgbm.pkl"
    joblib.dump(matcher, model_save_path)
    print(f"✅ Saved Trained Model: {model_save_path}", flush=True)

    thresh_save_path = models_dir / "best_threshold.txt"
    thresh_save_path.write_text(f"{best_thresh}\n", encoding="utf-8")
    print(f"✅ Saved Threshold: {thresh_save_path}", flush=True)

    print("\n✅ [STAGE 3 COMPLETE] Model trained, evaluated, and saved successfully!", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Feature extraction and LightGBM training with progress tracking")
    parser.add_argument("--cache-dir", type=str, default="processed_data")
    parser.add_argument("--candidates-cache-dir", type=str, default="cache")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--gt-file", type=str, default="student_resource/dataset/train/train_ground_truth.tsv")
    parser.add_argument("--max-train-entities", type=int, default=150000)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    args = parser.parse_args()

    run_stage3_train(
        cache_dir=Path(args.cache_dir),
        candidates_cache_dir=Path(args.candidates_cache_dir),
        models_dir=Path(args.models_dir),
        gt_file=Path(args.gt_file),
        max_train_entities=args.max_train_entities,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
