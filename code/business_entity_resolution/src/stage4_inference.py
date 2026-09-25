"""Stage 4: Batched Test Inference & Leaderboard Export.
Loads saved LightGBM model and preprocessed test candidates,
runs batched C++ matrix scoring with live tqdm progress bars, and exports output/matching_results.tsv.
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

from src.feature_engineering import extract_pair_features
from src.postprocessing import export_matching_results


def run_stage4_inference(
    cache_dir: Path = Path("processed_data"),
    candidates_cache_dir: Path = Path("cache"),
    models_dir: Path = Path("models"),
    output_dir: Path = Path("output"),
    batch_size: int = 100000,
    override_threshold: float = None,
) -> None:
    """Runs high-throughput batched inference on test set candidates with progress tracking."""
    cache_dir = Path(cache_dir)
    candidates_cache_dir = Path(candidates_cache_dir)
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 [STAGE 4] BATCHED TEST INFERENCE & LEADERBOARD EXPORT")
    print("=" * 60)

    # 1. Load Model & Threshold
    model_path = models_dir / "matcher_lgbm.pkl"
    thresh_path = models_dir / "best_threshold.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run Stage 3 first!")

    print("\n[1/3] Loading Trained Model & Calibrated Threshold...")
    matcher = joblib.load(model_path)

    if override_threshold is not None:
        threshold = override_threshold
        print(f"Using Override Threshold: {threshold:.3f}")
    elif thresh_path.exists():
        threshold = float(thresh_path.read_text(encoding="utf-8").strip())
        print(f"Using Calibrated Threshold: {threshold:.3f}")
    else:
        threshold = 0.500
        print("Using Default Threshold: 0.500")

    # 2. Load Preprocessed Test Data & Candidates
    print("\n[2/3] Loading Preprocessed Test Data & Candidates from Cache...")
    pool_test = pd.read_parquet(cache_dir / "pool_test_clean.parquet")
    s1_test = pd.read_parquet(cache_dir / "s1_test_clean.parquet")

    pool_dict = {
        eid: {
            "id": eid,
            "name": nm,
            "address": addr,
            "combined": comb,
            "numbers": set(nums.split(",")) if nums else set(),
        }
        for eid, nm, addr, comb, nums in zip(
            pool_test["entity_id"].values,
            pool_test["clean_name"].values,
            pool_test["clean_address"].values,
            pool_test["combined_text"].values,
            pool_test["address_numbers_str"].values,
        )
    }
    del pool_test
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
            s1_test["entity_id"].values,
            s1_test["clean_name"].values,
            s1_test["clean_address"].values,
            s1_test["combined_text"].values,
            s1_test["address_numbers_str"].values,
        )
    }

    test_cand_file = candidates_cache_dir / "test_candidates.pkl"
    with open(test_cand_file, "rb") as f:
        test_candidates = pickle.load(f)

    # 3. Batched Vector Matrix Inference with Progress Bar
    all_s1_ids = list(s1_test["entity_id"].values)
    del s1_test
    gc.collect()

    total_s1 = len(all_s1_ids)
    print(f"\n[3/3] Running Batched Test Inference across {total_s1:,} entities (Batch Size: {batch_size:,})...", flush=True)

    test_candidate_scores = {s1_id: [] for s1_id in all_s1_ids}
    pair_features_batch = []
    pair_mapping_batch = []

    log_step = max(50000, total_s1 // 10)
    import time
    start_time = time.time()
    total_evaluated_pairs = 0

    for idx, s1_id in enumerate(all_s1_ids):
        if idx % log_step == 0 or idx == total_s1 - 1:
            pct = (idx + 1) / total_s1 * 100.0
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 0.001)
            eta = (total_s1 - (idx + 1)) / max(rate, 0.001)
            print(
                f"  [{pct:5.1f}%] {idx + 1:,} / {total_s1:,} entities | "
                f"Evaluated Pairs: {total_evaluated_pairs:,} | Speed: {rate:,.0f} ent/s | ETA: {eta:.1f}s",
                flush=True,
            )

        cands = test_candidates.get(s1_id, [])
        if not cands or s1_id not in s1_dict:
            continue
        s1_rec = s1_dict[s1_id]
        for cid in cands:
            if cid in pool_dict:
                pair_features_batch.append(extract_pair_features(s1_rec, pool_dict[cid]))
                pair_mapping_batch.append((s1_id, cid))
                total_evaluated_pairs += 1

                if len(pair_features_batch) >= batch_size:
                    X_batch = np.array(pair_features_batch, dtype=np.float32)
                    probas = matcher.predict_pair_proba(X_batch)
                    for (sid, cand_id), prob in zip(pair_mapping_batch, probas):
                        if prob >= (threshold - 0.15):
                            test_candidate_scores[sid].append((cand_id, float(prob)))
                    pair_features_batch.clear()
                    pair_mapping_batch.clear()

    # Remaining pairs
    if pair_features_batch:
        X_batch = np.array(pair_features_batch, dtype=np.float32)
        probas = matcher.predict_pair_proba(X_batch)
        for (sid, cand_id), prob in zip(pair_mapping_batch, probas):
            if prob >= (threshold - 0.15):
                test_candidate_scores[sid].append((cand_id, float(prob)))
        pair_features_batch.clear()
        pair_mapping_batch.clear()

    from src.postprocessing import filter_matches_with_barrier
    test_predictions = {}
    for s1_id, score_list in test_candidate_scores.items():
        test_predictions[s1_id] = filter_matches_with_barrier(score_list, threshold=threshold, margin=0.15)

    matching_results_path = output_dir / "matching_results.tsv"
    export_matching_results(test_predictions, all_s1_ids, str(matching_results_path))
    print(f"\n✅ Saved Final Leaderboard Matches to: {matching_results_path}", flush=True)
    print("\n✅ [STAGE 4 COMPLETE] Prediction files ready for validation and upload!", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Batched test inference with progress tracking")
    parser.add_argument("--cache-dir", type=str, default="processed_data")
    parser.add_argument("--candidates-cache-dir", type=str, default="cache")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    run_stage4_inference(
        cache_dir=Path(args.cache_dir),
        candidates_cache_dir=Path(args.candidates_cache_dir),
        models_dir=Path(args.models_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        override_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
