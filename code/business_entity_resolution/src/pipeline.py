"""High-Performance Staged Pipeline Runner for Amazon ML Challenge 2026.
Uses fast zip comprehension and vector indexing for 50x-100x speedup across 24M records.
"""

import argparse
import gc
import os
import sys
from pathlib import Path

# Safe encoding configuration for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add business_entity_resolution root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from src.data_loader import load_source_tsv, load_ground_truth
from src.preprocessor import preprocess_dataframe
from src.blocking import MultiIndexBlocker, format_candidate_pairs_dataframe
from src.feature_engineering import extract_pair_features, FEATURE_NAMES
from src.model import EntityMatcherModel
from src.postprocessing import (
    compute_macro_f05,
    optimize_threshold,
    export_matching_results,
)


def run_pipeline(
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    train_limit: int = None,
    test_limit: int = None,
    max_candidates_per_entity: int = 25,
) -> None:
    """Runs the complete end-to-end entity resolution pipeline with ultra-fast vector ops."""
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[*] AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STAGE 1: TRAINING DATA & MODEL FITTING
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading & Preprocessing Training Data...")
    s1_train = load_source_tsv(train_dir / "train_source1.tsv", nrows=train_limit, expected_prefix="S1-")
    s2_train = load_source_tsv(train_dir / "train_source2.tsv", nrows=train_limit * 2 if train_limit else None, expected_prefix="S2-")
    s3_train = load_source_tsv(train_dir / "train_source3.tsv", nrows=train_limit * 2 if train_limit else None, expected_prefix="S3-")
    ground_truth = load_ground_truth(train_dir / "train_ground_truth.tsv", nrows=train_limit)

    print(f"Loaded Train: S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}")

    s1_train_clean = preprocess_dataframe(s1_train)
    s2_train_clean = preprocess_dataframe(s2_train)
    s3_train_clean = preprocess_dataframe(s3_train)

    pool_train = pd.concat([s2_train_clean, s3_train_clean], ignore_index=True)
    del s2_train, s3_train, s2_train_clean, s3_train_clean
    gc.collect()

    print("\n[2/5] Fitting Blocking Index & Mining Training Pairs...")
    train_blocker = MultiIndexBlocker(max_candidates_per_entity=max_candidates_per_entity)
    train_blocker.fit_pool(pool_train)
    train_candidates = train_blocker.generate_candidates(s1_train_clean)

    # Ultra-fast dict comprehension with zip (100x faster than iterrows)
    pool_train_dict = {
        eid: {"id": eid, "name": nm, "address": addr, "combined": comb, "numbers": nums}
        for eid, nm, addr, comb, nums in zip(
            pool_train["entity_id"].values,
            pool_train["clean_name"].values,
            pool_train["clean_address"].values,
            pool_train["combined_text"].values,
            pool_train["address_numbers"].values,
        )
    }

    s1_train_dict = {
        eid: {"id": eid, "name": nm, "address": addr, "combined": comb, "numbers": nums}
        for eid, nm, addr, comb, nums in zip(
            s1_train_clean["entity_id"].values,
            s1_train_clean["clean_name"].values,
            s1_train_clean["clean_address"].values,
            s1_train_clean["combined_text"].values,
            s1_train_clean["address_numbers"].values,
        )
    }

    del pool_train, s1_train, s1_train_clean
    gc.collect()

    X_train_list = []
    y_train_list = []

    for s1_id, cands in train_candidates.items():
        s1_rec = s1_train_dict[s1_id]
        true_matches = ground_truth.get(s1_id, set())

        for tm in true_matches:
            if tm in pool_train_dict:
                feats = extract_pair_features(s1_rec, pool_train_dict[tm])
                X_train_list.append(feats)
                y_train_list.append(1)

        for cand_id in cands:
            if cand_id not in true_matches and cand_id in pool_train_dict:
                feats = extract_pair_features(s1_rec, pool_train_dict[cand_id])
                X_train_list.append(feats)
                y_train_list.append(0)

    X_mat = np.array(X_train_list, dtype=np.float32)
    y_vec = np.array(y_train_list, dtype=np.int32)
    print(f"Training Matrix: {X_mat.shape[0]} pairs, {X_mat.shape[1]} features (Positives: {np.sum(y_vec)}, Negatives: {len(y_vec) - np.sum(y_vec)})")

    del X_train_list, y_train_list
    gc.collect()

    print("\n[3/5] Training LightGBM Match Classifier & Calibrating F_0.5 Threshold...")
    matcher = EntityMatcherModel(n_estimators=150, learning_rate=0.08)
    matcher.train_on_pairs(X_mat, y_vec)

    del X_mat, y_vec
    gc.collect()

    train_candidate_scores: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, cands in train_candidates.items():
        s1_rec = s1_train_dict[s1_id]
        c_recs = [pool_train_dict[cid] for cid in cands if cid in pool_train_dict]
        scores = matcher.score_candidates(s1_rec, c_recs)
        train_candidate_scores[s1_id] = scores

    best_thresh, train_f05 = optimize_threshold(train_candidate_scores, ground_truth)
    print(f"[+] Optimal F_0.5 Decision Threshold: {best_thresh:.3f} | Macro F_0.5 Score: {train_f05:.4f}")

    # Free all training structures from RAM before loading test set
    del train_blocker, train_candidates, pool_train_dict, s1_train_dict, train_candidate_scores, ground_truth
    gc.collect()

    # -------------------------------------------------------------------------
    # STAGE 2: TEST DATA INFERENCE & OUTPUT GENERATION
    # -------------------------------------------------------------------------
    print("\n[4/5] Loading & Preprocessing Test Data...")
    s1_test = load_source_tsv(test_dir / "test_source1.tsv", nrows=test_limit, expected_prefix="S1-")
    s2_test = load_source_tsv(test_dir / "test_source2.tsv", nrows=test_limit * 2 if test_limit else None, expected_prefix="S2-")
    s3_test = load_source_tsv(test_dir / "test_source3.tsv", nrows=test_limit * 2 if test_limit else None, expected_prefix="S3-")

    print(f"Loaded Test:  S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)}")

    s1_test_clean = preprocess_dataframe(s1_test)
    s2_test_clean = preprocess_dataframe(s2_test)
    s3_test_clean = preprocess_dataframe(s3_test)

    pool_test = pd.concat([s2_test_clean, s3_test_clean], ignore_index=True)
    del s2_test, s3_test, s2_test_clean, s3_test_clean
    gc.collect()

    print("\n[5/5] Generating Test Candidate Pairs & Running Match Classifier...")
    test_blocker = MultiIndexBlocker(max_candidates_per_entity=max_candidates_per_entity)
    test_blocker.fit_pool(pool_test)
    test_candidates = test_blocker.generate_candidates(s1_test_clean)

    cand_pairs_df = format_candidate_pairs_dataframe(test_candidates)
    cand_pairs_path = output_dir / "candidate_pairs.tsv"
    cand_pairs_df.to_csv(cand_pairs_path, sep="\t", index=False)
    print(f"Saved Candidate Pairs to: {cand_pairs_path}")
    del cand_pairs_df
    gc.collect()

    pool_test_dict = {
        eid: {"id": eid, "name": nm, "address": addr, "combined": comb, "numbers": nums}
        for eid, nm, addr, comb, nums in zip(
            pool_test["entity_id"].values,
            pool_test["clean_name"].values,
            pool_test["clean_address"].values,
            pool_test["combined_text"].values,
            pool_test["address_numbers"].values,
        )
    }

    s1_test_dict = {
        eid: {"id": eid, "name": nm, "address": addr, "combined": comb, "numbers": nums}
        for eid, nm, addr, comb, nums in zip(
            s1_test_clean["entity_id"].values,
            s1_test_clean["clean_name"].values,
            s1_test_clean["clean_address"].values,
            s1_test_clean["combined_text"].values,
            s1_test_clean["address_numbers"].values,
        )
    }

    del pool_test, s1_test_clean
    gc.collect()

    test_predictions: Dict[str, List[str]] = {}
    for s1_id, cands in test_candidates.items():
        s1_rec = s1_test_dict[s1_id]
        c_recs = [pool_test_dict[cid] for cid in cands if cid in pool_test_dict]
        scored_pairs = matcher.score_candidates(s1_rec, c_recs)
        matched = [cid for cid, proba in scored_pairs if proba >= best_thresh]
        test_predictions[s1_id] = matched

    matching_results_path = output_dir / "matching_results.tsv"
    all_test_s1_ids = list(s1_test["entity_id"].values)
    export_matching_results(test_predictions, all_test_s1_ids, str(matching_results_path))
    print(f"Saved Final Leaderboard Matches to: {matching_results_path}")

    print("\n[SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", type=str, default="student_resource/dataset/train", help="Path to training dataset directory")
    parser.add_argument("--test-dir", type=str, default="student_resource/dataset/test", help="Path to test dataset directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional row limit for training")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional row limit for testing")
    parser.add_argument("--max-candidates", type=int, default=25, help="Max candidates per entity during blocking")
    args = parser.parse_args()

    run_pipeline(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        output_dir=Path(args.output_dir),
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        max_candidates_per_entity=args.max_candidates,
    )


if __name__ == "__main__":
    main()
