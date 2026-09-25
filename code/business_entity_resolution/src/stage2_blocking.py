"""Stage 2: Candidate Blocking Module with Real-Time Progress Bars.
Loads preprocessed datasets from cache, builds multi-indexer keys,
generates candidate pairs with live percentage tracking, and exports candidate_pairs.tsv.
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

import numpy as np
import pandas as pd
from src.blocking import MultiIndexBlocker, format_candidate_pairs_dataframe


def run_stage2_blocking(
    cache_dir: Path = Path("processed_data"),
    output_dir: Path = Path("output"),
    candidates_cache_dir: Path = Path("cache"),
    max_candidates: int = 25,
    max_train_sample_entities: int = 150000,
) -> None:
    """Runs candidate generation with real-time progress bars and zero deserialization lag."""
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    candidates_cache_dir = Path(candidates_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔍 [STAGE 2] CANDIDATE BLOCKING & PAIR GENERATION")
    print("=" * 60)

    # 1. Train Blocking
    print("\n[1/2] Indexing Train Pool (10.3M rows) & Mining Training Candidates...")
    pool_train = pd.read_parquet(cache_dir / "pool_train_clean.parquet")
    train_blocker = MultiIndexBlocker(max_candidates_per_entity=max_candidates)
    train_blocker.fit_pool(pool_train, desc="Indexing Train Pool")
    del pool_train
    gc.collect()

    s1_train = pd.read_parquet(cache_dir / "s1_train_clean.parquet")
    # Sample 150k for ultra-fast training pair construction
    if len(s1_train) > max_train_sample_entities:
        np.random.seed(42)
        sample_indices = np.random.choice(len(s1_train), size=max_train_sample_entities, replace=False)
        s1_train_sampled = s1_train.iloc[sample_indices].copy()
    else:
        s1_train_sampled = s1_train

    train_candidates = train_blocker.generate_candidates(s1_train_sampled, desc="Mining Train Candidates")
    del s1_train, s1_train_sampled, train_blocker
    gc.collect()

    train_cand_file = candidates_cache_dir / "train_candidates.pkl"
    with open(train_cand_file, "wb") as f:
        pickle.dump(train_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Saved {len(train_candidates):,} Train Candidate Lists to: {train_cand_file}")
    del train_candidates
    gc.collect()

    # 2. Test Blocking
    print("\n[2/2] Indexing Test Pool (9.9M rows) & Mining Test Candidates (1.73M records)...")
    pool_test = pd.read_parquet(cache_dir / "pool_test_clean.parquet")
    test_blocker = MultiIndexBlocker(max_candidates_per_entity=max_candidates)
    test_blocker.fit_pool(pool_test, desc="Indexing Test Pool")
    del pool_test
    gc.collect()

    s1_test = pd.read_parquet(cache_dir / "s1_test_clean.parquet")
    test_candidates = test_blocker.generate_candidates(s1_test, desc="Mining Test Candidates")
    del s1_test, test_blocker
    gc.collect()

    test_cand_file = candidates_cache_dir / "test_candidates.pkl"
    with open(test_cand_file, "wb") as f:
        pickle.dump(test_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)

    cand_pairs_df = format_candidate_pairs_dataframe(test_candidates)
    cand_pairs_path = output_dir / "candidate_pairs.tsv"
    cand_pairs_df.to_csv(cand_pairs_path, sep="\t", index=False)
    print(f"✅ Saved Test Candidate Pairs to: {cand_pairs_path}")

    print("\n✅ [STAGE 2 COMPLETE] Blocking finished successfully!")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Candidate Blocking with progress bars")
    parser.add_argument("--cache-dir", type=str, default="processed_data")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--candidates-cache-dir", type=str, default="cache")
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--max-train-sample", type=int, default=150000)
    args = parser.parse_args()

    run_stage2_blocking(
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        candidates_cache_dir=Path(args.candidates_cache_dir),
        max_candidates=args.max_candidates,
        max_train_sample_entities=args.max_train_sample,
    )


if __name__ == "__main__":
    main()
