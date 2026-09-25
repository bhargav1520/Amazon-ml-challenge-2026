"""Stage 1: Preprocessing & Cache Module.
Loads raw TSV files, performs multilingual normalization, and saves cached parquet files
so you never have to re-process strings again.
"""

import argparse
import gc
import sys
from pathlib import Path

# Safe encoding configuration
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data_loader import load_source_tsv
from src.preprocessor import preprocess_dataframe


def run_stage1_preprocess(
    train_dir: Path,
    test_dir: Path,
    cache_dir: Path = Path("processed_data"),
    train_limit: int = None,
    test_limit: int = None,
) -> None:
    """Preprocesses all raw source datasets and saves to parquet files."""
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧹 [STAGE 1] PREPROCESSING RAW DATASETS & BUILDING CACHE")
    print("=" * 60)

    # 1. Train Source 1
    print("\n[1/4] Processing Train Source 1...")
    s1_train = load_source_tsv(train_dir / "train_source1.tsv", nrows=train_limit, expected_prefix="S1-")
    s1_train_clean = preprocess_dataframe(s1_train)
    # Convert set column to comma-string for fast parquet serialization
    s1_train_clean["address_numbers_str"] = s1_train_clean["address_numbers"].apply(lambda s: ",".join(sorted(s)))
    s1_train_clean.drop(columns=["address_numbers"]).to_parquet(cache_dir / "s1_train_clean.parquet", index=False)
    print(f"Saved: {cache_dir / 's1_train_clean.parquet'} ({len(s1_train_clean)} rows)")
    del s1_train, s1_train_clean
    gc.collect()

    # 2. Train Pool (Source 2 + Source 3)
    print("\n[2/4] Processing Train Pool (Source 2 & Source 3)...")
    s2_train = load_source_tsv(train_dir / "train_source2.tsv", nrows=train_limit * 2 if train_limit else None, expected_prefix="S2-")
    s2_clean = preprocess_dataframe(s2_train)
    del s2_train
    gc.collect()

    s3_train = load_source_tsv(train_dir / "train_source3.tsv", nrows=train_limit * 2 if train_limit else None, expected_prefix="S3-")
    s3_clean = preprocess_dataframe(s3_train)
    del s3_train
    gc.collect()

    pool_train = pd.concat([s2_clean, s3_clean], ignore_index=True)
    del s2_clean, s3_clean
    pool_train["address_numbers_str"] = pool_train["address_numbers"].apply(lambda s: ",".join(sorted(s)))
    pool_train.drop(columns=["address_numbers"]).to_parquet(cache_dir / "pool_train_clean.parquet", index=False)
    print(f"Saved: {cache_dir / 'pool_train_clean.parquet'} ({len(pool_train)} rows)")
    del pool_train
    gc.collect()

    # 3. Test Source 1
    print("\n[3/4] Processing Test Source 1...")
    s1_test = load_source_tsv(test_dir / "test_source1.tsv", nrows=test_limit, expected_prefix="S1-")
    s1_test_clean = preprocess_dataframe(s1_test)
    s1_test_clean["address_numbers_str"] = s1_test_clean["address_numbers"].apply(lambda s: ",".join(sorted(s)))
    s1_test_clean.drop(columns=["address_numbers"]).to_parquet(cache_dir / "s1_test_clean.parquet", index=False)
    print(f"Saved: {cache_dir / 's1_test_clean.parquet'} ({len(s1_test_clean)} rows)")
    del s1_test, s1_test_clean
    gc.collect()

    # 4. Test Pool (Source 2 + Source 3)
    print("\n[4/4] Processing Test Pool (Source 2 & Source 3)...")
    s2_test = load_source_tsv(test_dir / "test_source2.tsv", nrows=test_limit * 2 if test_limit else None, expected_prefix="S2-")
    s2_test_clean = preprocess_dataframe(s2_test)
    del s2_test
    gc.collect()

    s3_test = load_source_tsv(test_dir / "test_source3.tsv", nrows=test_limit * 2 if test_limit else None, expected_prefix="S3-")
    s3_test_clean = preprocess_dataframe(s3_test)
    del s3_test
    gc.collect()

    pool_test = pd.concat([s2_test_clean, s3_test_clean], ignore_index=True)
    del s2_test_clean, s3_test_clean
    pool_test["address_numbers_str"] = pool_test["address_numbers"].apply(lambda s: ",".join(sorted(s)))
    pool_test.drop(columns=["address_numbers"]).to_parquet(cache_dir / "pool_test_clean.parquet", index=False)
    print(f"Saved: {cache_dir / 'pool_test_clean.parquet'} ({len(pool_test)} rows)")
    del pool_test
    gc.collect()

    print("\n✅ [STAGE 1 COMPLETE] All preprocessed datasets successfully cached to:", cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Preprocess raw TSVs and build cache")
    parser.add_argument("--train-dir", type=str, default="student_resource/dataset/train")
    parser.add_argument("--test-dir", type=str, default="student_resource/dataset/test")
    parser.add_argument("--cache-dir", type=str, default="processed_data")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args()

    run_stage1_preprocess(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        cache_dir=Path(args.cache_dir),
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )


if __name__ == "__main__":
    main()
