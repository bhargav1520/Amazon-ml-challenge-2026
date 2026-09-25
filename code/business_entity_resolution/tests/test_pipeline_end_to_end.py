"""End-to-end integration tests for both pipeline.py and the 4-stage modular runners."""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline
from src.stage1_preprocess import run_stage1_preprocess
from src.stage2_blocking import run_stage2_blocking
from src.stage3_train import run_stage3_train
from src.stage4_inference import run_stage4_inference


@pytest.fixture
def dummy_dataset_dirs(tmp_path: Path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    # Dummy train data
    s1_train = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "123 Main St NY 10001", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Tata Motors", "business_address": "MG Road Mumbai 400001", "country": "India"},
        {"entity_id": "S1-3", "business_name": "Singleton Shop", "business_address": "Nowhere St 99999", "country": "US"},
    ])
    s2_train = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Acme Corporation", "business_address": "123 Main Street NY 10001", "country": "US"},
    ])
    s3_train = pd.DataFrame([
        {"entity_id": "S3-2", "business_name": "Tata Motors Ltd", "business_address": "MG Rd Mumbai 400001", "country": "India"},
    ])
    gt_train = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": "S3-2"},
        {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
    ])

    s1_train.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    s2_train.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    s3_train.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    gt_train.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    # Dummy test data
    s1_test = pd.DataFrame([
        {"entity_id": "S1-10", "business_name": "Acme Inc", "business_address": "123 Main Street New York 10001", "country": "US"},
        {"entity_id": "S1-20", "business_name": "Unknown Entity", "business_address": "Random Rd 11111", "country": "US"},
    ])
    s2_test = pd.DataFrame([
        {"entity_id": "S2-10", "business_name": "Acme Corp", "business_address": "123 Main St NY 10001", "country": "US"},
    ])
    s3_test = pd.DataFrame([
        {"entity_id": "S3-30", "business_name": "Google", "business_address": "Mountain View CA", "country": "US"},
    ])

    s1_test.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    s2_test.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    s3_test.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    return train_dir, test_dir


def test_pipeline_end_to_end(dummy_dataset_dirs, tmp_path: Path):
    train_dir, test_dir = dummy_dataset_dirs
    output_dir = tmp_path / "output"

    run_pipeline(
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        max_candidates_per_entity=10,
    )

    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"

    assert matching_tsv.exists()
    assert candidate_tsv.exists()

    df_match = pd.read_csv(matching_tsv, sep="\t", keep_default_na=False)
    assert len(df_match) == 2
    assert list(df_match.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert "S1-10" in df_match["source1_entity_id"].values
    assert "S1-20" in df_match["source1_entity_id"].values


def test_staged_pipeline_coordination(dummy_dataset_dirs, tmp_path: Path):
    """Tests the 4-stage sequential execution (Stage 1 -> 2 -> 3 -> 4) with cache passing."""
    train_dir, test_dir = dummy_dataset_dirs
    cache_dir = tmp_path / "processed_data"
    cand_cache_dir = tmp_path / "cache"
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "output_staged"

    # Stage 1: Preprocess
    run_stage1_preprocess(
        train_dir=train_dir,
        test_dir=test_dir,
        cache_dir=cache_dir,
    )
    assert (cache_dir / "s1_train_clean.parquet").exists()
    assert (cache_dir / "pool_train_clean.parquet").exists()
    assert (cache_dir / "s1_test_clean.parquet").exists()
    assert (cache_dir / "pool_test_clean.parquet").exists()

    # Stage 2: Blocking
    run_stage2_blocking(
        cache_dir=cache_dir,
        output_dir=output_dir,
        candidates_cache_dir=cand_cache_dir,
        max_candidates=10,
    )
    assert (cand_cache_dir / "train_candidates.pkl").exists()
    assert (cand_cache_dir / "test_candidates.pkl").exists()
    assert (output_dir / "candidate_pairs.tsv").exists()

    # Stage 3: Train
    run_stage3_train(
        cache_dir=cache_dir,
        candidates_cache_dir=cand_cache_dir,
        models_dir=models_dir,
        gt_file=train_dir / "train_ground_truth.tsv",
        max_train_entities=100,
        n_estimators=10,
    )
    assert (models_dir / "matcher_lgbm.pkl").exists()
    assert (models_dir / "best_threshold.txt").exists()

    # Stage 4: Inference
    run_stage4_inference(
        cache_dir=cache_dir,
        candidates_cache_dir=cand_cache_dir,
        models_dir=models_dir,
        output_dir=output_dir,
    )
    matching_tsv = output_dir / "matching_results.tsv"
    assert matching_tsv.exists()

    df_match = pd.read_csv(matching_tsv, sep="\t", keep_default_na=False)
    assert len(df_match) == 2
    assert list(df_match.columns) == ["source1_entity_id", "matched_entity_ids"]
