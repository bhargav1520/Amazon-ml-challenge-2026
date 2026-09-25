"""End-to-end integration test for pipeline.py"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline


def test_pipeline_end_to_end(tmp_path: Path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    output_dir = tmp_path / "output"

    train_dir.mkdir()
    test_dir.mkdir()

    # Create dummy train data
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

    # Create dummy test data
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

    # Run pipeline
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
