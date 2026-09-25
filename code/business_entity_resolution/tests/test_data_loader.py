"""Unit tests for data_loader.py"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add business_entity_resolution root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_source_tsv,
    load_ground_truth,
)


def test_load_source_tsv_valid(tmp_path: Path):
    tsv_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tAcme Corp\t123 Main St\tUS\n"
        "S1-002\tReliance Retail\tMG Road, Bangalore\tIndia\n"
    )
    test_file = tmp_path / "test_source1.tsv"
    test_file.write_text(tsv_content, encoding="utf-8")

    df = load_source_tsv(test_file, expected_prefix="S1-")
    assert len(df) == 2
    assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"]
    assert df.iloc[0]["business_name"] == "Acme Corp"
    assert df.iloc[1]["country"] == "India"


def test_load_source_tsv_prefix_mismatch(tmp_path: Path):
    tsv_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-001\tAcme Corp\t123 Main St\tUS\n"
    )
    test_file = tmp_path / "test_wrong_prefix.tsv"
    test_file.write_text(tsv_content, encoding="utf-8")

    with pytest.raises(ValueError, match="do not match prefix 'S1-'"):
        load_source_tsv(test_file, expected_prefix="S1-")


def test_load_ground_truth(tmp_path: Path):
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001,S3-002\n"
        "S1-002\t\n"
        "S1-003\tS2-005\n"
    )
    test_file = tmp_path / "train_ground_truth.tsv"
    test_file.write_text(gt_content, encoding="utf-8")

    gt = load_ground_truth(test_file)
    assert len(gt) == 3
    assert gt["S1-001"] == {"S2-001", "S3-002"}
    assert gt["S1-002"] == set()  # Singleton
    assert gt["S1-003"] == {"S2-005"}
