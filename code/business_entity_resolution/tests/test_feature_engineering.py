"""Unit tests for feature_engineering.py"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import (
    FEATURE_NAMES,
    extract_pair_features,
)


def test_extract_pair_features_exact_match():
    r1 = {
        "id": "S1-001",
        "name": "apple inc",
        "address": "1 infinite loop cupertino ca 95014",
        "combined": "apple inc 1 infinite loop cupertino ca 95014",
        "numbers": {"1", "95014"},
    }
    r2 = {
        "id": "S2-001",
        "name": "apple inc",
        "address": "1 infinite loop cupertino ca 95014",
        "combined": "apple inc 1 infinite loop cupertino ca 95014",
        "numbers": {"1", "95014"},
    }
    feats = extract_pair_features(r1, r2)
    assert len(feats) == len(FEATURE_NAMES)
    assert feats[FEATURE_NAMES.index("name_ratio")] == 1.0
    assert feats[FEATURE_NAMES.index("addr_ratio")] == 1.0
    assert feats[FEATURE_NAMES.index("has_matching_number")] == 1.0
    assert feats[FEATURE_NAMES.index("is_source_2")] == 1.0
    assert feats[FEATURE_NAMES.index("is_source_3")] == 0.0


def test_extract_pair_features_different():
    r1 = {
        "id": "S1-001",
        "name": "google llc",
        "address": "1600 amphitheatre pkwy mountain view ca 94043",
        "combined": "google llc 1600 amphitheatre pkwy mountain view ca 94043",
        "numbers": {"1600", "94043"},
    }
    r2 = {
        "id": "S3-999",
        "name": "starbucks coffee",
        "address": "2401 utah ave s seattle wa 98134",
        "combined": "starbucks coffee 2401 utah ave s seattle wa 98134",
        "numbers": {"2401", "98134"},
    }
    feats = extract_pair_features(r1, r2)
    assert len(feats) == len(FEATURE_NAMES)
    assert feats[FEATURE_NAMES.index("name_ratio")] < 0.3
    assert feats[FEATURE_NAMES.index("has_matching_number")] == 0.0
    assert feats[FEATURE_NAMES.index("is_source_3")] == 1.0
