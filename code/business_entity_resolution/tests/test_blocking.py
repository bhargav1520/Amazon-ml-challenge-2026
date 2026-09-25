"""Unit tests for blocking.py"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessor import preprocess_dataframe
from src.blocking import (
    MultiIndexBlocker,
    extract_blocking_tokens,
    format_candidate_pairs_dataframe,
)


def test_extract_blocking_tokens():
    tokens = extract_blocking_tokens("starbucks coffee roastery")
    assert "starbucks" in tokens
    assert "coffee" in tokens
    assert "starbucks_coffee" in tokens


def test_multi_index_blocker():
    s1_df = pd.DataFrame([
        {"entity_id": "S1-101", "business_name": "Microsoft Corp", "business_address": "1 Microsoft Way, Redmond, WA 98052", "country": "US"},
        {"entity_id": "S1-102", "business_name": "Tata Motors Ltd", "business_address": "Bombay House, Homi Mody Street 400001", "country": "India"},
        {"entity_id": "S1-103", "business_name": "Unknown Singleton", "business_address": "Nowhere Street 99999", "country": "US"},
    ])
    pool_df = pd.DataFrame([
        {"entity_id": "S2-201", "business_name": "Microsoft Corporation", "business_address": "Redmond WA 98052", "country": "US"},
        {"entity_id": "S3-301", "business_name": "Tata Motors", "business_address": "Mumbai 400001", "country": "India"},
        {"entity_id": "S2-202", "business_name": "Amazon LLC", "business_address": "Seattle WA", "country": "US"},
    ])

    s1_clean = preprocess_dataframe(s1_df)
    pool_clean = preprocess_dataframe(pool_df)

    blocker = MultiIndexBlocker(max_candidates_per_entity=10)
    blocker.fit_pool(pool_clean)
    cand_map = blocker.generate_candidates(s1_clean)

    assert "S2-201" in cand_map["S1-101"]
    assert "S3-301" in cand_map["S1-102"]
    assert cand_map["S1-103"] == []  # Singleton should have empty candidates

    cand_df = format_candidate_pairs_dataframe(cand_map)
    assert len(cand_df) == 3
    assert list(cand_df.columns) == ["source1_entity_id", "candidate_entity_ids"]
