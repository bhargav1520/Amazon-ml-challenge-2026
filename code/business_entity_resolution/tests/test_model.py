"""Unit tests for model.py"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import EntityMatcherModel
from src.feature_engineering import extract_pair_features, FEATURE_NAMES


def test_entity_matcher_model():
    r1 = {"id": "S1-1", "name": "google inc", "address": "1600 amphitheatre", "combined": "google inc 1600 amphitheatre", "numbers": {"1600"}}
    r_pos = {"id": "S2-1", "name": "google llc", "address": "1600 amphitheatre pkwy", "combined": "google llc 1600 amphitheatre pkwy", "numbers": {"1600"}}
    r_neg = {"id": "S3-2", "name": "starbucks", "address": "pike place", "combined": "starbucks pike place", "numbers": set()}

    f_pos = extract_pair_features(r1, r_pos)
    f_neg = extract_pair_features(r1, r_neg)

    X = np.tile(np.array([f_pos, f_neg], dtype=np.float32), (15, 1))
    y = np.tile(np.array([1, 0], dtype=np.int32), 15)

    model = EntityMatcherModel(n_estimators=10, min_child_samples=1)
    model.train_on_pairs(X, y)

    scores = model.score_candidates(r1, [r_pos, r_neg])
    assert len(scores) == 2
    # Probability for positive pair should be significantly higher than negative pair
    pos_score = [s for cid, s in scores if cid == "S2-1"][0]
    neg_score = [s for cid, s in scores if cid == "S3-2"][0]
    assert pos_score > neg_score
