"""Unit tests for postprocessing.py"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.postprocessing import (
    compute_entity_f05,
    compute_macro_f05,
    optimize_threshold,
)


def test_compute_entity_f05_perfect_match():
    preds = {"S2-1", "S3-2"}
    truth = {"S2-1", "S3-2"}
    assert compute_entity_f05(preds, truth) == 1.0


def test_compute_entity_f05_singletons():
    # True singleton predicted as singleton -> 1.0
    assert compute_entity_f05(set(), set()) == 1.0
    # True singleton predicted as match -> 0.0
    assert compute_entity_f05({"S2-1"}, set()) == 0.0
    # True match predicted as singleton -> 0.0
    assert compute_entity_f05(set(), {"S2-1"}) == 0.0


def test_compute_entity_f05_precision_heavy_example():
    # Example from PDF:
    # Model predicts [S2-00047, S2-00193, S3-00812]
    # Ground truth: [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 1.0 -> F_0.5 = 0.7142857...
    preds = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    score = compute_entity_f05(preds, truth)
    assert abs(score - 0.7142857) < 1e-4


def test_compute_macro_f05():
    preds = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
    }
    truth = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
    }
    assert compute_macro_f05(preds, truth) == 1.0


def test_optimize_threshold():
    candidate_scores = {
        "S1-1": [("S2-1", 0.95), ("S2-2", 0.20)],
        "S1-2": [("S2-3", 0.35)],  # True singleton
    }
    truth = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
    }
    best_thresh, best_score = optimize_threshold(candidate_scores, truth)
    assert best_score == 1.0
    assert best_thresh > 0.35  # Threshold should filter out S2-3 to award 1.0 on singleton
