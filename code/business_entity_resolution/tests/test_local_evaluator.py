"""Unit tests for local_evaluator.py"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.local_evaluator import run_local_evaluation


def test_local_evaluator_perfect_score():
    ground_truth = {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": {"S2-2"},
        "S1-3": set(),  # Singleton
        "S1-4": set(),  # Singleton
    }
    predictions = {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": {"S2-2"},
        "S1-3": set(),
        "S1-4": set(),
    }
    country_map = {"S1-1": "US", "S1-2": "India", "S1-3": "US", "S1-4": "France"}

    results = run_local_evaluation(predictions, ground_truth, country_map)
    assert results["macro_f05"] == 1.0
    assert results["singleton_accuracy"] == 1.0
    assert results["micro_precision"] == 1.0
    assert results["micro_recall"] == 1.0
    assert results["f05_US"] == 1.0
    assert results["f05_India"] == 1.0
    assert results["f05_France"] == 1.0
