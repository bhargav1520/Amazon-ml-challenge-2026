"""Local Competition Evaluation & Judging Suite for Amazon ML Challenge 2026.
Simulates official host judging with exact macro-F0.5 scoring, country breakdown,
singleton accuracy, and precision-recall audit.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from src.postprocessing import compute_entity_f05, compute_macro_f05


def run_local_evaluation(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    country_map: Dict[str, str] = None,
) -> Dict[str, float]:
    """Evaluates entity resolution predictions against ground truth.
    
    Args:
        predictions: Dict mapping s1_id -> Set of predicted matching entity IDs
        ground_truth: Dict mapping s1_id -> Set of true matching entity IDs (empty for singletons)
        country_map: Dict mapping s1_id -> country string (optional)
        
    Returns:
        Dict with overall F0.5, Precision, Recall, Singleton Accuracy, and Country-wise scores.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {}

    scores = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    singleton_total = 0
    singleton_correct = 0

    country_scores: Dict[str, List[float]] = {}

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f05 = compute_entity_f05(pred_set, true_set)
        scores.append(f05)

        if country_map and s1_id in country_map:
            c = country_map[s1_id]
            if c not in country_scores:
                country_scores[c] = []
            country_scores[c].append(f05)

        # Singleton tracking
        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1

        # Global Precision / Recall tracking
        tp = len(pred_set.intersection(true_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    macro_f05 = float(np.mean(scores))
    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 1.0
    micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0

    print("=" * 65)
    print("[RESULT] [LOCAL COMPETITION JUDGING REPORT]")
    print("=" * 65)
    print(f"[METRICS] Evaluated Entities:       {total_entities:,}")
    print(f"[SCORE] Overall Macro F_0.5 Score: {macro_f05:.4f}")
    print(f"[SCORE] Micro Precision:          {micro_prec:.4f}")
    print(f"[SCORE] Micro Recall:             {micro_rec:.4f}")
    print(f"[SHIELD] Singleton Accuracy:       {singleton_acc:.2%} ({singleton_correct:,} / {singleton_total:,})")
    print(f"[MISS] False Positives (Alarms): {fp_total:,} | False Negatives (Misses): {fn_total:,}")

    results = {
        "macro_f05": macro_f05,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "singleton_accuracy": singleton_acc,
    }

    if country_scores:
        print("\n[COUNTRY] [Country-Wise Performance Breakdown]")
        for c, c_scores in country_scores.items():
            c_f05 = float(np.mean(c_scores))
            results[f"f05_{c}"] = c_f05
            print(f"  - {c:12s}: Macro F_0.5 = {c_f05:.4f} ({len(c_scores):,} entities)")
    print("=" * 65)

    return results
