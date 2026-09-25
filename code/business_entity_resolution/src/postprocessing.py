"""Evaluation and post-processing module for Business Entity Resolution.
Implements the exact macro-averaged F_0.5 metric, threshold optimization,
singleton filtering, and TSV export.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd


def compute_entity_f05(
    predicted_matches: Set[str],
    true_matches: Set[str],
) -> float:
    """Computes F_0.5 score for a single Source 1 entity.
    
    Singletons:
    - True empty, predicted empty: 1.0
    - True empty, predicted non-empty: 0.0
    - True non-empty, predicted empty: 0.0
    """
    if len(true_matches) == 0 and len(predicted_matches) == 0:
        return 1.0
    if len(true_matches) == 0 or len(predicted_matches) == 0:
        return 0.0

    tp = len(predicted_matches.intersection(true_matches))
    if tp == 0:
        return 0.0

    precision = tp / len(predicted_matches)
    recall = tp / len(true_matches)

    denom = (0.25 * precision) + recall
    if denom == 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """Computes macro-average F_0.5 across all Source 1 entities in ground truth."""
    if not ground_truth:
        return 0.0

    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores.append(compute_entity_f05(pred_set, true_set))

    return float(np.mean(scores))


def optimize_threshold(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    threshold_range: Tuple[float, float, int] = (0.3, 0.95, 25),
) -> Tuple[float, float]:
    """Finds optimal decision threshold maximizing macro-average F_0.5 score.
    
    Args:
        candidate_scores: Dict mapping s1_id -> List of (cand_id, probability_score)
        ground_truth: Dict mapping s1_id -> Set of true matching IDs
        threshold_range: (min_thresh, max_thresh, steps)
        
    Returns:
        Tuple of (best_threshold, best_f05_score)
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_range[2])
    best_thresh = 0.5
    best_score = -1.0

    for thresh in thresholds:
        preds: Dict[str, Set[str]] = {}
        for s1_id, score_list in candidate_scores.items():
            matched = {cid for cid, score in score_list if score >= thresh}
            preds[s1_id] = matched

        score = compute_macro_f05(preds, ground_truth)
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh, best_score


def export_matching_results(
    predictions: Dict[str, List[str]],
    all_s1_ids: List[str],
    output_path: str,
) -> None:
    """Exports predictions to matching_results.tsv with explicit tab delimiter and single row per S1."""
    rows = []
    for s1_id in all_s1_ids:
        cands = predictions.get(s1_id, [])
        cands_str = ",".join(cands) if cands else ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": cands_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
