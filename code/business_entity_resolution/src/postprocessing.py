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


def filter_matches_with_barrier(
    score_list: List[Tuple[str, float]],
    threshold: float = 0.50,
    margin: float = 0.08,
    max_matches: int = 12,
) -> List[str]:
    """Filters candidate scores applying calibrated decision threshold with margin recovery.

    FIX (v2): Anchors the acceptance cutoff purely on top_score (not threshold).
    The old formula `max(threshold-margin, top_score-margin)` let entities with a
    borderline top_score accept far too many false positives.

    FIX (v3): max_matches raised 8 -> 12 to recover recall on legitimate high-degree
    clusters (e.g. large franchises/hotel chains with 9-12 branches).

    - If best candidate score < threshold -> singleton, return [].
    - If best candidate score >= threshold -> accept all within `margin` of top_score.
    - Hard cap at `max_matches` to prevent run-away FP cascades.
    """
    if not score_list:
        return []

    sorted_scores = sorted(score_list, key=lambda x: x[1], reverse=True)
    top_cid, top_score = sorted_scores[0]
    if top_score < threshold:
        return []

    if margin is None:
        valid = [cid for cid, score in sorted_scores if score >= threshold]
    else:
        # Anchor cutoff ONLY on top_score, never below threshold
        cutoff = max(threshold, top_score - margin)
        valid = [cid for cid, score in sorted_scores if score >= cutoff]

    # Hard cap to prevent FP cascades on ambiguous high-traffic entities
    return valid[:max_matches]


def optimize_threshold(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    threshold_range: Tuple[float, float, int] = (0.15, 0.95, 81),
    margin: float = 0.08,
    max_matches: int = 12,
) -> Tuple[float, float]:
    """Finds optimal decision threshold maximizing macro-average F_0.5 score."""
    thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_range[2])
    best_thresh = 0.50
    best_score = -1.0

    for thresh in thresholds:
        preds: Dict[str, Set[str]] = {}
        for s1_id, score_list in candidate_scores.items():
            matched = filter_matches_with_barrier(score_list, threshold=thresh, margin=margin, max_matches=max_matches)
            preds[s1_id] = set(matched)

        score = compute_macro_f05(preds, ground_truth)
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh, best_score


def optimize_country_thresholds(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    country_map: Dict[str, str],
    threshold_range: Tuple[float, float, int] = (0.15, 0.95, 81),
    margin: float = 0.08,
    max_matches: int = 12,
) -> Tuple[Dict[str, float], float]:
    """Finds per-country optimal decision thresholds maximizing macro-average F_0.5 score."""
    country_groups: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    country_gt: Dict[str, Dict[str, Set[str]]] = {}

    for s1_id, scores in candidate_scores.items():
        c = country_map.get(s1_id, "default")
        if c not in country_groups:
            country_groups[c] = {}
            country_gt[c] = {}
        country_groups[c][s1_id] = scores
        country_gt[c][s1_id] = ground_truth.get(s1_id, set())

    country_thresholds: Dict[str, float] = {}
    all_preds: Dict[str, Set[str]] = {}

    for c, c_cands in country_groups.items():
        c_gt = country_gt[c]
        c_best_th, _ = optimize_threshold(c_cands, c_gt, threshold_range=threshold_range, margin=margin, max_matches=max_matches)
        country_thresholds[c] = c_best_th

        for s1_id, score_list in c_cands.items():
            matched = filter_matches_with_barrier(score_list, threshold=c_best_th, margin=margin, max_matches=max_matches)
            all_preds[s1_id] = set(matched)

    total_macro_f05 = compute_macro_f05(all_preds, ground_truth)
    return country_thresholds, total_macro_f05


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
