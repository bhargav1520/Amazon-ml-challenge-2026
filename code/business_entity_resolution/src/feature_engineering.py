"""Feature engineering module for Business Entity Resolution.
Computes high-dimensional string distance, token set, phonetic, and numerical
features on candidate record pairs using rapidfuzz (C++ engine).
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance


FEATURE_NAMES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_levenshtein_norm",
    "name_len_diff",
    "name_len_ratio",
    "addr_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaro_winkler",
    "addr_levenshtein_norm",
    "addr_len_diff",
    "addr_len_ratio",
    "number_jaccard",
    "number_intersection_count",
    "has_matching_number",
    "comb_token_sort_ratio",
    "comb_token_set_ratio",
    "is_source_2",
    "is_source_3",
]


def extract_pair_features(
    s1_record: Dict[str, Any],
    cand_record: Dict[str, Any],
) -> List[float]:
    """Extracts pairwise feature vector between Source 1 record and a candidate record.
    
    Args:
        s1_record: Dict with keys 'name', 'address', 'combined', 'numbers', 'id'
        cand_record: Dict with keys 'name', 'address', 'combined', 'numbers', 'id'
        
    Returns:
        List of numerical feature values matching FEATURE_NAMES.
    """
    n1, n2 = s1_record["name"], cand_record["name"]
    a1, a2 = s1_record["address"], cand_record["address"]
    c1, c2 = s1_record["combined"], cand_record["combined"]
    nums1: Set[str] = s1_record.get("numbers", set())
    nums2: Set[str] = cand_record.get("numbers", set())
    cand_id: str = cand_record.get("id", "")

    # 1. Name Features
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_partial_ratio = fuzz.partial_ratio(n1, n2) / 100.0
    name_token_sort_ratio = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_token_set_ratio = fuzz.token_set_ratio(n1, n2) / 100.0
    name_jaro_winkler = float(distance.JaroWinkler.similarity(n1, n2))
    name_levenshtein_norm = float(distance.Levenshtein.normalized_similarity(n1, n2))
    name_len_diff = float(abs(len(n1) - len(n2)))
    max_nl = max(len(n1), len(n2), 1)
    name_len_ratio = float(min(len(n1), len(n2)) / max_nl)

    # 2. Address Features
    addr_ratio = fuzz.ratio(a1, a2) / 100.0
    addr_partial_ratio = fuzz.partial_ratio(a1, a2) / 100.0
    addr_token_sort_ratio = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_token_set_ratio = fuzz.token_set_ratio(a1, a2) / 100.0
    addr_jaro_winkler = float(distance.JaroWinkler.similarity(a1, a2))
    addr_levenshtein_norm = float(distance.Levenshtein.normalized_similarity(a1, a2))
    addr_len_diff = float(abs(len(a1) - len(a2)))
    max_al = max(len(a1), len(a2), 1)
    addr_len_ratio = float(min(len(a1), len(a2)) / max_al)

    # 3. Number / Postal Overlap
    if nums1 and nums2:
        inter = nums1.intersection(nums2)
        union = nums1.union(nums2)
        num_jaccard = float(len(inter) / len(union)) if union else 0.0
        num_inter_cnt = float(len(inter))
        has_match_num = 1.0 if len(inter) > 0 else 0.0
    else:
        num_jaccard = 0.0
        num_inter_cnt = 0.0
        has_match_num = 0.0

    # 4. Combined text & Origin metadata
    comb_sort = fuzz.token_sort_ratio(c1, c2) / 100.0
    comb_set = fuzz.token_set_ratio(c1, c2) / 100.0
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return [
        name_ratio,
        name_partial_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_jaro_winkler,
        name_levenshtein_norm,
        name_len_diff,
        name_len_ratio,
        addr_ratio,
        addr_partial_ratio,
        addr_token_sort_ratio,
        addr_token_set_ratio,
        addr_jaro_winkler,
        addr_levenshtein_norm,
        addr_len_diff,
        addr_len_ratio,
        num_jaccard,
        num_inter_cnt,
        has_match_num,
        comb_sort,
        comb_set,
        is_s2,
        is_s3,
    ]
