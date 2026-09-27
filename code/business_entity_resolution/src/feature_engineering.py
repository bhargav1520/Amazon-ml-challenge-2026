"""Feature engineering module for Business Entity Resolution.
Computes high-dimensional string distance, token set, phonetic Soundex, and numerical
features on candidate record pairs using rapidfuzz (C++ engine).

KEY ADDITIONS:
- name_token_jaccard: Token-level Jaccard (stronger than token_set_ratio for short names)
- name_prefix3_match: First 3 characters match (transliteration resilience)
- name_bigram_jaccard: Character bigram Jaccard (catches transpositions)
- addr_token_jaccard: Token-level address Jaccard
- name_is_empty / addr_is_empty: Null-signal features
- num_conflict_ratio: Fraction of conflicting address numbers (stronger penalty)
- name_char_ngram_jaccard: 3-gram Jaccard for phonetic/spelling robustness
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance


def soundex(token: str) -> str:
    """Computes standard 4-character Soundex phonetic code for string token."""
    if not token or not token.isalpha():
        return ""
    token = token.upper()
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    code = [token[0]]
    for char in token[1:]:
        digit = mapping.get(char, "")
        if digit and (not code or digit != code[-1]):
            code.append(digit)
    return ("".join(code) + "000")[:4]


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Returns the set of character n-grams from text."""
    clean = "".join(c for c in text.lower() if c.isalnum())
    if len(clean) < n:
        return set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}


def _token_jaccard(s1: str, s2: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _bigram_jaccard(s1: str, s2: str) -> float:
    """Word-bigram Jaccard similarity (order-invariant sorted pairs).
    
    FIX: Returns 0.0 when either side is a single-word (no bigrams possible).
    The old code returned 1.0 when BOTH sides had no bigrams, incorrectly
    treating e.g. 'Walmart' vs 'Target' as maximally similar.
    """
    t1 = s1.split()
    t2 = s2.split()
    def bigrams(toks):
        return {tuple(sorted([toks[i], toks[i+1]])) for i in range(len(toks)-1)}
    b1 = bigrams(t1)
    b2 = bigrams(t2)
    if not b1 or not b2:
        # Cannot form bigrams on either side — undefined, treat as 0 (no evidence of similarity)
        return 0.0
    return len(b1 & b2) / len(b1 | b2)


FEATURE_NAMES = [
    # 1. Name Similarities & Phonetics
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_levenshtein_norm",
    "name_len_diff",
    "name_len_ratio",
    "name_exact",
    "name_containment",
    "name_first_token_match",
    "name_phonetic_jaccard",
    "name_first_phonetic_match",
    # NEW: richer name features
    "name_token_jaccard",
    "name_prefix3_match",
    "name_bigram_jaccard",
    "name_char_ngram_jaccard",
    "name_is_empty",
    # 2. Address Similarities
    "addr_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaro_winkler",
    "addr_levenshtein_norm",
    "addr_len_diff",
    "addr_len_ratio",
    "addr_exact",
    "addr_containment",
    # NEW: richer address features
    "addr_token_jaccard",
    "addr_is_empty",
    # 3. Number / Postal Overlap & Penalty
    "number_jaccard",
    "number_intersection_count",
    "has_matching_number",
    "num_mismatch_penalty",
    "num_exact_match",
    "num_conflict_ratio",   # NEW: proportion of conflicting numbers
    # 4. Combined text & Origin metadata
    "comb_sort",
    "comb_set",
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

    # Safety: ensure strings
    n1, n2 = str(n1) if n1 else "", str(n2) if n2 else ""
    a1, a2 = str(a1) if a1 else "", str(a2) if a2 else ""
    c1, c2 = str(c1) if c1 else "", str(c2) if c2 else ""

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
    name_exact = 1.0 if (n1 and n2 and n1 == n2) else 0.0
    name_containment = 1.0 if (n1 and n2 and (n1 in n2 or n2 in n1)) else 0.0

    t1 = n1.split()
    t2 = n2.split()
    name_first_token_match = 1.0 if (t1 and t2 and t1[0] == t2[0]) else 0.0

    # Phonetics
    sx1 = [soundex(t) for t in t1 if len(t) >= 2]
    sx2 = [soundex(t) for t in t2 if len(t) >= 2]
    sx1_set = set(filter(None, sx1))
    sx2_set = set(filter(None, sx2))
    if sx1_set and sx2_set:
        name_phonetic_jaccard = float(len(sx1_set.intersection(sx2_set)) / len(sx1_set.union(sx2_set)))
    else:
        name_phonetic_jaccard = 0.0
    name_first_phonetic_match = 1.0 if (sx1 and sx2 and sx1[0] and sx1[0] == sx2[0]) else 0.0

    # NEW name features
    name_token_jaccard = _token_jaccard(n1, n2)
    # Prefix-3 match: first 3 alphanum chars match
    n1_clean = "".join(c for c in n1 if c.isalnum())
    n2_clean = "".join(c for c in n2 if c.isalnum())
    name_prefix3_match = 1.0 if (len(n1_clean) >= 3 and len(n2_clean) >= 3 and n1_clean[:3] == n2_clean[:3]) else 0.0
    name_bigram_jaccard = _bigram_jaccard(n1, n2)
    # Char 3-gram Jaccard
    ng1 = _char_ngrams(n1, 3)
    ng2 = _char_ngrams(n2, 3)
    if ng1 and ng2:
        name_char_ngram_jaccard = float(len(ng1 & ng2) / len(ng1 | ng2))
    else:
        name_char_ngram_jaccard = 0.0
    name_is_empty = 1.0 if (not n1 or not n2) else 0.0

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
    addr_exact = 1.0 if (a1 and a2 and a1 == a2) else 0.0
    addr_containment = 1.0 if (a1 and a2 and (a1 in a2 or a2 in a1)) else 0.0
    # NEW addr features
    addr_token_jaccard = _token_jaccard(a1, a2)
    addr_is_empty = 1.0 if (not a1 or not a2) else 0.0

    # 3. Number / Postal Overlap & Conflict Penalty
    if nums1 and nums2:
        # Both sides have numbers: full overlap metrics
        inter = nums1.intersection(nums2)
        union = nums1.union(nums2)
        num_jaccard = float(len(inter) / len(union)) if union else 0.0
        num_inter_cnt = float(len(inter))
        has_match_num = 1.0 if len(inter) > 0 else 0.0
        num_mismatch_penalty = 1.0 if len(inter) == 0 else 0.0
        num_exact_match = 1.0 if nums1 == nums2 else 0.0
        num_conflict_ratio = float(len(nums1 - nums2) / max(len(nums1), 1))
    elif nums1 and not nums2:
        # FIX (Bottleneck 4): Entity 1 has numbers but candidate has none.
        # Previously set to 0.0 (false "no conflict"). Now signals full conflict.
        num_jaccard = 0.0
        num_inter_cnt = 0.0
        has_match_num = 0.0
        num_mismatch_penalty = 1.0   # definite mismatch: s1 has numbers, cand doesn't
        num_exact_match = 0.0
        num_conflict_ratio = 1.0     # all s1 numbers are missing from candidate
    elif not nums1 and nums2:
        # Mirror case: candidate has numbers but s1 doesn't.
        num_jaccard = 0.0
        num_inter_cnt = 0.0
        has_match_num = 0.0
        num_mismatch_penalty = 1.0
        num_exact_match = 0.0
        num_conflict_ratio = 0.0     # s1 has no numbers, so conflict ratio is undefined → 0
    else:
        # Neither side has address numbers — truly neutral
        num_jaccard = 0.0
        num_inter_cnt = 0.0
        has_match_num = 0.0
        num_mismatch_penalty = 0.0
        num_exact_match = 0.0
        num_conflict_ratio = 0.0

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
        name_exact,
        name_containment,
        name_first_token_match,
        name_phonetic_jaccard,
        name_first_phonetic_match,
        name_token_jaccard,
        name_prefix3_match,
        name_bigram_jaccard,
        name_char_ngram_jaccard,
        name_is_empty,
        addr_ratio,
        addr_partial_ratio,
        addr_token_sort_ratio,
        addr_token_set_ratio,
        addr_jaro_winkler,
        addr_levenshtein_norm,
        addr_len_diff,
        addr_len_ratio,
        addr_exact,
        addr_containment,
        addr_token_jaccard,
        addr_is_empty,
        num_jaccard,
        num_inter_cnt,
        has_match_num,
        num_mismatch_penalty,
        num_exact_match,
        num_conflict_ratio,
        comb_sort,
        comb_set,
        is_s2,
        is_s3,
    ]
