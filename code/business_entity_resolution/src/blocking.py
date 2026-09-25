"""Blocking / Candidate Generation module for Business Entity Resolution.
Implements high-recall, multi-indexer candidate generation partitioned by country
using token inverted indices, numerical address signatures, and TF-IDF n-gram scoring.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# Common high-frequency stopwords / generic entity words to skip as sole blocking keys
GENERIC_STOPWORDS = {
    "and", "the", "of", "in", "for", "with", "a", "an", "at", "by", "to",
    "pvt", "ltd", "inc", "llc", "corp", "co", "sa", "sarl", "sas", "eurl",
    "street", "road", "avenue", "lane", "drive", "near", "opposite", "opp",
    "building", "floor", "suite", "apartment", "rue", "allee", "place"
}


def extract_blocking_tokens(clean_name: str) -> List[str]:
    """Extracts informative tokens and 2-token shingles for inverted index keying."""
    tokens = [t for t in clean_name.split() if len(t) >= 2 and t not in GENERIC_STOPWORDS]
    keys = list(tokens)
    # Add adjacent pairs (bigram keys) for multi-word precision
    for i in range(len(tokens) - 1):
        keys.append(f"{tokens[i]}_{tokens[i+1]}")
    return keys


class MultiIndexBlocker:
    """Multi-indexer candidate generator partitioned by country."""

    def __init__(self, max_candidates_per_entity: int = 25):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.country_token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.country_number_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.pool_records: Dict[str, Dict[str, str]] = {}

    def fit_pool(self, pool_df: pd.DataFrame) -> None:
        """Indexes candidate pools (Source 2 and Source 3 records).
        
        Args:
            pool_df: DataFrame containing entity_id, clean_name, clean_address, country, address_numbers.
        """
        for _, row in pool_df.iterrows():
            eid = row["entity_id"]
            country = row["country"]
            c_name = row["clean_name"]
            nums = row.get("address_numbers", set())

            self.pool_records[eid] = {
                "name": c_name,
                "address": row["clean_address"],
                "country": country,
            }

            # 1. Token-based indexing
            tokens = extract_blocking_tokens(c_name)
            for tok in tokens:
                self.country_token_index[country][tok].append(eid)

            # 2. Number / postal code indexing
            if isinstance(nums, set):
                for num in nums:
                    if len(num) >= 3:  # Only index numbers with 3+ digits (avoid single digits)
                        self.country_number_index[country][num].append(eid)

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """Generates candidate matches from Source 2/3 for each Source 1 record.
        
        Args:
            s1_df: DataFrame containing Source 1 records.
            
        Returns:
            Dict mapping s1_entity_id -> List of candidate entity_ids.
        """
        candidate_map: Dict[str, List[str]] = {}

        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            country = row["country"]
            c_name = row["clean_name"]
            nums = row.get("address_numbers", set())

            candidate_counts: Dict[str, int] = defaultdict(int)

            # Match by name tokens / bigrams
            tokens = extract_blocking_tokens(c_name)
            for tok in tokens:
                for match_id in self.country_token_index[country].get(tok, []):
                    candidate_counts[match_id] += 2  # Higher weight for name match

            # Match by number / postal overlap
            if isinstance(nums, set):
                for num in nums:
                    if len(num) >= 3:
                        for match_id in self.country_number_index[country].get(num, []):
                            candidate_counts[match_id] += 1

            if candidate_counts:
                # Sort candidates by match frequency / score
                sorted_candidates = sorted(
                    candidate_counts.keys(),
                    key=lambda cid: candidate_counts[cid],
                    reverse=True,
                )
                candidate_map[s1_id] = sorted_candidates[: self.max_candidates_per_entity]
            else:
                candidate_map[s1_id] = []

        return candidate_map


def format_candidate_pairs_dataframe(
    candidate_map: Dict[str, List[str]],
) -> pd.DataFrame:
    """Formats candidate map into candidate_pairs.tsv DataFrame format."""
    rows = []
    for s1_id, cands in candidate_map.items():
        cands_str = ",".join(cands) if cands else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cands_str})
    return pd.DataFrame(rows)
