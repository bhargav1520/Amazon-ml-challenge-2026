"""High-Performance Blocking / Candidate Generation module for Business Entity Resolution.
Features real-time percentage tracking, bounded inverted index postings, and zero-overhead candidate loops.
"""

import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd


GENERIC_STOPWORDS = {
    "and", "the", "of", "in", "for", "with", "a", "an", "at", "by", "to",
    "pvt", "ltd", "inc", "llc", "corp", "co", "sa", "sarl", "sas", "eurl",
    "street", "road", "avenue", "lane", "drive", "near", "opposite", "opp",
    "building", "floor", "suite", "apartment", "rue", "allee", "place", "de",
    "la", "le", "les", "du", "des", "en", "un", "une", "sur"
}


def extract_blocking_tokens(clean_name: str) -> List[str]:
    """Extracts informative tokens and 2-token shingles for inverted index keying."""
    if not clean_name:
        return []
    tokens = [t for t in clean_name.split() if len(t) >= 2 and t not in GENERIC_STOPWORDS]
    if not tokens:
        return []
    keys = list(tokens[:4])
    if len(tokens) >= 2:
        keys.append(f"{tokens[0]}_{tokens[1]}")
    return keys


class MultiIndexBlocker:
    """High-performance candidate generator partitioned by country with real-time percentage progress."""

    def __init__(self, max_candidates_per_entity: int = 25, max_postings_per_key: int = 200):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.max_postings_per_key = max_postings_per_key
        self.country_token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.country_number_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    def fit_pool(self, pool_df: pd.DataFrame, desc: str = "Indexing Candidate Pool") -> None:
        """Indexes candidate pools using fast native iteration with real-time percentage logs."""
        eids = pool_df["entity_id"].values
        countries = pool_df["country"].values
        names = pool_df["clean_name"].values
        
        # Support both address_numbers (set) and address_numbers_str (str)
        if "address_numbers_str" in pool_df.columns:
            raw_nums = pool_df["address_numbers_str"].values
            has_str_nums = True
        elif "address_numbers" in pool_df.columns:
            raw_nums = pool_df["address_numbers"].values
            has_str_nums = False
        else:
            raw_nums = [""] * len(pool_df)
            has_str_nums = True

        total_rows = len(eids)
        print(f"📌 {desc} ({total_rows:,} rows total)...", flush=True)
        log_step = max(100000, total_rows // 10)
        start_time = time.time()

        for i in range(total_rows):
            if i % log_step == 0 or i == total_rows - 1:
                pct = (i + 1) / total_rows * 100.0
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 0.001)
                eta = (total_rows - (i + 1)) / max(rate, 0.001)
                print(
                    f"  [{pct:5.1f}%] {i + 1:,} / {total_rows:,} indexed | "
                    f"Speed: {rate:,.0f} rows/s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                    flush=True,
                )

            eid = eids[i]
            country = countries[i]
            c_name = names[i]

            c_tok_idx = self.country_token_index[country]
            tokens = extract_blocking_tokens(c_name)
            for tok in tokens:
                postings = c_tok_idx[tok]
                if len(postings) < self.max_postings_per_key:
                    postings.append(eid)

            if has_str_nums:
                num_val = raw_nums[i]
                if num_val:
                    c_num_idx = self.country_number_index[country]
                    for num in num_val.split(","):
                        if len(num) >= 3:
                            num_postings = c_num_idx[num]
                            if len(num_postings) < self.max_postings_per_key:
                                num_postings.append(eid)
            else:
                nums_set = raw_nums[i]
                if isinstance(nums_set, set):
                    c_num_idx = self.country_number_index[country]
                    for num in nums_set:
                        if len(num) >= 3:
                            num_postings = c_num_idx[num]
                            if len(num_postings) < self.max_postings_per_key:
                                num_postings.append(eid)

        total_time = time.time() - start_time
        print(f"  [100.0%] Finished indexing {total_rows:,} rows in {total_time:.1f}s!\n", flush=True)

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        desc: str = "Generating Candidate Pairs",
    ) -> Dict[str, List[str]]:
        """Generates candidate matches for each Source 1 record with real-time percentage progress."""
        candidate_map: Dict[str, List[str]] = {}

        eids = s1_df["entity_id"].values
        countries = s1_df["country"].values
        names = s1_df["clean_name"].values

        if "address_numbers_str" in s1_df.columns:
            raw_nums = s1_df["address_numbers_str"].values
            has_str_nums = True
        elif "address_numbers" in s1_df.columns:
            raw_nums = s1_df["address_numbers"].values
            has_str_nums = False
        else:
            raw_nums = [""] * len(s1_df)
            has_str_nums = True

        total_rows = len(eids)
        print(f"📌 {desc} ({total_rows:,} entities total)...", flush=True)
        log_step = max(25000, total_rows // 10)
        start_time = time.time()

        for i in range(total_rows):
            if i % log_step == 0 or i == total_rows - 1:
                pct = (i + 1) / total_rows * 100.0
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 0.001)
                eta = (total_rows - (i + 1)) / max(rate, 0.001)
                print(
                    f"  [{pct:5.1f}%] {i + 1:,} / {total_rows:,} entities processed | "
                    f"Speed: {rate:,.0f} ent/s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                    flush=True,
                )

            s1_id = eids[i]
            country = countries[i]
            c_name = names[i]

            candidate_counts: Dict[str, int] = {}
            c_tok_idx = self.country_token_index.get(country, {})

            tokens = extract_blocking_tokens(c_name)
            for tok in tokens:
                postings = c_tok_idx.get(tok)
                if postings:
                    for match_id in postings:
                        candidate_counts[match_id] = candidate_counts.get(match_id, 0) + 2
                    if len(candidate_counts) >= 60:
                        break

            if len(candidate_counts) < 30 and has_str_nums:
                num_val = raw_nums[i]
                if num_val:
                    c_num_idx = self.country_number_index.get(country, {})
                    for num in num_val.split(","):
                        if len(num) >= 3:
                            num_postings = c_num_idx.get(num)
                            if num_postings:
                                for match_id in num_postings:
                                    candidate_counts[match_id] = candidate_counts.get(match_id, 0) + 1
                                if len(candidate_counts) >= 60:
                                    break

            if candidate_counts:
                if len(candidate_counts) <= self.max_candidates_per_entity:
                    candidate_map[s1_id] = list(candidate_counts.keys())
                else:
                    sorted_cands = sorted(
                        candidate_counts,
                        key=candidate_counts.get,
                        reverse=True,
                    )[: self.max_candidates_per_entity]
                    candidate_map[s1_id] = sorted_cands
            else:
                candidate_map[s1_id] = []

        total_time = time.time() - start_time
        print(f"  [100.0%] Finished candidate generation for {total_rows:,} entities in {total_time:.1f}s!\n", flush=True)
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
