"""Data loader module for Business Entity Resolution.
Handles loading TSV files with explicit tab separation, missing value imputation,
schema validation, and ground truth parsing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def load_source_tsv(
    filepath: Path,
    nrows: Optional[int] = None,
    expected_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Loads a source TSV file (Source 1, 2, or 3) with explicit tab delimiter.
    
    Args:
        filepath: Path to the TSV file.
        nrows: Optional row limit for fast debugging / prototyping.
        expected_prefix: Optional prefix check (e.g. 'S1-', 'S2-', 'S3-').
        
    Returns:
        pd.DataFrame with columns: entity_id, business_name, business_address, country.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Source file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )

    # Validate columns
    missing_cols = set(REQUIRED_SOURCE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {filepath}: {missing_cols}")

    # Fill NA / None with empty string
    df["business_name"] = df["business_name"].fillna("").astype(str).str.strip()
    df["business_address"] = df["business_address"].fillna("").astype(str).str.strip()
    df["country"] = df["country"].fillna("").astype(str).str.strip()
    df["entity_id"] = df["entity_id"].fillna("").astype(str).str.strip()

    if expected_prefix:
        valid_prefix = df["entity_id"].str.startswith(expected_prefix).all()
        if not valid_prefix:
            raise ValueError(
                f"Some entities in {filepath} do not match prefix '{expected_prefix}'"
            )

    return df


def load_ground_truth(
    filepath: Path,
    nrows: Optional[int] = None,
) -> Dict[str, Set[str]]:
    """Loads ground truth matching pairs into a dictionary for O(1) lookup.
    
    Args:
        filepath: Path to train_ground_truth.tsv
        nrows: Optional limit for testing.
        
    Returns:
        Dict mapping source1_entity_id -> Set of matched_entity_ids.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )

    missing_cols = set(REQUIRED_GROUND_TRUTH_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in ground truth: {missing_cols}")

    ground_truth: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        matches_str = row["matched_entity_ids"].strip()
        if matches_str:
            matches = {m.strip() for m in matches_str.split(",") if m.strip()}
        else:
            matches = set()
        ground_truth[s1_id] = matches

    return ground_truth
