"""Pairwise Matcher Model module for Business Entity Resolution.
Trains a high-precision gradient-boosted tree classifier with consistent feature names.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.feature_engineering import FEATURE_NAMES, extract_pair_features


class EntityMatcherModel:
    """Pairwise match classifier using LightGBM with strict feature name binding."""

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        max_depth: int = 7,
        num_leaves: int = 45,
        min_child_samples: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=0.6,
            random_state=random_state,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1,
        )
        self.feature_names = FEATURE_NAMES
        self.is_fitted = False

    def train_on_pairs(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fits the LightGBM classifier on feature DataFrame with explicit column names."""
        df_X = pd.DataFrame(X, columns=self.feature_names)
        self.model.fit(df_X, y)
        self.is_fitted = True

    def predict_pair_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Returns probability of match (class 1) using named DataFrame to prevent warnings."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")
        df_X = pd.DataFrame(X, columns=self.feature_names)
        probas = self.model.predict_proba(df_X)
        return probas[:, 1]

    def score_candidates(
        self,
        s1_record: Dict[str, Any],
        cand_records: List[Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        """Scores all candidate records for a given Source 1 record."""
        if not cand_records:
            return []

        X_pair = []
        c_ids = []
        for c_rec in cand_records:
            feats = extract_pair_features(s1_record, c_rec)
            X_pair.append(feats)
            c_ids.append(c_rec["id"])

        X_mat = np.array(X_pair, dtype=np.float32)
        probas = self.predict_pair_proba(X_mat)

        return list(zip(c_ids, [float(p) for p in probas]))
