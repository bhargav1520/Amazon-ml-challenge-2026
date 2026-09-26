"""Pairwise Matcher Model module for Business Entity Resolution.
Trains a high-precision dual gradient-boosted ensemble (LightGBM + CatBoost)
with Stratified K-Fold bagging to eliminate variance and maximize leaderboard precision.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from src.feature_engineering import FEATURE_NAMES, extract_pair_features


class EntityMatcherModel:
    """Pairwise match classifier using Stratified K-Fold LightGBM + CatBoost dual ensemble."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.06,
        max_depth: int = 7,
        num_leaves: int = 45,
        min_child_samples: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_folds: int = 3,
        enable_ensemble: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_folds = n_folds
        self.enable_ensemble = enable_ensemble and HAS_CATBOOST
        self.random_state = random_state

        self.lgb_models: List[lgb.LGBMClassifier] = []
        self.cb_models: List[Any] = []
        self.feature_names = FEATURE_NAMES
        self.is_fitted = False

    def train_on_pairs(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fits the K-Fold model ensemble on feature DataFrame with explicit column names."""
        df_X = pd.DataFrame(X, columns=self.feature_names)
        self.lgb_models.clear()
        self.cb_models.clear()

        # If data size allows, train across Stratified Folds
        if len(y) >= self.n_folds * 10:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            splits = list(skf.split(df_X, y))
        else:
            splits = [(np.arange(len(y)), np.arange(len(y)))]

        for fold_idx, (train_idx, _) in enumerate(splits):
            X_f, y_f = df_X.iloc[train_idx], y[train_idx]
            
            lgbm = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=1.0,
                random_state=self.random_state + fold_idx,
                n_jobs=-1,
                importance_type="gain",
                verbose=-1,
            )
            lgbm.fit(X_f, y_f)
            self.lgb_models.append(lgbm)

            if self.enable_ensemble:
                cb = CatBoostClassifier(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=min(self.max_depth, 7),
                    random_seed=self.random_state + fold_idx,
                    thread_count=-1,
                    verbose=False,
                )
                cb.fit(X_f, y_f)
                self.cb_models.append(cb)

        self.is_fitted = True

    def predict_pair_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Returns blended ensemble probability of match (class 1)."""
        if not self.is_fitted or not self.lgb_models:
            raise ValueError("Model is not fitted yet!")
        df_X = pd.DataFrame(X, columns=self.feature_names)

        lgb_preds = np.zeros(len(X), dtype=np.float32)
        for m in self.lgb_models:
            lgb_preds += m.predict_proba(df_X)[:, 1]
        lgb_preds /= len(self.lgb_models)

        if self.cb_models:
            cb_preds = np.zeros(len(X), dtype=np.float32)
            for m in self.cb_models:
                cb_preds += m.predict_proba(df_X)[:, 1]
            cb_preds /= len(self.cb_models)
            return 0.5 * lgb_preds + 0.5 * cb_preds

        return lgb_preds

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

