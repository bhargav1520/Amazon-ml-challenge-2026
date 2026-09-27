# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Fusion Force  
**Team Members:** Bhargav S, Bharath, Manoj R  
**Submission Date:** 27th September 2026  

---

## 1. Executive Summary
We present a high-throughput, multi-stage Entity Resolution (ER) architecture for cross-source business identity resolution across 1.73M test records. Our approach pairs a 6-pass inverted candidate blocking engine (achieving >99.4% candidate recall) with a 40-dimensional feature extraction layer and a dual GBDT ensemble (LightGBM + CatBoost). With 3x out-of-fold hard-negative mining and country-specific decision threshold calibration, our system achieves high precision (98.24%) and reliable singleton identification (89.65%) while scaling efficiently across millions of record pairs.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 12M+ cross-source records revealed key noise patterns:
1. **Jumbled Address Transpositions**: US records frequently invert components (State/City before Street) and vary unit/suite notations.
2. **Indian Address Colloquialisms & Abbreviations**: Pervasive use of `B/H` (behind), `Opp.` (opposite), `Ngr` (nagar), `Soc` (society), `Extn` (extension), and non-standard municipal house numbering.
3. **High Cluster Multiplicity**: A single Source 1 reference entity often maps to 4 to 7 records across Source 2 and Source 3.
4. **Unseen Country in Test**: The test set introduces `France`, requiring dynamic, non-hardcoded token selectivity.

### 2.2 Solution Strategy
**Approach Type:** Multi-Pass Inverted Blocking + 40-D C++ Feature Engineering + Dual GBDT Ensemble + Country Threshold Calibration  
**Core Innovation:** A 6-pass multi-indexer with IDF selectivity filtering and emergency prefix fallback ensures high candidate recall, while held-out out-of-sample hard-negative mining prevents tree overconfidence on difficult negative pairs.

---

## 3. Candidate Generation (Blocking)

To reduce the comparison space from 3.8 trillion Cartesian pairs to a clean candidate matrix:
- **Blocking keys used:**
  1. Rare Name Tokens (Dynamic IDF threshold $\le 0.005$)
  2. Character Bigrams of discriminative tokens
  3. Rare Address Tokens
  4. Street Numbers & 5/6-digit postal/PIN codes
  5. Character 3-gram and 4-gram shingles
  6. Emergency 5-character alphanumeric prefix index (handles zero-token transliterations)
- **Candidate pairs generated:** 80 candidates per entity across 1.73M test entities (average reduction ratio > 99.98%). Zero-candidate entities: 0.
- **Recall Preservation:** Postings lists capped with priority scoring; entities receiving zero index hits automatically fallback to prefix-5 candidate mining.

---

## 4. Matching Model

**Features used (40 Dimensions total):**
- **Name Features (15):** RapidFuzz token sort/set ratio, partial ratio, Levenshtein distance, Jaro-Winkler similarity, soundex phonetic Jaccard, character 3-gram Jaccard, first-token match.
- **Address Features (14):** Normalized edit distance, token-level Jaccard, length ratios, word-order invariant token sort ratio, emptiness indicators.
- **Numerical & Postal Signals (6):** Address number intersection, conflict ratio penalty, 5/6-digit PIN code equality/mismatch flags.
- **Origin & Combined (5):** Full record combined similarity and source origin flags (`is_source_2`, `is_source_3`).

**Model Architecture:**
- Dual GBDT Ensemble: LightGBM (600 trees, learning rate 0.05, `is_unbalance=True`, `max_depth=7`) + CatBoost (600 trees, depth 6).
- **Training Matrix:** 12,223,402 total pairs mined across 150,000 Source-1 entities with 3x hard-negative augmentation.
- **Hard-Negative Mining:** 20% held-out out-of-sample scoring to mine false-alarm negatives in the 0.3–0.8 probability band, up-sampled by 3x.
- **Threshold Selection:** Country-calibrated macro F_0.5 optimization evaluated strictly on 25,000 held-out validation entities.

---

## 5. Results & Error Analysis

- **Macro F_0.5 Score:** **0.7765** on 25,000 held-out validation entities.
- **Micro Precision:** **98.24%** (952 False Positives).
- **Micro Recall:** **61.26%** (33,594 False Negatives).
- **Singleton Accuracy:** **89.65%** (1,238 / 1,381).
- **Country Breakdown:**
  - US: Macro F_0.5 = **0.8101** (15,029 entities).
  - India: Macro F_0.5 = **0.7259** (9,971 entities).
- **Test Output Distribution (1,732,544 total entities):**
  - Singletons (no match): **279,741** (16.1%)
  - Entities with matches: **1,452,803** (83.9%)
  - Mean matches per entity: **2.20** (Max: 12 matches)
- **Error Analysis:**
  - *False Positives:* Primarily entities sharing identical commercial plaza addresses with generic business names.
  - *False Negatives:* Conservative threshold (`US: 0.81`, `India: 0.71`) prioritized precision over recall for borderline phonetic transliterations.

---

## 6. Conclusion
The combination of high-recall multi-pass blocking, order-invariant C++ feature engineering, and country-calibrated GBDT ensembling provides a robust, scalable solution for enterprise entity resolution capable of processing millions of records within minutes.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/`: Complete source code (preprocessing, blocking, feature extraction, model training, and batched inference).
- `code/business_entity_resolution/run_aws.py`: Automated 1-click end-to-end runner.
- `requirements.txt`: Pinned dependency configuration.
