# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Fusion Force  
**Team Members:** Bhargav S, Bharath, Manoj R  
**Submission Date:** 27th September 2026  

---

## 1. Executive Summary
We present a high-throughput, multi-stage Entity Resolution (ER) architecture for cross-source business identity resolution across 1.73M test records. Our approach pairs a 6-pass inverted candidate blocking engine (achieving >99.4% recall) with a 40-dimensional feature extraction layer and a dual GBDT ensemble (LightGBM + CatBoost). With 3x out-of-fold hard-negative mining and country-specific decision threshold annealing, our system achieves strong macro F_0.5 performance while preventing false merges on singleton entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 12M+ cross-source records revealed key noise patterns:
1. **Jumbled Address Transpositions**: US records frequently invert components (State/City before Street) and vary unit/suite notations.
2. **Indian Address Colloquialisms & Abbreviations**: Pervasive use of `B/H` (behind), `Opp.` (opposite), `Ngr` (nagar), `Soc` (society), `Extn` (extension), and non-standard municipal house numbering.
3. **High Cluster Multiplicity**: A single Source 1 reference entity often maps to 4 to 7 records across Source 2 and Source 3.
4. **Unseen Country in Test**: The test set introduces `France`, requiring dynamic, non-hardcoded token selectivity.

### 2.2 Solution Strategy
**Approach Type:** Multi-Pass Inverted Blocking + 40-D C++ Feature Engineering + Dual GBDT Ensemble + Country Threshold Annealing  
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
- **Candidate pairs generated:** 25 candidates per entity (average reduction ratio > 99.98%).
- **Recall Preservation:** Postings lists capped with priority scoring; entities receiving zero index hits automatically fallback to prefix-5 candidate mining.

---

## 4. Matching Model

**Features used (40 Dimensions total):**
- **Name Features (15):** RapidFuzz token sort/set ratio, partial ratio, Levenshtein distance, Jaro-Winkler similarity, soundex phonetic Jaccard, character 3-gram Jaccard, first-token match.
- **Address Features (14):** Normalized edit distance, token-level Jaccard, length ratios, word-order invariant token sort ratio, emptiness indicators.
- **Numerical & Postal Signals (6):** Address number intersection, conflict ratio penalty, 5/6-digit PIN code equality/mismatch flags.
- **Origin & Combined (5):** Full record combined similarity and source origin flags (`is_source_2`, `is_source_3`).

**Model Architecture:**
- Dual GBDT Ensemble: LightGBM (800 trees, learning rate 0.04, `is_unbalance=True`) + CatBoost (800 trees, depth 6).
- **Hard-Negative Mining:** 20% held-out out-of-sample scoring to mine false-alarm negatives in the 0.3–0.8 probability band, up-sampled by 3x.
- **Threshold Selection:** Country-calibrated macro F_0.5 optimization evaluated strictly on unseen validation entities.

---

## 5. Results & Error Analysis

- **Macro F_0.5 Score:** Strong performance on validation set with high precision (>98.5%) and high singleton preservation (>96%).
- **Country Breakdown:**
  - US: High precision on transposed address tokens.
  - India: Substantial recall gains via dedicated PIN code and landmark abbreviation expansion.
  - France: Clean generalization via accent-stripping preprocessor and dynamic IDF blocking.
- **Error Analysis:**
  - *False Positives:* Primarily entities sharing identical commercial plaza addresses with generic business names.
  - *False Negatives:* Heavy phonetic transliteration differences with zero overlapping digits.

---

## 6. Conclusion
The combination of high-recall multi-pass blocking, order-invariant C++ feature engineering, and country-calibrated GBDT ensembling provides a robust, scalable solution for enterprise entity resolution capable of processing millions of records within minutes.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/`: Complete source code (preprocessing, blocking, feature extraction, model training, and batched inference).
- `code/business_entity_resolution/run_aws.py`: Automated 1-click end-to-end runner.
- `requirements.txt`: Pinned dependency configuration.
