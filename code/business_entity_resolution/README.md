# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

**Team Name:** Fusion Force  
**Team Members:** Bhargav S, Bharath, Manoj R  
**Solution Repository:** `code/business_entity_resolution`  

---

## 1. Project Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources, each contributing partial, noisy fragments of information about the same real-world entities. 

This repository contains the complete, self-contained, and production-grade Machine Learning solution developed by team **Fusion Force** for the **Amazon ML Challenge 2026**. The pipeline performs high-throughput cross-source entity resolution across **1.73 million test entities** (~14 million evaluated candidate pairs) with high precision and strong macro $F_{0.5}$ score performance.

---

## 2. Directory & File Structure

The solution strictly adheres to the official competition directory specification:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv            # Final entity matches (Leaderboard scored)
│   └── candidate_pairs.tsv             # Blocking candidate set (Evaluator audited)
├── code/
│   └── business_entity_resolution/
│       ├── src/                        # Modular pipeline source code
│       │   ├── __init__.py             # Package marker
│       │   ├── preprocessor.py         # International text & address normalization
│       │   ├── data_loader.py          # High-speed TSV readers & format validators
│       │   ├── blocking.py             # 6-Pass multi-index inverted candidate blocker
│       │   ├── feature_engineering.py  # 40-D C++ RapidFuzz & phonetic feature extraction
│       │   ├── model.py                # Dual ensemble classifier (LightGBM + CatBoost)
│       │   ├── postprocessing.py       # Country thresholding & barrier filtering
│       │   ├── local_evaluator.py      # Macro F_0.5 scoring engine with singleton support
│       │   ├── pipeline.py             # End-to-end monolithic runner
│       │   ├── stage1_preprocess.py    # Stage 1: Data cleaning & columnar Parquet caching
│       │   ├── stage2_blocking.py      # Stage 2: Inverted candidate pair generation
│       │   ├── stage3_train.py         # Stage 3: Out-of-fold training & threshold search
│       │   └── stage4_inference.py     # Stage 4: Batched test inference & TSV export
│       ├── tests/                      # Automated test suite (21 unit/integration tests)
│       ├── run_aws.py                  # High-performance 1-click cloud execution runner
│       ├── requirements.txt            # Pinned environment dependencies
│       └── README.md                   # System documentation and execution guide
└── Documentation_template.md           # 1-2 page official methodology report
```

---

## 3. Pipeline Architecture & Technical Modules

```
[Raw TSV Sources] ──> [Stage 1: Preprocess] ──> [Stage 2: 6-Pass Blocking]
                                                        │
[Final TSV Outputs] <── [Stage 4: Batched Inference] <── [Stage 3: 40-D GBDT Ensemble]
```

### Module 1: Preprocessing & Text Normalization (`src/preprocessor.py`)
* **Legal Suffix Canonicalization:** Standardizes corporate suffixes (`LLC`, `Pvt Ltd`, `SARL`, `SAS`, `GmbH`) without aggressive over-normalization that destroys distinct corporate identities.
* **Indian Landmark Expansion:** Normalizes regional abbreviations (`B/H` $\rightarrow$ `behind`, `Opp.` $\rightarrow$ `opposite`, `Ngr` $\rightarrow$ `nagar`, `Soc` $\rightarrow$ `society`, `Extn` $\rightarrow$ `extension`, `Sec` $\rightarrow$ `sector`, `Marg`, `Taluk`).
* **International Support:** Strips diacritics and accents (Unicode `NFKD`) to seamlessly support open-set test records (including `France`).
* **Columnar Storage:** Exports optimized `.parquet` tables with Snappy compression for zero-copy memory mapping.

### Module 2: 6-Pass High-Recall Inverted Blocker (`src/blocking.py`)
Reduces the search space from **3.8 trillion Cartesian pairs** to high-confidence candidates:
1. **Rare Name Tokens:** Inverted index with dynamic IDF threshold ($\text{max\_fraction} \le 0.005$).
2. **Name Bigrams:** Captures adjacent word relationships.
3. **Rare Address Tokens:** Matches locality and street names.
4. **Number Index:** Extracts street numbers, unit numbers, and 5/6-digit PIN codes.
5. **Character 3/4-Gram Shingles:** Phonetic and spelling typo resilience.
6. **Prefix-5 Emergency Fallback:** Direct 5-character prefix matching preventing zero-candidate drops on rare transliterations.

### Module 3: 40-Dimensional Feature Engineering (`src/feature_engineering.py`)
Computes pairwise similarity vectors using C++ RapidFuzz engines:
* **Name Metrics (15):** Token Sort Ratio, Token Set Ratio, Partial Ratio, Levenshtein, Jaro-Winkler, Soundex Phonetic Jaccard, Char 3-gram Jaccard, First-token match.
* **Address Metrics (14):** Order-invariant token Jaccard, length ratios, normalized edit distances, emptiness flags.
* **Numerical & Postal Signals (6):** Address number intersection, conflict ratio penalties, PIN code matching.
* **Origin & Combined (5):** Full record combined text similarity and source origin flags.

### Module 4: Dual Ensemble & Decision Thresholding (`src/model.py`, `src/postprocessing.py`)
* **Model:** LightGBM ($600$ trees, learning rate $0.05$, `is_unbalance=True`) paired with CatBoost ($600$ trees, depth $6$).
* **Hard-Negative Mining:** Held-out out-of-sample scoring mining false alarms in the $0.30 – 0.80$ probability band, up-sampled by $3\times$.
* **Country Threshold Calibration:** Evaluates Macro $F_{0.5}$ across held-out validation entities with adaptive score barrier margins (`margin=0.08`, `max_matches=12`).

---

## 4. Setup & Installation

### Step 1: Environment Setup
Python 3.9+ is recommended. Create an isolated virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Step 2: Install Pinned Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. End-to-End Execution Guide

### Option A: 1-Click Automated Runner (Recommended)
Executes all 4 stages sequentially, verifies outputs, and validates submission rules:
```bash
python3 run_aws.py
```

### Option B: Modular Stage-by-Stage Execution

#### Stage 1: Preprocessing
```bash
python -m src.stage1_preprocess \
  --train-dir ../../student_resource/dataset/train \
  --test-dir ../../student_resource/dataset/test \
  --cache-dir processed_data
```

#### Stage 2: Candidate Blocking
```bash
python -m src.stage2_blocking \
  --cache-dir processed_data \
  --output-dir output \
  --candidates-cache-dir cache \
  --max-candidates 80 \
  --max-train-sample 150000
```

#### Stage 3: Model Training & Threshold Search
```bash
python -m src.stage3_train \
  --cache-dir processed_data \
  --candidates-cache-dir cache \
  --models-dir models \
  --gt-file ../../student_resource/dataset/train/train_ground_truth.tsv \
  --max-train-entities 150000 \
  --n-estimators 600 \
  --learning-rate 0.05 \
  --hard-negative-multiplier 3
```

#### Stage 4: Test Inference & Submission Generation
```bash
python -m src.stage4_inference \
  --cache-dir processed_data \
  --candidates-cache-dir cache \
  --models-dir models \
  --output-dir output \
  --batch-size 100000
```

---

## 6. Automated Testing & Verification

Run the full pytest suite (21 automated unit and integration tests):
```bash
pytest tests/ -v
```

Execute the official submission validator:
```bash
python ../../student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir ../../student_resource/dataset/test
```

---

## 7. Submission Package Generator

Generate the final official submission archive (`Fusion_Force_submission.zip`):
```bash
python package_submission.py --team-name "Fusion_Force"
```

---

## 8. Compliance & Constraints

* **License:** Fully open-source Apache 2.0 / MIT compliant (LightGBM, CatBoost, RapidFuzz, Scikit-Learn).
* **Model Parameters:** Well within the 8 Billion parameter ceiling (~120,000 tree parameters total).
* **Format:** Strictly tab-separated (`.tsv`) with UTF-8 encoding.
