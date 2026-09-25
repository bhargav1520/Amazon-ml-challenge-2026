# Kaggle Execution Guide: Amazon ML Challenge 2026

This guide explains how to run the pipeline on **Kaggle Notebooks (30 GB RAM)** either **all-at-once** OR **step-by-step with persistent caching**.

---

## 🚀 Step 1: Clone Repository & Install Dependencies
Run this in the first Kaggle cell:

```python
!git clone -b Bharath-solution https://github.com/bhargav1520/Amazon-ml-challenge-2026.git
%cd Amazon-ml-challenge-2026
!git pull

!pip install -r code/business_entity_resolution/requirements.txt
```

---

## 📌 Option A: Step-by-Step Execution with Caching (Recommended!)
If you run step-by-step, the preprocessed data and models are **saved to disk**. If anything stops, you never repeat earlier steps!

### 🧹 Cell 1: Stage 1 - Preprocessing (Run ONCE)
```python
# Cleans 24M records and saves to fast Parquet cache (~4 mins)
!python code/business_entity_resolution/src/stage1_preprocess.py \
  --train-dir "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/train" \
  --test-dir "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/test" \
  --cache-dir "processed_data"
```

### 🔍 Cell 2: Stage 2 - Candidate Blocking (Run ONCE)
```python
# Builds token indices and generates candidate pairs (~4 mins)
!python code/business_entity_resolution/src/stage2_blocking.py \
  --cache-dir "processed_data" \
  --output-dir "output" \
  --candidates-cache-dir "cache"
```

### 🧠 Cell 3: Stage 3 - Train LightGBM Model & Calibrate Threshold
```python
# Trains LightGBM, finds optimal F_0.5 threshold, and saves to models/ (~3 mins)
!python code/business_entity_resolution/src/stage3_train.py \
  --cache-dir "processed_data" \
  --candidates-cache-dir "cache" \
  --models-dir "models" \
  --gt-file "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/train/train_ground_truth.tsv"
```

### 🚀 Cell 4: Stage 4 - Batched Test Inference & TSV Export
```python
# Runs batched inference on test candidates and outputs matching_results.tsv (~3 mins)
!python code/business_entity_resolution/src/stage4_inference.py \
  --cache-dir "processed_data" \
  --candidates-cache-dir "cache" \
  --models-dir "models" \
  --output-dir "output"
```

---

## 📌 Option B: Run All Stages Together in One Command
```python
!python code/business_entity_resolution/src/pipeline.py \
  --train-dir "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/train" \
  --test-dir "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/test" \
  --output-dir "output"
```

---

## 📋 Step 3: Validate Outputs Before Submitting
```python
!python /kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/test"
```
👉 *Target output: `PASS (exit 0)`.*

---

## 📥 Step 4: Download Predictions
```python
from IPython.display import FileLink
display(FileLink("output/matching_results.tsv"))
display(FileLink("output/candidate_pairs.tsv"))
```
*(Or right-click `output/matching_results.tsv` in the Kaggle file explorer on the right panel and click Download).*
