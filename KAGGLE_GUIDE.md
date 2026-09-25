# Kaggle Execution Guide: Amazon ML Challenge 2026

This guide explains how to run the pipeline on **Kaggle Notebooks (30 GB RAM)** step-by-step with **persistent caching** and **real-time percentage tracking**.

---

## 🚀 Step 0: Pull Latest Code in Kaggle
Run this cell in Kaggle to pull the updated fast blocker with real-time percentage progress logs:

```python
%cd /kaggle/working/Amazon-ml-challenge-2026
!git checkout Bharath-solution
!git pull origin Bharath-solution
!pip install -r code/business_entity_resolution/requirements.txt
```

---

## 📌 Step-by-Step Execution with Caching & Live Percentage Updates

### 🧹 Cell 1: Stage 1 - Preprocessing (Already Completed in your notebook!)
```python
# Cleans 24M records and saves to fast Parquet cache (~4 mins)
!python -u code/business_entity_resolution/src/stage1_preprocess.py \
  --train-dir "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/train" \
  --test-dir "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/test" \
  --cache-dir "processed_data"
```

### 🔍 Cell 2: Stage 2 - Candidate Blocking (~2.5 mins with live % progress)
```python
# Builds token indices and generates candidate pairs with real-time percentages
!python -u code/business_entity_resolution/src/stage2_blocking.py \
  --cache-dir "processed_data" \
  --output-dir "output" \
  --candidates-cache-dir "cache"
```

### 🧠 Cell 3: Stage 3 - Train LightGBM Model & Calibrate Threshold (~2 mins)
```python
# Trains LightGBM, finds optimal F_0.5 threshold, and saves to models/
!python -u code/business_entity_resolution/src/stage3_train.py \
  --cache-dir "processed_data" \
  --candidates-cache-dir "cache" \
  --models-dir "models" \
  --gt-file "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/train/train_ground_truth.tsv"
```

### 🚀 Cell 4: Stage 4 - Batched Test Inference & TSV Export (~2 mins)
```python
# Runs batched C++ string scoring on test candidates and outputs matching_results.tsv
!python -u code/business_entity_resolution/src/stage4_inference.py \
  --cache-dir "processed_data" \
  --candidates-cache-dir "cache" \
  --models-dir "models" \
  --output-dir "output"
```

---

## 📋 Step 5: Validate Submission Outputs
```python
!python /kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir "/kaggle/input/datasets/pes2ug23cs121/ml-challenge-data/student_resource/dataset/test"
```
👉 *Target output: `PASS (exit 0)`.*

---

## 📥 Step 6: Download Predictions
```python
from IPython.display import FileLink
display(FileLink("output/matching_results.tsv"))
display(FileLink("output/candidate_pairs.tsv"))
```
*(Or right-click `output/matching_results.tsv` in the Kaggle file explorer on the right panel and click Download).*

