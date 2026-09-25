# Kaggle Execution Guide: Amazon ML Challenge 2026

This guide explains how to run the full training & inference pipeline on **Kaggle Notebooks** with **30 GB RAM and 4 vCPUs for free**.

---

## 🚀 Why Use Kaggle for this Challenge?
* **30 GB High-Speed RAM:** Easily loads and processes all 24.2 million records without memory crashes.
* **4 vCPUs & Free T4 GPU:** Parallel C++ `rapidfuzz` feature extraction runs in minutes.
* **Persistent Sessions:** Runs in the background without depending on your local machine.

---

## 📋 Step-by-Step Instructions

### Step 1: Push Code to your `Bharath-solution` Branch
On your local terminal, commit and push your code to your branch:

```bash
git checkout -B Bharath-solution
git add code/ HOW_TO_RUN.md README.md .gitignore package_submission.py KAGGLE_GUIDE.md
git commit -m "feat: Memory-optimized entity resolution pipeline for Kaggle & Local"
git push -u origin Bharath-solution
```

---

### Step 2: Upload Dataset to Kaggle (One-time Setup)
1. Zip your `student_resource` folder on your laptop:
   * Select `student_resource` folder $\rightarrow$ Right Click $\rightarrow$ Compress to ZIP (`student_resource.zip`).
2. Go to **[kaggle.com/datasets](https://www.kaggle.com/datasets)** $\rightarrow$ Click **New Dataset**.
3. Upload `student_resource.zip`, title it **`amazon-ml-2026-data`**, and set visibility to **Private**.
4. Click **Create**.

---

### Step 3: Create a Kaggle Notebook
1. Go to **[kaggle.com/code](https://www.kaggle.com/code)** $\rightarrow$ Click **New Notebook**.
2. On the right-side Settings panel:
   * **Accelerator:** GPU T4 x2 (or None / Standard CPU)
   * **Internet:** Turn **ON**
3. Click **+ Add Data** $\rightarrow$ Search for your dataset `amazon-ml-2026-data` $\rightarrow$ Click **Add**.

---

### Step 4: Run the Pipeline in Kaggle Cells

#### 📌 Cell 1: Clone Your Repository & Branch
```python
# Clone your specific branch
!git clone -b Bharath-solution https://github.com/bhargav1520/Amazon-ml-challenge-2026.git
%cd Amazon-ml-challenge-2026

# Install required packages
!pip install -r code/business_entity_resolution/requirements.txt
```

#### 📌 Cell 2: Locate Dataset & Run Pipeline
```python
# If dataset is in /kaggle/input/amazon-ml-2026-data/student_resource
import os

data_dir = "/kaggle/input/amazon-ml-2026-data/student_resource"
if not os.path.exists(data_dir):
    # Search for dataset path automatically
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_source1.tsv" in files:
            data_dir = os.path.dirname(os.path.dirname(root))
            break

print(f"Using Dataset Directory: {data_dir}")

!python code/business_entity_resolution/src/pipeline.py \
  --train-dir {data_dir}/dataset/train \
  --test-dir {data_dir}/dataset/test \
  --output-dir output
```

#### 📌 Cell 3: Validate Output Files
```python
!python {data_dir}/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir {data_dir}/dataset/test
```

#### 📌 Cell 4: Download Submission Files
```python
from IPython.display import FileLink
display(FileLink("output/matching_results.tsv"))
display(FileLink("output/candidate_pairs.tsv"))
```
*(Or download directly from Kaggle's `/kaggle/working/Amazon-ml-challenge-2026/output` file browser on the right panel).*
